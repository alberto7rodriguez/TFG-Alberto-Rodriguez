import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize
from scipy.special import comb

# =============================================================================
# --- 1. PARÁMETROS GLOBALES ---
# =============================================================================
N = 4            # Número de espines
n_levels = N + 1 # Niveles: 0, 1, 2, 3, 4 espines arriba
beta = 1.0
gamma = 1.0

# Fijamos g_i y K_ij usando las reglas matemáticas de espines libres
g_fixed = np.array([comb(N, k) for k in range(n_levels)])

K_fixed = np.zeros(n_levels - 1)
for k in range(n_levels - 1):
    K_fixed[k] = g_fixed[k] * (N - k)

print(f"Sistema de {N} Espines Libres:")
print(f"Degeneraciones (g) : {g_fixed}")
print(f"Conectividades (K) : {K_fixed}\n")

# =============================================================================
# --- 2. FUNCIÓN OBJETIVO (Solo optimiza Energías) ---
# =============================================================================
def calc_chi_independent(E_vars, n, g, K, beta, gamma):
    # E_vars tiene longitud n-1. E[0] = 0.
    E = np.zeros(n)
    E[1:] = E_vars
    
    # 1. Termodinámica (C)
    Z = np.sum(g * np.exp(-beta * E))
    P = (g * np.exp(-beta * E)) / Z
    
    E_mean = np.sum(P * E)
    E2_mean = np.sum(P * E**2)
    C = (beta**2) * (E2_mean - E_mean**2)
    
    if C < 1e-12: return 0.0
    
    # 2. Dinámica (tau)
    M = np.zeros((n, n))
    for i in range(n - 1):
        delta_E = E[i+1] - E[i]
        
        Gamma_up = gamma * (K[i] / g[i]) * (1.0 / (1.0 + np.exp(beta * delta_E)))
        Gamma_down = gamma * (K[i] / g[i+1]) * (1.0 / (1.0 + np.exp(-beta * delta_E)))
        
        M[i+1, i] = Gamma_up
        M[i, i+1] = Gamma_down

    np.fill_diagonal(M, 0)
    np.fill_diagonal(M, -np.sum(M, axis=0))
    
    evals = la.eigvals(M).real
    evals = np.sort(evals)
    lambda_1 = np.abs(evals[-2])
    tau = 1.0 / (lambda_1 + 1e-15)
    
    chi = C / tau
    return -chi

# =============================================================================
# --- 3. OPTIMIZACIÓN ---
# =============================================================================
# Condición: Energías ordenadas (E_i <= E_{i+1})
constraints = []
for i in range(n_levels - 2):
    constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i+1] - x[i]})

bounds = [(0, None) for _ in range(n_levels - 1)]

best_chi = -np.inf
best_E = None

for attempt in range(10):
    E_guess = np.sort(np.random.uniform(0.1, 5.0, n_levels - 1))
    
    res = minimize(calc_chi_independent, E_guess, args=(n_levels, g_fixed, K_fixed, beta, gamma), 
                   method='SLSQP', bounds=bounds, constraints=constraints)
    
    if res.success and -res.fun > best_chi:
        best_chi = -res.fun
        best_E = res.x

# =============================================================================
# --- 4. RESULTADOS ---
# =============================================================================
if best_E is not None:
    E_opt = np.zeros(n_levels)
    E_opt[1:] = best_E
    
    print("=== RESULTADO ÓPTIMO MACROSCÓPICO ===")
    print(f"Chi Máximo (C/tau): {best_chi:.6f}\n")
    print("Energías Óptimas:")
    for i in range(n_levels):
        print(f"E_{i} = {E_opt[i]:.4f}")
        
    print("\nGaps de Energía (Delta E):")
    for i in range(n_levels - 1):
        print(f"E_{i+1} - E_{i} = {E_opt[i+1] - E_opt[i]:.4f}")
else:
    print("Fallo en la optimización.")