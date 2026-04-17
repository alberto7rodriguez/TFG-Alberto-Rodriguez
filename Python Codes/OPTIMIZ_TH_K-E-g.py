import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize

# =============================================================================
# --- 1. PARÁMETROS GLOBALES ---
# =============================================================================
n_levels = 3         # Número de niveles macroscópicos (n)
N_spins = 5          # Número de espines equivalentes (N)
D_total = 2**N_spins # Dimensión total (D)
beta = 1.0           # Temperatura inversa
gamma = 1.0          # Constante de acoplamiento al baño

# =============================================================================
# --- 2. FUNCIÓN OBJETIVO (FÍSICA) ---
# =============================================================================
def calc_chi(x, n, N_spins, beta, gamma):
    """
    x es un vector de longitud (3n - 2):
      - [0 : n-1]     -> Energías E_2, ..., E_n (E_1 = 0)
      - [n-1 : 2n-1]  -> Degeneraciones g_1, ..., g_n
      - [2n-1 : 3n-2] -> Conectividades K_12, K_23, ..., K_{n-1, n}
    """
    E = np.zeros(n)
    E[1:] = x[:n-1]
    g = x[n-1 : 2*n-1]
    K = x[2*n-1 : 3*n-2]
    
    # --- A. Bloque Termodinámico (Capacidad Calorífica C) ---
    Z = np.sum(g * np.exp(-beta * E))
    P = (g * np.exp(-beta * E)) / Z
    
    E_mean = np.sum(P * E)
    E2_mean = np.sum(P * E**2)
    C = (beta**2) * (E2_mean - E_mean**2)
    
    # --- B. Bloque Dinámico (Tiempo de Relajación tau) ---
    M = np.zeros((n, n))
    
    for i in range(n - 1):
        delta_E = E[i+1] - E[i]
        
        # Tasa de subida y bajada usando el K_ij optimizado
        # Sumamos 1e-10 a g para evitar divisiones por 0
        Gamma_up = gamma * (K[i] / (g[i] + 1e-10)) * (1.0 / (1.0 + np.exp(beta * delta_E)))
        Gamma_down = gamma * (K[i] / (g[i+1] + 1e-10)) * (1.0 / (1.0 + np.exp(-beta * delta_E)))
        
        M[i+1, i] = Gamma_up
        M[i, i+1] = Gamma_down

    # Rellenar diagonal
    np.fill_diagonal(M, 0)
    np.fill_diagonal(M, -np.sum(M, axis=0))
    
    # Autovalores
    evals = la.eigvals(M).real
    evals = np.sort(evals)
    lambda_1 = np.abs(evals[-2])
    
    tau = 1.0 / (lambda_1 + 1e-12)
    
    chi = C / tau
    return -chi

# =============================================================================
# --- 3. RESTRICCIONES Y LÍMITES ---
# =============================================================================
eps = 1e-3 

# Límites: Energías >= 0, g_i >= eps, K_ij >= eps
bounds = [(0, None) for _ in range(n_levels - 1)] + \
         [(eps, D_total) for _ in range(n_levels)] + \
         [(eps, None) for _ in range(n_levels - 1)]

constraints = []

# 1. Conservación del espacio: sum(g) = D_total
constraints.append({'type': 'eq', 'fun': lambda x: np.sum(x[n_levels-1 : 2*n_levels-1]) - D_total})

# 2. Energías ordenadas: E_{i+1} >= E_i
for i in range(n_levels - 2):
    constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i+1] - x[i]})

# 3. Restricciones FÍSICAS de Localidad para K_ij
for i in range(n_levels - 1):
    # K_ij <= g_i * N
    constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[n_levels-1 + i] * N_spins - x[2*n_levels-1 + i]})
    # K_ij <= g_{i+1} * N
    constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[n_levels-1 + i+1] * N_spins - x[2*n_levels-1 + i]})
    # K_ij <= g_i * g_{i+1} (Límite topológico global)
    constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[n_levels-1 + i] * x[n_levels-1 + i+1] - x[2*n_levels-1 + i]})

# =============================================================================
# --- 4. OPTIMIZACIÓN MULTI-START ---
# =============================================================================
print(f"Iniciando optimización (E, g, K) para n={n_levels} y N={N_spins} espines...\n")

best_chi = -np.inf
best_x = None

for attempt in range(30):
    # Guesses aleatorios
    E_guess = np.sort(np.random.uniform(0.1, 5.0, n_levels - 1))
    
    g_raw = np.random.uniform(1.0, 10.0, n_levels)
    g_guess = (g_raw / np.sum(g_raw)) * D_total 
    
    # Guess para K_ij (un valor bajo para no romper restricciones de inicio)
    K_guess = np.random.uniform(0.1, 1.0, n_levels - 1)
    
    x0 = np.concatenate([E_guess, g_guess, K_guess])
    
    res = minimize(calc_chi, x0, args=(n_levels, N_spins, beta, gamma), 
                   method='SLSQP', bounds=bounds, constraints=constraints, 
                   options={'maxiter': 2000, 'ftol': 1e-7})
    
    if res.success and -res.fun > best_chi:
        best_chi = -res.fun
        best_x = res.x

# =============================================================================
# --- 5. RESULTADOS ---
# =============================================================================
if best_x is not None:
    E_opt = np.zeros(n_levels)
    E_opt[1:] = best_x[:n_levels-1]
    g_opt = best_x[n_levels-1 : 2*n_levels-1]
    K_opt = best_x[2*n_levels-1 : 3*n_levels-2]
    
    print("=== RESULTADO ÓPTIMO ENCONTRADO ===")
    print(f"Chi máximo (C/tau): {best_chi:.6f}\n")
    
    for i in range(n_levels):
        print(f"Nivel {i+1}: E = {E_opt[i]:.4f},  g = {g_opt[i]:.4f}")
        if i < n_levels - 1:
            # Calcular límite teórico para comparar
            limit = min(g_opt[i] * N_spins, g_opt[i+1] * N_spins, g_opt[i] * g_opt[i+1])
            print(f"   |-- K_{i+1}{i+2} = {K_opt[i]:.4f}  (Límite físico permitido: {limit:.4f})")
            
else:
    print("El optimizador no logró converger.")