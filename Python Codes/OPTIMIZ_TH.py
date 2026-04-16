import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize

# =============================================================================
# --- 1. PARÁMETROS GLOBALES ---
# =============================================================================
n_levels = 7         # Número de niveles macroscópicos (n)
N_spins = 7          # Número de espines equivalentes
D_total = 2**N_spins # Dimensión total del espacio de Hilbert
beta = 1.0           # Temperatura inversa
gamma = 1.0          # Constante de acoplamiento al baño

# =============================================================================
# --- 2. FUNCIÓN OBJETIVO (FÍSICA) ---
# =============================================================================
def calc_chi(x, n, beta, gamma):
    """
    x es un vector de longitud (2n - 1).
    Los primeros (n-1) elementos son las energías E_2, ..., E_n (E_1 = 0).
    Los últimos n elementos son las degeneraciones g_1, ..., g_n.
    """
    # Desempaquetar variables
    E = np.zeros(n)
    E[1:] = x[:n-1]
    g = x[n-1:]
    
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
        
        # Tasa de subida (limitada por localidad)
        Gamma_up = gamma / (1.0 + np.exp(beta * delta_E))
        
        # Tasa de bajada (penalizada por la relación de degeneraciones)
        # Evitamos divisiones por cero añadiendo un pequeño epsilon a g[i+1]
        Gamma_down = gamma * (g[i] / (g[i+1] + 1e-10)) * (1.0 / (1.0 + np.exp(-beta * delta_E)))
        
        # Llenar la matriz M (M[j, i] es la transición de i hacia j)
        M[i+1, i] = Gamma_up
        M[i, i+1] = Gamma_down

    # Rellenar la diagonal para conservar la probabilidad
    np.fill_diagonal(M, 0)
    np.fill_diagonal(M, -np.sum(M, axis=0))
    
    # Encontrar el gap espectral
    evals = la.eigvals(M).real
    evals = np.sort(evals) # Ordenados de menor (más negativo) a mayor (~0)
    
    # El autovalor más grande es 0. El segundo más grande es -lambda_1
    lambda_1 = np.abs(evals[-2])
    
    tau = 1.0 / (lambda_1 + 1e-12) # Evitar división por cero
    
    chi = C / tau
    
    # Devolvemos -chi porque scipy.minimize siempre busca el MÍNIMO
    return -chi

# =============================================================================
# --- 3. RESTRICCIONES Y LÍMITES ---
# =============================================================================
# Límite inferior para evitar g_i = 0 estricto
eps = 1e-3 

# Límite para las variables: Energías >= 0, Degeneraciones >= eps
bounds = [(0, None) for _ in range(n_levels - 1)] + [(eps, D_total) for _ in range(n_levels)]

# Restricciones (Constraints)
constraints = []

# 1. Conservación del espacio de Hilbert: sum(g) - D_total = 0
constraints.append({
    'type': 'eq',
    'fun': lambda x: np.sum(x[n_levels-1:]) - D_total
})

# 2. Energías ordenadas: E_3 - E_2 >= 0, etc.
for i in range(n_levels - 2):
    constraints.append({
        'type': 'ineq',
        'fun': lambda x, i=i: x[i+1] - x[i]
    })

# =============================================================================
# --- 4. MOTOR DE OPTIMIZACIÓN MULTI-START ---
# =============================================================================
print(f"Iniciando optimización para n={n_levels} niveles y N={N_spins}, por tanto D={D_total} estados totales...\n")

best_chi = -np.inf
best_x = None
num_starts = 20 # Número de intentos con diferentes semillas iniciales

for attempt in range(num_starts):
    # Generar un punto inicial aleatorio (Initial Guess)
    E_guess = np.sort(np.random.uniform(0.1, 5.0, n_levels - 1))
    
    # Generar g_guess aleatorias que sumen D_total
    g_raw = np.random.uniform(1.0, 10.0, n_levels)
    g_guess = (g_raw / np.sum(g_raw)) * D_total 
    
    x0 = np.concatenate([E_guess, g_guess])
    
    # Ejecutar el optimizador
    res = minimize(calc_chi, x0, args=(n_levels, beta, gamma), 
                   method='SLSQP', bounds=bounds, constraints=constraints, 
                   options={'maxiter': 1000, 'ftol': 1e-7})
    
    if res.success:
        chi_val = -res.fun
        if chi_val > best_chi:
            best_chi = chi_val
            best_x = res.x

# =============================================================================
# --- 5. RESULTADOS ---
# =============================================================================
if best_x is not None:
    E_opt = np.zeros(n_levels)
    E_opt[1:] = best_x[:n_levels-1]
    g_opt = best_x[n_levels-1:]
    
    print("=== RESULTADO ÓPTIMO ENCONTRADO ===")
    print(f"Chi máximo (C/tau): {best_chi:.6f}\n")
    print("Espectro de Energías (E):")
    for i, e in enumerate(E_opt):
        print(f"  E_{i+1} = {e:.4f}")
        
    print("\nDegeneraciones (g):")
    for i, g in enumerate(g_opt):
        print(f"  g_{i+1} = {g:.4f}  (Idealmente entero)")
        
    print(f"\nSuma total de g: {np.sum(g_opt):.2f} (Debe ser {D_total})")
else:
    print("El optimizador no logró converger en ninguno de los intentos.")