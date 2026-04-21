import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize

# =============================================================================
# --- 1. PARÁMETROS GLOBALES ---
# =============================================================================
n_levels = 9         # Número de niveles macroscópicos (n)
N_spins = 8          # Número de espines equivalentes
D_total = 2**N_spins # Dimensión total del espacio de Hilbert
beta = 1.0           # Temperatura inversa
gamma = 1.0          # Constante de acoplamiento al baño

# =============================================================================
# --- 2. FUNCIÓN OBJETIVO (FÍSICA) ---
# =============================================================================
import numpy as np
from scipy.special import comb
from scipy.optimize import minimize

def calc_chi_only_C(x, n, beta):
    """
    x es un vector de longitud (n-1) con las energías E_2, ..., E_n (E_1 = 0).
    Las degeneraciones g están fijadas por la topología ATA / Espines libres.
    """
    # 1. Desempaquetar variables (¡Solo Energías!)
    E = np.zeros(n)
    E[1:] = x
    
    # 2. Fijar las degeneraciones como la distribución binomial (Física del ATA)
    N_spins = n - 1
    g = np.array([comb(N_spins, i) for i in range(n)])
    
    # --- Bloque Termodinámico (Capacidad Calorífica C) ---
    # Evitar desbordamientos restando el mínimo (opcional pero seguro)
    E_shifted = E - np.min(E) 
    
    Z = np.sum(g * np.exp(-beta * E_shifted))
    P = (g * np.exp(-beta * E_shifted)) / Z
    
    E_mean = np.sum(P * E)
    E2_mean = np.sum(P * E**2)
    C = (beta**2) * (E2_mean - E_mean**2)
    
    # Hemos comentado todo el bloque dinámico de la matriz M
    # porque ahora solo nos interesa C.
    
    # Devolvemos -C para maximizar la capacidad calorífica
    return -C

# =============================================================================
# --- 3. RESTRICCIONES Y LÍMITES ---
# =============================================================================
# Límite inferior para las energías: E_i >= 0
bounds = [(0, None) for _ in range(n_levels - 1)]

# Restricciones
constraints = []

# ELIMINADA la restricción de sum(g) porque 'g' ya está fijada analíticamente.

# 1. Energías ordenadas: E_3 - E_2 >= 0, etc.
for i in range(n_levels - 2):
    constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i+1] - x[i]})

# =============================================================================
# --- 4. MOTOR DE OPTIMIZACIÓN MULTI-START ---
# =============================================================================
print(f"Iniciando optimización SÓLO de ENERGÍAS para n={n_levels} niveles y N={N_spins} (D={D_total})...\n")

best_x = None
best_C = -np.inf # Inicializamos el mejor valor de la Capacidad Calorífica
num_starts = 20  # Número de intentos con diferentes semillas iniciales

for attempt in range(num_starts):
    # Generar un punto inicial aleatorio SOLO para las n-1 energías
    x0_guess = np.sort(np.random.uniform(0.1, 5.0, n_levels - 1))
    
    res = minimize(calc_chi_only_C, x0_guess, args=(n_levels, beta), 
                   method='SLSQP', bounds=bounds, constraints=constraints)
    
    if res.success:
        C_val = -res.fun # Recuperamos el valor positivo de C
        if C_val > best_C:
            best_C = C_val
            best_x = res.x

# =============================================================================
# --- 5. RESULTADOS ---
# =============================================================================
if best_x is not None:
    # 1. Reconstruir las energías
    E_opt = np.zeros(n_levels)
    E_opt[1:] = best_x
    
    # 2. Reconstruir las degeneraciones fijas (Distribución Binomial)
    from scipy.special import comb
    g_opt = np.array([comb(N_spins, i) for i in range(n_levels)])
    
    print("=== RESULTADO ÓPTIMO ENCONTRADO ===")
    print(f"Capacidad Calorífica Máxima (C): {best_C:.6f}\n")
    
    print("Espectro de Energías (E):")
    for i, e in enumerate(E_opt):
        print(f"  E_{i+1} = {e:.4f}")
        
    print("\nDegeneraciones fijadas por modelo (g):")
    for i, g in enumerate(g_opt):
        print(f"  g_{i+1} = {g:.0f}")
        
    print(f"\nSuma total de g: {np.sum(g_opt):.0f} (Debe ser {D_total})")
    
    # Extra: Imprimir los gaps para ver si es una escalera perfecta (Espines libres) o parábola (ATA)
    print("\nGaps de Energía (Delta E):")
    for i in range(n_levels - 1):
        print(f"  E_{i+2} - E_{i+1} = {E_opt[i+1] - E_opt[i]:.4f}")
else:
    print("El optimizador no logró converger en ninguno de los intentos.")