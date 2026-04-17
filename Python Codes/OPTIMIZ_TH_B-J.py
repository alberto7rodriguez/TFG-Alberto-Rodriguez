import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize

# =============================================================================
# --- 1. PARÁMETROS GLOBALES ---
# =============================================================================
N = 4                 # Número de espines (Mantenlo <= 6 para velocidad)
D = 2**N              # Dimensión del espacio de Hilbert
beta = 1.0            # Temperatura inversa
gamma = 1.0           # Tasa base de Glauber

num_B = N             # N campos magnéticos locales
num_J = N * (N - 1) // 2 # N(N-1)/2 interacciones (todos con todos)
num_vars = num_B + num_J

# Pre-computar los microestados (de 0 a 2^N - 1) en formato de espines (+1, -1)
# Esto acelera el cálculo de energías dentro del bucle de optimización.
spins = np.zeros((D, N))
for s in range(D):
    for i in range(N):
        # Extraer el bit i-ésimo del entero s: si es 1 -> +1, si es 0 -> -1
        spins[s, i] = 1 if (s & (1 << i)) else -1

# =============================================================================
# --- 2. FUNCIÓN OBJETIVO (FÍSICA EXACTA) ---
# =============================================================================
def calc_chi_micro(x, N, D, spins, beta, gamma):
    # 1. Desempaquetar parámetros
    B = x[:num_B]
    J = x[num_B:]
    
    # 2. Calcular la energía de cada uno de los 2^N microestados
    E = np.zeros(D)
    for s in range(D):
        # Contribución de los campos locales B_i
        E[s] += np.sum(B * spins[s])
        
        # Contribución de las interacciones J_ij
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                E[s] += J[idx] * spins[s, i] * spins[s, j]
                idx += 1
                
    # 3. Bloque Termodinámico (C)
    # Restamos el mínimo para evitar desbordamientos numéricos en exp()
    E_shifted = E - np.min(E)
    weights = np.exp(-beta * E_shifted)
    Z = np.sum(weights)
    P = weights / Z
    
    E_mean = np.sum(P * E)
    E2_mean = np.sum(P * E**2)
    C = (beta**2) * (E2_mean - E_mean**2)
    
    # Si la capacidad calorífica es nula, chi es nulo
    if C < 1e-12:
        return 0.0
    
    # 4. Bloque Dinámico (Construcción de la matriz de Glauber Exacta)
    M = np.zeros((D, D))
    
    for s in range(D):
        for i in range(N):
            # Encontrar el estado 's_new' volteando el espín 'i'
            s_new = s ^ (1 << i) # Operador XOR a nivel de bits
            
            delta_E = E[s_new] - E[s]
            
            # Tasa de Glauber local (física real, sin ansätze raros)
            # Limitamos delta_E para evitar overflow en exp
            delta_E_clip = np.clip(delta_E, -100, 100) 
            rate = gamma / (1.0 + np.exp(beta * delta_E_clip))
            
            M[s_new, s] = rate

    # Conservación de probabilidad en la diagonal
    np.fill_diagonal(M, 0)
    np.fill_diagonal(M, -np.sum(M, axis=0))
    
    # 5. Gap espectral
    evals = la.eigvals(M).real
    evals = np.sort(evals) # evals[-1] es ~0, evals[-2] es -lambda_1
    
    lambda_1 = np.abs(evals[-2])
    tau = 1.0 / (lambda_1 + 1e-15)
    
    chi = C / tau
    return -chi

# =============================================================================
# --- 3. OPTIMIZACIÓN L-BFGS-B ---
# =============================================================================
# Ponemos límites a las interacciones para que no diverjan hacia el infinito
# [-5, 5] es un rango energético físico razonable para beta=1
bounds = [(-5.0, 5.0) for _ in range(num_vars)]

print(f"Buscando topología óptima para N={N} espines (Matriz {D}x{D})...")

best_chi = -np.inf
best_x = None

# Multi-start para evitar mínimos locales
for attempt in range(15):
    # Punto de partida aleatorio
    x0 = np.random.uniform(-1.0, 1.0, num_vars)
    
    res = minimize(calc_chi_micro, x0, args=(N, D, spins, beta, gamma), 
                   method='L-BFGS-B', bounds=bounds)
    
    if res.success and -res.fun > best_chi:
        best_chi = -res.fun
        best_x = res.x
        print(f"Intento {attempt+1}: Nuevo máximo encontrado -> Chi = {best_chi:.5f}")

# =============================================================================
# --- 4. RESULTADOS FINALES ---
# =============================================================================
if best_x is not None:
    B_opt = best_x[:num_B]
    J_opt = best_x[num_B:]
    
    print("\n" + "="*40)
    print("=== MAPA DEL TERMÓMETRO ÓPTIMO ===")
    print("="*40)
    print(f"Chi Máximo (C/tau): {best_chi:.6f}\n")
    
    print("--- Campos Locales (B_i) ---")
    for i in range(N):
        print(f"B_{i+1} = {B_opt[i]:.4f}")
        
    print("\n--- Interacciones (J_ij) ---")
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            print(f"J_{i+1},{j+1} = {J_opt[idx]:.4f}")
            idx += 1
            
    # Analizar el espectro resultante
    print("\n--- Análisis de la red ---")
    J_mean_abs = np.mean(np.abs(J_opt))
    B_mean_abs = np.mean(np.abs(B_opt))
    
    if J_mean_abs < 0.1 * B_mean_abs:
        print(">> El sistema tiende fuertemente a ESPINES INDEPENDIENTES (J ~ 0).")
    else:
        print(">> El sistema ha construido un modelo INTERACTUANTE.")
else:
    print("Error: El optimizador no convergió.")