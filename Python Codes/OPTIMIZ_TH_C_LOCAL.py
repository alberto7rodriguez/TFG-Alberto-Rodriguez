import numpy as np
from scipy.optimize import minimize

# =============================================================================
# --- 1. PARÁMETROS GLOBALES ---
# =============================================================================
N = 6                 # Número de espines (Mantenlo <= 6 para velocidad)
D = 2**N              # Dimensión del espacio de Hilbert
beta = 1.0            # Temperatura inversa

num_B = N             # N campos magnéticos locales
num_J = N * (N - 1) // 2 # N(N-1)/2 interacciones (todos con todos)
num_vars = num_B + num_J

# Pre-computar los microestados (de 0 a 2^N - 1) en formato de espines (+1, -1)
spins = np.zeros((D, N))
for s in range(D):
    for i in range(N):
        spins[s, i] = 1 if (s & (1 << i)) else -1

# =============================================================================
# --- 2. FUNCIÓN OBJETIVO (SOLO TERMODINÁMICA) ---
# =============================================================================
def calc_C_micro(x, N, D, spins, beta):
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
                
    # 3. Bloque Termodinámico (Capacidad Calorífica C)
    E_shifted = E - np.min(E)
    weights = np.exp(-beta * E_shifted)
    Z = np.sum(weights)
    P = weights / Z
    
    E_mean = np.sum(P * E)
    E2_mean = np.sum(P * E**2)
    C = (beta**2) * (E2_mean - E_mean**2)
    
    # Devolvemos -C para maximizar
    return -C

# =============================================================================
# --- 3. OPTIMIZACIÓN L-BFGS-B ---
# =============================================================================
# Límites a las interacciones. Como no hay penalización de tiempo, 
# el optimizador podría empujar los valores a los límites para maximizar el gap.
bounds = [(-10.0, 10.0) for _ in range(num_vars)]

print(f"Buscando topología óptima para MAXIMIZAR C con N={N} espines...")

best_C = -np.inf
best_x = None

# Multi-start para evitar mínimos locales
for attempt in range(20):
    # Punto de partida aleatorio
    x0 = np.random.uniform(-1.0, 1.0, num_vars)
    
    res = minimize(calc_C_micro, x0, args=(N, D, spins, beta), 
                   method='L-BFGS-B', bounds=bounds)
    
    if res.success and -res.fun > best_C:
        best_C = -res.fun
        best_x = res.x
        print(f"Intento {attempt+1}: Nuevo máximo encontrado -> C = {best_C:.5f}")

# =============================================================================
# --- 4. RESULTADOS FINALES ---
# =============================================================================
if best_x is not None:
    B_opt = best_x[:num_B]
    J_opt = best_x[num_B:]
    
    print("\n" + "="*40)
    print("=== MAPA DEL TERMÓMETRO ÓPTIMO (SOLO C) ===")
    print("="*40)
    print(f"Capacidad Calorífica Máxima (C): {best_C:.6f}\n")
    
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