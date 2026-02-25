#Direct calculations of the thermalization times for the Star Model

import numpy as np
import scipy.linalg as la
from scipy.special import comb

# --- 1. Your Data ---
a_values = [-0.711, 0.000, 0.894, 2.015, 3.398, 5.070, 7.052, 9.358, 11.998, 14.977, 18.297, 21.960, 25.967, 30.318, 35.013, 40.053, 45.438, 51.168, 57.243, 63.664, 70.431, 77.543, 85.001, 92.805, 100.956, 109.452, 118.294, 127.483, 137.018, 146.899, 157.127, 167.701, 178.621, 189.887, 201.500, 213.460, 225.766, 238.418, 251.417, 264.762, 278.453, 292.492, 306.876, 321.607, 336.685, 352.109, 367.879, 383.996, 400.460]
b_values = [0.711, 0.797, 0.894, 1.007, 1.133, 1.267, 1.410, 1.560, 1.714, 1.872, 2.033, 2.196, 2.361, 2.527, 2.693, 2.861, 3.029, 3.198, 3.367, 3.537, 3.707, 3.877, 4.048, 4.218, 4.389, 4.560, 4.732, 4.903, 5.075, 5.246, 5.418, 5.590, 5.762, 5.934, 6.106, 6.278, 6.450, 6.623, 6.795, 6.967, 7.140, 7.312, 7.485, 7.657, 7.830, 8.002, 8.175, 8.348, 8.520]

# --- 2. The General Matrix Builder ---
def build_star_model_matrix(N, a_list, b_list, beta=1.0, gamma=1.0):
    a = a_list[N - 2]
    b = b_list[N - 2]
    
    energies = np.zeros(2 * N)
    for k in range(N):
        energies[k] = -a                                 
        energies[k + N] = a + 2 * b * (2 * k - N + 1)    
        
    def f(delta_E):
        # Using np.clip to prevent overflow warnings for large energy gaps
        return 1.0 / (1.0 + np.exp(np.clip(beta * delta_E, -700, 700)))
        
    M = np.zeros((2 * N, 2 * N))
    
    for k in range(N):
        down_idx = k
        up_idx = k + N
        
        # Central Spin Flips
        delta_E_up = energies[up_idx] - energies[down_idx]
        M[up_idx, down_idx] = gamma * f(delta_E_up)     
        M[down_idx, up_idx] = gamma * f(-delta_E_up)    
        
        # Satellite Flips in DOWN manifold
        if k < N - 1:
            delta_E_k_plus = energies[down_idx + 1] - energies[down_idx]
            M[down_idx + 1, down_idx] = gamma * (N - 1 - k) * f(delta_E_k_plus)
        if k > 0:
            delta_E_k_minus = energies[down_idx - 1] - energies[down_idx]
            M[down_idx - 1, down_idx] = gamma * k * f(delta_E_k_minus)

        # Satellite Flips in UP manifold
        if k < N - 1:
            delta_E_k_plus = energies[up_idx + 1] - energies[up_idx]
            M[up_idx + 1, up_idx] = gamma * (N - 1 - k) * f(delta_E_k_plus)
        if k > 0:
            delta_E_k_minus = energies[up_idx - 1] - energies[up_idx]
            M[up_idx - 1, up_idx] = gamma * k * f(delta_E_k_minus)

    # Probability conservation (columns sum to 0)
    np.fill_diagonal(M, 0)
    np.fill_diagonal(M, -np.sum(M, axis=0))
    
    return M

# --- 3. Calculate Thermalization Times ---
target_N_values = [3, 5, 7, 9, 11, 13, 15]
thermalization_times = []

print(f"{'N':<5} | {'Thermalization Time (tau)':<30}")
print("-" * 40)

for N in target_N_values:
    # Build the matrix
    M = build_star_model_matrix(N, a_values, b_values)
    
    # Calculate eigenvalues
    # We use eigvals and take the real part, as stochastic matrices have real eigenvalues
    eigenvalues = np.real(la.eigvals(M))
    
    # Sort them in descending order (closest to 0 comes first)
    sorted_evals = np.sort(eigenvalues)[::-1]
    
    # sorted_evals[0] will be ~0. 
    # sorted_evals[1] is the spectral gap (lambda_1)
    lambda_1 = sorted_evals[1]
    
    # Calculate tau
    tau = -1.0 / lambda_1
    thermalization_times.append(tau)
    
    print(f"{N:<5} | {tau:<30.6f}")