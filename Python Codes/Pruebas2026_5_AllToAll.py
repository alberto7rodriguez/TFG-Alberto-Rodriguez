import numpy as np
import scipy.linalg as la
from scipy.special import comb

# --- 1. System Parameters ---
J_values = [0.7112, 0.496, 0.3769, 0.3019, 0.2506, 0.2135, 0.1856, 0.1638, 0.1464]
# Indices: N=2 is index 0. Therefore, index = N - 2.

# --- 2. The ATA Matrix Builder ---
def build_ata_matrix(N, J_list, beta=1.0, gamma=1.0):
    J = J_list[N - 2]
    
    # We have N + 1 macroscopic states (n = 0, 1, ..., N)
    energies = np.zeros(N + 1)
    degeneracies = np.zeros(N + 1)
    
    for n in range(N + 1):
        # Degeneracy is exactly N choose n
        degeneracies[n] = comb(N, n, exact=True)
        
        # Calculate energy based on your exact formula
        energies[n] = J * (-(N * (N + 1)) / 2.0 + 2 * (n + 1) * (N - n))
        
    def f(delta_E):
        return 1.0 / (1.0 + np.exp(np.clip(beta * delta_E, -700, 700)))
        
    M = np.zeros((N + 1, N + 1))
    
    for n in range(N + 1):
        # Jump n -> n + 1
        if n < N:
            delta_E_plus = energies[n + 1] - energies[n]
            M[n + 1, n] = gamma * (N - n) * f(delta_E_plus)
            
        # Jump n -> n - 1
        if n > 0:
            delta_E_minus = energies[n - 1] - energies[n]
            M[n - 1, n] = gamma * n * f(delta_E_minus)

    # Ensure probability conservation (columns sum to 0)
    np.fill_diagonal(M, 0)
    np.fill_diagonal(M, -np.sum(M, axis=0))
    
    return M, energies, degeneracies

# --- 3. Main Loop: Validation & Thermalization Times ---
# We will loop from N=3 up to N=10 (since J_values only goes up to N=10)
target_N_values = [3,4,5,6,7,8,9,10]
beta = 1.0

print(f"{'N':<5} | {'Gibbs Match Check':<25} | {'Thermalization Time (tau)':<25}")
print("-" * 60)

for N in target_N_values:
    M, energies, degeneracies = build_ata_matrix(N, J_values, beta=beta)
    
    # -- A. Calculate Steady State from Matrix (Null Space) --
    null_vector = la.null_space(M)
    P_steady = np.abs(null_vector[:, 0])
    P_steady /= np.sum(P_steady)
    
    # -- B. Calculate Analytical Gibbs State --
    shifted_energies = energies - np.min(energies)
    boltzmann_factors = degeneracies * np.exp(-beta * shifted_energies)
    Z = np.sum(boltzmann_factors)
    P_gibbs = boltzmann_factors / Z
    
    # -- C. Verify Match --
    max_error = np.max(np.abs(P_steady - P_gibbs))
    if max_error < 1e-10:
        match_status = "Pass (Error < 1e-10)"
    else:
        match_status = f"FAIL (Error: {max_error:.2e})"
    
    # -- D. Calculate Thermalization Time --
    eigenvalues = np.real(la.eigvals(M))
    sorted_evals = np.sort(eigenvalues)[::-1]
    
    # lambda_0 is ~0 (steady state). lambda_1 dictates the thermalization time.
    lambda_1 = sorted_evals[1]
    tau = -1.0 / lambda_1
    
    print(f"{N:<5} | {match_status:<25} | {tau:<25.6f}")