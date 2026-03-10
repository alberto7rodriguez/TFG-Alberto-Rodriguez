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


import matplotlib.pyplot as plt

# --- 4. Comparative Evolution of Fisher Information for Beta ---

# Parameters for the evolution plot
N_values_to_plot = [3, 5, 7]
times = np.linspace(0, 75, 200)
d_beta = 0.001

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Scaling of Tau (Recalculated for all target_N_values)
all_taus = []
for n_val in target_N_values:
    M_scale, _, _ = build_ata_matrix(n_val, J_values, beta=beta)
    evals_scale = np.sort(np.real(la.eigvals(M_scale)))[::-1]
    all_taus.append(-1.0 / evals_scale[1])

ax1.plot(target_N_values, all_taus, 'o-', color='tab:red', linewidth=2)
ax1.set_xlabel(r'$N$', fontsize=12)
ax1.set_ylabel(r'$\tau$', fontsize=12)
ax1.set_title(r'Thermalization times scaling with $N$', fontsize=14)
ax1.grid(True, linestyle='--', alpha=0.6)

# Plot 2: Fisher Information Evolution for N = 3, 5, 7
for N_test in N_values_to_plot:
    fisher_beta = []
    # Initial state: Uniform distribution (Fisher Info = 0)
    p0 = np.ones(N_test + 1) / (N_test + 1)
    
    # Pre-calculate matrices for beta and beta + d_beta to speed up the time loop
    M_b, _, _ = build_ata_matrix(N_test, J_values, beta=beta)
    evals, evecs = la.eig(M_b)
    evecs_inv = la.inv(evecs)
    
    M_b_plus, _, _ = build_ata_matrix(N_test, J_values, beta=beta + d_beta)
    evals_p, evecs_p = la.eig(M_b_plus)
    evecs_inv_p = la.inv(evecs_p)

    for t in times:
        # P(t) at beta
        exp_dt = np.diag(np.exp(evals * t))
        pt = np.real(evecs @ exp_dt @ evecs_inv @ p0)
        
        # P(t) at beta + d_beta
        exp_dt_p = np.diag(np.exp(evals_p * t))
        pt_plus = np.real(evecs_p @ exp_dt_p @ evecs_inv_p @ p0)
        
        # Numerical derivative and Fisher Info
        dp_dbeta = (pt_plus - pt) / d_beta
        fi = np.sum((dp_dbeta**2) / (pt + 1e-12))
        fisher_beta.append(fi)
    
    ax2.plot(times, fisher_beta, label=f'N = {N_test}', linewidth=2)

ax2.set_xlabel(r'$t$', fontsize=12)
ax2.set_xscale('function', functions=(lambda x: np.power(x, 0.5), lambda x: np.power(x, 2)))
ax2.set_ylabel(r'$F_{\beta}(t)$', fontsize=12)
ax2.set_title(r'Evolution of $F_{\beta}$ ($p_n(0)=1/(N+1)$)', fontsize=14)
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.6)

fig.suptitle(r'All-To-All Model', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.show()