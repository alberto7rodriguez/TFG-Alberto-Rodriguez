#Thermalization times + Gibbs convergence check for the Star Model for arbitrary N

import numpy as np
import scipy.linalg as la
from scipy.special import comb
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- 1. Your Data ---
a_values = [-0.711, 0.000, 0.894, 2.015, 3.398, 5.070, 7.052, 9.358, 11.998, 14.977, 18.297, 21.960, 25.967, 30.318, 35.013, 40.053, 45.438, 51.168, 57.243, 63.664, 70.431, 77.543, 85.001, 92.805, 100.956, 109.452, 118.294, 127.483, 137.018, 146.899, 157.127, 167.701, 178.621, 189.887, 201.500, 213.460, 225.766, 238.418, 251.417, 264.762, 278.453, 292.492, 306.876, 321.607, 336.685, 352.109, 367.879, 383.996, 400.460]
b_values = [0.711, 0.797, 0.894, 1.007, 1.133, 1.267, 1.410, 1.560, 1.714, 1.872, 2.033, 2.196, 2.361, 2.527, 2.693, 2.861, 3.029, 3.198, 3.367, 3.537, 3.707, 3.877, 4.048, 4.218, 4.389, 4.560, 4.732, 4.903, 5.075, 5.246, 5.418, 5.590, 5.762, 5.934, 6.106, 6.278, 6.450, 6.623, 6.795, 6.967, 7.140, 7.312, 7.485, 7.657, 7.830, 8.002, 8.175, 8.348, 8.520]

# --- 2. The General Matrix Builder ---
def build_star_model_matrix(N, a_list, b_list, beta=1.0, gamma=1.0):
    a = a_list[N - 2]
    b = b_list[N - 2]
    
    energies = np.zeros(2 * N)
    degeneracies = np.zeros(2 * N)
    
    for k in range(N):
        deg = comb(N - 1, k, exact=True)
        degeneracies[k] = deg
        degeneracies[k + N] = deg
        
        energies[k] = -a                                 
        energies[k + N] = a + 2 * b * (2 * k - N + 1)    
        
    def f(delta_E):
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

    # Probability conservation
    np.fill_diagonal(M, 0)
    np.fill_diagonal(M, -np.sum(M, axis=0))
    
    return M, energies, degeneracies

# --- 3. Main Loop: Validation & Thermalization Times ---
target_N_values = [3, 5, 7, 9, 11, 13, 15]
beta = 1.0
taus_list = []
print(f"{'N':<5} | {'Gibbs Match Check':<20} | {'Thermalization Time (tau)':<25}")
print("-" * 55)

for N in target_N_values:
    M, energies, degeneracies = build_star_model_matrix(N, a_values, b_values, beta=beta)
    
    # -- A. Calculate Steady State from Matrix --
    null_vector = la.null_space(M)
    P_steady = np.abs(null_vector[:, 0])
    P_steady /= np.sum(P_steady)
    
    # -- B. Calculate Analytical Gibbs State --
    # Shift energies to prevent exponential overflow
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
    lambda_1 = sorted_evals[1]
    tau = -1.0 / lambda_1
    taus_list.append(tau)

    print(f"{N:<5} | {match_status:<20} | {tau:<25.6f}")


# --- (Make sure your previous target_N_values, taus_list, a_values, b_values, build_star_model_matrix are defined above this) ---

# --- 1. Data Preparation and Regression (for ax1) ---
N_values = np.array(target_N_values)
taus = np.array(taus_list)

def modelo_exponencial(x, a, b):
    return a * np.exp(b * x)

popt, _ = curve_fit(modelo_exponencial, N_values, taus)
a, b = popt

residuals = taus - modelo_exponencial(N_values, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((taus - np.mean(taus))**2)
r_squared = 1 - (ss_res / ss_tot)

x_fit = np.linspace(min(N_values), max(N_values), 100)
y_fit = modelo_exponencial(x_fit, a, b)

# --- 2. Create the Figure and Subplots ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Plot 1: Exponential Regression on ax1 ---
ax1.scatter(N_values, taus, color='red', marker='x', label='Obtained times', zorder=5)
ax1.plot(x_fit, y_fit, color='blue', label=f'$\\tau = {a:.4f} \\cdot e^{{{b:.4f}N}}$')

ax1.set_xlabel(r'$N$', fontsize=12)
ax1.set_ylabel(r'$\tau$', fontsize=12)
ax1.set_title(r'Thermalization times scaling with $N$', fontsize=14)
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.6)
# ax1.set_yscale('log') # Uncomment if you want logarithmic scale later

# --- Plot 2: Fisher Information Evolution on ax2 ---
N_values_to_plot = [3, 5, 7] 
d_beta = 0.001
times = np.linspace(0, 160, 200) 

for N_test in N_values_to_plot:
    fisher_beta = []
    
    # Initial state: Uniform distribution over 2N states
    p0 = np.ones(2 * N_test) / (2 * N_test)
    M_b, _, _ = build_star_model_matrix(N_test, a_values, b_values, beta=beta)

    '''Initial state: Ground State
    M_b, energies, _ = build_star_model_matrix(N_test, a_values, b_values, beta=beta)
    p0 = np.zeros(2 * N_test)
    ground_state_idx = np.argmin(energies) # Finds the index of the minimum energy
    p0[ground_state_idx] = 1.0
    '''

    evals, evecs = la.eig(M_b)
    evecs_inv = la.inv(evecs)
    
    M_b_plus, _, _ = build_star_model_matrix(N_test, a_values, b_values, beta=beta + d_beta)
    evals_p, evecs_p = la.eig(M_b_plus)
    evecs_inv_p = la.inv(evecs_p)

    for t in times:
        # P(t) at beta
        exp_dt = np.diag(np.exp(evals * t))
        pt = np.real(evecs @ exp_dt @ evecs_inv @ p0)
        
        # P(t) at beta + d_beta
        exp_dt_p = np.diag(np.exp(evals_p * t))
        pt_plus = np.real(evecs_p @ exp_dt_p @ evecs_inv_p @ p0)
        
        # Numerical derivative and Fisher Info calculation
        dp_dbeta = (pt_plus - pt) / d_beta
        fi = np.sum((dp_dbeta**2) / (pt + 1e-12))
        fisher_beta.append(fi)
    
    ax2.plot(times, fisher_beta, label=f'N = {N_test}', linewidth=2.5)

ax2.set_xlabel(r'$t$', fontsize=12)
ax2.set_xscale('function', functions=(lambda x: np.power(x, 0.5), lambda x: np.power(x, 2)))
ax2.set_ylabel(r'$F_{\beta}(t)$', fontsize=12)
ax2.set_title(r'Evolution of $F_{\beta}$ ($p_n(0)=1/2N$)', fontsize=14)
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.6)

fig.suptitle(r'Star Model Model', fontsize=16, fontweight='bold')

# --- Final Layout Adjustments and Print Statements ---
plt.tight_layout()
plt.show()

print(f"Ecuación ajustada: tau = {a:.4f} * exp({b:.4f} * N)")
print(f"R^2: {r_squared:.4f}")