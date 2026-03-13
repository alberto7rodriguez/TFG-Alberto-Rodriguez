import numpy as np
import scipy.linalg as la
from scipy.special import comb
import matplotlib.pyplot as plt

# --- 1. Datos de los Modelos ---
a_values = [-0.711, 0.000, 0.894, 2.015, 3.398, 5.070, 7.052, 9.358, 11.998, 14.977, 18.297, 21.960, 25.967, 30.318, 35.013, 40.053, 45.438, 51.168, 57.243, 63.664, 70.431, 77.543, 85.001, 92.805, 100.956, 109.452, 118.294, 127.483, 137.018, 146.899, 157.127, 167.701, 178.621, 189.887, 201.500, 213.460, 225.766, 238.418, 251.417, 264.762, 278.453, 292.492, 306.876, 321.607, 336.685, 352.109, 367.879, 383.996, 400.460]
b_values = [0.711, 0.797, 0.894, 1.007, 1.133, 1.267, 1.410, 1.560, 1.714, 1.872, 2.033, 2.196, 2.361, 2.527, 2.693, 2.861, 3.029, 3.198, 3.367, 3.537, 3.707, 3.877, 4.048, 4.218, 4.389, 4.560, 4.732, 4.903, 5.075, 5.246, 5.418, 5.590, 5.762, 5.934, 6.106, 6.278, 6.450, 6.623, 6.795, 6.967, 7.140, 7.312, 7.485, 7.657, 7.830, 8.002, 8.175, 8.348, 8.520]
J_values = [0.7112, 0.4961, 0.3769, 0.3018, 0.2506, 0.2135, 0.1856, 0.1638, 0.1464, 0.1321, 0.1203, 0.1103, 0.1018, 0.0945]

beta = 1.0

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

    np.fill_diagonal(M, 0)
    np.fill_diagonal(M, -np.sum(M, axis=0))
    return M, energies, degeneracies

def build_ata_matrix(N, J_list, beta=1.0, gamma=1.0):
    J = J_list[N - 2]
    energies = np.zeros(N + 1)
    degeneracies = np.zeros(N + 1)
    
    for n in range(N + 1):
        degeneracies[n] = comb(N, n, exact=True)
        energies[n] = J * (-(N * (N + 1)) / 2.0 + 2 * (n + 1) * (N - n))
        
    def f(delta_E):
        return 1.0 / (1.0 + np.exp(np.clip(beta * delta_E, -700, 700)))
        
    M = np.zeros((N + 1, N + 1))
    
    for n in range(N + 1):
        if n < N:
            delta_E_plus = energies[n + 1] - energies[n]
            M[n + 1, n] = gamma * (N - n) * f(delta_E_plus)
        if n > 0:
            delta_E_minus = energies[n - 1] - energies[n]
            M[n - 1, n] = gamma * n * f(delta_E_minus)

    np.fill_diagonal(M, 0)
    np.fill_diagonal(M, -np.sum(M, axis=0))
    return M, energies, degeneracies

# ==============================================================================
# --- EXTRA PLOT: Precisión Transitoria vs QFI Absoluta (Lado a Lado) ---
# ==============================================================================
fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(15, 6))

N_values_to_plot = [3, 5, 7] 
d_beta = 0.001

# ------------------------------------------------------------------------------
# 1. SUBPLOT IZQUIERDO: STAR MODEL
# ------------------------------------------------------------------------------
ax2 = ax1.twinx()
times_star = np.linspace(0.1, 160, 200) 

for N_test in N_values_to_plot:
    chi_t = []
    fisher_t = []
    
    p0 = np.ones(2 * N_test) / (2 * N_test) # 2N estados
    M_b, _, _ = build_star_model_matrix(N_test, a_values, b_values, beta=beta)
    evals, evecs = la.eig(M_b)
    evecs_inv = la.inv(evecs)
    
    M_b_plus, _, _ = build_star_model_matrix(N_test, a_values, b_values, beta=beta + d_beta)
    evals_p, evecs_p = la.eig(M_b_plus)
    evecs_inv_p = la.inv(evecs_p)

    for t in times_star:
        exp_dt = np.diag(np.exp(evals * t))
        pt = np.real(evecs @ exp_dt @ evecs_inv @ p0)
        
        exp_dt_p = np.diag(np.exp(evals_p * t))
        pt_plus = np.real(evecs_p @ exp_dt_p @ evecs_inv_p @ p0)
        
        dp_dbeta = (pt_plus - pt) / d_beta
        fi = np.sum((dp_dbeta**2) / (pt + 1e-12))
        
        fisher_t.append(fi)
        chi_t.append(fi / t)
        
    line, = ax1.plot(times_star, chi_t, label=f'$N = {N_test}$ ($\chi$)', linewidth=2.5)
    max_chi = max(chi_t)
    opt_time = times_star[chi_t.index(max_chi)]
    ax1.plot(opt_time, max_chi, marker='*', color=line.get_color(), markersize=12)
    ax2.plot(times_star, fisher_t, linestyle='--', color=line.get_color(), alpha=0.3, linewidth=2.5)

ax1.set_xlabel(r'$t$', fontsize=13)
ax1.set_ylabel(r'$\chi(t) = \mathcal{F}_{\beta}(t) / t$', fontsize=13)
ax1.set_xscale('function', functions=(lambda x: np.power(x, 0.5), lambda x: np.power(x, 2)))
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc='upper right', fontsize=11)
ax2.set_ylabel(r'$\mathcal{F}_{\beta}(t)$', fontsize=13, color='gray')
ax2.tick_params(axis='y', labelcolor='gray')
ax1.set_title(r'Star Model', fontsize=15, fontweight='bold')


# ------------------------------------------------------------------------------
# 2. SUBPLOT DERECHO: ALL-TO-ALL MODEL
# ------------------------------------------------------------------------------
ax4 = ax3.twinx()
times_ata = np.linspace(0.1, 80, 200) # Tiempo más corto porque termaliza antes

for N_test in N_values_to_plot:
    chi_t = []
    fisher_t = []
    
    p0 = np.ones(N_test + 1) / (N_test + 1) # N+1 estados
    M_b, _, _ = build_ata_matrix(N_test, J_values, beta=beta)
    evals, evecs = la.eig(M_b)
    evecs_inv = la.inv(evecs)
    
    M_b_plus, _, _ = build_ata_matrix(N_test, J_values, beta=beta + d_beta)
    evals_p, evecs_p = la.eig(M_b_plus)
    evecs_inv_p = la.inv(evecs_p)

    for t in times_ata:
        exp_dt = np.diag(np.exp(evals * t))
        pt = np.real(evecs @ exp_dt @ evecs_inv @ p0)
        
        exp_dt_p = np.diag(np.exp(evals_p * t))
        pt_plus = np.real(evecs_p @ exp_dt_p @ evecs_inv_p @ p0)
        
        dp_dbeta = (pt_plus - pt) / d_beta
        fi = np.sum((dp_dbeta**2) / (pt + 1e-12))
        
        fisher_t.append(fi)
        chi_t.append(fi / t)
        
    line, = ax3.plot(times_ata, chi_t, label=f'$N = {N_test}$ ($\chi$)', linewidth=2.5)
    max_chi = max(chi_t)
    opt_time = times_ata[chi_t.index(max_chi)]
    ax3.plot(opt_time, max_chi, marker='*', color=line.get_color(), markersize=12)
    ax4.plot(times_ata, fisher_t, linestyle='--', color=line.get_color(), alpha=0.3, linewidth=2.5)

ax3.set_xlabel(r'$t$', fontsize=13)
ax3.set_xscale('function', functions=(lambda x: np.power(x, 0.5), lambda x: np.power(x, 2)))
ax3.grid(True, linestyle='--', alpha=0.6)
ax3.legend(loc='lower right', fontsize=11) # Movido para que no tape las curvas
ax4.set_ylabel(r'$\mathcal{F}_{\beta}(t)$', fontsize=13, color='gray')
ax4.tick_params(axis='y', labelcolor='gray')
ax3.set_title(r'All-To-All Model', fontsize=15, fontweight='bold')

plt.tight_layout()
plt.show()