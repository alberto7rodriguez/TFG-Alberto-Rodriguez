import numpy as np
import scipy.linalg as la
from scipy.special import comb
import matplotlib.pyplot as plt

# --- 1. Datos de los Modelos ---
a_values = [-0.711, 0.000, 0.894, 2.015, 3.398, 5.070, 7.052, 9.358, 11.998, 14.977, 18.297, 21.960, 25.967, 30.318, 35.013, 40.053, 45.438, 51.168, 57.243, 63.664, 70.431, 77.543, 85.001, 92.805, 100.956, 109.452, 118.294, 127.483, 137.018, 146.899, 157.127, 167.701, 178.621, 189.887, 201.500, 213.460, 225.766, 238.418, 251.417, 264.762, 278.453, 292.492, 306.876, 321.607, 336.685, 352.109, 367.879, 383.996, 400.460]
b_values = [0.711, 0.797, 0.894, 1.007, 1.133, 1.267, 1.410, 1.560, 1.714, 1.872, 2.033, 2.196, 2.361, 2.527, 2.693, 2.861, 3.029, 3.198, 3.367, 3.537, 3.707, 3.877, 4.048, 4.218, 4.389, 4.560, 4.732, 4.903, 5.075, 5.246, 5.418, 5.590, 5.762, 5.934, 6.106, 6.278, 6.450, 6.623, 6.795, 6.967, 7.140, 7.312, 7.485, 7.657, 7.830, 8.002, 8.175, 8.348, 8.520]
J_values = [0.7112, 0.4961, 0.3769, 0.3018, 0.2506, 0.2135, 0.1856, 0.1638, 0.1464, 0.1321, 0.1203, 0.1103, 0.1018, 0.0945]

# --- 2. Constructores de Matrices ---
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

# --- 3. Función QFI en Equilibrio ---
def calculate_equilibrium_QFI(energies, degeneracies, beta=1.0, delta_beta=0.001):
    def get_gibbs(b):
        shifted_E = energies - np.min(energies)
        boltzmann = degeneracies * np.exp(-b * shifted_E)
        return boltzmann / np.sum(boltzmann)
    
    p_plus = get_gibbs(beta + delta_beta)
    p_minus = get_gibbs(beta - delta_beta)
    dp_dbeta = (p_plus - p_minus) / (2 * delta_beta)
    
    p_eq = get_gibbs(beta)
    valid_indices = p_eq > 1e-15
    qfi_eq = np.sum((dp_dbeta[valid_indices]**2) / p_eq[valid_indices])
    return qfi_eq

# --- 4. Bucle Principal: Extracción de chi ---
target_N_values = list(range(3, 16))
beta = 1.0

chi_ata_list = []
chi_star_list = []

print(f"{'N':<5} | {'Chi (All-To-All)':<20} | {'Chi (Star Model)':<20}")
print("-" * 50)

for N in target_N_values:
    # --- Evaluación All-To-All ---
    M_ata, E_ata, deg_ata = build_ata_matrix(N, J_values, beta=beta)
    evals_ata = np.sort(np.real(la.eigvals(M_ata)))[::-1]
    tau_ata = -1.0 / evals_ata[1]
    qfi_ata = calculate_equilibrium_QFI(E_ata, deg_ata, beta=beta)
    chi_ata = qfi_ata / tau_ata
    chi_ata_list.append(chi_ata)
    
    # --- Evaluación Star Model ---
    M_star, E_star, deg_star = build_star_model_matrix(N, a_values, b_values, beta=beta)
    evals_star = np.sort(np.real(la.eigvals(M_star)))[::-1]
    tau_star = -1.0 / evals_star[1]
    qfi_star = calculate_equilibrium_QFI(E_star, deg_star, beta=beta)
    chi_star = qfi_star / tau_star
    chi_star_list.append(chi_star)
    
    print(f"{N:<5} | {chi_ata:<20.6f} | {chi_star:<20.6f}")

# --- 5. Graficar el Trade-off ---
plt.figure(figsize=(9, 6))

plt.plot(target_N_values, chi_ata_list, 's-', color='orange', linewidth=2.5, markersize=8, label='All-To-All Model')
plt.plot(target_N_values, chi_star_list, 'o-', color='darkblue', linewidth=2.5, markersize=8, label='Star Model')

plt.title(r'Precision per unit time ($\chi = \mathcal{F}_{eq} / \tau$)', fontsize=15)
plt.xlabel(r'$N$', fontsize=13)
plt.ylabel(r'$\chi$', fontsize=13)

# Escala logarítmica fundamental para ver la caída exponencial del Star Model
#plt.yscale('log') 

plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(fontsize=12)
plt.tight_layout()

plt.show()

# ==============================================================================
# --- EXTRA PLOT: Precisión Transitoria vs QFI Absoluta (Doble Eje) ---
# ==============================================================================
fig, ax1 = plt.subplots(figsize=(9, 6))

# Crear un segundo eje Y que comparte el mismo eje X
ax2 = ax1.twinx()

N_values_to_plot = [3, 5, 7] 
d_beta = 0.001
times = np.linspace(0.1, 160, 200) 

for N_test in N_values_to_plot:
    chi_t = []
    fisher_t = []
    
    p0 = np.ones(2 * N_test) / (2 * N_test)
    M_b, _, _ = build_star_model_matrix(N_test, a_values, b_values, beta=beta)

    evals, evecs = la.eig(M_b)
    evecs_inv = la.inv(evecs)
    
    M_b_plus, _, _ = build_star_model_matrix(N_test, a_values, b_values, beta=beta + d_beta)
    evals_p, evecs_p = la.eig(M_b_plus)
    evecs_inv_p = la.inv(evecs_p)

    for t in times:
        exp_dt = np.diag(np.exp(evals * t))
        pt = np.real(evecs @ exp_dt @ evecs_inv @ p0)
        
        exp_dt_p = np.diag(np.exp(evals_p * t))
        pt_plus = np.real(evecs_p @ exp_dt_p @ evecs_inv_p @ p0)
        
        dp_dbeta = (pt_plus - pt) / d_beta
        fi = np.sum((dp_dbeta**2) / (pt + 1e-12))
        
        fisher_t.append(fi)
        chi_t.append(fi / t)
        
    # Graficar chi(t) en el eje IZQUIERDO (Sólido)
    line, = ax1.plot(times, chi_t, label=f'$N = {N_test}$ ($\chi$)', linewidth=2.5)
    
    # Marcar el pico de chi(t)
    max_chi = max(chi_t)
    opt_time = times[chi_t.index(max_chi)]
    ax1.plot(opt_time, max_chi, marker='*', color=line.get_color(), markersize=12)
    
    # Graficar F_beta(t) en el eje DERECHO (Punteado y clarito)
    # Usamos el MISMO color extrayéndolo de 'line.get_color()'
    ax2.plot(times, fisher_t, linestyle='--', color=line.get_color(), alpha=0.3, linewidth=2.5)

# Configuración del eje izquierdo (chi)
ax1.set_xlabel(r'$t$', fontsize=13)
ax1.set_ylabel(r'$\chi(t) = \mathcal{F}_{\beta}(t) / t$', fontsize=13)
ax2.set_xscale('function', functions=(lambda x: np.power(x, 0.5), lambda x: np.power(x, 2)))
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc='upper right', fontsize=11)

# Configuración del eje derecho (Fisher absoluta)
ax2.set_ylabel(r'$\mathcal{F}_{\beta}(t)$', fontsize=13, color='gray')
ax2.tick_params(axis='y', labelcolor='gray')

plt.title(r'Transient $\chi(t)$', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()