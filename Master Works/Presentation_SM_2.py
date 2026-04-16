import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from scipy.optimize import root_scalar

# ====================================================================
# 1. Core Solver Function
# ====================================================================
def calculate_energies(N, a_tilde):
    """Runs the GP+LHY solver and LDA for a given N and a_tilde."""
    
    g_tilde = 4 * np.pi * a_tilde
    g_lhy = (128 * np.sqrt(np.pi) / 3) * (a_tilde**(5/2))

    # Grid Setup (Keeping it slightly coarser here for faster looping)
    R_max = 12.0       
    N_grid = 800      
    r = np.linspace(0.001, R_max, N_grid) 
    dr = r[1] - r[0]
    V_trap = 0.5 * r**2

    # Kinetic matrix
    main_diag =  1.0 / dr**2 * np.ones(N_grid)
    off_diag  = -0.5 / dr**2 * np.ones(N_grid - 1)
    T_matrix = diags([off_diag, main_diag, off_diag], [-1, 0, 1])

    # Initial guess
    chi = r * np.exp(-0.5 * r**2)
    chi /= np.sqrt(4 * np.pi * np.trapz(chi**2, r))

    tolerance = 1e-6
    max_iter = 100
    mixing_alpha = 0.1 

    # --- SIMULATION ---
    for _ in range(max_iter):
        chi[0] = 0.0 
        phi = np.zeros_like(chi)
        phi[1:] = chi[1:] / r[1:]
        phi[0] = chi[1] / dr 
        
        physical_density = N * (phi**2)
        
        V_eff_diag = V_trap + g_tilde * physical_density + g_lhy * (physical_density**(3/2))
        H = T_matrix + diags([V_eff_diag], [0])
        
        _, eigenvectors = eigsh(H, k=1, which='SA')
        chi_new = eigenvectors[:, 0]
        
        if chi_new[10] < 0: chi_new = -chi_new
        chi_new /= np.sqrt(4 * np.pi * np.trapz(chi_new**2, r))
        
        if np.max(np.abs(chi_new - chi)) < tolerance:
            chi = chi_new
            break
            
        chi = (1 - mixing_alpha) * chi + mixing_alpha * chi_new
        chi /= np.sqrt(4 * np.pi * np.trapz(chi**2, r))

    # Calculate Simulated Energy
    chi[0] = 0.0
    phi_final = np.zeros_like(chi)
    phi_final[1:] = chi[1:] / r[1:]
    phi_final[0] = chi[1] / dr
    n_sim = (phi_final**2) * N

    kin_energy = np.trapz(chi * (T_matrix @ chi), r) * 4 * np.pi
    pot_energy = np.trapz(V_trap * n_sim * 4 * np.pi * r**2, r) / N
    int_energy = np.trapz(0.5 * g_tilde * (n_sim**2) * 4 * np.pi * r**2, r) / N
    lhy_energy = np.trapz((2/5) * g_lhy * (n_sim**(5/2)) * 4 * np.pi * r**2, r) / N
    E_sim = kin_energy + pot_energy + int_energy + lhy_energy

    # --- LDA PREDICTION ---
    def equation_of_state(n, mu_loc):
        return g_tilde * n + g_lhy * (n**(1.5)) - mu_loc

    def get_lda_density(mu):
        n_lda = np.zeros_like(r)
        for i, r_val in enumerate(r):
            mu_loc = mu - V_trap[i]
            if mu_loc > 0:
                sol = root_scalar(equation_of_state, args=(mu_loc,), bracket=[0, mu_loc/g_tilde + 1])
                n_lda[i] = sol.root
        return n_lda

    def particle_number_error(mu):
        return np.trapz(get_lda_density(mu) * 4 * np.pi * r**2, r) - N

    mu_tf_guess = 0.5 * (15 * N * g_tilde / (4 * np.pi))**(2/5)
    sol_mu = root_scalar(particle_number_error, bracket=[mu_tf_guess*0.3, mu_tf_guess*3.0])
    
    n_lda = get_lda_density(sol_mu.root)
    E_lda = np.trapz((V_trap * n_lda + 0.5 * g_tilde * n_lda**2 + (2/5) * g_lhy * n_lda**(2.5)) * 4 * np.pi * r**2, r) / N

    return E_sim, E_lda

# ====================================================================
# 2. Run Loops and Plotting
# ====================================================================
# Define the parameter space
N_values = [1000, 3000, 5000, 7000, 10000]
a_values = [0.005, 0.01, 0.02]

plt.figure(figsize=(10, 6))
colors = ['blue', 'red', 'green']

print("Starting simulations... this may take a minute.")

for idx, a_tilde in enumerate(a_values):
    E_sim_list = []
    E_lda_list = []
    
    print(f"Calculating for a/a_ho = {a_tilde}...")
    for N in N_values:
        E_sim, E_lda = calculate_energies(N, a_tilde)
        E_sim_list.append(E_sim)
        E_lda_list.append(E_lda)
        
    # Plot Simulation (solid lines) and LDA (dashed lines)
    plt.plot(N_values, E_sim_list, 'o-', color=colors[idx], label=f'GP+LHY (a={a_tilde})')
    plt.plot(N_values, E_lda_list, '--', color=colors[idx], alpha=0.7)

# Dummy line for LDA legend entry
plt.plot([], [], '--', color='gray', label='LDA Prediction')

plt.title("E/N vs. N for different values of a (Ground state)")
plt.xlabel(r"$N$")
plt.ylabel(r"$E/N$ (units of $\hbar\omega$)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
print("Simulations complete!")