import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from scipy.optimize import root_scalar

# ====================================================================
# 1. Physical Parameters and Grid Setup
# ====================================================================
N = 50          # Number of particles
a_tilde = 0.01     # Dimensionless scattering length (a / a_ho)

# Dimensionless coupling constants
g_tilde = 4 * np.pi * a_tilde
g_lhy =  (128 * np.pi**(1/2) * a_tilde**(5/2))/3

# Radial Grid Setup (1D)
R_max = 10.0       # Maximum radius
N_grid = 1000      # Number of grid points
r = np.linspace(0.001, R_max, N_grid) # Start slightly above 0 to avoid division by zero
dr = r[1] - r[0]

# External harmonic potential V(r)
V_trap = 0.5 * r**2

# ====================================================================
# 2. GP + LHY Solver (Self-Consistent Diagonalization)
# ====================================================================
# Kinetic energy matrix (Finite Differences for -1/2 d^2/dr^2)
main_diag =  1.0 / dr**2 * np.ones(N_grid)
off_diag  = -0.5 / dr**2 * np.ones(N_grid - 1)
T_matrix = diags([off_diag, main_diag, off_diag], [-1, 0, 1])

# Initial guess for the wavefunction chi(r) = r * phi(r) (Harmonic oscillator ground state)
chi = r * np.exp(-0.5 * r**2)
# Normalize: 4 * pi * integral(|chi|^2 dr) = 1
norm_factor = np.sqrt(4 * np.pi * np.trapz(chi**2, r))
chi /= norm_factor

tolerance = 1e-6
max_iter = 100
mixing_alpha = 0.1 # Mixing parameter to prevent oscillating convergence

for iteration in range(max_iter):
    # Strictly enforce the boundary condition at the origin to kill numerical noise
    chi[0] = 0.0 
    
    # Safely calculate phi = chi / r using L'Hopital's rule for the first point
    phi = np.zeros_like(chi)
    phi[1:] = chi[1:] / r[1:]
    phi[0] = chi[1] / dr  # L'Hopital: limit as r->0 is the derivative
    
    # Calculate physical density safely
    density_norm = phi**2
    physical_density = N * density_norm
    
    # Non-linear potentials
    V_mean_field = g_tilde * physical_density
    V_lhy = g_lhy * (physical_density**(3/2))
    
    # Total effective potential diagonal
    V_eff_diag = V_trap + V_mean_field + V_lhy
    V_matrix = diags([V_eff_diag], [0])
    
    # Full Hamiltonian
    H = T_matrix + V_matrix
    
    # Find the lowest eigenvalue and eigenvector
    eigenvalues, eigenvectors = eigsh(H, k=1, which='SA')
    chi_new = eigenvectors[:, 0]
    
    # Enforce positive wavefunction
    if chi_new[10] < 0:
        chi_new = -chi_new
        
    # Normalize new chi
    norm_factor = np.sqrt(4 * np.pi * np.trapz(chi_new**2, r))
    chi_new /= norm_factor
    
    # Check convergence
    diff = np.max(np.abs(chi_new - chi))
    if diff < tolerance:
        print(f"Converged after {iteration} iterations.")
        chi = chi_new
        break
        
    # Mix old and new to stabilize self-consistency
    chi = (1 - mixing_alpha) * chi + mixing_alpha * chi_new
    chi /= np.sqrt(4 * np.pi * np.trapz(chi**2, r))

# Final simulated density calculation (Safe version)
chi[0] = 0.0
phi_final = np.zeros_like(chi)
phi_final[1:] = chi[1:] / r[1:]
phi_final[0] = chi[1] / dr
n_sim = (phi_final**2) * N

# Energy Functional Calculation E/N
# Kinetic: int (-1/2 * chi * chi'') dr 
kin_energy = np.trapz(chi * (T_matrix @ chi), r) * 4 * np.pi
pot_energy = np.trapz(V_trap * n_sim * 4 * np.pi * r**2, r) / N
int_energy = np.trapz(0.5 * g_tilde * (n_sim**2) * 4 * np.pi * r**2, r) / N
lhy_energy = np.trapz((2/5) * g_lhy * (n_sim**(5/2)) * 4 * np.pi * r**2, r) / N

E_per_N_sim = kin_energy + pot_energy + int_energy + lhy_energy
print(f"Simulated E/N: {E_per_N_sim:.5f} hbar*omega")

# ====================================================================
# 3. Local Density Approximation (LDA)
# ====================================================================
def equation_of_state(n, mu_loc):
    """ Equation to solve: g*n + g_lhy*n**(3/2) - mu_loc = 0 """
    return g_tilde * n + g_lhy * (n**(1.5)) - mu_loc

def get_lda_density(mu):
    n_lda = np.zeros_like(r)
    for i, r_val in enumerate(r):
        mu_loc = mu - V_trap[i]
        if mu_loc > 0:
            # Solve for local density using a root finder
            sol = root_scalar(equation_of_state, args=(mu_loc,), bracket=[0, mu_loc/g_tilde + 1])
            n_lda[i] = sol.root
    return n_lda

def particle_number_error(mu):
    n_lda = get_lda_density(mu)
    N_calc = np.trapz(n_lda * 4 * np.pi * r**2, r)
    return N_calc - N

# Find the chemical potential mu that gives exactly N particles
# Bracket guesses based on Thomas-Fermi approximation
mu_tf_guess = 0.5 * (15 * N * g_tilde / (4 * np.pi))**(2/5)
sol_mu = root_scalar(particle_number_error, bracket=[mu_tf_guess*0.5, mu_tf_guess*2.0])
mu_lda = sol_mu.root

# Calculate final LDA density and Energy
n_lda = get_lda_density(mu_lda)
E_lda = np.trapz((V_trap * n_lda + 0.5 * g_tilde * n_lda**2 + (2/5) * g_lhy * n_lda**(2.5)) * 4 * np.pi * r**2, r) / N
print(f"LDA E/N:       {E_lda:.5f} hbar*omega")

# ====================================================================
# 4. Plotting Results
# ====================================================================
plt.figure(figsize=(10, 6))
plt.plot(r[2:], n_sim[2:], label='GP + LHY Simulation', linewidth=2)
plt.plot(r, n_lda, '--', label='LDA Prediction', linewidth=2)

plt.title(f"Density profile ($N = {N}$, $a/a_{{ho}} = {a_tilde}$)")
plt.xlabel(r"$\tilde{r}$ (units of $a_{ho}$)")
plt.ylabel(r"$n(r)$ (units of $a_{ho}^{-3}$)")
plt.xlim(0, 5) # Adjust based on trap size
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()