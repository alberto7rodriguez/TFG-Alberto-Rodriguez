import numpy as np
import scipy.linalg as la

# --- 1. System Parameters ---
N = 4
a = 0.894
b = 0.894
beta = 1.0   # Inverse temperature
gamma = 1.0  # Thermalization rate

# --- 2. State Definitions and Energies ---
# We have 8 states. Let's index them 0 to 7:
# Indices 0-3: Central Spin DOWN, k = 0, 1, 2, 3
# Indices 4-7: Central Spin UP,   k = 0, 1, 2, 3

energies = np.zeros(4)
degeneracies = np.array([1, 3, 3, 1, 1, 3, 3, 1])

# Calculate energies based on your Hamiltonian
for k in range(4):
    # Central DOWN
    energies[k] = -a
    # Central UP: a + 2b(k - (N-1-k)) -> a + 2b(2k - 3)
    energies[k+4] = a + 2 * b * (2 * k - 3)

# --- 3. The Glauber Rate Function ---
def f(delta_E):
    return 1.0 / (1.0 + np.exp(beta * delta_E))

# --- 4. Build the Rate Matrix M ---
M = np.zeros((8, 8))

for k in range(4):
    down_idx = k
    up_idx = k + 4
    
    # A. Central Spin Flips (k stays the same)
    delta_E_up = energies[up_idx] - energies[down_idx]
    M[up_idx, down_idx] = gamma * f(delta_E_up)     # Down -> Up
    M[down_idx, up_idx] = gamma * f(-delta_E_up)    # Up -> Down
    
    # B. Satellite Flips in DOWN manifold (Central spin stays DOWN)
    if k < 3:
        delta_E_k_plus = energies[down_idx + 1] - energies[down_idx] # Always 0
        M[down_idx + 1, down_idx] = gamma * (3 - k) * f(delta_E_k_plus)
    if k > 0:
        delta_E_k_minus = energies[down_idx - 1] - energies[down_idx] # Always 0
        M[down_idx - 1, down_idx] = gamma * k * f(delta_E_k_minus)

    # C. Satellite Flips in UP manifold (Central spin stays UP)
    if k < 3:
        delta_E_k_plus = energies[up_idx + 1] - energies[up_idx]
        M[up_idx + 1, up_idx] = gamma * (3 - k) * f(delta_E_k_plus)
    if k > 0:
        delta_E_k_minus = energies[up_idx - 1] - energies[up_idx]
        M[up_idx - 1, up_idx] = gamma * k * f(delta_E_k_minus)

# Ensure probability conservation (columns must sum to 0)
for j in range(8):
    M[j, j] = -np.sum(M[:, j])

# --- 5. Find the Steady State (Null Space) ---
null_vector = la.null_space(M)

# The null space might return a negative vector, so we take the absolute value
# and normalize it so the probabilities sum to 1.
P_steady = np.abs(null_vector[:, 0])
P_steady /= np.sum(P_steady)

# --- 6. Calculate the Analytical Gibbs State ---
boltzmann_factors = degeneracies * np.exp(-beta * energies)
Z = np.sum(boltzmann_factors)
P_gibbs = boltzmann_factors / Z

total_prob_down = 0
# --- 7. Compare the Results ---
print(f"{'State (s_0, k)':<15} | {'Steady State (Matrix)':<25} | {'Analytical Gibbs':<25}")
print("-" * 70)
for k in range(4):
    total_prob_down += P_steady[k]
    print(f"DOWN, k={k:<6} | {P_steady[k]:<25.8f} | {P_gibbs[k]:<25.8f}")
for k in range(4):
    print(f"UP,   k={k:<6} | {P_steady[k+4]:<25.8f} | {P_gibbs[k+4]:<25.8f}")

print(f"Total probability of having the spin down: {total_prob_down:<25.8f}")