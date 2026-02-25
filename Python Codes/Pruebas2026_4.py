#Analytic solutions for the different probabilites for the Star Model

import numpy as np
import scipy.linalg as la
from scipy.special import comb

def print_analytical_dynamics(N, a, b, beta=1.0, gamma=1.0, initial_state_idx=0):
    print(f"--- Analytical Dynamics for N={N}, a={a}, b={b} ---")
    
    # 1. State Definitions and Energies (2N states)
    energies = np.zeros(2 * N)
    for k in range(N):
        energies[k] = -a                                 
        energies[k + N] = a + 2 * b * (2 * k - N + 1)    
        
    def f(delta_E):
        return 1.0 / (1.0 + np.exp(np.clip(beta * delta_E, -700, 700)))
        
    # 2. Build the Rate Matrix M
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

    # 3. Eigendecomposition
    evals, evecs = la.eig(M)
    
    # Detailed balance guarantees real eigenvalues, but we cast to real to remove 0j
    evals = np.real(evals)
    evecs = np.real(evecs)
    
    # Sort eigenvalues in descending order (so ~0 is first, then the slowest decays)
    sort_idx = np.argsort(evals)[::-1]
    evals = evals[sort_idx]
    evecs = evecs[:, sort_idx]

    # 4. Define Initial State and find Coefficients (C = V^-1 * P_0)
    P_init = np.zeros(2 * N)
    P_init[initial_state_idx] = 1.0  # 100% probability in the chosen starting state
    
    C = la.solve(evecs, P_init)

    # 5. Print the Analytical Expressions
    for i in range(2 * N):
        state_label = f"DOWN, k={i}" if i < N else f"UP, k={i-N}"
        
        # Start building the string for this specific state
        expr = f"P_{{{state_label}}}(t) ="
        
        for m in range(2 * N):
            amplitude = C[m] * evecs[i, m]
            
            # Filter out terms with effectively zero amplitude
            if abs(amplitude) > 1e-5:
                lam = evals[m]
                
                # If lambda is basically 0, it's the steady state constant
                if abs(lam) < 1e-10:
                    expr += f" {amplitude:+.4f}"
                else:
                    expr += f" {amplitude:+.4f} * exp({lam:.4f}*t)"
                    
        print(expr)
    print("\n")

# --- Run for N=3 and N=5 ---
# For N=3 (index 1 in your list -> a=0.000, b=0.797)
print_analytical_dynamics(N=3, a=0.000, b=0.797, initial_state_idx=0)

# For N=5 (index 3 in your list -> a=2.015, b=1.007)
print_analytical_dynamics(N=5, a=2.015, b=1.007, initial_state_idx=0)