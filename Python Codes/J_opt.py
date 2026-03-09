import numpy as np
from scipy.special import comb
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

def calculate_heat_capacity(J, N, beta=1.0):
    """
    Calculates the heat capacity of the All-To-All model for a given J.
    """
    energies = np.zeros(N + 1)
    degeneracies = np.zeros(N + 1)
    
    for n in range(N + 1):
        degeneracies[n] = comb(N, n, exact=True)
        # Your corrected All-To-All energy formula
        energies[n] = J * (-(N * (N + 1)) / 2.0 + 2 * (n + 1) * (N - n))
        
    # Shift energies to prevent numerical overflow in the exponentials
    shifted_energies = energies - np.min(energies)
    boltzmann_factors = degeneracies * np.exp(-beta * shifted_energies)
    Z = np.sum(boltzmann_factors)
    
    # Probabilities
    p_n = boltzmann_factors / Z
    
    # Energy Expectation Values
    # Note: We must use the original unshifted energies to calculate the true variance
    E_mean = np.sum(p_n * energies)
    E_squared_mean = np.sum(p_n * (energies**2))
    
    # Heat Capacity (Energy Variance * beta^2)
    energy_variance = E_squared_mean - (E_mean**2)
    heat_capacity = (beta**2) * energy_variance
    
    return heat_capacity

def objective_function(J, N, beta=1.0):
    """
    Scipy minimizers look for minimums. To maximize Heat Capacity, 
    we minimize the negative Heat Capacity.
    """
    return -calculate_heat_capacity(J, N, beta)

# --- Main Optimization Loop ---
target_N_values = list(range(2, 16)) # Let's go up to N=15 now!
beta = 1.0

optimal_J_list = []
max_C_list = []

print(f"{'N':<5} | {'Optimal J':<15} | {'Max Heat Capacity (C)':<25}")
print("-" * 50)

for N in target_N_values:
    # We use minimize_scalar with bounded method. 
    # We know J is roughly between 0.01 and 2.0 based on your manual data.
    result = minimize_scalar(
        objective_function, 
        bounds=(0.01, 2.0), 
        args=(N, beta), 
        method='bounded'
    )
    
    best_J = result.x
    max_C = -result.fun # Convert back to positive heat capacity
    
    optimal_J_list.append(best_J)
    max_C_list.append(max_C)
    
    print(f"{N:<5} | {best_J:<15.6f} | {max_C:<25.6f}")

# Optional: Print the exact python list you can copy-paste into your other script!
print("\n--- Copy-Paste ready lists ---")
print(f"J_values = {[round(j, 4) for j in optimal_J_list]}")