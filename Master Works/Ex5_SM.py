import numpy as np
import matplotlib.pyplot as plt

def calculate_energy(pos):
    N = pos.shape[0]
    e_harmonic = np.sum(pos**2)
    e_coulomb = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            distance = np.linalg.norm(pos[i] - pos[j])
            e_coulomb += 1.0 / distance
            
    return e_harmonic + e_coulomb

def simulated_annealing(N, T_init=10.0, T_final=0.0001, cooling_rate=0.999, steps_per_iter=50):
    pos = np.random.uniform(-1, 1, (N, 2))
    current_energy = calculate_energy(pos)
    
    T = T_init
    delta_t = 0.5
    
    history_energy = []
    history_temp = []
    iteration = 0
    while T > T_final:
        accepted_moves = 0
        for _ in range(steps_per_iter):
            move = (2 * np.random.rand(N, 2) - 1) * delta_t
            proposed_pos = pos + move
            proposed_energy = calculate_energy(proposed_pos)
            delta_E = proposed_energy - current_energy
            if delta_E <= 0 or np.random.rand() < np.exp(-delta_E / T):
                pos = proposed_pos
                current_energy = proposed_energy
                accepted_moves += 1
                
        history_energy.append(current_energy / N)
        history_temp.append(T)
        acceptance_prob = accepted_moves / steps_per_iter
        if acceptance_prob < 0.2:
            delta_t *= 0.95 
        elif acceptance_prob > 0.8:
            delta_t *= 1.05 
            
        T *= cooling_rate
        iteration += 1
        
    return pos, current_energy, history_energy, history_temp

def run_and_plot(N):
    print(f"N = {N} charges")
    best_pos, final_energy, E_hist, T_hist = simulated_annealing(N)
    print(f"Final Energy per particle (E/N): {final_energy / N:.5f}")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(E_hist, color='red', label='Energy per particle (E/N)')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('E/N', color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(T_hist, color='blue', alpha=0.5, label='Temperature (T)')
    ax1_twin.set_ylabel('Temperature', color='blue')
    ax1_twin.tick_params(axis='y', labelcolor='blue')
    ax1.set_title(f'Annealing Progress (N={N})')
    ax2.scatter(best_pos[:, 0], best_pos[:, 1], color='black', s=100)
    ax2.set_aspect('equal', 'box')
    ax2.set_title(f'Optimal Configuration (N={N})')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    distances = np.linalg.norm(best_pos, axis=1)
    max_dist = np.max(distances)
    circle = plt.Circle((0, 0), max_dist, fill=False, color='gray', linestyle='--')
    ax2.add_artist(circle)
    plt.tight_layout()
    plt.show()

for N in [5, 20, 26]:
    run_and_plot(N)