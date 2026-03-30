import numpy as np

N = 242; rho = 0.96; L = np.sqrt(N / rho)

def init_grid(N, L):
    pos = np.zeros((N, 2))
    n_side = int(np.ceil(np.sqrt(N)))
    spacing = L / n_side
    idx = 0
    for i in range(n_side):
        for j in range(n_side):
            if idx < N:
                pos[idx] = [i * spacing + spacing/2, j * spacing + spacing/2]
                idx += 1
    return pos

def apply_pbc_dist(dx, dy, L):
    dx -= np.floor(dx / L + 0.5) * L
    dy -= np.floor(dy / L + 0.5) * L
    return dx, dy

def calc_total_energy(pos, L):
    energy = 0.0
    for i in range(N):
        dx = pos[i+1:, 0] - pos[i, 0]
        dy = pos[i+1:, 1] - pos[i, 1]
        dx, dy = apply_pbc_dist(dx, dy, L)
        r2 = dx**2 + dy**2
        r2 = np.where(r2 < 0.01, 0.01, r2)
        u = 4.0 * ((1.0 / r2**6) - (1.0 / r2**3))
        energy += np.sum(u)
    return energy

def calc_delta_e(pos, idx, new_p, L):
    dx = pos[:, 0] - pos[idx, 0]
    dy = pos[:, 1] - pos[idx, 1]
    dx, dy = apply_pbc_dist(dx, dy, L)
    r2_old = dx**2 + dy**2
    r2_old[idx] = np.inf 
    
    dx_new = pos[:, 0] - new_p[0]
    dy_new = pos[:, 1] - new_p[1]
    dx_new, dy_new = apply_pbc_dist(dx_new, dy_new, L)
    r2_new = dx_new**2 + dy_new**2
    r2_new[idx] = np.inf
    
    u_old = 4.0 * ((1.0 / r2_old**6) - (1.0 / r2_old**3))
    u_new = 4.0 * ((1.0 / r2_new**6) - (1.0 / r2_new**3))
    return np.sum(u_new) - np.sum(u_old)

def metropolis_sweep(pos, L, T, delta):
    accepted = 0
    for _ in range(N):
        idx = np.random.randint(N)
        new_p = pos[idx] + delta * (2.0 * np.random.rand(2) - 1.0)
        new_p[0] -= np.floor(new_p[0] / L) * L
        new_p[1] -= np.floor(new_p[1] / L) * L
        
        dE = calc_delta_e(pos, idx, new_p, L)
        if dE <= 0 or np.random.rand() < np.exp(-dE / T):
            pos[idx] = new_p
            accepted += 1
    return pos, accepted / N

def block_average(data, num_blocks=10):
    block_size = len(data) // num_blocks
    block_means = []
    for i in range(num_blocks):
        block = data[i*block_size : (i+1)*block_size]
        block_means.append(np.mean(block))
    error = np.std(block_means) / np.sqrt(num_blocks)
    return error

temperatures = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
eq_steps = 2000   
prod_steps = 5000  

print(f"{'T*':<6} | {'Iterations':<10} | {'Delta':<7} | {'Accept Rate':<12} | {'E/N':<8} | {'Error (Block)'}")
print("-" * 65)

current_pos = init_grid(N, L)

for T in temperatures:
    delta = 0.1 # Initial guess
    for step in range(eq_steps):
        current_pos, acc = metropolis_sweep(current_pos, L, T, delta)
        if step % 100 == 0:
            if acc < 0.45: delta *= 0.9
            elif acc > 0.55: delta *= 1.1

    # 2. Production Run
    energies_per_particle = []
    total_acc = 0
    
    for step in range(prod_steps):
        current_pos, acc = metropolis_sweep(current_pos, L, T, delta)
        total_acc += acc
        
        u_pot = calc_total_energy(current_pos, L)
        
        e_per_particle = T + (u_pot / N) 
        energies_per_particle.append(e_per_particle)
        
    final_acc_rate = total_acc / prod_steps
    avg_E = np.mean(energies_per_particle)
    err_E = block_average(energies_per_particle, num_blocks=20)
    
    print(f"{T:<6.1f} | {prod_steps:<10} | {delta:<7.3f} | {final_acc_rate:<12.1%} | {avg_E:<8.3f} | ± {err_E:.4f}")