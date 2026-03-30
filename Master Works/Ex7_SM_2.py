import numpy as np
import matplotlib.pyplot as plt

N = 242
rho = 0.96
L = np.sqrt(N / rho)

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
    for _ in range(N):
        idx = np.random.randint(N)
        new_p = pos[idx] + delta * (2.0 * np.random.rand(2) - 1.0)
        new_p[0] -= np.floor(new_p[0] / L) * L
        new_p[1] -= np.floor(new_p[1] / L) * L
        dE = calc_delta_e(pos, idx, new_p, L)
        if dE <= 0 or np.random.rand() < np.exp(-dE / T):
            pos[idx] = new_p
    return pos

def calc_gr(pos, L, bins, r_max):
    hist = np.zeros(bins)
    dr = r_max / bins
    
    for i in range(N):
        dx = pos[i+1:, 0] - pos[i, 0]
        dy = pos[i+1:, 1] - pos[i, 1]
        dx, dy = apply_pbc_dist(dx, dy, L)
        r = np.sqrt(dx**2 + dy**2)
        
        valid_r = r[r < r_max]
        bin_indices = (valid_r / dr).astype(int)
        for b in bin_indices:
            if b < bins:
                hist[b] += 2.0 
                
    return hist

def run_simulation(T, delta, eq_steps, prod_steps, sample_freq=10):
    global pos
    for _ in range(eq_steps):
        pos = metropolis_sweep(pos, L, T, delta)
        
    bins = 100
    r_max = L / 2.0
    dr = r_max / bins
    r_centers = np.linspace(dr/2, r_max - dr/2, bins)
    gr_accum = np.zeros(bins)
    measurements = 0
    
    for step in range(prod_steps):
        pos = metropolis_sweep(pos, L, T, delta)
        if step % sample_freq == 0:
            gr_accum += calc_gr(pos, L, bins, r_max)
            measurements += 1

    ideal_gas_shell = rho * 2 * np.pi * r_centers * dr * N
    gr_final = gr_accum / (measurements * ideal_gas_shell)
    return r_centers, gr_final

pos = init_grid(N, L)
r_solid, gr_solid = run_simulation(T=1.0, delta=0.1, eq_steps=2000, prod_steps=2000)
r_gas, gr_gas = run_simulation(T=3.0, delta=0.2, eq_steps=3000, prod_steps=2000)
plt.figure(figsize=(8, 5))
plt.plot(r_solid, gr_solid, color='blue', label='T* = 1.0 (Solid)')
plt.plot(r_gas, gr_gas, color='black', label='T* = 3.0 (Gas)')
plt.xlim(0, L/2)
plt.ylim(0, 6)
plt.xlabel("r / σ")
plt.ylabel("g(r)")
plt.title("Radial Distribution Function")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()