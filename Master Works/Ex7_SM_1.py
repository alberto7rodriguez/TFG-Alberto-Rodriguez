import numpy as np
import matplotlib.pyplot as plt

N = 242
rho = 0.96
L = np.sqrt(N / rho)

def apply_pbc_dist(dx, dy, L):
    dx = dx - np.floor(dx / L + 0.5) * L
    dy = dy - np.floor(dy / L + 0.5) * L
    return dx, dy

def calc_delta_e(pos, idx, new_p, L):
    dx_old = pos[:, 0] - pos[idx, 0]
    dy_old = pos[:, 1] - pos[idx, 1]
    dx_old, dy_old = apply_pbc_dist(dx_old, dy_old, L)
    r2_old = dx_old**2 + dy_old**2
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
        move = delta * (2.0 * np.random.rand(2) - 1.0)
        new_p = pos[idx] + move
        new_p[0] = new_p[0] - np.floor(new_p[0] / L) * L
        new_p[1] = new_p[1] - np.floor(new_p[1] / L) * L
        
        dE = calc_delta_e(pos, idx, new_p, L)
        
        if dE <= 0 or np.random.rand() < np.exp(-dE / T):
            pos[idx] = new_p
    return pos

pos = np.random.rand(N, 2) * L 
T_anneal = 5.0
delta = 0.5

for step in range(10000):
    pos = metropolis_sweep(pos, L, T_anneal, delta)
    T_anneal *= 0.999 

crystal_pos = np.copy(pos)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(crystal_pos[:,0], crystal_pos[:,1], c='blue', edgecolors='black')
plt.title("Crystal phase (low T)")
plt.xlim(0, L); plt.ylim(0, L)
plt.gca().set_aspect('equal')

T_gas = 3.0 
delta = 0.15 

for step in range(1000):
    pos = metropolis_sweep(pos, L, T_gas, delta)

gas_pos = np.copy(pos)

plt.subplot(1, 2, 2)
plt.scatter(gas_pos[:,0], gas_pos[:,1], c='red', edgecolors='black')
plt.title(f"Gas phase (T* = {T_gas})")
plt.xlim(0, L); plt.ylim(0, L)
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.show()
