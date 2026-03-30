import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. System Parameters & Initialization
# ==========================================
N = 242               # Number of particles, chosen to reduce commensurability issues [cite: 173]
rho = 0.96            # Density [cite: 168]
L = np.sqrt(N / rho)  # Box length for a 2D system [cite: 168, 181]

def init_positions(N, L):
    """Generate initial random positions inside the box[cite: 227]."""
    return np.random.rand(N, 2) * L

def apply_pbc_dist(dx, dy, L):
    """Apply minimal image convention for distances[cite: 222, 224, 225]."""
    dx = dx - np.floor(dx / L + 0.5) * L
    dy = dy - np.floor(dy / L + 0.5) * L
    return dx, dy

def apply_pbc_pos(pos, L):
    """Keep particles inside the primary simulation box[cite: 218, 220, 221]."""
    pos[:, 0] = pos[:, 0] - np.floor(pos[:, 0] / L) * L
    pos[:, 1] = pos[:, 1] - np.floor(pos[:, 1] / L) * L
    return pos

# ==========================================
# 2. Energy Calculations
# ==========================================
def calc_total_energy(pos, L):
    """Calculate the total Lennard-Jones potential energy of the system[cite: 163, 182]."""
    energy = 0.0
    for i in range(N):
        dx = pos[i+1:, 0] - pos[i, 0]
        dy = pos[i+1:, 1] - pos[i, 1]
        dx, dy = apply_pbc_dist(dx, dy, L)
        r2 = dx**2 + dy**2
        
        # Avoid division by zero if particles overlap heavily
        r2 = np.where(r2 < 0.1, 0.1, r2) 
        
        # U_LJ(r) = 4 * ((1/r^12) - (1/r^6))
        u = 4.0 * ((1.0 / r2**6) - (1.0 / r2**3))
        energy += np.sum(u)
    return energy

def calc_delta_e(pos, idx, new_p, L):
    """Calculate the energy difference if particle 'idx' is moved to 'new_p'[cite: 119]."""
    # Distances to old position
    dx_old = pos[:, 0] - pos[idx, 0]
    dy_old = pos[:, 1] - pos[idx, 1]
    dx_old, dy_old = apply_pbc_dist(dx_old, dy_old, L)
    r2_old = dx_old**2 + dy_old**2
    r2_old[idx] = np.inf # Ignore self-interaction
    
    # Distances to new position
    dx_new = pos[:, 0] - new_p[0]
    dy_new = pos[:, 1] - new_p[1]
    dx_new, dy_new = apply_pbc_dist(dx_new, dy_new, L)
    r2_new = dx_new**2 + dy_new**2
    r2_new[idx] = np.inf
    
    # Calculate difference
    u_old = 4.0 * ((1.0 / r2_old**6) - (1.0 / r2_old**3))
    u_new = 4.0 * ((1.0 / r2_new**6) - (1.0 / r2_new**3))
    
    return np.sum(u_new) - np.sum(u_old)

# ==========================================
# 3. Metropolis Algorithm
# ==========================================
def metropolis_sweep(pos, L, T, delta):
    """Perform one Monte Carlo sweep (N trial moves)[cite: 39, 116, 117]."""
    accepted = 0
    for _ in range(N):
        idx = np.random.randint(N)
        # Generate trial move [cite: 49, 50]
        move = delta * (2.0 * np.random.rand(2) - 1.0)
        new_p = pos[idx] + move
        
        # Apply PBC to trial position
        new_p[0] = new_p[0] - np.floor(new_p[0] / L) * L
        new_p[1] = new_p[1] - np.floor(new_p[1] / L) * L
        
        dE = calc_delta_e(pos, idx, new_p, L)
        
        # Acceptance criterion [cite: 33, 121]
        if dE <= 0 or np.random.rand() < np.exp(-dE / T):
            pos[idx] = new_p
            accepted += 1
            
    return pos, accepted / N

# ==========================================
# 4. Observables: Radial Distribution g(r)
# ==========================================
def calc_gr(pos, L, bins=100, r_max=None):
    """Calculate the radial distribution function g(r)[cite: 182, 208]."""
    if r_max is None:
        r_max = L / 2.0 # Minimum image convention limit
        
    hist = np.zeros(bins)
    dr = r_max / bins
    
    for i in range(N):
        dx = pos[i+1:, 0] - pos[i, 0]
        dy = pos[i+1:, 1] - pos[i, 1]
        dx, dy = apply_pbc_dist(dx, dy, L)
        r = np.sqrt(dx**2 + dy**2)
        
        # Bin the distances
        valid_r = r[r < r_max]
        bin_indices = (valid_r / dr).astype(int)
        for b in bin_indices:
            if b < bins:
                hist[b] += 2.0 # Count both i->j and j->i
                
    # Normalize histogram to get g(r) [cite: 182, 210]
    r_centers = np.linspace(dr/2, r_max - dr/2, bins)
    ideal_gas_counts = rho * 2 * np.pi * r_centers * dr * N
    gr = hist / ideal_gas_counts
    
    return r_centers, gr

# ==========================================
# 5. Main Execution Scripts
# ==========================================
if __name__ == "__main__":
    print(f"Initializing system with N={N}, L={L:.2f}, rho={rho}")
    pos = init_positions(N, L)
    
    # --- Phase A: Simulated Annealing ---
    print("\nStarting Simulated Annealing to find crystal structure... [cite: 169]")
    T_anneal = 5.0
    delta = 0.5
    for step in range(1000):
        pos, acc = metropolis_sweep(pos, L, T_anneal, delta)
        T_anneal *= 0.99 # Slowly decrease temperature [cite: 108, 122]
        if step % 200 == 0:
            print(f"Anneal Step {step}, T={T_anneal:.4f}, Energy/N={calc_total_energy(pos,L)/N:.2f}")
    
    crystal_pos = np.copy(pos)
    
    # Plot crystal snapshot [cite: 193]
    plt.figure(figsize=(5,5))
    plt.scatter(crystal_pos[:,0], crystal_pos[:,1], c='b', s=20)
    plt.title("Crystal Configuration (After Annealing)")
    plt.xlim(0, L); plt.ylim(0, L)
    plt.show()

    # --- Phase B: Production Runs ---
    temperatures = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0] # [cite: 184]
    results = {}
    
    for T in temperatures:
        print(f"\n--- Running MC for T* = {T} ---")
        pos = np.copy(crystal_pos) # Start from crystal for consistency
        
        # Equilibration & Auto-tuning Delta (Aim for ~50% acceptance) [cite: 179]
        delta = 0.1
        for eq_step in range(500):
            pos, acc = metropolis_sweep(pos, L, T, delta)
            if eq_step % 100 == 0:
                if acc < 0.4: delta *= 0.9
                elif acc > 0.6: delta *= 1.1
                
        print(f"Equilibrated. Final delta: {delta:.3f}, Acceptance: {acc:.2%}")
        
        # Production
        prod_steps = 1000
        energies = []
        gr_accum = np.zeros(100)
        
        for step in range(prod_steps):
            pos, acc = metropolis_sweep(pos, L, T, delta)
            
            # Measure every 10 sweeps to reduce correlation [cite: 79]
            if step % 10 == 0:
                e_tot = calc_total_energy(pos, L)
                # Kinetic contribution (2D): (2/2)k_BT [cite: 182]
                e_per_particle = T + (e_tot / N) 
                energies.append(e_per_particle)
                
                r_bins, gr = calc_gr(pos, L, bins=100)
                gr_accum += gr
                
        # Average results [cite: 180]
        avg_E = np.mean(energies)
        std_E = np.std(energies)
        avg_gr = gr_accum / (prod_steps / 10)
        
        print(f"Results for T*={T}: E/N = {avg_E:.3f} +/- {std_E:.3f}")
        results[T] = (r_bins, avg_gr)

    # --- Plotting g(r) ---
    plt.figure(figsize=(8,5))
    for T in [1.0, 3.0]: # Plotting specific temperatures to match example [cite: 236-254]
        r, gr = results[T]
        plt.plot(r, gr, label=f'T* = {T}')
        
    plt.title("Radial Distribution Function g(r) [cite: 188]")
    plt.xlabel("r / sigma")
    plt.ylabel("g(r)")
    plt.legend()
    plt.grid(True)
    plt.show()