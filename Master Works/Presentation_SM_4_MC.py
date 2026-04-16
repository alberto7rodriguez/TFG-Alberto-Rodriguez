import numpy as np
import matplotlib.pyplot as plt

def run_vmc(N, a_tilde, n_steps=5000, equilibration=1000):
    def get_distances(R):
        diff = R[:, np.newaxis, :] - R[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        return diff, dist

    def trial_wavefunction_squared(R):
        _, dist = get_distances(R)
        if np.any(dist < a_tilde): return 0.0
        trap_part = np.exp(-np.sum(R**2))
        jastrow_part = 1.0
        for i in range(N):
            for j in range(i + 1, N):
                jastrow_part *= (1.0 - a_tilde / dist[i, j])**2
        return trap_part * jastrow_part

    R = np.random.randn(N, 3) * 0.5 
    while trial_wavefunction_squared(R) == 0:
        R = np.random.randn(N, 3) * 0.5

    current_prob = trial_wavefunction_squared(R)
    g2_distances = []
    step_size = 0.5

    print(f"Running VMC for a = {a_tilde}...")
    for step in range(n_steps + equilibration):
        R_new = R + (np.random.rand(N, 3) - 0.5) * step_size
        new_prob = trial_wavefunction_squared(R_new)
        
        if new_prob > 0:
            ratio = new_prob / current_prob
            if ratio >= 1.0 or np.random.rand() < ratio:
                R = R_new
                current_prob = new_prob
                
        if step >= equilibration:
            _, dist = get_distances(R)
            for i in range(N):
                for j in range(i + 1, N):
                    g2_distances.append(dist[i, j])
                    
    return g2_distances

# ====================================================================
# Run Simulations
# ====================================================================
N = 100
a_values = [0.01, 0.1]
colors = ['orange', 'purple']
distances_data = []

for a in a_values:
    distances_data.append(run_vmc(N, a))

# ====================================================================
# Plotting
# ====================================================================
plt.figure(figsize=(10, 6))

for idx, a_tilde in enumerate(a_values):
    # 1. Histogram extraction
    counts, bins = np.histogram(distances_data[idx], bins=80, range=(0.005, 2.5), density=True)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    
    # 2. Divide by ideal trap background
    background = (bin_centers**2) * np.exp(-bin_centers**2 / 2)
    g2_extracted = counts / background
    g2_extracted /= np.mean(g2_extracted[-15:]) # Normalize at tail
    
    # 3. Plot Scatter
    plt.plot(bin_centers, g2_extracted, 'o', color=colors[idx], alpha=0.8, 
             label=f'VMC ($a={a_tilde}$)')
    
    # 4. Plot Analytical Theory
    r_theory = np.linspace(0.001, 2.5, 500)
    g2_theory = np.where(r_theory >= a_tilde, (1 - a_tilde/r_theory)**2, 0)
    plt.plot(r_theory, g2_theory, '--', color=colors[idx], linewidth=2, 
             label=f'Theory ($a={a_tilde}$)')

plt.title(f"$g_2(r)$ in dilute & strongly interacting regime ($N={N}$)")
plt.xlabel(r"$r$ (units of $a_{ho}$)")
plt.ylabel(r"$g_2(r)$")
plt.xlim(0, 2.0)
plt.ylim(-0.05, 1.2)
plt.legend(ncol=2) # 2-column legend to keep it tidy
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()