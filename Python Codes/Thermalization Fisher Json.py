import json
import numpy as np
import matplotlib.pyplot as plt

# Load JSON files
with open(r"C:\Users\alber\OneDrive\Escritorio\TFG\JSON\prob_functions_beta1.json", "r") as f1, \
     open(r"C:\Users\alber\OneDrive\Escritorio\TFG\JSON\prob_functions_beta1.0008.json", "r") as f2, \
     open(r"C:\Users\alber\OneDrive\Escritorio\TFG\JSON\prob_ground_beta1.json", "r") as f3, \
     open(r"C:\Users\alber\OneDrive\Escritorio\TFG\JSON\prob_ground_beta1.0008.json", "r") as f4:
              
 
    prob_uniform_beta1 = json.load(f1)
    prob_uniform_beta1p0008 = json.load(f2)
    prob_ground_beta1 = json.load(f3)
    prob_ground_beta1p0008 = json.load(f4)

# Parameters
h = 0.0008
t_vals = np.linspace(0.01, 50, 500)
N_list = sorted(map(int, prob_uniform_beta1.keys()))
fisher_uniform = {}
fisher_ground = {}
Fishers_Gibbs = {
    3: 1.6564888757700964,
    5: 3.202834457001724,
    7: 5.473113087581061,
    9: 8.728544713963085,
    11: 12.999992031339094,
    13: 18.260988346336685,
    15: 24.490847768304718,
    18: 35.64744230502836,
    20: 44.28326799218033
}

# Loop through each N
for N in N_list:
    F_uniform_t = []
    F_ground_t = []
    for t in t_vals:
        F1 = 0.0
        F2 = 0.0
        for n in range(N + 1):
            expr_uni_beta1 = prob_uniform_beta1[str(N)][str(n)]
            expr_uni_beta2 = prob_uniform_beta1p0008[str(N)][str(n)]
            expr_gr_beta1 = prob_ground_beta1[str(N)][str(n)]
            expr_gr_beta2 = prob_ground_beta1p0008[str(N)][str(n)]

            try:
                p1_uni = eval(expr_uni_beta1, {"np": np, "t": t})
                p2_uni = eval(expr_uni_beta2, {"np": np, "t": t})
                p1_gr = eval(expr_gr_beta1, {"np": np, "t": t})
                p2_gr = eval(expr_gr_beta2, {"np": np, "t": t})
            except Exception:
                p1_uni = p2_uni = p1_gr = p2_gr = 1e-12

            dp_dbeta_uni = (p2_uni - p1_uni) / h
            dp_dbeta_gr = (p2_gr - p1_gr) / h

            if p1_uni > 1e-12:
                F1 += (dp_dbeta_uni**2) / p1_uni
            if p1_gr > 1e-12:
                F2 += (dp_dbeta_gr**2) / p1_gr

        F_uniform_t.append(F1)
        F_ground_t.append(F2)

    fisher_uniform[N] = F_uniform_t
    fisher_ground[N] = F_ground_t

# Plotting
colors = ['#2980B9', '#E67E22', '#27AE60', '#8E44AD', '#C0392B', '#F1C40F', '#3498DB', '#E67E22', '#9B59B6']
plt.figure(figsize=(10, 6))
i = 0
for N in N_list:
    plt.plot(t_vals, fisher_ground[N], label=f"N={N}", color=colors[i])
    plt.plot(t_vals, fisher_uniform[N], linestyle='--', alpha=0.3, color=colors[i])
    '''if N in Fishers_Gibbs:
        plt.axhline(y=Fishers_Gibbs[N], linestyle=':', alpha=0.2, color=colors[i])'''
    i += 1
plt.xlabel("t")
plt.ylabel(r"$\mathcal{F}_\beta(t)$")
plt.xscale('function', functions=(lambda x: np.power(x, 0.5), lambda x: np.power(x, 2)))
plt.xlim(0.09, 50)
plt.ylim(0.001, 55)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.11), ncol=5, frameon=False)
plt.grid(True)
plt.tight_layout()
plt.show()