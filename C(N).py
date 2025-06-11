import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp
e = np.e

mp.dps = 50 

def safe_exp(x):
    """Evita overflow al calcular exponenciales grandes."""
    return np.exp(x) if x < 700 else np.exp(700)

def C_opt(N,x):
    return e**x * x**2 * (2**N - 1)/((2**N - 1 + e**x)**2) 

def C_star(N, a, b):
    beta = 1
    cosh_2b = np.cosh(2 * b * beta)
    sinh_2b = np.sinh(2 * b * beta)
    exp_2a = safe_exp(2 * a * beta) 

    cosh_2b_N = cosh_2b ** N if N < 100 else np.exp(N * np.log(cosh_2b))

    numerator = -4 * (cosh_2b ** (N - 2)) * (
        2 * (N - 1) * a * b * exp_2a * (cosh_2b ** 2) * sinh_2b
        - (N - 1) * (b ** 2) * cosh_2b_N
        - ((N - 1) ** 2 * b ** 2 + a ** 2) * exp_2a * (cosh_2b ** 3)
        + (N - 2) * (N - 1) * (b ** 2) * exp_2a * cosh_2b
    )

    denominator = (cosh_2b_N + exp_2a * cosh_2b) ** 2

    return numerator / denominator if denominator != 0 else np.nan  

def C_star_asympt(N):
    return ((N - 1) ** 2 * (np.log(2)) ** 2) / 4

a_values = [-0.711, 0.000, 0.894, 2.015, 3.398, 5.070, 7.052, 9.358, 11.998, 14.977, 18.297, 21.960, 25.967, 30.318, 35.013, 40.053, 45.438, 51.168, 57.243, 63.664, 70.431, 77.543, 85.001, 92.805, 100.956, 109.452, 118.294, 127.483, 137.018, 146.899, 157.127, 167.701, 178.621, 189.887, 201.500, 213.460, 225.766, 238.418, 251.417, 264.762, 278.453, 292.492, 306.876, 321.607, 336.685, 352.109, 367.879, 383.996, 400.460]
b_values = [0.711, 0.797, 0.894, 1.007, 1.133, 1.267, 1.410, 1.560, 1.714, 1.872, 2.033, 2.196, 2.361, 2.527, 2.693, 2.861, 3.029, 3.198, 3.367, 3.537, 3.707, 3.877, 4.048, 4.218, 4.389, 4.560, 4.732, 4.903, 5.075, 5.246, 5.418, 5.590, 5.762, 5.934, 6.106, 6.278, 6.450, 6.623, 6.795, 6.967, 7.140, 7.312, 7.485, 7.657, 7.830, 8.002, 8.175, 8.348, 8.520]

N_opt_values = list(range(1, 51))
x_opt_values = [2.399357, 2.844989, 3.332611, 3.856799, 4.411763, 4.991966, 5.5925, 6.209242, 6.838849, 7.478676, 8.126653, 8.781173, 9.44099, 10.105132, 10.772838, 11.443508, 12.116666, 12.791929, 13.468985, 14.147583, 14.827513, 15.508603, 16.190709, 16.873707, 17.557496, 18.241986, 18.927103, 19.612781, 20.298964, 20.985603, 21.672654, 22.360079, 23.047847, 23.735926, 24.42429, 25.112917, 25.801785, 26.490876, 27.180173, 27.86966, 28.559323, 29.249151, 29.939132, 30.629256, 31.319513, 32.009895, 32.700393, 33.391001, 34.081712, 34.772519]
C_opt_values = []
C_non_int = []

N_star_values = list(range(2, 51))
C_star_values = []

for i in range(0,len(N_opt_values)):
    C = C_opt(N_opt_values[i], x_opt_values[i])
    C_opt_values.append(C)
    C_non_int.append(0.44*N_opt_values[i])

for i in range(len(N_star_values)):
    if N_star_values[i] <= 32:
        C = C_star(N_star_values[i], a_values[i], b_values[i])
    else:
        C = C_star_asympt(N_star_values[i])  
    C_star_values.append(C)

xtick_values = [ 2, 5, 10, 20, 50]
ytick_values = [ 1, 2, 5, 10, 20, 50, 100, 200]
inset_xtick_values = [ 2, 4, 6, 8, 10]
inset_ytick_values = [ 2, 6, 10]

C_all = [1.02349, 1.70583, 2.4613, 3.27434, 4.1345, 5.03421, 5.967779, 6.9308103, 7.9198015]
N_all = [2,3,4,5,6,7,8,9,10]

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(N_opt_values, C_opt_values, linestyle='-', color='r', label='Optimal bound $\mathcal{C}_{opt}$')
ax.plot(N_opt_values, C_non_int, linestyle='-', color='g', label='No interaction bound')
ax.scatter(N_star_values, C_star_values, s=11 , marker='o', color='b', label='Star Model $\mathcal{C}_{Star}$')
ax.scatter(0, 0, s=11, color='orange', label='All-To-All Model $\mathcal{C}_{all}$')
ax.fill_between(N_opt_values, C_opt_values, max(C_opt_values), color='r', alpha=0.2)
ax.fill_between(N_opt_values, C_non_int, 0, color='g', alpha=0.1)

ax.set_xlabel('N')
ax.set_ylabel('$\mathcal{C}_{max}$')
ax.tick_params(axis='both', direction='in', length=6)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.set_xlim(1, max(N_opt_values))
ax.set_ylim(0.45, max(C_opt_values))
ax.set_xscale('function', functions=(lambda x: np.power(x, 0.1), lambda x: np.power(x, 1/0.1)))
ax.set_yscale('function', functions=(lambda y: np.power(y, 0.07), lambda y: np.power(y, 1/0.07)))
ax.set_xticks(xtick_values)
ax.set_xticklabels([str(i) for i in xtick_values])
ax.set_yticks(ytick_values)
ax.set_yticklabels([str(i) for i in ytick_values])
ax.grid(True, alpha=0.3)

legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4)
legend.get_frame().set_linewidth(0)
legend.get_frame().set_facecolor('none')


inset_ax = fig.add_axes([0.18, 0.45, 0.3, 0.4])  
inset_ax.plot(N_opt_values, C_opt_values, linestyle='-', color='r',alpha = 0.6, linewidth=0.7)
inset_ax.scatter(N_star_values, C_star_values, marker='_', s=125, color='b')
inset_ax.scatter(N_all, C_all, marker='_', s=200, color='orange')
inset_ax.tick_params(axis='both', direction='in', length=6, labelsize=8)
inset_ax.set_xticks(inset_xtick_values)
inset_ax.set_xticklabels([str(i) for i in inset_xtick_values])
inset_ax.set_yticks(inset_ytick_values)
inset_ax.set_yticklabels([str(i) for i in inset_ytick_values])
inset_ax.spines['top'].set_visible(True)
inset_ax.spines['right'].set_visible(True)
inset_ax.set_xlim(1, 10.5)
inset_ax.set_ylim(0.45, 13)
inset_ax.set_xscale('function', functions=(lambda x: np.power(x, 1.2), lambda x: np.power(x, 1/1.2)))
inset_ax.grid(True, alpha=0.3)

plt.show()


