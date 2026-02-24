import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig

# Star Model parameters
a_values = [-0.711, 0.000, 0.894, 2.015, 3.398, 5.070, 7.052, 9.358, 11.998, 14.977,
            18.297, 21.960, 25.967, 30.318, 35.013, 40.053, 45.438, 51.168, 57.243, 63.664,
            70.431, 77.543, 85.001, 92.805, 100.956, 109.452, 118.294, 127.483, 137.018,
            146.899, 157.127, 167.701, 178.621, 189.887, 201.500, 213.460, 225.766,
            238.418, 251.417, 264.762, 278.453, 292.492, 306.876, 321.607, 336.685,
            352.109, 367.879, 383.996, 400.460]

b_values = [0.711, 0.797, 0.894, 1.007, 1.133, 1.267, 1.410, 1.560, 1.714, 1.872,
            2.033, 2.196, 2.361, 2.527, 2.693, 2.861, 3.029, 3.198, 3.367, 3.537,
            3.707, 3.877, 4.048, 4.218, 4.389, 4.560, 4.732, 4.903, 5.075, 5.246,
            5.418, 5.590, 5.762, 5.934, 6.106, 6.278, 6.450, 6.623, 6.795, 6.967,
            7.140, 7.312, 7.485, 7.657, 7.830, 8.002, 8.175, 8.348, 8.520]

J_values = [0.7112, 0.496, 0.3769, 0.3019, 0.2506, 0.2135, 0.1856, 0.1638, 0.1464]

def f(x, beta):
    return 1 / (1 + np.exp(beta * x))

def U_star(N, n, a, b):
    return -a if n == 0 else a + 2 * b * (2 * (n - 1) - (N - 1))

def M_star(N, beta, a, b):
    M = np.zeros((N + 1, N + 1))
    for n in range(N + 1):
        if n == 0:
            delta = U_star(N, 1, a, b) - U_star(N, 0, a, b)
            r_up = f(delta, beta)
            M[n, n] -= r_up
            M[n+1, n] += r_up
        if 0 < n < N:
            delta = U_star(N, n+1, a, b) - U_star(N, n, a, b)
            r_up = (N - n) * f(delta, beta)
            M[n, n] -= r_up
            M[n+1, n] += r_up
        if n > 0:
            delta = U_star(N, n-1, a, b) - U_star(N, n, a, b)
            r_down = (2**(N - 1) if n == 1 else n - 1) * f(delta, beta)
            M[n, n] -= r_down
            M[n-1, n] += r_down
    return M

def U_all(N, n, J):
    return J * (-N * (N + 1) * 0.5 + 2 * (n + 1) * (N - n))

def M_all(N, beta, J):
    M = np.zeros((N + 1, N + 1))
    for n in range(N + 1):
        if n < N:
            delta = U_all(N, n + 1, J) - U_all(N, n, J)
            r_up = (N - n) * f(delta, beta)
            M[n, n] -= r_up
            M[n + 1, n] += r_up
        if n > 0:
            delta = U_all(N, n - 1, J) - U_all(N, n, J)
            r_down = n * f(delta, beta)
            M[n, n] -= r_down
            M[n - 1, n] += r_down
    return M

# Compute Star Model data
N_star = list(range(2, 51))
inv_lambda_star = []
for N in N_star:
    a = a_values[N - 2]
    b = b_values[N - 2]
    M = M_star(N, beta=1, a=a, b=b)
    eigvals = np.real(eig(M)[0])
    max_nonzero = max([val for val in eigvals if abs(val) > 1e-10])
    inv_lambda_star.append(1 / abs(max_nonzero))

# Compute All-To-All data (N = 2 to 10)
N_all = list(range(2, 11))
inv_lambda_all = []
for i, N in enumerate(N_all):
    J = J_values[i]
    M = M_all(N, beta=1, J=J)
    eigvals = np.real(eig(M)[0])
    max_nonzero = max([val for val in eigvals if abs(val) > 1e-10])
    inv_lambda_all.append(1 / abs(max_nonzero))

# Main plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(N_star, inv_lambda_star, marker='o', markersize = 4, label='Star Model')
ax.plot(N_all, inv_lambda_all, marker='o', markersize = 4, label='All-To-All Model')
ax.set_xlabel(r"$N$")
ax.set_ylabel(r"$1 / |\lambda_{1}|$")
ax.grid(True, alpha= 0.5)
ax.tick_params(axis='both', direction='in')

# Inset plot
inset_ax = fig.add_axes([0.43, 0.35, 0.52, 0.52])
inset_ax.plot(N_star, inv_lambda_star, marker='o', markersize = 3, color='tab:blue')
inset_ax.tick_params(axis='both', labelsize=8, direction='in')
inset_ax.set_yticks([1.00, 1.02, 1.05, 1.07])
inset_ax.grid(True, alpha = 0.5)

# Legend
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.085), ncol=2, frameon=False)

plt.tight_layout()
plt.show()