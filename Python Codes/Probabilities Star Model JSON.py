import numpy as np
from scipy.linalg import expm, eig, inv
import json

# Predefined values
a_values = [-0.711, 0.000, 0.894, 2.015, 3.398, 5.070, 7.052, 9.358, 11.998, 14.977,
            18.297, 21.960, 25.967, 30.318, 35.013, 40.053, 45.438, 51.168, 57.243, 63.664,
            70.431, 77.543, 85.001, 92.805, 100.956, 109.452, 118.294, 127.483, 137.018, 146.899,
            157.127, 167.701, 178.621, 189.887, 201.500, 213.460, 225.766, 238.418, 251.417, 264.762,
            278.453, 292.492, 306.876, 321.607, 336.685, 352.109, 367.879, 383.996, 400.460]

b_values = [0.711, 0.797, 0.894, 1.007, 1.133, 1.267, 1.410, 1.560, 1.714, 1.872,
            2.033, 2.196, 2.361, 2.527, 2.693, 2.861, 3.029, 3.198, 3.367, 3.537,
            3.707, 3.877, 4.048, 4.218, 4.389, 4.560, 4.732, 4.903, 5.075, 5.246,
            5.418, 5.590, 5.762, 5.934, 6.106, 6.278, 6.450, 6.623, 6.795, 6.967,
            7.140, 7.312, 7.485, 7.657, 7.830, 8.002, 8.175, 8.348, 8.520]

# Helper functions
def U(N, n, a, b):
    return -a if n == 0 else a + 2 * b * (2 * (n - 1) - (N - 1))

def f(x, beta):
    return 1 / (1 + np.exp(beta * x))

def transition_matrix(N, beta, a, b):
    M = np.zeros((N + 1, N + 1))
    for n in range(N + 1):
        if n == 0:
            delta_U = U(N, n + 1, a, b) - U(N, n, a, b)
            rate_up = f(delta_U, beta)
            M[n, n] -= rate_up
            M[n + 1, n] += rate_up
        if n < N and n != 0:
            delta_U = U(N, n + 1, a, b) - U(N, n, a, b)
            rate_up = (N - n) * f(delta_U, beta)
            M[n, n] -= rate_up
            M[n + 1, n] += rate_up
        if n > 0:
            delta_U = U(N, n - 1, a, b) - U(N, n, a, b)
            rate_down = (2 ** (N - 1) if n == 1 else (n - 1)) * f(delta_U, beta)
            M[n, n] -= rate_down
            M[n - 1, n] += rate_down
    return M

# Main computation loop
target_Ns = [3, 5, 7, 9, 11, 13, 15, 18, 20]
beta = 1
prob_functions = {}

for N in target_Ns:
    a = a_values[N - 2]
    b = b_values[N - 2]
    p0 = np.full(N + 1, 0)
    #excited = int(N/2)
    p0[0] = 1
    #p0=np.full(N+1, 1/(N+1))
    M = transition_matrix(N, beta, a, b)
    eigvals, eigvecs = eig(M)
    V_inv = inv(eigvecs)
    coeffs = V_inv @ p0

    prob_functions[N] = {}
    for n in range(N + 1):
        terms = []
        for k in range(N + 1):
            amplitude = eigvecs[n, k] * coeffs[k]
            if np.abs(amplitude) > 1e-6:
                real_part = np.real(amplitude)
                decay = np.real(eigvals[k])
                terms.append(f"{real_part:.8f} * np.exp({decay:.8f} * t)")
        prob_functions[N][n] = " + ".join(terms)

# Save as JSON
import os
output_path = r"C:\Users\alber\OneDrive\Escritorio\TFG\JSON\prob_ground_beta1.json"
with open(output_path, "w") as f:
    json.dump(prob_functions, f, indent=2)

output_path
