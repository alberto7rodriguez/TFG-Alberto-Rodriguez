import numpy as np
from scipy.linalg import expm
from scipy.linalg import eig, inv

J_values = [0.7112, 0.496, 0.3769, 0.3019, 0.2506, 0.2135, 0.1856, 0.1638, 0.1464]

def U(N,n):
    return J*(-N*(N+1)*0.5 + 2*(n+1)*(N-n)) 

def f(x, beta):
    return 1 / (1 + np.exp(beta * x))

def transition_matrix(N, beta):
    M = np.zeros((N+1, N+1))

    for n in range(N+1):
        # Transitions from n to n+1
        if n < N:
            delta_U = U(N,n+1) - U(N,n)
            rate_up = (N - n) * f(delta_U, beta)
            M[n, n] -= rate_up
            M[n+1, n] += rate_up

        # Transitions from n to n-1
        if n > 0:
            delta_U = U(N,n-1) - U(N,n)
            rate_down = n * f(delta_U, beta)
            M[n, n] -= rate_down
            M[n-1, n] += rate_down

    return M

# Parameters
N = 4
J = J_values[N-2]
beta = 1
t = 20.0  # time at which we evaluate the solution

for n in range(N+1):
    print(f"E_{n} = {U(N,n)}")

# Initial condition: start in state 0 (e.g., all probability in p_0)
p0 = np.full(N+1, 1/(N+1))

# Compute matrix exponential
M = transition_matrix(N, beta)
eMt = expm(M * t)

# Final state: p(t) = exp(M t) @ p0
pt = eMt @ p0
'''
print(M)
print("Probability distribution p(t) at t =", t)
print(np.round(pt, 5))
print("Sum of probabilities:", np.sum(pt))  # should be ~1
'''

eigvals, eigvecs = eig(M)
V_inv = inv(eigvecs)

# Project initial condition into eigenbasis
coeffs = V_inv @ p0

# Compose each p_n(t) as a sum of exponentials
print("Explicit time dependence of p_n(t):")
for n in range(N+1):
    terms = []
    for k in range(N+1):
        amplitude = eigvecs[n, k] * coeffs[k]
        if np.abs(amplitude) > 1e-6:
            real_part = np.real(amplitude)
            decay = np.real(eigvals[k])
            terms.append(f"{real_part:.8f} * np.exp({decay:.8f} * t)")
    print(f"p_{n}(t) = " + " + ".join(terms))

