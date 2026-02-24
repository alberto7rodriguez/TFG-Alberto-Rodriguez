import numpy as np
from scipy.linalg import expm
from scipy.linalg import eig, inv


a_values = ['list of a values']
b_values = ['list of b values']

def U(N,n):
    if n == 0:
        return - a
    else:
        return a + 2* b* (2*(n-1) - (N - 1)) 

def f(x, beta):
    return 1 / (1 + np.exp(beta * x))

def transition_matrix(N, beta):
    M = np.zeros((N+1, N+1))

    for n in range(N+1):
        # Transitions from n to n+1
        if n == 0:
           delta_U = U(N,n+1) - U(N,n)
           rate_up = f(delta_U, beta)
           M[n, n] -= rate_up
           M[n+1, n] += rate_up
           
        if n < N and n != 0:
            delta_U = U(N,n+1) - U(N,n)
            rate_up = (N-n)*f(delta_U, beta)
            M[n, n] -= rate_up
            M[n+1, n] += rate_up

        # Transitions from n to n-1
        if n > 0:
            if n==1:
                delta_U = U(N,n-1) - U(N,n)
                rate_down = 2**(N-1) * f(delta_U, beta) 
                M[n, n] -= rate_down
                M[n-1, n] += rate_down
                
            else:
                delta_U = U(N,n-1) - U(N,n)
                rate_down = (n-1)*f(delta_U, beta) 
                M[n, n] -= rate_down
                M[n-1, n] += rate_down

    return M

# Parameters
N = 17
a = a_values[N-2]
b = b_values[N-2]
beta = 1
t = 20.0  # time at which we evaluate the solution

for n in range(N+1):
    print(f"E_{n} = {U(N,n)}")

# Initial condition: start in state 0 (e.g., all probability in p_0)
p0 = np.full(N+1, 1/(N+1))
'''
p0 = np.zeros(N+1)
p0[0] = 1
p0[1] = 0
p0[2] = 0
'''

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
    print()

