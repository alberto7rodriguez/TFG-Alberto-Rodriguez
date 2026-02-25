import numpy as np
from math import comb

N = 4
beta = 1

J_values = [0.7112, 0.496, 0.3769, 0.3019, 0.2506, 0.2135]

a_values = [-0.711, 0.000, 0.894, 2.015, 3.398, 5.070]
b_values = [0.711, 0.797, 0.894, 1.007, 1.133, 1.267]

J = J_values[N-2]
a = a_values[N-2]
b = b_values[N-2]


def U(sistema,N,n):
    if sistema == 1: #All-To-All
        return J*(-N*(N+1)*0.5 + 2*(n+1)*(N-n))
    if sistema == 2: #Star Model
        if n == 0:
            return - a
        else:
            return a + 2* b* (2*(n-1) - (N - 1))

def g(sistema, n):
    if sistema == 1:
        return comb(N, n)
    if sistema == 2:
        k = n-1
        if k == -1:
            return 2**(N-1)
        else:
            return comb(N-1, k)        

def gibbs_distribution(sistema, N, beta):
    energies = np.array([U(sistema, N, n) for n in range(N+1)])
    degeneracies = np.array([g(sistema, n) for n in range(N+1)])
    weights = degeneracies * np.exp(-beta * energies)
    Z = np.sum(weights)
    probabilities = weights / Z
    return probabilities

p_all_gibbs = gibbs_distribution(1, N, beta)
p_star_gibbs = gibbs_distribution(2, N, beta)

"""
print("Gibbs (thermal) probabilities for the All-To-All Model:")
for n, p in enumerate(p_all_gibbs):
    print(f"p_{n} = {p:.5f}")
print("Sum of probabilities:", np.sum(p_all_gibbs))  # should be ~1


E_all_gibbs = np.sum(p_all_gibbs * np.array([U(1, N, n) for n in range(N+1)]))
print("Energy of the Gibbs state:", E_all_gibbs)

print()
"""
print("Gibbs (thermal) probabilities for the Star Model:")
for n, p in enumerate(p_star_gibbs):
    print(f"p_{n} = {p:.5f}")
print("Sum of probabilities:", np.sum(p_star_gibbs))  # should be ~1

E_star_gibbs = np.sum(p_star_gibbs * np.array([U(2, N, n) for n in range(N+1)]))
print("Energy of the Gibbs state:", E_star_gibbs)











