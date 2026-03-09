import numpy as np
from math import comb

N = 15
beta = 1

#J_values = [0.7112, 0.496, 0.3769, 0.3019, 0.2506, 0.2135]
J_values = np.zeros(40).tolist()

a_values = [-0.711, 0.000, 0.894, 2.015, 3.398, 5.070, 7.052, 9.358, 11.998, 14.977, 18.297, 21.960, 25.967, 30.318, 35.013, 40.053, 45.438, 51.168, 57.243, 63.664, 70.431, 77.543, 85.001, 92.805, 100.956, 109.452, 118.294, 127.483, 137.018, 146.899, 157.127, 167.701, 178.621, 189.887, 201.500, 213.460, 225.766, 238.418, 251.417, 264.762, 278.453, 292.492, 306.876, 321.607, 336.685, 352.109, 367.879, 383.996, 400.460]
b_values = [0.711, 0.797, 0.894, 1.007, 1.133, 1.267, 1.410, 1.560, 1.714, 1.872, 2.033, 2.196, 2.361, 2.527, 2.693, 2.861, 3.029, 3.198, 3.367, 3.537, 3.707, 3.877, 4.048, 4.218, 4.389, 4.560, 4.732, 4.903, 5.075, 5.246, 5.418, 5.590, 5.762, 5.934, 6.106, 6.278, 6.450, 6.623, 6.795, 6.967, 7.140, 7.312, 7.485, 7.657, 7.830, 8.002, 8.175, 8.348, 8.520]

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











