import numpy as np
import sympy as sp
from scipy.special import comb

beta = sp.symbols('β')

N_values = [2,3,4,5,6,7]
J_values = [0.7112, 0.496, 0.3769, 0.3019, 0.2506, 0.2135]
a_values = [-0.711, 0.000, 0.894, 2.015, 3.398, 5.070]
b_values = [0.711, 0.797, 0.894, 1.007, 1.133, 1.267]


'''Star Model'''

print()
print('STAR MODEL')
print()  
    
for N in N_values:
    index = N-2

    print('Energy values for N =',N)
    Z = 0
    E_deg = -a_values[index]
    deg = 2**(N-1)
    Z = deg * sp.exp(-E_deg * beta)
    print(f"Spin off | E_deg = {E_deg:6.3f} | Degeneracy = {deg}")
    for k in range(N):
            E_k = a_values[index] + 2 * b_values[index] * (2 * k - (N - 1))
            degeneracion = int(comb(N-1, k))
            Z += degeneracion * sp.exp(-E_k * beta) 
            print(f"k = {k:0d} | E_{k:0d} = {E_k:6.3f} | Degeneracy = {degeneracion}")
    sp.latex(Z)
    print(f"Z = {Z}")
    print()
    

'''All-To-All Model'''

print()
print('All-To-All Model')
print()

for N in N_values:
    
    index = N - 2
    print('Energy values for N =',N , '- With J =', J_values[index])
    Z = 0
    
    for k in range(N+1):
            E_k = J_values[index] * (-0.5*N*(N+1) + 2*(k+1)*(N-k))
            degeneracion = int(comb(N, k))
            Z += degeneracion * sp.exp(-E_k * beta) 
            print(f"k = {k:0d} | E_{k:0d} = {E_k:6.3f} | Degeneracy = {degeneracion}")
    sp.latex(Z)
    print(f"Z = {Z}")
    print()