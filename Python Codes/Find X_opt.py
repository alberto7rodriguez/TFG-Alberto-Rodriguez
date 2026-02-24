import numpy as np
from scipy.optimize import fsolve

# Lista para almacenar los valores de x encontrados
x_values = []

# Iterar sobre N de 1 a 33
for N in range(1, 51):
    D = 2**N  # Calcular D
    
    # Valores iniciales para buscar soluciones
    initial_guesses = [-3, 0, 3]

    # Definir la ecuación a resolver
    def equation(x):
        return np.exp(x) - (D - 1) * (x + 2) / (x - 2)

    # Encontrar soluciones numéricas
    solutions = [fsolve(equation, x0)[0] for x0 in initial_guesses]
    unique_solutions = np.unique(np.round(solutions, decimals=6))

    # Seleccionar la solución positiva
    for x_i in unique_solutions:
        if x_i > 0:
            x_values.append(x_i)
            break  # Solo tomamos la primera solución positiva encontrada

# Mostrar la lista de valores de x encontrados
x_values

print(x_values)
