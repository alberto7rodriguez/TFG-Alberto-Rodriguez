import matplotlib.pyplot as plt
import numpy as np

# 1. CONFIGURACIÓN DE DATOS
temperaturas = [10, 50, 150]
reff_1 = 2.553  # Distancia de referencia del modelo Feff

# --- MODELO 1: Solo Single Scattering (SS) ---
dr1_ss = [-0.00718591, -0.00695958, -0.00424897]  # Sustituye con tus dr_1
err_dr1_ss = [0.00371005, 0.00370213, 0.00585493] # Sustituye con +/- de dr_1
ss1_ss = [0.00349473, 0.00358410, 0.00539554]     # Sustituye con tus ss1
err_ss1_ss = [0.00044328, 0.00044529, 0.00092248] # Sustituye con +/- de ss1

# --- MODELO 2: SS + Otros términos (Multiple Scattering) ---
dr1_ms = [-0.00536002, -0.00458188, -0.00357527]  # Sustituye con tus dr_1 del 2º fit
err_dr1_ms = [0.00363396, 0.00244577, 0.00373670] # Sustituye con +/-
ss1_ms = [0.00362564, 0.00372811, 0.00555463]     # Sustituye con tus ss1 del 2º fit
err_ss1_ms = [0.00043627, 0.00031230, 0.00058999] # Sustituye con +/-

# Calcular R final (R = Reff + deltaR)
r1_ss = [reff_1 + x for x in dr1_ss]
r1_ms = [reff_1 + x for x in dr1_ms]

# 2. CREACIÓN DE LAS GRÁFICAS
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Gráfica A: Distancia R vs T ---
ax1.errorbar(temperaturas, r1_ss, yerr=err_dr1_ss, fmt='-o', color='blue', 
             capsize=5, label='Solo Single Scattering')
ax1.errorbar(temperaturas, r1_ms, yerr=err_dr1_ms, fmt='--s', color='cyan', 
             capsize=5, label='SS + Multiple Scattering')

ax1.set_xlabel('T (K)')
ax1.set_ylabel('R (Å)')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.6)

# --- Gráfica B: Sigma^2 vs T ---
ax2.errorbar(temperaturas, ss1_ss, yerr=err_ss1_ss, fmt='-o', color='red', 
             capsize=5, label='Solo Single Scattering')
ax2.errorbar(temperaturas, ss1_ms, yerr=err_ss1_ms, fmt='--s', color='orange', 
             capsize=5, label='SS + Multiple Scattering')

ax2.set_xlabel('T(K)')
ax2.set_ylabel('$\sigma^2$ (Å$^2$)')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()