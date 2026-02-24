import numpy as np
import matplotlib.pyplot as plt

e = np.e
T_array = np.arange(0.001, 1.001, 0.001)  # Evitamos que T_i sea 0
T = T_array.tolist()

N_array = np.arange(2, 12, 2)
N = N_array.tolist()

F = []

F_ho = []

for N_i in N:
  F_i = []
  for T_i in T:
      x = 1 / T_i
      if x > 200:
          F_ii = 0
      else:
          F_ii = ((e**x) * (x**4) * (N_i - 1)) / ((N_i - 1 + e**x)**2)
      F_i.append(F_ii)
  F.append(F_i)
  
for T_i in T:
    Fi_ho = (1/(4*(T_i)**4))*(1/((np.sinh(1/(2*T_i)))**2))
    F_ho.append(Fi_ho)

F_N2 = F[0]
F_N4 = F[1]
F_N6 = F[2]
F_N8 = F[3]
F_N10 = F[4]

F_normalN2 = []
F_normalN10 = []

for i in F_N2:
  F_normalN2.append(i / max(F_N2))

for i in F_N10:
  F_normalN10.append(i / max(F_N10))


# Main plot
fig, ax = plt.subplots(figsize=(10, 4.5))

# Main curves
ax.plot(T, F_ho, label='Harm. Osc.', linestyle='--', color='silver', alpha=0.8)
ax.plot(T, F_N2, label='D=2')
ax.plot(T, F_N4, label='D=4')
ax.plot(T, F_N6, label='D=6')
ax.plot(T, F_N8, label='D=8')
ax.plot(T, F_N10, label='D=10', color='purple')

# Main plot formatting
ax.tick_params(axis='both', direction='in', length=6)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.set_xlim(min(T), max(T))
ax.set_ylim(0, 36)
ax.set_xlabel(r"$\mathcal{T}$")
ax.set_ylabel(r"$\mathcal{F}$")

# Legend at the top, outside the main plot
legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.11), ncol=6)
legend.get_frame().set_linewidth(0)

# Inset plot
inset_ax = fig.add_axes([0.515, 0.39, 0.37, 0.46])  # [left, bottom, width, height] in figure coords
inset_ax.plot(T, F_normalN2, label='D=2', linestyle='--')
inset_ax.plot(T, F_normalN10, label='D=10', color='purple')

inset_ax.tick_params(axis='both', direction='in', length=6)
inset_ax.spines['top'].set_visible(True)
inset_ax.spines['right'].set_visible(True)
inset_ax.set_xlim(min(T), max(T))
inset_ax.set_ylim(0, 1.1)
inset_ax.set_ylabel(r"$\mathcal{F}/\mathcal{F}_{\mathrm{max}}$", fontsize=8)
inset_ax.tick_params(labelsize=8)
inset_ax.legend(fontsize=8)

plt.show()
