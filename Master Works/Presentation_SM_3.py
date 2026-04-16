import numpy as np
import matplotlib.pyplot as plt

# Define dimensionless scattering length (a / a_ho)
# We use a slightly larger 'a' here just to make the cutoff visually obvious on the plot
a_tilde = 0.05 

# Create an array of distances (r / a_ho)
r = np.linspace(0.001, 1.0, 1000)

# Calculate g_2(r)
g2 = np.zeros_like(r)

# Apply the hard-sphere condition
for i, r_val in enumerate(r):
    if r_val > a_tilde:
        g2[i] = (1 - a_tilde / r_val)**2
    else:
        g2[i] = 0.0 # Strict hard-sphere cutoff

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(r, g2, 'b-', linewidth=2.5, label=r'$g_2(r) = (1 - a/r)^2$')

# Highlight the cutoff region
plt.axvspan(0, a_tilde, color='red', alpha=0.2, label='Excluded Volume ($r < a$)')
plt.axvline(x=a_tilde, color='red', linestyle='--')

plt.title(f"Two-Body Correlation Function (Hard Sphere $a/a_{{ho}} = {a_tilde}$)")
plt.xlabel(r"Relative Distance $r$ (units of $a_{ho}$)")
plt.ylabel(r"$g_2(r)$")
plt.xlim(0, 0.5)
plt.ylim(-0.05, 1.1)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()