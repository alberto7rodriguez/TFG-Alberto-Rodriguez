import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

# Data
x_values = np.array([3, 5, 7, 9, 11, 15, 18, 20])
y_values = np.array([10.684084084084084, 10.803903903903903, 11.283183183183182,
                     11.602702702702702, 11.722522522522523, 12.12192192192192,
                     13.599699699699698, 14.358558558558558])

x_not = [13]
y_not = [10.244744744744745]

# Models
def linear(x, a, b): return a * x + b
def log_model(x, a, b): return a * np.log(x) + b
def power_model(x, a, b): return a * x**b
def exp_model(x, a, b, c): return a * np.exp(b * x) + c

models = {
    "Linear": (linear, 2),
    "Logarithmic": (log_model, 2),
    "Power": (power_model, 2),
    "Exponential": (exp_model, 3)
}

results = {}
x_fit = np.linspace(min(x_values), max(x_values), 200)

# Fit each model and calculate R²
for name, (func, _) in models.items():
    try:
        popt, _ = curve_fit(func, x_values, y_values, maxfev=5000)
        y_pred = func(x_values, *popt)
        r2 = r2_score(y_values, y_pred)
        results[name] = {
            "params": popt,
            "r2": r2,
            "y_fit": func(x_fit, *popt)
        }
    except RuntimeError:
        continue

# Best model
best_model_name, best_model_data = max(results.items(), key=lambda x: x[1]['r2'])

# Plot
plt.figure(figsize=(9, 4))
plt.scatter(x_values, y_values, marker='x', color="black")
plt.scatter(x_not, y_not, marker='x', color="red")
for name, res in results.items():
    plt.plot(x_fit, res["y_fit"], label=f"{name} (R²={res['r2']:.4f})")
plt.ylabel("$t_{th}$")
plt.xlabel("$N$")
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.tick_params(axis='both', direction='in', length=6)
plt.xticks([3, 5, 7, 9, 11, 13, 15, 18, 20])
plt.xlim(2.5, 20.5)
plt.show()


# Output best model info
print("Best Model:", best_model_name)
print("Parameters:", best_model_data["params"])
print("R² Score:", best_model_data["r2"])

