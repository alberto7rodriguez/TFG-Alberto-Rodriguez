import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import simpson
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution

g = 10.0         
M = 1.0          
T = 2.0            
x_stand = 1.00      
x_takeoff = 1.05    
h = 1.50            
L_max = 0.82        
sigma_minus = 0.045 
sigma_plus = 0.09   
w0 = 0.15          
ws = 0.85           
w_sigma = 0.09      


def alpha_max(x):
    if x >= x_takeoff:
        return 0.0
    elif x <= L_max:
        return g + 4 * g * np.exp(-((x - L_max)**2) / (2 * sigma_minus**2))
    else:
        return g + 4 * g * np.exp(-((x - L_max)**2) / (2 * sigma_plus**2))

def w(x):
    return w0 + ws * (1 - np.exp(-((x - L_max)**2) / (2 * w_sigma**2)))

def dw_dx(x):
    return ws * np.exp(-((x - L_max)**2) / (2 * w_sigma**2)) * ((x - L_max) / w_sigma**2)

def get_optimal_control(x, p2):
    alpha_free = p2 / (2 * w(x))
    a_max = alpha_max(x)
    return np.clip(alpha_free, 0.0, a_max)

def jump_dynamics(t, state):
    x, v, p1, p2 = state
    alpha = get_optimal_control(x, p2)
    dxdt = v
    dvdt = alpha - g
    dp1dt = dw_dx(x) * (alpha**2)
    dp2dt = -p1
    return [dxdt, dvdt, dp1dt, dp2dt]

def simulate_trajectory(t0, lambda1, lambda2):
    initial_state = [x_stand, 0.0, lambda1, lambda2]
    if t0 >= T:
        return None
    
    sol = solve_ivp(
        fun=jump_dynamics,
        t_span=(t0, T),
        y0=initial_state,
        method='LSODA',
        max_step=0.01,  
        dense_output=True
    )
    
    t_eval = sol.t
    states = sol.y
    x_vals = states[0]
    p2_vals = states[3]
    
    alpha_vals = np.array([get_optimal_control(x, p2) for x, p2 in zip(x_vals, p2_vals)])
    
    return t_eval, states, alpha_vals

v_takeoff = 3.0
t2 = 1.7
kappa_v = 0.5
sigma_x = 0.05
sigma_t = 0.08
A1 = 5.0
A2 = 20.0
A3 = 15.0
A4 = 25.0
B1 = 20.0
B2 = 10.0
B3 = 20.0
dx_launch = 0.025
v_min_launch = 2.0
v_scale = 1.0
delta_x = 0.03
x_safe = 0.65
Gamma_x = 25.0
Gamma_v = 10.0

def evaluate_trajectory(t_eval, states, alpha_vals, A_marker_weight=1.0):
    if t_eval is None or len(t_eval) == 0:
        return -1e6 
        
    x = states[0]
    v = states[1]
    v_min = np.min(v)
    x_min = np.min(x)
    x_max = np.max(x)
    
    d_t = np.sqrt((x - x_takeoff)**2 + kappa_v * (v - v_takeoff)**2)
    idx_star = np.argmin(d_t)
    d_star = d_t[idx_star]
    t_star = t_eval[idx_star]
    Q_counter = A1 * min(max(0, -v_min) / 2.0, 1.0)
    Q_takeoff = A2 * np.exp(-(d_star**2) / (sigma_x**2))
    Q_timing = A3 * np.exp(-((t_star - t2)**2) / (sigma_t**2))
    
    valid_launch_mask = (np.abs(x - x_takeoff) < dx_launch) & (v > v_min_launch)
    if np.any(valid_launch_mask):
        best_launch_v = np.max(v[valid_launch_mask])
        Q_launch = A4 * min((best_launch_v - v_min_launch) / v_scale, 1.0)
        Q_weak = 0.0
    else:
        Q_launch = 0.0
        Q_weak = B3 
        
    Q_overshoot = B1 * max(0, x_max - x_takeoff - delta_x)**2
    Q_crouch = B2 * max(0, x_safe - x_min)**2

    effort_integrand = np.array([w(xi) * (ai**2) for xi, ai in zip(x, alpha_vals)])
    muscular_effort = simpson(y=effort_integrand, x=t_eval)
    
    terminal_penalty = Gamma_x * (x[-1] - h)**2 + Gamma_v * (v[-1])**2
    P_final = -muscular_effort - terminal_penalty 
    marker_sum = Q_counter + Q_takeoff + Q_timing + Q_launch - Q_overshoot - Q_crouch - Q_weak
    Q_total = P_final + A_marker_weight * marker_sum
    
    return Q_total


def objective_function(params, A_weight):
    t0, lambda1, lambda2 = params
    result = simulate_trajectory(t0, lambda1, lambda2)
    
    if result is None:
        return 1e6 
        
    t_eval, states, alpha_vals = result
    Q_total = evaluate_trajectory(t_eval, states, alpha_vals, A_marker_weight=A_weight)
    
    return -Q_total

def plot_optimal_trajectory(t, states, alpha):
    x = states[0]
    v = states[1]
    a_max_vals = [alpha_max(xi) for xi in x]
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    axs[0].plot(t, x, 'b-', linewidth=2, label="CM height $x(t)$")
    axs[0].axhline(y=x_stand, color='k', linestyle=':', label="Standing (1.00m)")
    axs[0].axhline(y=x_takeoff, color='r', linestyle='--', label="Takeoff (1.05m)")
    axs[0].axhline(y=h, color='g', linestyle='--', label="Rim target (1.50m)")
    axs[0].set_ylabel("Height (m)")
    axs[0].set_title("CM trajectory")
    axs[0].legend()
    axs[0].grid(True)
    
    axs[1].plot(t, v, 'm-', linewidth=2, label="$v(t)$")
    axs[1].axhline(y=0, color='k', linestyle='-')
    axs[1].set_ylabel("Velocity (m/s)")
    axs[1].set_title("Velocity")
    axs[1].legend()
    axs[1].grid(True)
    
    axs[2].plot(t, alpha, 'g-', linewidth=2, label="Exerted force $\\alpha^*(t)$")
    axs[2].plot(t, a_max_vals, 'r--', alpha=0.6, label="Max available force $\\alpha_{max}(x)$")
    axs[2].set_xlabel("t (s)")
    axs[2].set_ylabel("Force / Mass (m/s²)")
    axs[2].set_title("Control Effort")
    axs[2].legend()
    axs[2].grid(True)
    
    plt.tight_layout()
    plt.show()

print("--- Phase 1: Global Exploration with Markers ON (A = 500) ---")

bounds = [(0.8, 1.5), (-40.0, -30.0), (0.6, 1.0)]

res_phase1 = differential_evolution(
    objective_function, 
    bounds, 
    args=(500.0,), 
    strategy='best1bin',
    maxiter=50,     
    popsize=15,     
    disp=True,      
    tol=0.01
)

best_params_phase1 = res_phase1.x
print(f"Phase 1 Best Params (t0, lambda1, lambda2): {best_params_phase1}")

t_p1, states_p1, alpha_p1 = simulate_trajectory(*best_params_phase1)
print("Plotting Phase 1 (Guided trajectory)... close the window to run Phase 2.")
plot_optimal_trajectory(t_p1, states_p1, alpha_p1)
print("--- Phase 2: Fine-Tuning Final Payoff (A = 0) ---")

res_phase2 = minimize(
    objective_function, 
    best_params_phase1, 
    args=(0.0,),   
    method='L-BFGS-B', 
    bounds=bounds,     
    options={'disp': True}
)

best_params_final = res_phase2.x
best_Q_final = -res_phase2.fun

print(f"Final Optimal Params (t0, lambda1, lambda2): {best_params_final}")
print(f"Maximum Total Final Payoff (P_final): {best_Q_final:.2f}")
final_t, final_states, final_alpha = simulate_trajectory(*best_params_final)
print("Plotting Phase 2 (Final Fine-Tuned Trajectory)...")

plot_optimal_trajectory(final_t, final_states, final_alpha)
