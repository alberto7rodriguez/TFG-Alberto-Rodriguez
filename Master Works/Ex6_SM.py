import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import simpson
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution

# --- 1. Physical Constants & Parameters ---
g = 10.0            # Gravity [cite: 12]
M = 1.0             # Normalized mass [cite: 16]
T = 2.0             # Target time at the rim [cite: 7]

# State targets and bounds
x_stand = 1.00      # Standing height [cite: 34]
x_takeoff = 1.05    # Takeoff height [cite: 36]
h = 1.50            # Target rim height [cite: 38]

# Force model parameters
L_max = 0.82        # Optimal crouch height [cite: 41, 42]
sigma_minus = 0.045 # [cite: 50]
sigma_plus = 0.09   # [cite: 50]

# Weight function parameters for muscular effort
w0 = 0.15           # [cite: 59]
ws = 0.85           # [cite: 59]
w_sigma = 0.09      # [cite: 59]

# --- 2. Helper Functions ---

def alpha_max(x):
    """Maximum upward force depending on center of mass height [cite: 46-54]."""
    if x >= x_takeoff:
        return 0.0
    elif x <= L_max:
        return g + 4 * g * np.exp(-((x - L_max)**2) / (2 * sigma_minus**2))
    else:
        return g + 4 * g * np.exp(-((x - L_max)**2) / (2 * sigma_plus**2))

def w(x):
    """Weight function for muscular effort[cite: 58]."""
    return w0 + ws * (1 - np.exp(-((x - L_max)**2) / (2 * w_sigma**2)))

def dw_dx(x):
    """Derivative of the weight function w.r.t x[cite: 87]."""
    return ws * np.exp(-((x - L_max)**2) / (2 * w_sigma**2)) * ((x - L_max) / w_sigma**2)

def get_optimal_control(x, p2):
    """Calculates the optimal clipped control alpha*(t) [cite: 93-95]."""
    alpha_free = p2 / (2 * w(x))
    a_max = alpha_max(x)
    return np.clip(alpha_free, 0.0, a_max)

# --- 3. ODE System & Integrator ---

def jump_dynamics(t, state):
    """
    Computes the derivatives of the state and costate variables [cite: 79-86].
    state = [x, v, p1, p2]
    """
    x, v, p1, p2 = state
    
    # Get the bounded optimal control for the current state
    alpha = get_optimal_control(x, p2)
    
    # State equations [cite: 79, 80]
    dxdt = v
    dvdt = alpha - g
    
    # Costate equations [cite: 81-86]
    dp1dt = dw_dx(x) * (alpha**2)
    dp2dt = -p1
    
    return [dxdt, dvdt, dp1dt, dp2dt]

def simulate_trajectory(t0, lambda1, lambda2):
    """
    Integrates the trajectory from t0 to T[cite: 108].
    Returns the time array, state array, and computed control array.
    """
    # Initial conditions at t0 [cite: 104-106]
    initial_state = [x_stand, 0.0, lambda1, lambda2]
    
    # Ensure t0 doesn't exceed T
    if t0 >= T:
        return None
    
    # Integrate using Radau or LSODA (stiff solvers handle clipping bounds well)
    sol = solve_ivp(
        fun=jump_dynamics,
        t_span=(t0, T),
        y0=initial_state,
        method='LSODA',
        max_step=0.01,  # Keep steps small to catch the takeoff event accurately
        dense_output=True
    )
    
    # Reconstruct the control history for plotting and payoff calculation
    t_eval = sol.t
    states = sol.y
    x_vals = states[0]
    p2_vals = states[3]
    
    alpha_vals = np.array([get_optimal_control(x, p2) for x, p2 in zip(x_vals, p2_vals)])
    
    return t_eval, states, alpha_vals

# --- 4. Marker Parameters ---
# Takeoff targets [cite: 126-129]
v_takeoff = 3.0
t2 = 1.7
kappa_v = 0.5
sigma_x = 0.05
sigma_t = 0.08

# Marker Weights (A and B coefficients) [cite: 139-163]
A1 = 5.0
A2 = 20.0
A3 = 15.0
A4 = 25.0
B1 = 20.0
B2 = 10.0
B3 = 20.0

# Thresholds [cite: 150-159]
dx_launch = 0.025
v_min_launch = 2.0
v_scale = 1.0
delta_x = 0.03
x_safe = 0.65

# Final payoff weights 
Gamma_x = 25.0
Gamma_v = 10.0

# --- 5. Payoff Evaluation ---

def evaluate_trajectory(t_eval, states, alpha_vals, A_marker_weight=1.0):
    """
    Evaluates the trajectory and calculates the total score Q [cite: 164-165].
    """
    if t_eval is None or len(t_eval) == 0:
        return -1e6 # Heavy penalty for failed integration
        
    x = states[0]
    v = states[1]
    
    # Global extrema [cite: 130]
    v_min = np.min(v)
    x_min = np.min(x)
    x_max = np.max(x)
    
    # Takeoff-state distance [cite: 131-132]
    d_t = np.sqrt((x - x_takeoff)**2 + kappa_v * (v - v_takeoff)**2)
    idx_star = np.argmin(d_t)
    d_star = d_t[idx_star]
    t_star = t_eval[idx_star]
    
    # --- Rewards ---
    # 1. Countermovement [cite: 138-139]
    Q_counter = A1 * min(max(0, -v_min) / 2.0, 1.0)
    
    # 2. Takeoff-state [cite: 140-141]
    Q_takeoff = A2 * np.exp(-(d_star**2) / (sigma_x**2))
    
    # 3. Timing [cite: 142-143]
    Q_timing = A3 * np.exp(-((t_star - t2)**2) / (sigma_t**2))
    
    # 4. Launch [cite: 144-153]
    valid_launch_mask = (np.abs(x - x_takeoff) < dx_launch) & (v > v_min_launch)
    if np.any(valid_launch_mask):
        # Find the best valid launch velocity
        best_launch_v = np.max(v[valid_launch_mask])
        Q_launch = A4 * min((best_launch_v - v_min_launch) / v_scale, 1.0)
        Q_weak = 0.0
    else:
        Q_launch = 0.0
        # Weak-jump penalty applies if no launch detected [cite: 160-163]
        Q_weak = B3 
        
    # --- Penalties ---
    # 5. Overshoot [cite: 156-157]
    Q_overshoot = B1 * max(0, x_max - x_takeoff - delta_x)**2
    
    # 6. Excessive crouch [cite: 158-159]
    Q_crouch = B2 * max(0, x_safe - x_min)**2
    
    # --- Final Payoff (P_final) --- [cite: 68-71]
    # Note: Terminal conditions are usually formulated as penalties.
    # We subtract the terminal distance so that maximizing P_final means hitting the target.
    effort_integrand = np.array([w(xi) * (ai**2) for xi, ai in zip(x, alpha_vals)])
    muscular_effort = simpson(y=effort_integrand, x=t_eval)
    
    terminal_penalty = Gamma_x * (x[-1] - h)**2 + Gamma_v * (v[-1])**2
    P_final = -muscular_effort - terminal_penalty 
    
    # --- Total Score Q --- [cite: 164-165]
    marker_sum = Q_counter + Q_takeoff + Q_timing + Q_launch - Q_overshoot - Q_crouch - Q_weak
    Q_total = P_final + A_marker_weight * marker_sum
    
    return Q_total

# --- 7. The Optimization Loop ---

def objective_function(params, A_weight):
    """
    The function we want to MINIMIZE. 
    Since our goal is to MAXIMIZE the total score Q, we return -Q.
    params = [t0, lambda1, lambda2]
    """
    t0, lambda1, lambda2 = params
    
    # Run the simulation
    result = simulate_trajectory(t0, lambda1, lambda2)
    
    # If the integration failed or t0 >= T, return a massive penalty
    if result is None:
        return 1e6 
        
    t_eval, states, alpha_vals = result
    
    # Calculate the total score
    Q_total = evaluate_trajectory(t_eval, states, alpha_vals, A_marker_weight=A_weight)
    
    return -Q_total

def plot_optimal_trajectory(t, states, alpha):
    x = states[0]
    v = states[1]
    
    # Calculate the maximum available force at each point for visualization
    a_max_vals = [alpha_max(xi) for xi in x]
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # 1. Height Plot
    axs[0].plot(t, x, 'b-', linewidth=2, label="CM Height $x(t)$")
    axs[0].axhline(y=x_stand, color='k', linestyle=':', label="Standing (1.00m)")
    axs[0].axhline(y=x_takeoff, color='r', linestyle='--', label="Takeoff (1.05m)")
    axs[0].axhline(y=h, color='g', linestyle='--', label="Rim Target (1.50m)")
    axs[0].set_ylabel("Height (m)")
    axs[0].set_title("Optimal Jump: Center of Mass Trajectory")
    axs[0].legend()
    axs[0].grid(True)
    
    # 2. Velocity Plot
    axs[1].plot(t, v, 'm-', linewidth=2, label="Vertical Velocity $v(t)$")
    axs[1].axhline(y=0, color='k', linestyle='-')
    axs[1].set_ylabel("Velocity (m/s)")
    axs[1].set_title("Optimal Jump: Velocity")
    axs[1].legend()
    axs[1].grid(True)
    
    # 3. Control Force Plot
    axs[2].plot(t, alpha, 'g-', linewidth=2, label="Exerted Force $\\alpha^*(t)$")
    axs[2].plot(t, a_max_vals, 'r--', alpha=0.6, label="Max Available Force $\\alpha_{max}(x)$")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Force / Mass (m/s²)")
    axs[2].set_title("Optimal Jump: Control Effort")
    axs[2].legend()
    axs[2].grid(True)
    
    plt.tight_layout()
    plt.show()

# --- Phase 1: Global Exploration with Markers ON (A = 500) ---
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

# Plot Phase 1 to prove it works!
t_p1, states_p1, alpha_p1 = simulate_trajectory(*best_params_phase1)
print("Plotting Phase 1 (Guided trajectory)... close the window to run Phase 2.")
plot_optimal_trajectory(t_p1, states_p1, alpha_p1)

# --- Phase 2: Fine-Tuning (Markers OFF) ---
print("--- Phase 2: Fine-Tuning Final Payoff (A = 0) ---")

# We use L-BFGS-B here as it handles bounds well for fine-tuning
res_phase2 = minimize(
    objective_function, 
    best_params_phase1, 
    args=(0.0,),   
    method='L-BFGS-B', 
    bounds=bounds,      # <-- THIS WAS MISSING BEFORE!
    options={'disp': True}
)

best_params_final = res_phase2.x
best_Q_final = -res_phase2.fun

print(f"Final Optimal Params (t0, lambda1, lambda2): {best_params_final}")
print(f"Maximum Total Final Payoff (P_final): {best_Q_final:.2f}")

# Plot the Final Phase 2 Trajectory
final_t, final_states, final_alpha = simulate_trajectory(*best_params_final)
print("Plotting Phase 2 (Final Fine-Tuned Trajectory)...")

plot_optimal_trajectory(final_t, final_states, final_alpha)
