import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
N = 30           # Grid size (30x30 mesh)
M = 500          # Number of random walkers per grid point
V_left = 10.0    # Potential on the left boundary
V_other = 1.0    # Potential on the top, right, and bottom boundaries

# Initialize the potential matrix
# We will store the calculated potentials here.
V = np.zeros((N, N))

# --- Set Boundary Conditions ---
# y-axis is the first index (rows), x-axis is the second index (columns)
V[:, 0] = V_left       # Left boundary (x = 0)
V[:, -1] = V_other     # Right boundary (x = N-1)
V[0, :] = V_other      # Bottom boundary (y = 0)
V[-1, :] = V_other     # Top boundary (y = N-1)

# --- Define the Random Walk Function ---
def random_walk_potential(start_x, start_y, M, N):
    """
    Runs M random walkers from (start_x, start_y) until they hit a boundary.
    Returns the average boundary potential.
    """
    total_boundary_potential = 0.0
    
    # Possible moves: (dx, dy) -> Right, Left, Up, Down
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    for _ in range(M):
        x, y = start_x, start_y
        
        # Walk until hitting a boundary
        while 0 < x < N - 1 and 0 < y < N - 1:
            dx, dy = moves[np.random.randint(0, 4)] # Pick a random direction
            x += dx
            y += dy
            
        # Check which boundary was hit and add the corresponding potential
        if x == 0:
            total_boundary_potential += V_left
        else:
            total_boundary_potential += V_other
            
    # Calculate the average potential for this starting point
    return total_boundary_potential / M

# --- Run the Simulation ---
print(f"Starting simulation on a {N}x{N} grid with {M} walkers per point...")

# We only loop through the interior points (1 to N-2)
# because the boundaries are already fixed.
for j in range(1, N - 1):      # y-coordinates
    for i in range(1, N - 1):  # x-coordinates
        V[j, i] = random_walk_potential(i, j, M, N)

print("Simulation complete! Generating heatmap...")

# --- Plot the Heatmap ---
plt.figure(figsize=(8, 6))
# Using 'gnuplot' or 'plasma' colormap to match the style in the document
heatmap = plt.imshow(V, origin='lower', extent=[0, 1, 0, 1], cmap='plasma', interpolation='bilinear')
plt.colorbar(heatmap, label='Potential')
plt.title('Electrostatic Potential via Discrete Random Walk')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()