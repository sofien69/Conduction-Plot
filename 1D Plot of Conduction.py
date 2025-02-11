import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
Lx, Ly, Lz = 1.0, 1.0, 1.0  # Dimensions of the 3D object (meters)
Nx, Ny, Nz = 20, 20, 20     # Number of points in each dimension
dx, dy, dz = Lx / (Nx - 1), Ly / (Ny - 1), Lz / (Nz - 1)  # Step sizes
T_initial = 300             # Initial temperature (K)
T_boundary = 400            # Boundary temperature (K)
k = 200                     # Thermal conductivity (W/m·K)
rho = 8000                  # Density (kg/m³)
cp = 500                    # Specific heat capacity (J/kg·K)
dt = 0.01                   # Time step (s)
t_max = 0.5                  # Total simulation time (s)

# Stability criterion for explicit method
alpha = k / (rho * cp)  # Thermal diffusivity (m²/s)
stability_criterion = alpha * dt * (1 / dx**2 + 1 / dy**2 + 1 / dz**2)
if stability_criterion > 0.5:
    raise ValueError("Stability condition not satisfied. Reduce dt or increase spatial resolution.")

# Initialization
T = np.ones((Nx, Ny, Nz)) * T_initial
T[:, :, 0] = T_boundary  # Set one face to boundary temperature

# Simulation loop
time_steps = int(t_max / dt)

for _ in range(time_steps):
    T_new = T.copy()
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            for k in range(1, Nz - 1):
                T_new[i, j, k] = T[i, j, k] + alpha * dt * (
                    (T[i + 1, j, k] - 2 * T[i, j, k] + T[i - 1, j, k]) / dx**2 +
                    (T[i, j + 1, k] - 2 * T[i, j, k] + T[i, j - 1, k]) / dy**2 +
                    (T[i, j, k + 1] - 2 * T[i, j, k] + T[i, j, k - 1]) / dz**2
                )
    T = T_new

# Visualization
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
z = np.linspace(0, Lz, Nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

def plot_3d_temperature(T):
    """Plot the temperature distribution in 3D."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Select a 2D slice from the 3D data (e.g., middle of the z-axis)
    slice_index = Nz // 2
    temp_slice = T[:, :, slice_index]
    
    surf = ax.plot_surface(X[:, :, slice_index], Y[:, :, slice_index], temp_slice, cmap='hot')
    ax.set_title("3D Heat Transfer (Middle Slice)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Temperature (K)")
    fig.colorbar(surf, shrink=0.5, aspect=10, label="Temperature (K)")
    plt.show()

# Plot the result
plot_3d_temperature(T)
