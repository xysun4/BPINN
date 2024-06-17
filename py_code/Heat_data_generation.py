import numpy as np
from numpy.random import default_rng

# Define constants
N_m = 5000  # Number of measurements
k = 0.5  # Decay constant
L = np.pi  # Spatial domain bound
T_end = 1  # Time domain bound
noise_level = 0.01  # Noise level for observations
N_sb = 128  # Number of boundary points in space
N_tb = 128  # Number of boundary points in time
N_int = int(100**2)  # Number of interior points

# Function to calculate the exact solution
def u_star(x, t):
    return np.sin(x) * np.exp(-k * t)

# Initialize random number generator
rng = default_rng(seed=1)

# Generate measurement points in space and time
x_vals = rng.uniform(0, L, N_m)
t_vals = rng.uniform(0, T_end, N_m)

# Generate sensor points in space and time
x_vals_sensor = rng.uniform(0, L, N_m)
t_vals_sensor = rng.uniform(0, T_end, N_m)
u_true = u_star(x_vals_sensor, t_vals_sensor) 

# Add noise to the true observations
mea_sig = noise_level * np.std(u_true)
y_obs_int = u_true + rng.normal(0, mea_sig, u_true.shape)

# Generate boundary condition points when t = 0 and t = 1
t_vals_tb = np.random.choice([0, T_end], N_tb, replace=True)
x_vals_tb = rng.uniform(0, L, N_tb)
u_vals_tb = u_star(x_vals_tb, t_vals_tb)

# Generate boundary condition points when x = 0 or L
t_vals_sb = rng.uniform(0, T_end, N_sb)
x_vals_sb = np.random.choice([0, L], N_sb, replace=True)
u_vals_sb = u_star(x_vals_sb, t_vals_sb)

# Generate interior points
x_vals_int = rng.uniform(0, L, N_int)
t_vals_int = rng.uniform(0, T_end, N_int)

# Generating the required data arrays
d_tb = np.column_stack((x_vals_tb, t_vals_tb, u_vals_tb))
d_sb = np.column_stack((x_vals_sb, t_vals_sb, u_vals_sb))
data = np.column_stack((x_vals_sensor, t_vals_sensor, y_obs_int))
inter = np.column_stack((x_vals, t_vals))
