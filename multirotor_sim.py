import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Parameters
def get_params():
    I = 1.0    # Moment of inertia
    g = 9.8    # Gravitational acceleration (m/s^2)
    m = 1.0    # Multirotor mass
    return I, g, m

# Controller (no input for verification)
def controller(x):
    return np.array([0.0, 0.0])

# Dynamics: Xdot = f(X, u)
def multirotor_system(x, t, controller):
    I, g, m = get_params()

    u = controller(x)
    dx = np.zeros(6)
    dx[0] = x[2]                             # x1_dot = v1
    dx[1] = x[3]                             # x2_dot = v2
    dx[2] = -(1/m) * u[0] * np.sin(x[4])     # v1_dot = -u1*sin(theta)/m
    dx[3] = (1/m) * u[0] * np.cos(x[4]) - g  # v2_dot = u1*cos(theta)/m - g
    dx[4] = x[5]                             # theta_dot = omega
    dx[5] = (1/I) * u[1]                     # omega_dot = u2/I
    return dx

# -Simulation
def simulate_multirotor(x0, tfinal, timestep, controller):
    t = np.arange(0, tfinal, timestep)
    sol = odeint(multirotor_system, x0, t, args=(controller,))
    return sol

# Plot
def plot_trajectories(sol, timestep, save_path=None):
    t = np.arange(0, sol.shape[0] * timestep, timestep)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(t, sol[:, 0], label='x1 (horizontal position)')
    axes[0].plot(t, sol[:, 1], label='x2 (vertical position)')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Position (m)')
    axes[0].set_title('Multirotor Position Trajectories')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t, sol[:, 4], label='theta (angle from horizontal)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Angle (rad)')
    axes[1].set_title('Multirotor Angle Trajectories')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

# Run verification simulation
x0 = [0, 2, 10, 10, 0, 0]
tfinal = 5.0
timestep = 0.01

solvec = simulate_multirotor(x0, tfinal, timestep, controller)
plot_trajectories(solvec, timestep, save_path='verification_plot.png')
