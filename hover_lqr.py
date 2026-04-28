import numpy as np
from scipy.integrate import odeint
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt

# Parameters
def get_params():
    I = 1.0    # Moment of inertia
    g = 9.8    # Gravitational acceleration (m/s^2)
    m = 1.0    # Multirotor mass
    return I, g, m

# Linearization matrices
A = np.array([
    [0, 0, 1, 0, 0,    0],
    [0, 0, 0, 1, 0,    0],
    [0, 0, 0, 0, -9.8, 0],
    [0, 0, 0, 0, 0,    0],
    [0, 0, 0, 0, 0,    1],
    [0, 0, 0, 0, 0,    0]
])

B = np.array([
    [0, 0],
    [0, 0],
    [0, 0],
    [1, 0],
    [0, 0],
    [0, 1]
])

# LQR gain
Q = np.diag([10, 10, 1, 1, 10, 1])   # state error penalty
R = np.diag([1, 1])                  # input cost

P = solve_continuous_are(A, B, Q, R)
K = -np.linalg.inv(R) @ B.T @ P

print("LQR gain K:")
print(np.round(K, 4))

# Nonlinear dynamics
def multirotor_system(x, t, controller):
    I, g, m = get_params()
    u = controller(x)
    dx = np.zeros(6)
    dx[0] = x[2]
    dx[1] = x[3]
    dx[2] = -(1/m) * u[0] * np.sin(x[4])
    dx[3] = (1/m) * u[0] * np.cos(x[4]) - g
    dx[4] = x[5]
    dx[5] = (1/I) * u[1]
    return dx

# Simulation
def simulate_multirotor(x0, tfinal, timestep, controller):
    t = np.arange(0, tfinal, timestep)
    sol = odeint(multirotor_system, x0, t, args=(controller,))
    return t, sol

# Euler integration (for noise testing)
def simulate_euler(x0, tfinal, dt, controller, noise_std=0.0):
    I, g, m = get_params()
    t = np.arange(0, tfinal, dt)
    x = np.array(x0, dtype=float)
    sol = [x.copy()]
    for i in range(1, len(t)):
        u = controller(x)
        dx = np.zeros(6)
        dx[0] = x[2]
        dx[1] = x[3]
        dx[2] = -(1/m) * u[0] * np.sin(x[4])
        dx[3] = (1/m) * u[0] * np.cos(x[4]) - g
        dx[4] = x[5]
        dx[5] = (1/I) * u[1]
        # Process noise on accelerations
        dx[2] += np.random.normal(0, noise_std)
        dx[3] += np.random.normal(0, noise_std)
        dx[5] += np.random.normal(0, noise_std)
        x = x + dx * dt
        sol.append(x.copy())
    return t, np.array(sol)

# LQR controller
def make_lqr_controller(x_star, u_star, K, noise_std=0.0):
    def controller(x):
        x_tilde = x - x_star
        u_tilde = K @ x_tilde
        u = u_star + u_tilde
        # Add noise if specified
        if noise_std > 0:
            u = u + np.random.normal(0, noise_std, size=2)
        return u
    return controller

# Plot
def plot_hover(results, title, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    for idx, (label, t, sol, x_star) in enumerate(results):
        c = colors[idx]
        axes[0].plot(t, sol[:, 0], '-', color=c, label=f'{label} x1')
        axes[0].plot(t, sol[:, 1], '--', color=c, label=f'{label} x2')
        axes[0].axhline(y=x_star[0], color=c, linestyle=':', alpha=0.3)
        axes[0].axhline(y=x_star[1], color=c, linestyle=':', alpha=0.3)

    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Position (m)')
    axes[0].set_title('Position vs Time')
    axes[0].legend(fontsize=10, loc='lower right')
    axes[0].grid(True)

    for idx, (label, t, sol, x_star) in enumerate(results):
        c = colors[idx]
        axes[1].plot(t, np.degrees(sol[:, 4]), color=c, label=f'{label} theta')

    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Angle (deg)')
    axes[1].set_title('Roll Angle vs Time')
    axes[1].legend(fontsize=10)
    axes[1].grid(True)

    #plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

# Run simulations
I, g, m = get_params()
u_star = np.array([m * g, 0.0]) # equilibrium input
tfinal = 10.0
timestep = 0.01

# Test 1: Several hover positions (no noise)
hover_targets = [
    ("Hover (0,5)",   np.array([0, 5, 0, 0, 0, 0]),   np.array([2, 3, 1, 1, 0.1, 0])),
    ("Hover (3,10)",  np.array([3, 10, 0, 0, 0, 0]),   np.array([5, 8, -1, 2, -0.1, 0])),
    ("Hover (-2,7)",  np.array([-2, 7, 0, 0, 0, 0]),   np.array([0, 5, 2, -1, 0.05, 0])),
]

results_no_noise = []
for label, x_star, x0 in hover_targets:
    ctrl = make_lqr_controller(x_star, u_star, K, noise_std=0.0)
    t, sol = simulate_multirotor(x0, tfinal, timestep, ctrl)
    results_no_noise.append((label, t, sol, x_star))

plot_hover(results_no_noise, "LQR Hover Controller (No Noise)", "hover_no_noise.png")

# Test 2: With noise (Euler integration)
results_noise = []
for label, x_star, x0 in hover_targets:
    ctrl = make_lqr_controller(x_star, u_star, K, noise_std=0.0)  # no noise on input
    t, sol = simulate_euler(x0, tfinal, timestep, ctrl, noise_std=0.5)
    results_noise.append((label + " +noise", t, sol, x_star))

plot_hover(results_noise, "LQR Hover Controller (With Noise)", "hover_with_noise.png")