import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Parameters
def get_params():
    I = 1.0    # Moment of inertia
    g = 9.8    # Gravitational acceleration (m/s^2)
    m = 1.0    # Multirotor mass
    return I, g, m

# Reference trajectory
def ref_trajectory(t, w):
    """Returns y*, dy*, ddy*, dddy*, ddddy* for y*(t) = [cos(wt), sin(wt)]."""
    c, s = np.cos(w*t), np.sin(w*t)
    y      = np.array([ c,       s])
    dy     = np.array([-w*s,     w*c])
    ddy    = np.array([-w**2*c, -w**2*s])
    dddy   = np.array([ w**3*s, -w**3*c])
    ddddy  = np.array([ w**4*c,  w**4*s])
    return y, dy, ddy, dddy, ddddy

# Tracking controller
def make_tracking_controller(w, lam):
    """Returns a controller function for dynamic feedback linearization tracking."""
    def controller(x_ext, t):
        x1, x2, v1, v2, theta, omega, x7, x8 = x_ext
        I, g, m = get_params()
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)

        # Output derivatives from current state
        y    = np.array([x1, x2])
        dy   = np.array([v1, v2])
        ddy  = np.array([-x7*sin_t, x7*cos_t - g])
        dddy = np.array([-x8*sin_t - x7*omega*cos_t,
                          x8*cos_t - x7*omega*sin_t])

        # State-only terms in y^(4)
        b = np.array([
            -2*x8*omega*cos_t + x7*omega**2*sin_t,
            -2*x8*omega*sin_t - x7*omega**2*cos_t
        ])

        # Decoupling matrix
        A_ext = np.array([
            [-sin_t, -x7*cos_t],
            [ cos_t, -x7*sin_t]
        ])

        # Reference trajectory
        y_ref, dy_ref, ddy_ref, dddy_ref, ddddy_ref = ref_trajectory(t, w)

        # Tracking errors
        e    = y    - y_ref
        ed   = dy   - dy_ref
        edd  = ddy  - ddy_ref
        eddd = dddy - dddy_ref

        # Gains from (s + lambda)^4
        k0 = lam**4
        k1 = 4*lam**3
        k2 = 6*lam**2
        k3 = 4*lam

        # Virtual input
        nu = ddddy_ref - k3*eddd - k2*edd - k1*ed - k0*e

        # Feedback linearizing control law: u = A_ext^{-1}(nu - b)
        try:
            u_new = np.linalg.solve(A_ext, nu - b)
        except np.linalg.LinAlgError:
            u_new = np.array([0.0, 0.0])

        return u_new  # [u1_tilde, u2]
    return controller

# Extended system dynamics (8 states)
def extended_system(x_ext, t, controller):
    I, g, m = get_params()
    x1, x2, v1, v2, theta, omega, x7, x8 = x_ext

    u1_tilde, u2 = controller(x_ext, t)

    dx = np.zeros(8)
    dx[0] = v1                                # x1_dot
    dx[1] = v2                                # x2_dot
    dx[2] = -(1/m)*x7*np.sin(theta)           # v1_dot
    dx[3] = (1/m)*x7*np.cos(theta) - g        # v2_dot
    dx[4] = omega                             # theta_dot
    dx[5] = (1/I)*u2                          # omega_dot
    dx[6] = x8                                # x7_dot = x8
    dx[7] = u1_tilde                          # x8_dot = u1_tilde
    return dx

# Simulation
def simulate_extended(x0, tfinal, timestep, controller):
    t = np.arange(0, tfinal, timestep)
    sol = odeint(extended_system, x0, t, args=(controller,))
    return t, sol

# Plotting
def plot_tracking(t, sol, w, save_path):
    y_ref = np.array([ref_trajectory(ti, w)[0] for ti in t])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: trajectory in x1-x2 plane
    axes[0].plot(sol[:, 0], sol[:, 1], 'b-', label='Actual', linewidth=1.5)
    axes[0].plot(y_ref[:, 0], y_ref[:, 1], 'r--', label='Reference', linewidth=1.5)
    axes[0].plot(sol[0, 0], sol[0, 1], 'go', markersize=8, label='Start')
    axes[0].set_xlabel('x1 (m)')
    axes[0].set_ylabel('x2 (m)')
    axes[0].set_title('Trajectory in x1-x2 Plane')
    axes[0].legend()
    axes[0].set_aspect('equal')
    axes[0].grid(True)

    # Right: position vs time
    axes[1].plot(t, sol[:, 0], 'b-', label='x1 (actual)')
    axes[1].plot(t, sol[:, 1], 'b--', label='x2 (actual)')
    axes[1].plot(t, y_ref[:, 0], 'r-', alpha=0.5, label='x1* (ref)')
    axes[1].plot(t, y_ref[:, 1], 'r--', alpha=0.5, label='x2* (ref)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Position (m)')
    axes[1].set_title('Position vs Time')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

# Run
I, g, m = get_params()
w = 1.0       # Circular trajectory speed
lam = 2.0     # Pole placement parameter
tfinal = 15.0
timestep = 0.001

ctrl = make_tracking_controller(w, lam)

# Initial conditions: [x1, x2, v1, v2, theta, omega, x7(=u1), x8(=u1_dot)]
ics = [
    ("ic1", [1.0, 0.0, 0.0, w, 0.0, 0.0, g, 0.0]),
    ("ic2", [2.0, 1.0, 0.0, 0.0, 0.0, 0.0, g, 0.0]),
    ("ic3", [0.0, 3.0, -1.0, 0.0, 0.1, 0.0, g, 0.0]),
]

for label, x0 in ics:
    print(f"Simulating {label}...")
    t, sol = simulate_extended(x0, tfinal, timestep, ctrl)
    plot_tracking(t, sol, w, f"tracking_{label}.png")

print("Done!")

# With noise (Euler integration)
def simulate_euler_ext(x0, tfinal, dt, controller, noise_std=0.0):
    I, g, m = get_params()
    t = np.arange(0, tfinal, dt)
    x = np.array(x0, dtype=float)
    sol = [x.copy()]
    for i in range(1, len(t)):
        dx = extended_system(x, t[i], controller)
        noise = np.zeros(8)
        noise[2] = np.random.normal(0, noise_std)
        noise[3] = np.random.normal(0, noise_std)
        noise[5] = np.random.normal(0, noise_std)
        x = x + (dx + noise) * dt
        sol.append(x.copy())
    return t, np.array(sol)

print("Simulating ic2 with noise...")
t, sol = simulate_euler_ext(ics[1][1], tfinal, timestep, ctrl, noise_std=0.5)
plot_tracking(t, sol, w, "tracking_noise.png")