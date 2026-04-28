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
    c, s = np.cos(w*t), np.sin(w*t)
    y      = np.array([ c,       s])
    dy     = np.array([-w*s,     w*c])
    ddy    = np.array([-w**2*c, -w**2*s])
    dddy   = np.array([ w**3*s, -w**3*c])
    ddddy  = np.array([ w**4*c,  w**4*s])
    return y, dy, ddy, dddy, ddddy

# Nominal tracking controller (from Part II)
def nominal_controller(x_ext, t, w, lam):
    x1, x2, v1, v2, theta, omega, x7, x8 = x_ext
    I, g, m = get_params()
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    # Output derivatives from state
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

    # Reference and tracking errors
    y_ref, dy_ref, ddy_ref, dddy_ref, ddddy_ref = ref_trajectory(t, w)
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

    # Feedback linearizing law: u = A_ext^{-1}(nu - b)
    try:
        u_nom = np.linalg.solve(A_ext, nu - b)
    except np.linalg.LinAlgError:
        u_nom = np.array([0.0, 0.0])

    return u_nom  # [u1_tilde, u2]

# HOCBF computation
def compute_hocbf(x_ext, L, gammas):
    x1, x2, v1, v2, theta, omega, x7, x8 = x_ext
    g1, g2, g3, g4 = gammas
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    I, g, m = get_params()

    # h and its derivatives
    h    = L - x1
    h_d1 = -v1
    h_d2 = x7*sin_t
    h_d3 = x8*sin_t + x7*omega*cos_t

    b_h = 2*x8*omega*cos_t - x7*omega**2*sin_t
    L_g_h4 = np.array([sin_t, x7*cos_t])

    # HOCBF chain
    psi0 = h
    psi1 = h_d1 + g1*h
    psi2 = h_d2 + (g1+g2)*h_d1 + g1*g2*h
    psi3 = (h_d3
            + (g1+g2+g3)*h_d2
            + (g1*g2 + g3*(g1+g2))*h_d1
            + g1*g2*g3*h)

    # psi3_dot split into state part (L_f) and input part (L_g)
    c_h_d2 = g1*g2 + g3*(g1+g2)
    L_f_psi3 = (b_h
                + (g1+g2+g3)*h_d3
                + c_h_d2*h_d2
                + g1*g2*g3*h_d1)
    L_g_psi3 = L_g_h4

    return psi3, L_f_psi3, L_g_psi3

# BF-QP safety filter
def cbf_qp_filter(u_nom, x_ext, L, gammas):
    g4 = gammas[3]
    psi3, L_f_psi3, L_g_psi3 = compute_hocbf(x_ext, L, gammas)

    # Constraint
    a = L_g_psi3
    c = -(L_f_psi3 + g4*psi3)

    if a @ u_nom >= c:
        return u_nom

    a_norm_sq = a @ a
    if a_norm_sq < 1e-10:
        return u_nom

    mu = u_nom + ((c - a @ u_nom) / a_norm_sq) * a
    return mu

# Controller
def make_tracking_controller(w, lam):
    def controller(x_ext, t):
        return nominal_controller(x_ext, t, w, lam)
    return controller

def make_cbf_controller(w, lam, L, gammas):
    def controller(x_ext, t):
        u_nom = nominal_controller(x_ext, t, w, lam)
        u_safe = cbf_qp_filter(u_nom, x_ext, L, gammas)
        return u_safe
    return controller

# Extended system dynamics
def extended_system(x_ext, t, controller):
    I, g, m = get_params()
    x1, x2, v1, v2, theta, omega, x7, x8 = x_ext
    u1_tilde, u2 = controller(x_ext, t)

    dx = np.zeros(8)
    dx[0] = v1
    dx[1] = v2
    dx[2] = -(1/m)*x7*np.sin(theta)
    dx[3] = (1/m)*x7*np.cos(theta) - g
    dx[4] = omega
    dx[5] = (1/I)*u2
    dx[6] = x8
    dx[7] = u1_tilde
    return dx

# xtended system with parameter mismatch
def extended_system_mismatch(x_ext, t, controller):
    m_true = 1.1    # controller assumes 1.0
    g_true = 10.0   # controller assumes 9.8
    I_true = 1.0

    x1, x2, v1, v2, theta, omega, x7, x8 = x_ext
    u1_tilde, u2 = controller(x_ext, t)

    dx = np.zeros(8)
    dx[0] = v1
    dx[1] = v2
    dx[2] = -(1/m_true)*x7*np.sin(theta)
    dx[3] = (1/m_true)*x7*np.cos(theta) - g_true
    dx[4] = omega
    dx[5] = (1/I_true)*u2
    dx[6] = x8
    dx[7] = u1_tilde
    return dx

# Simulation
def simulate_extended(x0, tfinal, timestep, controller, dynamics=None):
    if dynamics is None:
        dynamics = extended_system
    t = np.arange(0, tfinal, timestep)
    sol = odeint(dynamics, x0, t, args=(controller,))
    return t, sol

# Euler integration
def simulate_euler_ext(x0, tfinal, dt, controller, dynamics=None, noise_std=0.0):
    if dynamics is None:
        dynamics = extended_system
    t = np.arange(0, tfinal, dt)
    x = np.array(x0, dtype=float)
    sol = [x.copy()]
    for i in range(1, len(t)):
        dx = dynamics(x, t[i], controller)
        noise = np.zeros(8)
        noise[2] = np.random.normal(0, noise_std)
        noise[3] = np.random.normal(0, noise_std)
        noise[5] = np.random.normal(0, noise_std)
        x = x + (dx + noise) * dt
        sol.append(x.copy())
    return t, np.array(sol)

# Plot
def plot_cbf_tracking(t, sol, w, L, save_path, sol_no_cbf=None):
    """Plot trajectory with safety boundary."""
    y_ref = np.array([ref_trajectory(ti, w)[0] for ti in t])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: trajectory in x1-x2 plane
    if sol_no_cbf is not None:
        axes[0].plot(sol_no_cbf[:, 0], sol_no_cbf[:, 1], 'g-',
                     alpha=0.4, linewidth=1, label='Without CBF')
    axes[0].plot(sol[:, 0], sol[:, 1], 'b-', label='With CBF', linewidth=1.5)
    axes[0].plot(y_ref[:, 0], y_ref[:, 1], 'r--', label='Reference', linewidth=1.5)
    axes[0].axvline(x=L, color='k', linestyle='-', linewidth=2, label=f'Barrier (L={L})')
    axes[0].fill_betweenx([-2, 2], L, 2, alpha=0.1, color='red')
    axes[0].plot(sol[0, 0], sol[0, 1], 'go', markersize=8, label='Start')
    axes[0].set_xlabel('x1 (m)')
    axes[0].set_ylabel('x2 (m)')
    axes[0].set_title('Trajectory in x1-x2 Plane')
    axes[0].legend(fontsize=8)
    axes[0].set_aspect('equal')
    axes[0].grid(True)

    # Right: x1 position vs time with barrier
    axes[1].plot(t, sol[:, 0], 'b-', label='x1 (with CBF)', linewidth=1.5)
    if sol_no_cbf is not None:
        axes[1].plot(t, sol_no_cbf[:, 0], 'g-', alpha=0.4, label='x1 (without CBF)')
    axes[1].plot(t, y_ref[:, 0], 'r--', alpha=0.5, label='x1* (ref)')
    axes[1].axhline(y=L, color='k', linestyle='-', linewidth=2, label=f'Barrier (L={L})')
    axes[1].fill_between(t, L, L+0.5, alpha=0.1, color='red')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Position (m)')
    axes[1].set_title('Horizontal Position vs Time')
    axes[1].legend(fontsize=8)
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

# Run simulations
I, g, m = get_params()
w = 1.0        # Circular trajectory speed
lam = 2.0      # Tracking pole placement
L = 0.6        # Safety barrier x1 <= L
tfinal = 20.0
timestep = 0.001

# HOCBF gains [gamma1, gamma2, gamma3, gamma4]
gammas = [5.0, 5.0, 5.0, 5.0]

# Initial condition: start inside safe set (x1 = 0.5 < L = 0.6)
x0 = [0.5, 0.0, 0.0, w, 0.0, 0.0, g, 0.0]

# 1. Without CBF
print("Simulating without CBF...")
ctrl_nom = make_tracking_controller(w, lam)
t, sol_no_cbf = simulate_extended(x0, tfinal, timestep, ctrl_nom)

# With CBF
print("Simulating with CBF (no noise)...")
ctrl_cbf = make_cbf_controller(w, lam, L, gammas)
t, sol_cbf = simulate_extended(x0, tfinal, timestep, ctrl_cbf)
plot_cbf_tracking(t, sol_cbf, w, L, "cbf_no_noise.png", sol_no_cbf=sol_no_cbf)

# With CBF + noise
print("Simulating with CBF + noise...")
t, sol_cbf_noise = simulate_euler_ext(x0, tfinal, timestep, ctrl_cbf, noise_std=0.3)
plot_cbf_tracking(t, sol_cbf_noise, w, L, "cbf_with_noise.png")

# 4. With CBF + parameter mismatch
print("Simulating with CBF + parameter mismatch (m=1.1, g=10.0)...")
t, sol_cbf_mismatch = simulate_extended(
    x0, tfinal, timestep, ctrl_cbf, dynamics=extended_system_mismatch)
plot_cbf_tracking(t, sol_cbf_mismatch, w, L, "cbf_mismatch.png")

print("Done!")