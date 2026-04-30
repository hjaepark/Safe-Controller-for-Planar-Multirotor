import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os

# Parameters
def get_params():
    I = 1.0    # Moment of inertia
    g = 9.8    # Gravitational acceleration (m/s^2)
    m = 1.0    # Multirotor mass
    return I, g, m

def ref_trajectory_figure8(t, omega=1.0, c=1.0, t_offset=0.0):
    tt = t + t_offset

    # First output: x1 = sin(2*omega*tt), amplitude 1, double frequency
    y1     =                np.sin(2*omega*tt)
    y1d1   =      2 * omega       * np.cos(2*omega*tt)
    y1d2   =     -4 * omega**2    * np.sin(2*omega*tt)
    y1d3   =     -8 * omega**3    * np.cos(2*omega*tt)
    y1d4   =     16 * omega**4    * np.sin(2*omega*tt)

    # Second output: x2 = sin(omega*tt) + c, amplitude 1, base frequency
    y2     =      np.sin(omega*tt) + c
    y2d1   =      omega    * np.cos(omega*tt)
    y2d2   = -1 * omega**2 * np.sin(omega*tt)
    y2d3   = -1 * omega**3 * np.cos(omega*tt)
    y2d4   =      omega**4 * np.sin(omega*tt)

    y      = np.array([y1,   y2  ])
    dy     = np.array([y1d1, y2d1])
    ddy    = np.array([y1d2, y2d2])
    dddy   = np.array([y1d3, y2d3])
    ddddy  = np.array([y1d4, y2d4])
    return y, dy, ddy, dddy, ddddy

# Nominal feedback linearization controller
def nominal_controller(x_ext, t, omega_ref, c_ref, lam, t_offset=0.0):
    x1, x2, v1, v2, theta, omega, x7, x8 = x_ext
    I, g, m = get_params()
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

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

    # Reference and tracking errors (with phase shift t_offset)
    y_ref, dy_ref, ddy_ref, dddy_ref, ddddy_ref = ref_trajectory_figure8(
        t, omega_ref, c_ref, t_offset)
    e    = y    - y_ref
    ed   = dy   - dy_ref
    edd  = ddy  - ddy_ref
    eddd = dddy - dddy_ref

    k0 = lam**4
    k1 = 4*lam**3
    k2 = 6*lam**2
    k3 = 4*lam

    nu = ddddy_ref - k3*eddd - k2*edd - k1*ed - k0*e

    # u = A_ext^{-1}(nu - b)
    try:
        u_nom = np.linalg.solve(A_ext, nu - b)
    except np.linalg.LinAlgError:
        u_nom = np.array([0.0, 0.0])
    return u_nom

# HOCBF computation
def compute_hocbf(x_ext, L, gammas):
    x1, x2, v1, v2, theta, omega, x7, x8 = x_ext
    g1, g2, g3, g4 = gammas
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    I, g, m = get_params()

    h    = L - x1
    h_d1 = -v1
    h_d2 = x7*sin_t
    h_d3 = x8*sin_t + x7*omega*cos_t

    b_h = 2*x8*omega*cos_t - x7*omega**2*sin_t
    L_g_h4 = np.array([sin_t, x7*cos_t])

    # Psi_i = dot Psi_{i-1} + gamma_i * Psi_{i-1}
    psi0 = h
    psi1 = h_d1 + g1*h
    psi2 = h_d2 + (g1+g2)*h_d1 + g1*g2*h
    psi3 = (h_d3
            + (g1+g2+g3)*h_d2
            + (g1*g2 + g3*(g1+g2))*h_d1
            + g1*g2*g3*h)

    # psi3_dot = L_f_psi3 + L_g_psi3 * u
    c_h_d2 = g1*g2 + g3*(g1+g2)
    L_f_psi3 = (b_h
                + (g1+g2+g3)*h_d3
                + c_h_d2*h_d2
                + g1*g2*g3*h_d1)
    L_g_psi3 = L_g_h4

    return psi3, L_f_psi3, L_g_psi3

# CBF-QP safety filter
def cbf_qp_filter(u_nom, x_ext, L, gammas):
    g4 = gammas[3]
    psi3, L_f_psi3, L_g_psi3 = compute_hocbf(x_ext, L, gammas)

    a = L_g_psi3
    c = -(L_f_psi3 + g4*psi3)

    if a @ u_nom >= c:
        return u_nom

    a_norm_sq = a @ a
    if a_norm_sq < 1e-10:
        return u_nom
    return u_nom + ((c - a @ u_nom) / a_norm_sq) * a

# Controller
def make_tracking_controller(omega_ref, c_ref, lam, t_offset=0.0):
    def controller(x_ext, t):
        return nominal_controller(x_ext, t, omega_ref, c_ref, lam, t_offset)
    return controller

def make_cbf_controller(omega_ref, c_ref, lam, L_cbf, gammas, t_offset=0.0):
    def controller(x_ext, t):
        u_nom = nominal_controller(x_ext, t, omega_ref, c_ref, lam, t_offset)
        return cbf_qp_filter(u_nom, x_ext, L_cbf, gammas)
    return controller

# Extended 8-state dynamics
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

# Simulation
def simulate_extended(x0, tfinal, timestep, controller):
    t = np.arange(0, tfinal, timestep)
    sol = odeint(extended_system, x0, t, args=(controller,))
    return t, sol

def plot_figure8_tracking(t, sol, omega_ref, c_ref, save_path, t_offset=0.0):
    y_ref = np.array([ref_trajectory_figure8(ti, omega_ref, c_ref, t_offset)[0]
                      for ti in t])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: trajectory in x1-x2 plane
    axes[0].plot(sol[:, 0], sol[:, 1], 'b-', label='Actual', linewidth=1.5)
    axes[0].plot(y_ref[:, 0], y_ref[:, 1], 'r--', label='Reference', linewidth=1.5)
    axes[0].plot(sol[0, 0], sol[0, 1], 'go', markersize=8, label='Start')
    axes[0].set_xlabel('x1 (m)')
    axes[0].set_ylabel('x2 (m)')
    axes[0].set_title('Trajectory in x1-x2 Plane')
    axes[0].legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=9)
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
    axes[1].legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=9)
    axes[1].grid(True)

    plt.tight_layout(rect=[0, 0, 0.93, 1])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    abs_path = os.path.abspath(save_path)
    print(f"Figure saved to: {abs_path}")
    plt.show()


def plot_cbf_figure8(t, sol_cbf, sol_no_cbf, omega_ref, c_ref, L, save_path,
                     t_offset=0.0):
    y_ref = np.array([ref_trajectory_figure8(ti, omega_ref, c_ref, t_offset)[0]
                      for ti in t])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: trajectory in x1-x2 plane with safety boundary
    axes[0].plot(sol_no_cbf[:, 0], sol_no_cbf[:, 1], color='orange',
                 alpha=0.7, linewidth=1.5, label='Without CBF')
    axes[0].plot(sol_cbf[:, 0], sol_cbf[:, 1], 'b-', linewidth=1.8, label='With CBF')
    axes[0].plot(y_ref[:, 0], y_ref[:, 1], 'r--', alpha=0.7,
                 linewidth=1.5, label='Reference')
    axes[0].axvline(x=L, color='k', linestyle='-', linewidth=2,
                    label=f'Barrier (L={L})')

    # Shade the unsafe region (right of x1 = L)
    y_lo = min(sol_cbf[:, 1].min(), sol_no_cbf[:, 1].min(), y_ref[:, 1].min()) - 0.3
    y_hi = max(sol_cbf[:, 1].max(), sol_no_cbf[:, 1].max(), y_ref[:, 1].max()) + 0.3
    x_hi = max(sol_no_cbf[:, 0].max(), y_ref[:, 0].max()) + 0.2
    axes[0].fill_betweenx([y_lo, y_hi], L, x_hi, alpha=0.12, color='red')

    axes[0].plot(sol_cbf[0, 0], sol_cbf[0, 1], 'go', markersize=8, label='Start')
    axes[0].set_xlabel('x1 (m)')
    axes[0].set_ylabel('x2 (m)')
    axes[0].set_title('Trajectory in x1-x2 Plane')
    axes[0].legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=9)
    axes[0].set_aspect('equal')
    axes[0].grid(True)

    # Right: x1 vs time, showing how CBF clips x1 at L
    axes[1].plot(t, sol_no_cbf[:, 0], color='orange', alpha=0.7,
                 linewidth=1.5, label='x1 (without CBF)')
    axes[1].plot(t, sol_cbf[:, 0], 'b-', linewidth=1.8, label='x1 (with CBF)')
    axes[1].plot(t, y_ref[:, 0], 'r--', alpha=0.5, label='x1* (ref)')
    axes[1].axhline(y=L, color='k', linestyle='-', linewidth=2,
                    label=f'Barrier (L={L})')
    axes[1].fill_between(t, L, max(sol_no_cbf[:, 0].max(), y_ref[:, 0].max()) + 0.2,
                         alpha=0.12, color='red')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('x1 (m)')
    axes[1].set_title('Horizontal Position vs Time')
    axes[1].legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=9)
    axes[1].grid(True)

    plt.tight_layout(rect=[0, 0, 0.93, 1])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    abs_path = os.path.abspath(save_path)
    print(f"Figure saved to: {abs_path}")
    plt.show()

if __name__ == '__main__':
    I, g, m = get_params()

    # Reference parameters
    omega_ref = 0.5
    c_ref     = 1.0
    lam       = 2.0

    L = 0.6

    cbf_margin = 0.02
    L_cbf      = L - cbf_margin

    # CBF gains
    gammas = [3.0, 3.0, 3.0, 3.0]

    tfinal   = 8 * np.pi
    timestep = 0.001

    x0_partII = [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, m*g, 0.0]

    print(f"Part II initial state (off the reference):")
    print(f"  x1(0) = {x0_partII[0]:.3f} m,  x2(0) = {x0_partII[1]:.3f} m")
    print(f"  v1(0) = {x0_partII[2]:.3f} m/s,  v2(0) = {x0_partII[3]:.3f} m/s")
    print()

    print("Simulating figure-8 tracking (Part II, no CBF)...")
    ctrl_nom_partII = make_tracking_controller(omega_ref, c_ref, lam, t_offset=0.0)
    t, sol_partII = simulate_extended(x0_partII, tfinal, timestep, ctrl_nom_partII)
    plot_figure8_tracking(t, sol_partII, omega_ref, c_ref, "tracking_figure8.png", t_offset=0.0)

    t_offset_partIII = np.pi
    
    # Compute the initial state on the reference at t = 0 with the offset.
    y_ref0, dy_ref0, _, _, _ = ref_trajectory_figure8(
        0.0, omega_ref, c_ref, t_offset_partIII)
    x0_partIII = [
        y_ref0[0],     # x1: 0.0 (same position as before)
        y_ref0[1],     # x2: c_ref (figure-8 center)
        dy_ref0[0],    # v1: -1.0 m/s (moves leftward, away from boundary)
        dy_ref0[1],    # v2: -0.5 m/s
        0.0,           # theta
        0.0,           # omega
        m * g,         # x7: hover thrust
        0.0,           # x8
    ]

    print(f"Part III initial state (on the reference, moving away from boundary):")
    print(f"  x1(0) = {x0_partIII[0]:.3f} m,  x2(0) = {x0_partIII[1]:.3f} m")
    print(f"  v1(0) = {x0_partIII[2]:.3f} m/s,  v2(0) = {x0_partIII[3]:.3f} m/s")
    print(f"  Safety check: x1(0) = {x0_partIII[0]:.3f} < L = {L}  -> "
          f"{'OK' if x0_partIII[0] < L else 'VIOLATED'}")
    print(f"  CBF margin: enforcing x1 <= {L_cbf:.3f} (display boundary at {L})")
    print()

    print("Simulating figure-8 with nominal controller...")
    ctrl_nom_partIII = make_tracking_controller(omega_ref, c_ref, lam,
                                                t_offset=t_offset_partIII)
    t, sol_no_cbf_partIII = simulate_extended(x0_partIII, tfinal, timestep,
                                              ctrl_nom_partIII)

    print("\nSimulating figure-8 tracking with CBF...")
    ctrl_cbf = make_cbf_controller(omega_ref, c_ref, lam, L_cbf, gammas,
                                   t_offset=t_offset_partIII)
    t, sol_cbf = simulate_extended(x0_partIII, tfinal, timestep, ctrl_cbf)

    x1_max = sol_cbf[:, 0].max()
    print(f"\nForward invariance check on CBF trajectory:")
    print(f"  max(x1) = {x1_max:.4f}  (displayed limit: {L})  -> "
          f"{'OK' if x1_max <= L + 1e-3 else 'VIOLATED'}")

    plot_cbf_figure8(t, sol_cbf, sol_no_cbf_partIII, omega_ref, c_ref, L,
                     "cbf_figure8.png", t_offset=t_offset_partIII)

    print("\nDone!")