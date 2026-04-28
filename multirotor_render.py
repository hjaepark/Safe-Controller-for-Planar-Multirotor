import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib import animation
from scipy.ndimage import rotate

# Parameters
def get_params():
    I = 1.0    # Moment of inertia
    g = 9.8    # Gravitational acceleration (m/s^2)
    m = 1.0    # Multirotor mass
    return I, g, m

# Controller (no input)
def controller(x):
    return np.array([0.0, 0.0])

# Dynamics
def multirotor_system(x, t, controller):
    I, g, m = get_params()
    u = controller(x)
    dx = np.zeros(6)
    dx[0] = x[2]                             # x1_dot = v1
    dx[1] = x[3]                             # x2_dot = v2
    dx[2] = -(1/m) * u[0] * np.sin(x[4])     # v1_dot
    dx[3] = (1/m) * u[0] * np.cos(x[4]) - g  # v2_dot
    dx[4] = x[5]                             # theta_dot = omega
    dx[5] = (1/I) * u[1]                     # omega_dot
    return dx

# Simulation
def simulate_multirotor(x0, tfinal, timestep, controller):
    t = np.arange(0, tfinal, timestep)
    sol = odeint(multirotor_system, x0, t, args=(controller,))
    return sol

# Animation
def make_animation(dt, solvec):
    posx = solvec[:, 0]
    posy = solvec[:, 1]
    theta = solvec[:, 4]
    Nt = solvec.shape[0]

    fig, ax = plt.subplots(figsize=(8, 6))
    margin = 2
    ax.set_xlim(min(posx) - margin, max(posx) + margin)
    ax.set_ylim(min(posy) - margin, max(posy) + margin)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlabel('x1 (m)')
    ax.set_ylabel('x2 (m)')
    ax.set_title('Planar Multirotor Simulation')

    # Multirotor image
    multirotor_image = mpimg.imread("multirotor.png")
    imagebox = OffsetImage(multirotor_image, zoom=0.2)
    ab = AnnotationBbox(imagebox, (0, 0), frameon=False)
    ax.add_artist(ab)

    # Trail
    trail, = ax.plot([], [], '-', color='orange', alpha=0.5, lw=1.5)

    # Time text
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes,
                        fontsize=12, verticalalignment='top')

    trail_x, trail_y = [], []

    def init():
        trail.set_data([], [])
        time_text.set_text('')
        trail_x.clear()
        trail_y.clear()
        return ab, trail, time_text

    def animate(i):
        cx, cy, th = posx[i], posy[i], theta[i]

        # Rotate image and update position
        rotated_img = rotate(multirotor_image, np.degrees(th), reshape=True)
        rotated_img = np.clip(rotated_img, 0, 1)
        imagebox.set_data(rotated_img)
        ab.xybox = (cx, cy)

        # Trail
        trail_x.append(cx)
        trail_y.append(cy)
        trail.set_data(trail_x, trail_y)

        time_text.set_text(f'Time = {i*dt:.1f} s')
        return ab, trail, time_text

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                    frames=Nt, interval=dt*1000*0.5, blit=False)
    return fig, anim

# Run
x0 = [0, 2, 10, 10, 0, 0]
tfinal = 5.0
timestep = 0.01

solvec = simulate_multirotor(x0, tfinal, timestep, controller)

# Save as gif
fig, anim = make_animation(timestep, solvec)
anim.save('linearized_multirotor_sim.gif', fps=30, writer='pillow')
print("Saved: linearized_multirotor_sim.gif")
plt.close()

# Uncomment to watch live:
# fig, anim = make_animation(timestep, solvec)
# plt.show()