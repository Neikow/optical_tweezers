import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import pymunk as pm
import pymunk.matplotlib_util

# Modèle de Mie / Modèle de Rayleigh

#
# Constants
#

pi = np.pi

# Laser properties
wl = 400 * 10 ** -9  # m (wavelength)
k = 2 * pi / wl

w0 = 50e-9  # m (beam waist)
zR = pi * w0 ** 2 / wl  # m (Rayleigh range)
I0 = 600  # W/m^2 (peak intensity)

# Particle properties
mass = 10e-15  # kg (mass of a particle)
n1 = 1.59  # refractive index of the particle
radius = 1e-8

r_i = w0  # m (initial radius)
z_i = 2 * zR  # m (initial distance)


# Medium properties
c = 299792458  # m/s (speed of light)
n0 = 1.00  # refractive index of free space
m = n0 / n1
M = ((m ** 2 - 1) / (m ** 2 + 2)) ** 2
eta0 = 377  # (ohm) impedance of the medium

# Simulation properties
N = 400  # precision of the simulation
dt = 0.05  # s (time step)
max_distance: float = zR * 3  # m (distance of the simulation)
max_radius: float = w0 * 1.5  # m (maximum radius of the simulation)

# Medium setup
R, Z = np.meshgrid(np.linspace(-max_radius, max_radius, N), np.linspace(-max_distance, max_distance, N))
r, z, w, Is = sp.symbols('r z w Is')

# Symbolic calculations
sym_w = w0 * sp.sqrt(1 + (z / zR) ** 2)
sym_I = I0 * ((w0 / w) ** 2 * sp.exp(-2 * r ** 2 / w ** 2)).subs(w, sym_w)
sym_F_grad = [2 * pi * n0 * radius ** 3 / c * M * sp.diff(sym_I, var) for var in [r, z]]
sym_F_scatter = (8 * pi * n0 * k ** 4 * radius ** 6 * Is * M / (3 * c)).subs(Is, sym_I)

# Numerical calculations
num_I = sp.lambdify([r, z], sym_I)
num_F_grad = sp.lambdify([r, z], sym_F_grad)
num_F_scatter = sp.lambdify([r, z], sym_F_scatter)

# Simulation
I_at = num_I(R, Z)

space = pm.Space()
body = pm.Body()
body.position = r_i, z_i

poly = pm.Circle(body, radius)
poly.mass = mass
space.add(body, poly)

# Plotting
fig, ax = plt.subplots(1, 1)
fig.suptitle('Optical Force on a Spherical Particle', fontsize=16)
fig.tight_layout(pad=3.0)

draw_options = pm.matplotlib_util.DrawOptions(ax)

c = ax.imshow(I_at, cmap='magma', extent=(-max_radius, max_radius, -max_distance, max_distance))
fig.colorbar(c, label='I (W/m^2)', ax=ax)


def setup_axis(axis: plt.Axes):
    axis.clear()
    axis.set_aspect('equal')
    axis.set_xlabel('r (m)')
    axis.set_ylabel('z (m)')
    axis.imshow(I_at, cmap='magma', extent=(-max_radius, max_radius, -max_distance, max_distance))
    ax.text(0.01, 0.95, f'Position:\nr={body.position.x:.3e}, z={body.position.y:.3e}', transform=ax.transAxes, color='white')


# exit on close
fig.canvas.mpl_connect('close_event', lambda _: exit())


def brownian_force():
    # Brownian force
    force_norm = 1e-21

    def rd():
        return np.random.normal(0, 1) * force_norm

    force = rd(), rd()
    return force


def friction_force(b: pm.Body):
    friction_coefficient = 0.5e-6
    force = -friction_coefficient * b.velocity * abs(b.velocity)
    return force


def scatter_force(b: pm.Body):
    return [0, num_F_scatter(b.position.x, b.position.y)]


def grad_force(b: pm.Body):
    force = num_F_grad(b.position.x, b.position.y)
    return force


def isnan(_x):
    return np.isnan(_x) or np.isinf(_x)


def set_random_position(_):
    body.position = (np.random.uniform(-max_radius + 2 * radius, max_radius - 2 * radius),
                     np.random.uniform(-max_distance + 2 * radius, max_distance - 2 * radius))
    body.velocity = (0, 0)


b_random_position = plt.Button(plt.axes((0.1, 0.1, 0.1, 0.075)), 'Random position')
b_random_position.on_clicked(set_random_position)

position_history = []

display_particle = True
display_forces = True
display_path = False


def toggle_particle(_):
    global display_particle
    display_particle = not display_particle


def toggle_forces(_):
    global display_forces
    display_forces = not display_forces


def toggle_path(_):
    global display_path
    display_path = not display_path


b_toggle_particle = plt.Button(plt.axes((0.1, 0.8, 0.1, 0.075)), 'Toggle particle')
b_toggle_particle.on_clicked(toggle_particle)

b_toggle_forces = plt.Button(plt.axes((0.1, 0.7, 0.1, 0.075)), 'Toggle forces')
b_toggle_forces.on_clicked(toggle_forces)

b_toggle_path = plt.Button(plt.axes((0.1, 0.6, 0.1, 0.075)), 'Toggle path')
b_toggle_path.on_clicked(toggle_path)

frame = 0

while True:
    frame += 1
    setup_axis(ax)
    x, y = body.position.x, body.position.y

    if frame % 5 == 0:
        position_history.append((x, y))

    if display_path:
        ax.plot(*zip(*position_history), color='crimson')

    if isnan(x) or isnan(y):
        raise Exception('Particle out of bounds')

    space.step(dt)

    forces = [['Scattering', scatter_force(body), 'crimson'],
              ['Friction', friction_force(body), 'blueviolet'],
              ['Gradient', grad_force(body), 'royalblue']]

    if display_particle:
        ax.add_patch(plt.Circle((x, y), radius, color='white', fill=True))

    arrows = []
    texts = []
    # Apply forces
    for i, (name, f, color) in enumerate(forces):
        force_scaling = 1e-8 / 1e-23
        if display_forces:
            arrows.append(ax.arrow(x, y, f[0] * force_scaling, f[1] * force_scaling, color=color, width=0.1e-8, ))
        body.apply_force_at_local_point(f, (0, 0))

        # Add text on the left of the plot
        texts.append(plt.text(0, -1 * (i + 2), f'{name}:\n{np.linalg.norm(f):.3e} N ({f[0]:.3e} N, {f[1]:.3e} N)'))

    if display_forces:
        ax.legend(arrows, [f[0] for f in forces])

    plt.pause(0.016)

    for text in texts:
        text.remove()

    if y > max_distance or y < -max_distance or x > max_radius or x < -max_radius:
        raise Exception('Particle out of bounds')