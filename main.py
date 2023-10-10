import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import pymunk as pm
import pymunk.matplotlib_util

#
# Constants
#

pi = np.pi

# Laser properties
wl = 400 * 10 ** -9  # m (wavelength)
k = 2 * pi / wl

w0 = 50e-9  # m (beam waist)
zR = pi * w0 ** 2 / wl  # m (Rayleigh range)
I0 = 200000  # W/m^2 (peak intensity)

# Particle properties
mass = 10e-15  # kg (mass of a particle)
n1 = 1.59  # refractive index of the particle
radius = 10e-9

r_i = w0  # m (initial radius)
z_i = zR / 3  # m (initial distance)

# Medium properties
c = 299792458  # m/s (speed of light)
n0 = 1.00  # refractive index of free space
m = n0 / n1
M = ((m ** 2 - 1) / (m ** 2 + 2)) ** 2
eta0 = 377  # (ohm) impedance of the medium

# Simulation properties
N = 500  # precision of the simulation
dt = 0.01  # s (time step)
max_distance: float = zR * 3  # m (distance of the simulation)
max_radius: float = w0 * 1.5  # m (maximum radius of the simulation)

# Medium setup
R, Z = np.meshgrid(np.linspace(-max_radius, max_radius, N), np.linspace(-max_distance, max_distance, N))
r, z, w, Is = sp.symbols('r z w Is')

# Symbolic calculations
sym_w = w0 * sp.sqrt(1 + (z / zR) ** 2)
sym_I = I0 * ((w0 / w) ** 2 * sp.exp(-2 * r ** 2 / w ** 2)).subs(w, sym_w)
sym_F_grad = [2 * pi * n0 * radius ** 3 / c * M * sp.diff(sym_I, var) for var in [r, z]]
# sym_F_scatter = (8 * pi * n0 * k ** 4 * radius ** 6 * Is * M / (3 * c)).subs(Is, sym_I)

# Numerical calculations
num_I = sp.lambdify([r, z], sym_I)
num_F_grad = sp.lambdify([r, z], sym_F_grad)
# num_F_scatter = sp.lambdify([r, z], sym_F_scatter, 'numpy')

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

c = ax.imshow(I_at, cmap='hot', extent=(-max_radius, max_radius, -max_distance, max_distance))
fig.colorbar(c, label='I (W/m^2)', ax=ax)


def setup_axis(axis: plt.Axes):
    axis.clear()
    axis.set_aspect('equal')
    axis.set_xlabel('x (m)')
    axis.set_ylabel('z (m)')
    axis.imshow(I_at, cmap='hot', extent=(-max_radius, max_radius, -max_distance, max_distance))
    ax.text(0.01, 0.95, f'x={body.position.x:.3e}, y={body.position.y:.3e}', transform=ax.transAxes, color='white')


# exit on close
fig.canvas.mpl_connect('close_event', lambda _: exit())


def brownian_force():
    # Brownian force
    force = 1e-21

    def rd():
        return np.random.normal(0, 1) * force

    f = rd(), rd()
    return f


def friction_force(b: pm.Body):
    friction_coefficient = 10e-6
    f = -friction_coefficient * b.velocity * abs(b.velocity)
    return f


# def scatter_force(b: pm.Body):
#     return F_scatter_at(b.position.x, b.position.y)

def grad_force(b: pm.Body):
    f = num_F_grad(b.position.x, b.position.y)
    return f


def isnan(_x):
    return np.isnan(_x) or np.isinf(_x)


while True:
    setup_axis(ax)
    x, y = body.position.x, body.position.y

    if isnan(x) or isnan(y):
        raise Exception('Particle out of bounds')

    # Apply forces
    body.apply_force_at_local_point(brownian_force(), (0, 0))
    body.apply_force_at_local_point(friction_force(body), (0, 0))
    body.apply_force_at_local_point(grad_force(body), (0, 0))

    space.step(dt)
    space.debug_draw(draw_options)

    plt.pause(0.001)

    if y > max_distance or y < -max_distance or x > max_radius or x < -max_radius:
        raise Exception('Particle out of bounds')

