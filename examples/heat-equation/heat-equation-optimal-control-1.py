"""
Finite element solver in Firedrake for the time-dependent heat equation, with optimal control of a forcing term.

"""
from astropy.constants.codata2010 import alpha
from firedrake import *
import matplotlib.pyplot as plt
from firedrake_adjoint import *
from pyadjoint import *

num_cells = 50
mesh = UnitSquareMesh(num_cells, num_cells)
dt = 0.001
V = FunctionSpace(mesh, "CG", 2)

u = Function(V, name="State")
u_init = Function(V, name="Initial condition")
u_desired = Function(V, name="Desired state")
m = Function(V, name="Control")
v = TestFunction(V)

x, y = SpatialCoordinate(mesh)
# Set a Gaussian initial condition
alpha = 100
u_init.interpolate(exp(-alpha * ((x - 0.5) ** 2 + (y - 0.5) ** 2)))
u.assign(u_init)

# Set a time-dependent desired state
def u_desired_expr(t):
    return exp(-alpha * ((x - 0.5) ** 2 + (y - 0.5) ** 2)) * exp(0.1*t)

def du_dt(u_, u, dt):
    return (u_ - u) / dt