"""
Finite element solver in Firedrake for the time-dependent heat equation, with optimal control of a forcing term.

"""
import os
from firedrake import *
import matplotlib.pyplot as plt
from firedrake.adjoint import *
from pyadjoint import ParametrisedReducedFunctional
from pyadjoint.optimization.tao_solver import MinimizationProblem, TAOSolver
from petsc4py import PETSc
PETSc.Sys.popErrorHandler()
continue_annotation()

num_cells = 50
mesh = UnitSquareMesh(num_cells, num_cells)
dt = 0.001
T = 0.01
k = 0.1
V = FunctionSpace(mesh, "CG", 2)

u = Function(V, name="State")
u_new = Function(V, name="Solution at new time step")
u_init = Function(V, name="Initial condition")
u_desired = Function(V, name="Desired state")
m = Function(V, name="Control")
v = TestFunction(V)

x, y = SpatialCoordinate(mesh)
# Set a Gaussian initial condition
alpha = 100
init_expr = exp(-alpha * ((x - 0.5) ** 2 + (y - 0.5) ** 2))
u_init.interpolate(init_expr)
u.assign(u_init)

# Set a time-dependent desired state
def u_desired_expr(t):
    return exp(-alpha * ((x - 0.5) ** 2 + (y - 0.5) ** 2)) * exp(0.1*t)

def du_dt(u_, u, dt):
    return (u_ - u) / dt

# Set up an initial guess for the control
m.interpolate(Constant(0.0))



F = inner(du_dt(u_new, u, dt), v)*dx + k*inner(grad(u_new), grad(v))*dx - inner(m, v)*dx
bc = DirichletBC(V, 0.0, "on_boundary")
J = 0

num_windows = 10
total_steps = int(round(T / dt))
window_size = total_steps // num_windows
remainder_steps = total_steps % num_windows

# Time-stepping loop to be run in each window
def window_time_loop(steps_in_window, t, J):
    for _ in range(steps_in_window):
        solve(F == 0, u_new, bc)
        u.assign(u_new)
        t += dt

        # Add the "loss" functional
        # \int_0^T exp(-0.1*t)*0.5*||u_desired(t) - u(t)||^2 + 0.01*||m||^2 dt        
        J += assemble(exp(-0.1*t)*0.5*inner(u_desired_expr(t) - u, u_desired_expr(t) - u)*dx + 0.01*inner(m, m)*dx)
    
    return J

def set_TAO_solver(Jhat):
    problem = MinimizationProblem(Jhat)
    parameters = { 'method': 'nls',
                   'max_it': 20,
                   'fatol' : 0.0,
                   'frtol' : 0.0,
                   'gatol' : 1e-9,
                   'grtol' : 0.0
                   }
    solver = TAOSolver(problem, parameters=parameters)
    return solver

def get_optimal_control(solver):
    m_opt = solver.solve()
    return m_opt

# Loop over the time windows
t = 0.0
J = 0
outfile = VTKFile("output/heat_equation_optimal_control.pvd")

for window in range(num_windows):
    print(f"Optimizing over window {window+1}/{num_windows} with {window_size} time steps")
    # Set up the functional and optimizer initially
    if window == 0:
        u.assign(u_init)
        steps_in_window = window_size + (1 if window < remainder_steps else 0)
        J = window_time_loop(steps_in_window, t, J)
        Jhat = ParametrisedReducedFunctional(J, Control(m), u_init)
        solver = set_TAO_solver(Jhat)
        m_opt = get_optimal_control(solver)
    else:
        u.assign(u_init)
        steps_in_window = window_size + (1 if window < remainder_steps else 0)
    # Time-step forward through the current window, accumulate the "loss" functional
        J = window_time_loop(steps_in_window, t, J)
        Jhat.update_parameters(u_init)
        m_opt = get_optimal_control(solver)
        m.assign(m_opt)
        u_desired.interpolate(u_desired_expr(t))
    
    u_init.assign(u)
    outfile.write(u, m, u_desired)


