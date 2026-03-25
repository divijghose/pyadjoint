"""
Finite element solver in Firedrake for the two-dimensional, time-dependent heat equation, with optimal control of a forcing term.
The equation solved is
\partial_t u - k \Delta u = m
where u is the temperature, k is the diffusivity, and m is a control term. The goal is to find the optimal control m that minimizes the functional
\int_0^T exp(-lambda*t)*0.5*||u_desired(t) - u(t)||^2 + 0.01*||m||^2 dt
where u_desired is a time-dependent desired state.

Model predictive control
The optimization is performed in the context of model predictive control. We divide the time
interval [0, T] into a number of windows. Consider the first window 

m0      m1     m2      m3     m4
|--dt--|--dt--|--dt--|--dt--|        W0 

We start with an intial guess for the list of controls [m0, m1, m2, m3, m4] and solve the forward model over the first window W0 to compute the functional. To avoid confusion, going forwards in time in this part is being called time-hopping.
J = \sum_{i=0}^{t_w0} exp(-lambda*ti)*0.5*||u_desired(ti) - u(ti)||^2 + 0.01*||m||^2 dt. 
We then assemble a ParametrisedReducedFunctional with J as the functional, 
[m0, m1, m2, m3, m4] as the controls, and u_init (the initial condition for the window) as the parameter. We then optimize over the controls in [m0, m1, m2, m3, m4] to find an optimal set of controls for that window, [m0_opt, m1_opt, m2_opt, m3_opt, m4_opt].

m0_opt m1_opt m2_opt m3_opt  m4_opt
|--dt--|--dt--|--dt--|--dt--|        W0

We then perform a time-stepping, using m0_opt as the control for the first time step. We also move the window forward in time,
henceforth referred to as time-leaping.

         
|--dt--|--dt--|--dt--|--dt--|        W0

        m1_opt m2_opt m3_opt m4_opt   m5
--> u1 |--dt--|--dt--|--dt--|--dt--|        W1

The new controls are initialised as the previous optimal controls, and an additional control is initialized as 0 (since the exponential
term in the funtional means that this contribution will be small anyway). The initial condition for the next window is set to the solution at the end of the current window, and the parametrized reduced functional is updated accordingly. 

In this manner, we hop, step and leap through time.
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
m_list = [Function(V, name=f"Control at time step {i}") for i in range(int(T/dt)+1)]
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
for m_i in m_list:
    m_i.interpolate(Constant(0.0))



F = inner(du_dt(u_new, u, dt), v)*dx + k*inner(grad(u_new), grad(v))*dx - inner(m, v)*dx
bc = DirichletBC(V, 0.0, "on_boundary")
J = 0

num_windows = 10
total_steps = int(round(T / dt))
window_size = total_steps // num_windows
remainder_steps = total_steps % num_windows

t_actual = 0.0 # Keeps track of the actual time, only incremented during a time-step.
# Time-stepping loop to be run in each window
def time_hop_loop(steps_in_window, t_init, J):
    t_current = t_init # Keeps track of the time in a time-hop loop, incremented at each time-hop.
    for i in range(steps_in_window):
        m.assign(m_list[i])
        solve(F == 0, u_new, bc)
        u.assign(u_new)
        t_current += dt
        # Add the "loss" functional
        # \int_0^T exp(-0.1*t)*0.5*||u_desired(t) - u(t)||^2 + 0.01*||m||^2 dt        
        J += assemble(exp(-0.1*t_current)*0.5*inner(u_desired_expr(t_current) - u, u_desired_expr(t_current) - u)*dx + 0.01*inner(m, m)*dx)

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
        J = time_hop_loop(steps_in_window, t, J)
        Jhat = ParametrisedReducedFunctional(J, Control(m), u_init)
        solver = set_TAO_solver(Jhat)
        m_opt = get_optimal_control(solver)
    else:
        u.assign(u_init)
        steps_in_window = window_size + (1 if window < remainder_steps else 0)
    # Time-step forward through the current window, accumulate the "loss" functional
        J = time_hop_loop(steps_in_window, t, J)
        Jhat.update_parameters(u_init)
        m_opt = get_optimal_control(solver)
        m.assign(m_opt)
        u_desired.interpolate(u_desired_expr(t))
    
    u_init.assign(u)
    outfile.write(u, m, u_desired)


