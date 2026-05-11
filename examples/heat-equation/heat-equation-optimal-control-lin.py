"""
Finite element solver in Firedrake for the two-dimensional, time-dependent heat equation, with optimal control of a forcing term.
The equation solved is
\partial_t u - k \Delta u = m
where u is the temperature, k is the diffusivity, and m is a control term. The goal is to find the optimal control m that minimizes the functional
\int_0^T exp(-lambda*t)*weight*||u_desired(t) - u(t)||^2 + ||m||^2 dt
where u_desired is a time-dependent desired state.

Model predictive control
The optimization is performed in the context of model predictive control. We divide the time
interval [0, T] into a number of windows. Consider the first window 

m0      m1     m2      m3     m4
|--dt--|--dt--|--dt--|--dt--|        W0 

We start with an intial guess for the list of controls [m0, m1, m2, m3, m4] and solve the forward model over the first window W0 to compute the functional. To avoid confusion, going forwards in time in this part is being called time-hopping.
J = \sum_{i=0}^{t_w0} exp(-lambda*ti)*misfit_weight*||u_desired(ti) - u(ti)||^2 + ||m||^2 dt. 
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
import csv
import os
# Set OMP_NUM_THREADS to 1 to avoid warnings
os.environ["OMP_NUM_THREADS"] = "1"
from fcntl import flock, LOCK_EX, LOCK_UN
from firedrake import *
import matplotlib.pyplot as plt
from firedrake.adjoint import *
from pyadjoint import ParametrisedReducedFunctional
from pyadjoint.optimization.tao_solver import MinimizationProblem, TAOSolver
# from petsc4py import PETSc
from firedrake.petsc import PETSc
PETSc.Sys.popErrorHandler()
continue_annotation()


opts = PETSc.Options()
verbose = opts.getBool("--verbose", default=False)
pvdOutput = opts.getBool("--pvd-output", default=True)

k = 0.01
num_cells = 50
mesh = UnitSquareMesh(num_cells, num_cells)
dt = 0.001 # Time step size
T = opts.getReal("--final-time", default=0.01) # Final time
window_size = opts.getInt("--window-size", default=5) # Number of time steps in each window
window_step = opts.getInt("--window-step", default=1) # Number of time steps to step forward in each window. Must be less than or equal to window_size.
assert window_step <= window_size, "The window step must be less than or equal to the window size."

outfile_path = opts.getString("--outfile-path", default="output")
summary_csv_path = opts.getString("--summary-csv-path", default="")
decay_constant = opts.getReal("--decay-constant", default=0.1) # Time decay constant
lambda_t = decay_constant/((window_size+window_step)*dt)
#TODO: Rewrite the time-decay parameter as a function of the window size and the window step
misfit_weight = opts.getReal("--misfit-weight", default=1.0) # Regularization parameter for the deviation of the state from the desired state
if not os.path.exists(outfile_path):
    os.makedirs(outfile_path, exist_ok=True)
if pvdOutput:
    outfile = VTKFile(f"{outfile_path}/heat_equation_optimal_control.pvd")


V = FunctionSpace(mesh, "CG", 2)
u = TrialFunction(V)
u_new = Function(V, name="Solution at new time step")
u_new_hop = Function(V, name="Solution at new time step during time-hop")
u_init = Function(V, name="Initial condition")
u_desired = Function(V, name="Desired state")
m = Function(V, name="Control")
print(f"Control: {m}")
m_list = [Function(V, name=f"Control at time hop {i}") for i in range(window_size)]
v = TestFunction(V)
u_point_wise_error = Function(V, name="Pointwise error")


x, y = SpatialCoordinate(mesh)
t_actual = 0.0 # Keeps track of the actual time, only incremented during a time-step.
# Set a Gaussian initial condition
alpha = 100
init_expr = exp(-alpha * ((x - 0.5) ** 2 + (y - 0.5) ** 2))
u_init.interpolate(init_expr)
u_new.assign(u_init)
u_new_hop.assign(u_init)

# Set a time-dependent desired state
def u_desired_expr(t):
    return exp(-alpha * ((x - 0.5) ** 2 + (y - 0.5) ** 2)) * exp(0.1*t)

def du_dt(u_, u, dt):
    return (u_ - u) / dt

# Set up an initial guess for the control
for m_i in m_list:
    # m_i.interpolate(Constant(0.0))
    m_i.interpolate(0.1*exp(-(alpha*10.0) * ((x - 0.5) ** 2 + (y - 0.5) ** 2))) # A small Gaussian in the middle of the domain as an initial guess for the control.

a = (dt*inner(grad(u), grad(v))*k + inner(u, v))*dx
L = inner(m, v)*dx

# F = inner(du_dt(u_new, u, dt), v)*dx + k*inner(grad(u_new), grad(v))*dx - inner(m, v)*dx
bc = DirichletBC(V, 0.0, "on_boundary")

solver_heat_step = LinearVariationalSolver(LinearVariationalProblem(a, L, u_new, bcs=bc))

solver_heat_hop = LinearVariationalSolver(LinearVariationalProblem(a, L, u_new_hop, bcs=bc), solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})

def loss_functional(t_current, t_window):
    u_desired.interpolate(u_desired_expr(t_current))
    return assemble(((exp(-lambda_t*t_window)*misfit_weight*inner(u_desired - u_new_hop, u_desired - u_new_hop)) + inner(m, m))*dx)


# Time-stepping loop to be run in each window
def time_hop_loop(controls, t_init, J):
    t_current = t_init # Keeps track of the time in a time-hop loop, incremented at each time-hop.
    t_window = 0.0 # Keeps track of the time within the current window.
    for m_i in controls:
        m.assign(m_i)
        solver_heat_hop.solve()
        t_current += dt
        t_window += dt
        # Add the "loss" functional
        # \int_0^T exp(-0.1*t)*0.5*||u_desired(t) - u(t)||^2 + 0.01*||m||^2 dt        
        J+= loss_functional(t_current, t_window)
    return J

def time_step_loop(m_opt, t_init):
    t_current = t_init
    m.assign(m_opt)
    solver_heat_step.solve()
    t_current += dt
    return t_current

def set_TAO_solver(Jhat):
    problem = MinimizationProblem(Jhat)
    parameters = { 'method': 'lbfgs',
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

def append_summary_row(summary_path, row):
    if not summary_path:
        return

    summary_dir = os.path.dirname(summary_path)
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)

    fieldnames = [
        "outfile_path",
        "window_size",
        "window_step",
        "decay_constant",
        "misfit_weight",
        "final_time",
        "final_l2_error",
        "final_linf_error",
    ]

    with open(summary_path, "a+", newline="") as summary_file:
        flock(summary_file.fileno(), LOCK_EX)
        try:
            summary_file.seek(0, os.SEEK_END)
            file_is_empty = summary_file.tell() == 0
            writer = csv.DictWriter(summary_file, fieldnames=fieldnames)
            if file_is_empty:
                writer.writeheader()
            writer.writerow(row)
            summary_file.flush()
        finally:
            flock(summary_file.fileno(), LOCK_UN)

u_desired.interpolate(u_desired_expr(t_actual))
u_point_wise_error.interpolate(abs(u_desired - u_new))
m.assign(m_list[0])
if pvdOutput:
    outfile.write(u_new, m, u_desired, u_point_wise_error)

#TODO: Plot the relative l2 error wrt time 

window_num = 0
l2_errors = []
linf_errors = []
J = 0
PETSc.Sys.Print(f"Starting experiment with window size {window_size}, window step {window_step}, decay constant {decay_constant}, misfit weight {misfit_weight}")
while t_actual < T:
    if verbose:
        PETSc.Sys.Print(f"Starting window {window_num+1} at time {t_actual:.4f}")
    if window_num == 0:
        u_new.assign(u_init)
        J = time_hop_loop(m_list, t_actual, J)
        Jhat = ParametrisedReducedFunctional(J, [Control(m_i) for m_i in m_list], u_init)
        solver = set_TAO_solver(Jhat)
        m_opt = get_optimal_control(solver)
        # exit(0)

        for i in range(window_step):
            t_actual = time_step_loop(m_opt[i], t_actual)

            u_desired.interpolate(u_desired_expr(t_actual))
            u_point_wise_error.interpolate(abs(u_desired - u_new))
            if pvdOutput:
                outfile.write(u_new, m, u_desired, u_point_wise_error)
            l2_error = norm(u_desired - u_new)
            linf_error = max(u_point_wise_error.dat.data)
            l2_errors.append(l2_error)
            linf_errors.append(linf_error)
            if verbose:
                PETSc.Sys.Print(f"Time {t_actual:.4f}, L2 error: {l2_error:.6f}, L-infinity error: {linf_error:.6f}")
        u_init.assign(u_new)
        window_num += 1
        m_list[:-window_step] = (m_opt[window_step:])
        for i in range(window_step):
            m_list[-(i+1)].interpolate(0.1*exp(-(alpha*10.0) * ((x - 0.5) ** 2 + (y - 0.5) ** 2)))
        
        Jhat.update_parameters(u_init)
    else:
        u_new.assign(u_init)
        J = time_hop_loop(m_list, t_actual, J)
        m_opt = get_optimal_control(solver)
        for i in range(window_step):
            t_actual = time_step_loop(m_opt[i], t_actual)
            u_desired.interpolate(u_desired_expr(t_actual))
            u_point_wise_error.interpolate(abs(u_desired - u_new))
            if pvdOutput:
                outfile.write(u_new, m, u_desired, u_point_wise_error)
            l2_error = norm(u_desired - u_new)
            linf_error = max(u_point_wise_error.dat.data)
            if verbose:
                PETSc.Sys.Print(f"Time {t_actual:.4f}, L2 error: {l2_error:.6f}, L-infinity error: {linf_error:.6f}")
            l2_errors.append(l2_error)
            linf_errors.append(linf_error)
        u_init.assign(u_new)
        window_num += 1
        m_list[:-window_step] = (m_opt[window_step:])
        for i in range(window_step):
            m_list[-(i+1)].interpolate(0.1*exp(-(alpha*10.0) * ((x - 0.5) ** 2 + (y - 0.5) ** 2)))

        Jhat.update_parameters(u_init)

# Plot the relative l2 error over time
plt.figure()
plt.plot(np.arange(len(l2_errors))*dt, l2_errors, label="l2 error")
plt.plot(np.arange(len(linf_errors))*dt, linf_errors, label="L-inf error")
plt.xlabel("Time")
plt.ylabel("Error")
plt.title(f"Error over time for window size {window_size}, window step {window_step}, \n decay constant {decay_constant}, misfit weight {misfit_weight}")
plt.legend()
plt.savefig(f"{outfile_path}/error_plot.png")

final_l2_error = l2_errors[-1] if l2_errors else norm(u_desired - u)
final_linf_error = linf_errors[-1] if linf_errors else max(u_point_wise_error.dat.data)
append_summary_row(
    summary_csv_path,
    {
        "outfile_path": outfile_path,
        "window_size": window_size,
        "window_step": window_step,
        "decay_constant": decay_constant,
        "misfit_weight": misfit_weight,
        "final_time": t_actual,
        "final_l2_error": final_l2_error,
        "final_linf_error": final_linf_error,
    },
)
