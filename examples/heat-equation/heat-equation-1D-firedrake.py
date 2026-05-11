import os
# Set OMP_NUM_THREADS to 1 to avoid warnings
os.environ["OMP_NUM_THREADS"] = "1"
from firedrake import *
import matplotlib.pyplot as plt
from firedrake.petsc import PETSc
PETSc.Sys.popErrorHandler()


opts = PETSc.Options()
verbose = opts.getBool("--verbose", default=False)
pvdOutput = opts.getBool("--pvd-output", default=True)

k = 0.1
num_cells = 50
mesh = UnitIntervalMesh(num_cells)
dt = 0.001 # Time step size
T = opts.getReal("--final-time", default=0.01) # Final time

outfile_path = opts.getString("--outfile-path", default="output")
if not os.path.exists(outfile_path):
    os.makedirs(outfile_path, exist_ok=True)

if pvdOutput:
    outfile = VTKFile(f"{outfile_path}/heat_equation_optimal_control.pvd")


V = FunctionSpace(mesh, "CG", 2)
u = Function(V, name="State")
u_new = Function(V, name="Solution at new time step")
u_init = Function(V, name="Initial condition")
v = TestFunction(V)
u_point_wise_error = Function(V, name="Pointwise error")
m = Function(V, name="Forcing term")

x = SpatialCoordinate(mesh)
t_actual = 0.0 # Keeps track of the actual time, only incremented during a time-step.
# Set a Gaussian initial condition
alpha = 100
init_expr = exp(-alpha * ((x[0] - 0.5) ** 2))
u_init.interpolate(init_expr)
u.interpolate(init_expr)


def du_dt(u_, u, dt):
    return (u_ - u) / dt

# Set up an initial guess for the control
m.interpolate(10.0)

F = inner(du_dt(u_new, u, dt), v)*dx + k*inner(grad(u_new), grad(v))*dx - inner(m, v)*dx
bc = DirichletBC(V, 0.0, "on_boundary")



def time_step_loop(t_init):
    t_current = t_init
    solve(F == 0, u_new, bc)
    u.assign(u_new)
    t_current += dt
    return t_current

if pvdOutput:
    outfile.write(u)


while t_actual < T:
    print(f"Time: {t_actual}")
    t_actual = time_step_loop(t_actual)
    if pvdOutput:
        outfile.write(u)


