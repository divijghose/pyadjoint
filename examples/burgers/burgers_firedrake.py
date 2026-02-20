from firedrake import *
from firedrake.adjoint import *
from pyadjoint import *

from pyadjoint.reduced_functional import ParametrisedReducedFunctional
continue_annotation()
# from pyadjoint import *
n = 30
mesh = UnitIntervalMesh(n)
timestep = Constant(1.0/n)
steps = 10
# nu = Constant(.0001)
mu = Constant(0.01)

x, = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, "CG", 2)
ic = project(sin(2.0*pi*x), V, name="ic")
nu = project(Constant(0.00001), V, name="nu")

u_old = Function(V, name="u_old")
u_new = Function(V, name="u_new")
v = TestFunction(V)

u_old.assign(ic)

F = ((u_new-u_old)/timestep*v + u_new*u_new.dx(0)*v + nu*u_new.dx(0)*v.dx(0))*dx
bc = DirichletBC(V, 0.0, "on_boundary")
problem = NonlinearVariationalProblem(F, u_new, bcs=bc)
solver = NonlinearVariationalSolver(problem)

J = assemble(ic*ic*dx)

for _ in range(steps):
    solver.solve()
    u_old.assign(u_new)
    J += assemble(u_new*u_new*dx)
pause_annotation()
print(round(J, 3))

get_working_tape().progress_bar = ProgressBar

Jhat_r = ParametrisedReducedFunctional(J, Control(ic), parameters=nu)

ic_new = project(sin(pi*x), V)
J_new = Jhat_r(ic_new)
print(round(J_new, 3))
new_nu = project(Constant(20), V)
Jhat_r.parameter_update(new_nu)

J_new = Jhat_r(ic_new)

print(round(J_new, 3))
ic_new = project(sin(2.0*pi*x), V)
J_new = Jhat_r(ic_new)
print(round(J_new, 3))

dm = assemble(project(Constant(1.), V))
hessian = Jhat_r.hessian()
rate = taylor_test(Jhat_r, ic_new, dm)




