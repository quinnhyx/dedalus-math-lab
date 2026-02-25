import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

# Parameters
Lx, Lz = 4, 1
# Nx, Nz = 128, 256
Nx, Nz = 256, 64
Reynolds = 500
Schmidt = 1
dealias = 3/2
stop_sim_time = 20
timestepper = d3.RK222
max_timestep = 1e-2
dtype = np.float64
A = 1

# Bases
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.Chebyshev(coords['z'], size=Nz, bounds=(-Lz/2, Lz/2), dealias=dealias)

# Fields
p = dist.Field(name='p', bases=(xbasis,zbasis))
s = dist.Field(name='s', bases=(xbasis,zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))
tau_p = dist.Field(name='tau_p')
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)
tau_s1 = dist.Field(name='tau_s1', bases=xbasis)
tau_s2 = dist.Field(name='tau_s2', bases=xbasis)

# Lift operator
lift_basis = zbasis.derivative_basis(1)
lift = lambda tau: d3.Lift(tau, lift_basis, -1)

# Substitutions
nu = 1 / Reynolds
D = nu / Schmidt
x, z = dist.local_grids(xbasis, zbasis)
ex, ez = coords.unit_vector_fields(dist)

grad_u = d3.grad(u)+ez*lift(tau_u1)
# gradu = grad_u
grad_s = d3.grad(s) + ez*lift(tau_s1)


# Problem
problem = d3.IVP([u, s, p, tau_p, tau_u1, tau_u2, tau_s1, tau_s2], namespace=locals())
# problem.add_equation("dt(u) + grad(p) - nu*lap(u) = - u@grad(u)")
problem.add_equation(
    "dt(u) + grad(p) - nu*div(grad_u) + lift(tau_u2) = - u@grad(u)"
)
problem.add_equation("dt(s) - D*div(grad_s) + lift(tau_s2) = - u@grad(s)")
# problem.add_equation("div(u) + tau_p = 0")
problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("integ(p) = 0") # Pressure gauge
problem.add_equation(f"u(z={-Lz/2}) = 0")
problem.add_equation(f"u(z={+Lz/2}) = 0")

# Tracer BCs (Dirichlet baseline)
problem.add_equation(f"s(z={-Lz/2}) = 0")
problem.add_equation(f"s(z={+Lz/2}) = 0")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
# Background shear matching no-slip walls at z = ±Lz/2
# u['g'][0] = A * np.sin(np.pi * (z + Lz/2) / Lz)  # u_x(z)
u['g'][0] = np.tanh(z / 0.05)
u['g'][1] = 0
s['g'] = u['g'][0]
u['g'][1] += 0.1 * np.sin(2*np.pi*x/Lx) * np.exp(-(z-0.5)**2/0.01)
u['g'][1] += 0.1 * np.sin(2*np.pi*x/Lx) * np.exp(-(z+0.5)**2/0.01)


# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots-boundary', sim_dt=0.1, max_writes=1)
snapshots.add_task(s, name='tracer')
snapshots.add_task(p, name='pressure')
snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.2, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property((u@ez)**2, name='w2')

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            max_w = np.sqrt(flow.max('w2'))
            logger.info('Iteration=%i, Time=%e, dt=%e, max(w)=%f' %(solver.iteration, solver.sim_time, timestep, max_w))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
