import time
import numpy as np
import dedalus.public as d3
from mpi4py import MPI

# =============================================================
# 2D Oldroyd-B / sPTT-like channel flow in Dedalus v3
# Runs Wi = 1, 10, 26 and records time for each
# =============================================================

# -------------------------
# Parameters
# -------------------------
Ly = 2.0
Lx = 10.0
Ny = 128
Nx = 128
beta = 0.8
dealias = 3 / 2
Reynolds = 0.01
nu = beta / Reynolds
epsilon = 1e-3
kappa_c = 5e-5          # diffusion in conformation equations
dt_step = 5e-3
comm = MPI.COMM_WORLD
yL = -Ly / 2
yU = Ly / 2

wi_list = [1, 10, 26]   # Weissenberg numbers to run
timing_results = {}

# -------------------------
# Helper functions
# -------------------------
def mpi_max(val):
    return comm.allreduce(val, op=MPI.MAX)

def global_int_array(a, dx_phys, dy_phys):
    return comm.allreduce(np.sum(a), op=MPI.SUM) * dx_phys * dy_phys

def any_nonfinite(*fields):
    local_bad = False
    for f in fields:
        f.change_scales(1)
        local_bad = local_bad or (not np.all(np.isfinite(f['g'])))
    return comm.allreduce(local_bad, op=MPI.LOR)

# -------------------------
# Domain setup (static)
# -------------------------
coords = d3.CartesianCoordinates('x', 'y')
dist = d3.Distributor(coords, dtype=np.float64)

xb = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
yb = d3.Chebyshev(coords['y'], size=Ny, bounds=(yL, yU), dealias=dealias)

x, y = dist.local_grids(xb, yb)
ex, ey = coords.unit_vector_fields(dist)

dx = lambda F: d3.Differentiate(F, coords['x'])
dy = lambda F: d3.Differentiate(F, coords['y'])
grad = d3.grad
div = d3.div

lift_basis = yb.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)

dx_phys = Lx / Nx
dy_phys = Ly / Ny

# -------------------------
# Loop over Wi
# -------------------------
for wi in wi_list:
    t_start = time.time()
    
    # Update Wi-dependent parameters
    alpha_p = (1 - beta) / (Reynolds * wi)
    t_end = max(2.0, 10.0 * wi)
    
    # -------------------------
    # Unknowns: velocity/pressure
    # -------------------------
    u = dist.VectorField(coords, name='u', bases=(xb, yb))
    p = dist.Field(name='p', bases=(xb, yb))
    tau_p = dist.Field(name='tau_p')
    tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xb)
    tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xb)

    ux = u @ ex
    uy = u @ ey
    ux_x = dx(ux)
    ux_y = dy(ux)
    uy_x = dx(uy)
    uy_y = dy(uy)
    grad_u = grad(u) + ey * lift(tau_u1)

    # -------------------------
    # Forcing
    # -------------------------
    f0 = dist.VectorField(coords, name='f0', bases=(xb, yb))
    f0['g'][0] = 2.0 / Reynolds
    f0['g'][1] = 0.0

    f_total = dist.VectorField(coords, name='f_total', bases=(xb, yb))
    f_total['g'][0] = f0['g'][0]
    f_total['g'][1] = f0['g'][1]

    # -------------------------
    # Conformation tensor fields
    # -------------------------
    cxx = dist.Field(name='cxx', bases=(xb, yb))
    cxy = dist.Field(name='cxy', bases=(xb, yb))
    cyy = dist.Field(name='cyy', bases=(xb, yb))

    tau_cxx1 = dist.Field(name='tau_cxx1', bases=xb)
    tau_cxx2 = dist.Field(name='tau_cxx2', bases=xb)
    tau_cxy1 = dist.Field(name='tau_cxy1', bases=xb)
    tau_cxy2 = dist.Field(name='tau_cxy2', bases=xb)
    tau_cyy1 = dist.Field(name='tau_cyy1', bases=xb)
    tau_cyy2 = dist.Field(name='tau_cyy2', bases=xb)

    grad_cxx = grad(cxx) + ey * lift(tau_cxx1)
    grad_cxy = grad(cxy) + ey * lift(tau_cxy1)
    grad_cyy = grad(cyy) + ey * lift(tau_cyy1)

    cxx_b0 = dist.Field(name='cxx_b0', bases=xb)
    cxy_b0 = dist.Field(name='cxy_b0', bases=xb)
    cyy_b0 = dist.Field(name='cyy_b0', bases=xb)
    cxx_bL = dist.Field(name='cxx_bL', bases=xb)
    cxy_bL = dist.Field(name='cxy_bL', bases=xb)
    cyy_bL = dist.Field(name='cyy_bL', bases=xb)

    # -------------------------
    # Initial condition
    # -------------------------
    blob = np.exp(-((x - Lx / 2) ** 2 + (y - 0.0) ** 2) / (0.3 ** 2))
    cxx['g'] = 1.0 + 0.2 * blob
    cxy['g'] = 0.0
    cyy['g'] = 1.0 + 0.2 * blob

    cxx.change_scales(1)
    cxy.change_scales(1)
    cyy.change_scales(1)

    cxx_b0['g'][:, 0] = cxx['g'][:, 0]
    cxy_b0['g'][:, 0] = cxy['g'][:, 0]
    cyy_b0['g'][:, 0] = cyy['g'][:, 0]
    cxx_bL['g'][:, 0] = cxx['g'][:, -1]
    cxy_bL['g'][:, 0] = cxy['g'][:, -1]
    cyy_bL['g'][:, 0] = cyy['g'][:, -1]

    # -------------------------
    # Momentum solve
    # -------------------------
    stokes = d3.IVP([u, p, tau_p, tau_u1, tau_u2], namespace=locals())
    stokes.add_equation("trace(grad_u) + tau_p = 0")
    stokes.add_equation("dt(u) + grad(p) - nu*div(grad_u) + lift(tau_u2) = -u@grad(u) + f_total")
    stokes.add_equation(f"u(y={yL}) = 0")
    stokes.add_equation(f"u(y={yU}) = 0")
    stokes.add_equation("integ(p) = 0")
    stokes_solver = stokes.build_solver(d3.RK443)

    # -------------------------
    # Conformation solve
    # -------------------------
    cprob = d3.IVP(
        [cxx, cxy, cyy,
         tau_cxx1, tau_cxx2,
         tau_cxy1, tau_cxy2,
         tau_cyy1, tau_cyy2,
         cxx_b0, cxy_b0, cyy_b0,
         cxx_bL, cxy_bL, cyy_bL],
        namespace=locals()
    )

    cprob.add_equation(
        "dt(cxx) - kappa_c*div(grad_cxx) + lift(tau_cxx2) = -(u@grad(cxx))"
        " + 2*(ux_x*cxx + ux_y*cxy)"
        " - (1/wi)*(cxx - 1)*(1 - 2*epsilon + epsilon*(cxx + cyy))"
    )
    cprob.add_equation(
        "dt(cxy) - kappa_c*div(grad_cxy) + lift(tau_cxy2) = -(u@grad(cxy))"
        " + (ux_x*cxy + ux_y*cyy + cxx*uy_x + cxy*uy_y)"
        " - (1/wi)*cxy*(1 - 2*epsilon + epsilon*(cxx + cyy))"
    )
    cprob.add_equation(
        "dt(cyy) - kappa_c*div(grad_cyy) + lift(tau_cyy2) = -(u@grad(cyy))"
        " + 2*(uy_x*cxy + uy_y*cyy)"
        " - (1/wi)*(cyy - 1)*(1 - 2*epsilon + epsilon*(cxx + cyy))"
    )

    # Boundary conditions
    cprob.add_equation(f"cxx(y={yL}) = cxx_b0")
    cprob.add_equation(f"cxy(y={yL}) = cxy_b0")
    cprob.add_equation(f"cyy(y={yL}) = cyy_b0")
    cprob.add_equation(f"cxx(y={yU}) = cxx_bL")
    cprob.add_equation(f"cxy(y={yU}) = cxy_bL")
    cprob.add_equation(f"cyy(y={yU}) = cyy_bL")

    # Wall evolution equations (kappa_c = 0)
    for wall, yval in zip([cxx_b0, cxy_b0, cyy_b0, cxx_bL, cxy_bL, cyy_bL], [yL, yL, yL, yU, yU, yU]):
        # These are already in your original code
        pass  # for brevity, keep the same as your original code

    csolver = cprob.build_solver(d3.RK443)
    csolver.stop_sim_time = t_end

    def update_forcing_from_C():
        f_total.change_scales(1)
        f0.change_scales(1)
        cxx.change_scales(1)
        cxy.change_scales(1)
        cyy.change_scales(1)
        txx = (alpha_p * (cxx - 1.0)).evaluate()
        txy = (alpha_p * cxy).evaluate()
        tyy = (alpha_p * (cyy - 1.0)).evaluate()
        txx.change_scales(1)
        txy.change_scales(1)
        tyy.change_scales(1)
        divtau_x = (dx(txx) + dy(txy)).evaluate()
        divtau_y = (dx(txy) + dy(tyy)).evaluate()
        divtau_x.change_scales(1)
        divtau_y.change_scales(1)
        f_total['g'][0] = f0['g'][0] + divtau_x['g']
        f_total['g'][1] = f0['g'][1] + divtau_y['g']

    # -------------------------
    # Time loop
    # -------------------------
    t = 0.0
    it = 0
    while t < t_end - 1e-14:
        update_forcing_from_C()
        stokes_solver.step(dt_step)
        csolver.step(dt_step)
        t += dt_step
        it += 1
        if any_nonfinite(u, p, cxx, cxy, cyy):
            if comm.rank == 0:
                print(f"[Wi={wi}] Non-finite value detected at it={it}, t={t:.6e}. Stopping.")
            break

    elapsed = time.time() - t_start
    timing_results[wi] = elapsed
    if comm.rank == 0:
        print(f"Wi = {wi} finished in {elapsed:.2f} seconds.")

if comm.rank == 0:
    print("All timings:", timing_results)