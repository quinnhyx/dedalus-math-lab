import numpy as np
import dedalus.public as d3
from mpi4py import MPI
import h5py

# =============================================================
# Restart options
# =============================================================
restart_mode = "from_snapshot"   # "none" or "from_snapshot"
restart_file = "snapshots-channel-restart/snapshots-channel-restart_s2.h5"
restart_index = -1

segment_duration = 20.0

# =============================================================
# Parameters
# =============================================================
Ly, Lx = 2.0, 10.0
Ny, Nx = 128, 128

beta = 0.8
Reynolds = 0.01
nu = beta / Reynolds

epsilon = 1e-3
kappa_c = 5e-5
wi = 1.0

dt_step = 5e-3
dealias = 3/2

alpha_p = (1 - beta) / (Reynolds * wi)

comm = MPI.COMM_WORLD
rank = comm.rank

yL, yU = -Ly/2, Ly/2

def mpi_max(val):
    return comm.allreduce(val, op=MPI.MAX)

# =============================================================
# Domain
# =============================================================
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

# =============================================================
# Fields
# =============================================================
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

# =============================================================
# Forcing
# =============================================================
f0 = dist.VectorField(coords, name='f0', bases=(xb, yb))
f0['g'][0] = 2.0 / Reynolds
f0['g'][1] = 0.0

f_total = dist.VectorField(coords, name='f_total', bases=(xb, yb))
f_total['g'] = f0['g']

# =============================================================
# Conformation tensor
# =============================================================
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

# Wall fields
cxx_b0 = dist.Field(name='cxx_b0', bases=xb)
cxy_b0 = dist.Field(name='cxy_b0', bases=xb)
cyy_b0 = dist.Field(name='cyy_b0', bases=xb)

cxx_bL = dist.Field(name='cxx_bL', bases=xb)
cxy_bL = dist.Field(name='cxy_bL', bases=xb)
cyy_bL = dist.Field(name='cyy_bL', bases=xb)

# =============================================================
# Initial condition OR restart
# =============================================================
if restart_mode == "from_snapshot":

    with h5py.File(restart_file, "r") as f:

        # ---- scalar fields ----
        cxx.load_from_hdf5(f, index=restart_index, task="cxx")
        cxy.load_from_hdf5(f, index=restart_index, task="cxy")
        cyy.load_from_hdf5(f, index=restart_index, task="cyy")

        # ---- velocity (IMPORTANT PATH FIX) ----
        ux = f["tasks/ux"][restart_index]
        uy = f["tasks/uy"][restart_index]

        u['g'][0] = ux
        u['g'][1] = uy

        # ---- time ----
        t = float(f["scales/sim_time"][restart_index])

    u.change_scales(1)

    if rank == 0:
        print(f"[Restart] Loaded at t = {t:.6f}")

else:
    blob = np.exp(-((x - Lx/2)**2 + y**2)/(0.3**2))
    cxx['g'] = 1.0 + 0.2 * blob
    cxy['g'] = 0.0
    cyy['g'] = 1.0 + 0.2 * blob

    u['g'] = 0
    p['g'] = 0

    t = 0.0

# =============================================================
# FIXED wall initialization (important!)
# =============================================================
cxx.change_scales(1)
cxy.change_scales(1)
cyy.change_scales(1)

cxx_b0['g'][:, 0] = cxx['g'][:, 0]
cxy_b0['g'][:, 0] = cxy['g'][:, 0]
cyy_b0['g'][:, 0] = cyy['g'][:, 0]

cxx_bL['g'][:, 0] = cxx['g'][:, -1]
cxy_bL['g'][:, 0] = cxy['g'][:, -1]
cyy_bL['g'][:, 0] = cyy['g'][:, -1]

# =============================================================
# Solvers
# =============================================================
stokes = d3.IVP([u, p, tau_p, tau_u1, tau_u2], namespace=locals())
stokes.add_equation("trace(grad_u) + tau_p = 0")
stokes.add_equation("dt(u) + grad(p) - nu*div(grad_u) + lift(tau_u2) = -u@grad(u) + f_total")
stokes.add_equation(f"u(y={yL}) = 0")
stokes.add_equation(f"u(y={yU}) = 0")
stokes.add_equation("integ(p) = 0")

stokes_solver = stokes.build_solver(d3.RK443)

cprob = d3.IVP(
        [cxx, cxy, cyy,
        tau_cxx1, tau_cxx2,
        tau_cxy1, tau_cxy2,
        tau_cyy1, tau_cyy2,
        cxx_b0, cxy_b0, cyy_b0,
        cxx_bL, cxy_bL, cyy_bL],
        namespace=locals()
    )

# Interior PDEs
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

# Boundary matching: interior field takes the wall-value field at each wall
cprob.add_equation(f"cxx(y={yL}) = cxx_b0")
cprob.add_equation(f"cxy(y={yL}) = cxy_b0")
cprob.add_equation(f"cyy(y={yL}) = cyy_b0")

cprob.add_equation(f"cxx(y={yU}) = cxx_bL")
cprob.add_equation(f"cxy(y={yU}) = cxy_bL")
cprob.add_equation(f"cyy(y={yU}) = cyy_bL")

# Wall evolution with kappa_c = 0
# Bottom wall y = yL
cprob.add_equation(
    f"dt(cxx_b0) = 2*(ux_x(y={yL})*cxx_b0 + ux_y(y={yL})*cxy_b0)"
    " - (1/wi)*(cxx_b0 - 1)*(1 - 2*epsilon + epsilon*(cxx_b0 + cyy_b0))"
)

cprob.add_equation(
    f"dt(cxy_b0) = ux_x(y={yL})*cxy_b0 + ux_y(y={yL})*cyy_b0 + cxx_b0*uy_x(y={yL}) + cxy_b0*uy_y(y={yL})"
    " - (1/wi)*cxy_b0*(1 - 2*epsilon + epsilon*(cxx_b0 + cyy_b0))"
)

cprob.add_equation(
    f"dt(cyy_b0) = 2*(uy_x(y={yL})*cxy_b0 + uy_y(y={yL})*cyy_b0)"
    " - (1/wi)*(cyy_b0 - 1)*(1 - 2*epsilon + epsilon*(cxx_b0 + cyy_b0))"
)

# Top wall y = yU
cprob.add_equation(
    f"dt(cxx_bL) = 2*(ux_x(y={yU})*cxx_bL + ux_y(y={yU})*cxy_bL)"
    " - (1/wi)*(cxx_bL - 1)*(1 - 2*epsilon + epsilon*(cxx_bL + cyy_bL))"
)

cprob.add_equation(
    f"dt(cxy_bL) = ux_x(y={yU})*cxy_bL + ux_y(y={yU})*cyy_bL + cxx_bL*uy_x(y={yU}) + cxy_bL*uy_y(y={yU})"
    " - (1/wi)*cxy_bL*(1 - 2*epsilon + epsilon*(cxx_bL + cyy_bL))"
)

cprob.add_equation(
    f"dt(cyy_bL) = 2*(uy_x(y={yU})*cxy_bL + uy_y(y={yU})*cyy_bL)"
    " - (1/wi)*(cyy_bL - 1)*(1 - 2*epsilon + epsilon*(cxx_bL + cyy_bL))"
)

csolver = cprob.build_solver(d3.RK443)


# Sync time
stokes_solver.sim_time = t
csolver.sim_time = t
csolver.stop_sim_time = t + segment_duration
stokes_solver.stop_sim_time = t + segment_duration


# =============================================================
# Output
# =============================================================
snap = csolver.evaluator.add_file_handler(
    "snapshots-channel-restart",
    sim_dt=0.1,
    max_writes=50,
    mode="overwrite" if restart_mode == "none" else "append"
)

snap.add_task(u @ ex, name="ux")
snap.add_task(u @ ey, name="uy")
snap.add_task(p, name="p")
snap.add_task(cxx, name="cxx")
snap.add_task(cxy, name="cxy")
snap.add_task(cyy, name="cyy")

# =============================================================
# Coupling
# =============================================================
def update_forcing_from_C():
        """
        Update momentum forcing:
            f_total = f0 + div(tau_poly),
        where tau_poly = alpha_p * (C - I).
        """
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

# =============================================================
# Safety check
# =============================================================
def any_nonfinite(*fields):
    for f in fields:
        f.change_scales(1)
        if not np.isfinite(f['g']).all():
            return True
    return False

# =============================================================
# Time loop
# =============================================================
it = 0

while csolver.proceed:

    update_forcing_from_C()

    stokes_solver.step(dt_step)
    csolver.step(dt_step)

    t = csolver.sim_time
    it += 1

    if any_nonfinite(u, p, cxx, cxy, cyy):
        if comm.rank == 0:
            print(f"Non-finite value detected at it={it}, t={t:.6e}. Stopping.")
        break

    if it % 100 == 0:
        cxx.change_scales(1)
        cxy.change_scales(1)
        cyy.change_scales(1)

        tr = cxx['g'] + cyy['g']
        tr_max = mpi_max(np.max(tr))
        cxy_abs_max = mpi_max(np.max(np.abs(cxy['g'])))

        if comm.rank == 0:
            print(
                f"[C] it={it:5d} t={t:.4f}  "
                f"max(trC)={tr_max:.6e}  max(|cxy|)={cxy_abs_max:.6e}"
            )

# =========================
# SAVE FINAL STATE (for comparison)
# =========================
u.change_scales(1)
cxx.change_scales(1)
cxy.change_scales(1)
cyy.change_scales(1)

final_data = {
    "u": u['g'].copy(),
    "cxx": cxx['g'].copy(),
    "cxy": cxy['g'].copy(),
    "cyy": cyy['g'].copy()
}

np.savez(f"final_state_{restart_mode}.npz", **final_data)

if rank == 0:
    print(f"[Saved] final_state_{restart_mode}.npz")