import numpy as np
import dedalus.public as d3
from mpi4py import MPI
import os
import re
import glob
import h5py
from steady_state_detector import SteadyStateDetector

def find_latest_checkpoint(base_folder):
    candidates = glob.glob(os.path.join(base_folder, "*_s*"))

    latest_time = -np.inf
    latest_file = None

    for path in candidates:

        # ---------- file checkpoint ----------
        if path.endswith(".h5"):
            try:
                with h5py.File(path, 'r') as h5:
                    t = h5['scales/sim_time'][-1]
            except Exception:
                continue

        # ---------- mpi folder ----------
        elif os.path.isdir(path):
            h5_files = glob.glob(os.path.join(path, "*_p0.h5"))

            if not h5_files:
                continue

            f = h5_files[0]

            try:
                with h5py.File(f, 'r') as h5:
                    t = h5['scales/sim_time'][-1]
            except Exception:
                continue

        else:
            continue

        # ---------- largest time ----------
        if t > latest_time:
            latest_time = t
            latest_file = path if path.endswith(".h5") else f

    if latest_file is None:
        raise RuntimeError(f"No valid checkpoint found in {base_folder}")

    print(f"[DEBUG] latest checkpoint: {latest_file}, t={latest_time}")

    return latest_file

# =============================================================
# Parameters
# =============================================================
Ly = 2.0
Lx = 10.0
Ny = 1024
Nx = 256
beta = 0.8
dealias = 3/2
Reynolds = 0.01
nu = beta / Reynolds
epsilon = 1e-3

kappa_c = 5e-5
wi = 80.0
dt_step = 5e-3


t0 = 0.0
t1_target = 1100.0

checkpoint_dt = 0.1
snapshot_dt = 0.1


# Stop slightly past the nominal segment end so a checkpoint at t1_target
# is actually written when t1_target lies on the checkpoint grid.
stop_pad = 0.51 * dt_step
t1_stop = t1_target + stop_pad

# =============================================================
# Restart options
# =============================================================
restart_mode = "from_checkpoint"   # "none" or "from_checkpoint"

restart_index = -1

# Sparse runtime monitoring / guardrails
log_iter = 200
nan_check_iter = 100
monitor_iter = 100

alpha_p = (1 - beta) / (Reynolds * wi)
yL = -Ly / 2
yU =  Ly / 2

# =============================================================
# MPI
# =============================================================
comm = MPI.COMM_WORLD
rank = comm.rank

def mpi_max(val):
    return comm.allreduce(val, op=MPI.MAX)

def mpi_min(val):
    return comm.allreduce(val, op=MPI.MIN)

def monitor_stats():
    # 统一 scale
    u.change_scales(1)
    cxx.change_scales(1)
    cxy.change_scales(1)
    cyy.change_scales(1)

    # ---- physical quantities ----
    tr = cxx['g'] + cyy['g']
    det = cxx['g'] * cyy['g'] - cxy['g']**2
    usq = u['g'][0]**2 + u['g'][1]**2

    # ---- MPI global reduction ----
    tr_max = mpi_max(np.max(tr))
    tr_min = mpi_min(np.min(tr))
    det_min = mpi_min(np.min(det))
    umax = mpi_max(np.sqrt(np.max(usq)))
    cxy_abs_max = mpi_max(np.max(np.abs(cxy['g'])))

    return tr_max, tr_min, det_min, umax, cxy_abs_max

def monitor_tensor_stats():

    # --------------------------------------------------------
    # Ensure physical scale
    # --------------------------------------------------------
    cxx.change_scales(1)
    cxy.change_scales(1)
    cyy.change_scales(1)

    # --------------------------------------------------------
    # Local arrays
    # --------------------------------------------------------
    cxx_g = cxx['g']
    cxy_g = cxy['g']
    cyy_g = cyy['g']

    # --------------------------------------------------------
    # L2 norms
    # --------------------------------------------------------
    cxx_norm_local = np.sqrt(np.sum(cxx_g**2))
    cxy_norm_local = np.sqrt(np.sum(cxy_g**2))
    cyy_norm_local = np.sqrt(np.sum(cyy_g**2))

    # --------------------------------------------------------
    # Global L2 norms
    # --------------------------------------------------------
    cxx_norm = np.sqrt(comm.allreduce(cxx_norm_local**2, op=MPI.SUM))
    cxy_norm = np.sqrt(comm.allreduce(cxy_norm_local**2, op=MPI.SUM))
    cyy_norm = np.sqrt(comm.allreduce(cyy_norm_local**2, op=MPI.SUM))

    # --------------------------------------------------------
    # Maximum values
    # --------------------------------------------------------
    cxx_max = mpi_max(np.max(cxx_g))
    cxy_max = mpi_max(np.max(cxy_g))
    cyy_max = mpi_max(np.max(cyy_g))

    return (
        cxx_norm,
        cxy_norm,
        cyy_norm,
        cxx_max,
        cxy_max,
        cyy_max,
    )

def any_nonfinite(*fields):
    local_bad = False
    for f in fields:
        f.change_scales(1)
        local_bad = local_bad or (not np.all(np.isfinite(f['g'])))
    return comm.allreduce(local_bad, op=MPI.LOR)

def compute_kinetic_energy():
    u.change_scales(1)
    usq = u['g'][0]**2 + u['g'][1]**2

    local_sum = np.sum(usq)
    local_count = usq.size

    total_sum = comm.allreduce(local_sum, op=MPI.SUM)
    total_count = comm.allreduce(local_count, op=MPI.SUM)

    return 0.5 * total_sum / total_count

# ============================================================
# Domain / fields
# ============================================================
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

# Momentum fields
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

# Forcing
f0 = dist.VectorField(coords, name='f0', bases=(xb, yb))
f0['g'][0] = 2.0 / Reynolds
f0['g'][1] = 0.0

f_total = dist.VectorField(coords, name='f_total', bases=(xb, yb))
f_total['g'][0] = f0['g'][0]
f_total['g'][1] = f0['g'][1]

# Conformation fields
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

# Wall fields (x-only)
cxx_b0 = dist.Field(name='cxx_b0', bases=xb)
cxy_b0 = dist.Field(name='cxy_b0', bases=xb)
cyy_b0 = dist.Field(name='cyy_b0', bases=xb)
cxx_bL = dist.Field(name='cxx_bL', bases=xb)
cxy_bL = dist.Field(name='cxy_bL', bases=xb)
cyy_bL = dist.Field(name='cyy_bL', bases=xb)

# ============================================================
# Initial condition
# ============================================================
if restart_mode == "none":
    blob = np.exp(-((x - Lx / 2) ** 2 + y ** 2) / (0.3 ** 2))
    cxx['g'] = 1.0 + 0.2 * blob
    cxy['g'] = 0.0
    cyy['g'] = 1.0 + 0.2 * blob

    cxx.change_scales(1)
    cxy.change_scales(1)
    cyy.change_scales(1)

    # MPI-safe wall initialization: evaluate boundary traces as x-only fields
    cxx_b0_tmp = cxx(y=yL).evaluate()
    cxy_b0_tmp = cxy(y=yL).evaluate()
    cyy_b0_tmp = cyy(y=yL).evaluate()

    cxx_bL_tmp = cxx(y=yU).evaluate()
    cxy_bL_tmp = cxy(y=yU).evaluate()
    cyy_bL_tmp = cyy(y=yU).evaluate()

    cxx_b0_tmp.change_scales(1)
    cxy_b0_tmp.change_scales(1)
    cyy_b0_tmp.change_scales(1)

    cxx_bL_tmp.change_scales(1)
    cxy_bL_tmp.change_scales(1)
    cyy_bL_tmp.change_scales(1)

    cxx_b0.change_scales(1)
    cxy_b0.change_scales(1)
    cyy_b0.change_scales(1)

    cxx_bL.change_scales(1)
    cxy_bL.change_scales(1)
    cyy_bL.change_scales(1)

    cxx_b0['g'] = cxx_b0_tmp['g']
    cxy_b0['g'] = cxy_b0_tmp['g']
    cyy_b0['g'] = cyy_b0_tmp['g']

    cxx_bL['g'] = cxx_bL_tmp['g']
    cxy_bL['g'] = cxy_bL_tmp['g']
    cyy_bL['g'] = cyy_bL_tmp['g']

# ============================================================
# Problems / solvers
# ============================================================
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

cprob.add_equation(f"cxx(y={yL}) = cxx_b0")
cprob.add_equation(f"cxy(y={yL}) = cxy_b0")
cprob.add_equation(f"cyy(y={yL}) = cyy_b0")
cprob.add_equation(f"cxx(y={yU}) = cxx_bL")
cprob.add_equation(f"cxy(y={yU}) = cxy_bL")
cprob.add_equation(f"cyy(y={yU}) = cyy_bL")

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

# ============================================================
# Restart handling
# ============================================================
if restart_mode == "from_checkpoint":
    file_handler_mode = "append"   # append to existing files, which will be overwritten from the restart point onward

    stokes_restart_file = find_latest_checkpoint("checkpoints_stokes-wi40")
    conf_restart_file   = find_latest_checkpoint("checkpoints_conf-wi40")

    print("Using stokes:", stokes_restart_file)
    print("Using conf:", conf_restart_file)

    stokes_solver.load_state(stokes_restart_file, index=restart_index)
    csolver.load_state(conf_restart_file, index=restart_index)

    if abs(stokes_solver.sim_time - csolver.sim_time) > 1e-12:
        raise RuntimeError(
            f"Restart mismatch: stokes={stokes_solver.sim_time}, conf={csolver.sim_time}"
        )

    if rank == 0:
        print(f"[Restart OK] resumed at t = {stokes_solver.sim_time:.6f}")

else:
    file_handler_mode = "overwrite"

    if rank == 0:
        print(f"[Fresh run] starting at t = {t0:.6f}")

stokes_solver.stop_sim_time = t1_stop
csolver.stop_sim_time = t1_stop

# ============================================================
# Output handlers
# ============================================================
snapshots = csolver.evaluator.add_file_handler(
    "snapshots-wi40",
    sim_dt=snapshot_dt,
    max_writes=50,
    mode=file_handler_mode,
)
snapshots.add_task(u @ ex, name="ux")
snapshots.add_task(u @ ey, name="uy")
snapshots.add_task(p, name="p")
snapshots.add_task(cxx, name="cxx")
snapshots.add_task(cxy, name="cxy")
snapshots.add_task(cyy, name="cyy")
snapshots.add_task(cxx + cyy, name="trC")

checkpoints_stokes = stokes_solver.evaluator.add_file_handler(
    "checkpoints_stokes-wi40",
    sim_dt=checkpoint_dt,
    max_writes=10,
    mode=file_handler_mode,
)
checkpoints_stokes.add_tasks(stokes_solver.state)

checkpoints_conf = csolver.evaluator.add_file_handler(
    "checkpoints_conf-wi40",
    sim_dt=checkpoint_dt,
    max_writes=10,
    mode=file_handler_mode,
)
checkpoints_conf.add_tasks(csolver.state)

# ============================================================
# Diagnostics / guardrails
# ============================================================
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

def check_finite_state():
    bad = []
    fields = {
        'u': u,
        'p': p,
        'tau_p': tau_p,
        'tau_u1': tau_u1,
        'tau_u2': tau_u2,
        'cxx': cxx,
        'cxy': cxy,
        'cyy': cyy,
        'tau_cxx1': tau_cxx1,
        'tau_cxx2': tau_cxx2,
        'tau_cxy1': tau_cxy1,
        'tau_cxy2': tau_cxy2,
        'tau_cyy1': tau_cyy1,
        'tau_cyy2': tau_cyy2,
        'cxx_b0': cxx_b0,
        'cxy_b0': cxy_b0,
        'cyy_b0': cyy_b0,
        'cxx_bL': cxx_bL,
        'cxy_bL': cxy_bL,
        'cyy_bL': cyy_bL,
        'f_total': f_total,
    }
    for name, fld in fields.items():
        fld.change_scales(1)
        if not np.isfinite(fld['g']).all():
            bad.append(name)
    return bad

# =============================================================
# Loop
# =============================================================
it = 0
detector = SteadyStateDetector(threshold=1e-10, duration=10.0)
steady_state_reached = False

while stokes_solver.proceed and csolver.proceed:

    update_forcing_from_C()

    stokes_solver.step(dt_step)
    csolver.step(dt_step)

    it += 1
    t = csolver.sim_time    

    # ---- NaN check ----
    if it % 50 == 0:
        if any_nonfinite(u, p, cxx, cxy, cyy):
            if rank == 0:
                print(f"[ERROR] Non-finite detected at it={it}, t={t:.6e}")
            break

    # ---- monitor ----
    if it % 100 == 0:
        E = compute_kinetic_energy()
        tr_max, tr_min, det_min, umax, cxy_abs_max = monitor_stats()
        cxx_norm, cxy_norm, cyy_norm, cxx_max, cxy_max, cyy_max = monitor_tensor_stats()

        if rank == 0:
            print(
                f"[STAT] it={it:5d} t={t:.4f} "
                f"E={E:.3e} "
                f"tr=[{tr_min:.2e},{tr_max:.2e}] "
                f"det_min={det_min:.2e} "
                f"u_max={umax:.2e} "
                f"cxy_max={cxy_abs_max:.2e}"
            )
            print(f"||cxx||_2 = {cxx_norm:.6e}   max(cxx) = {cxx_max:.6e}")
            print(f"||cxy||_2 = {cxy_norm:.6e}   max(cxy) = {cxy_max:.6e}")
            print(f"||cyy||_2 = {cyy_norm:.6e}   max(cyy) = {cyy_max:.6e}")

        # ---- steady check----
        if not steady_state_reached and detector.update(E, t):
            steady_state_reached = True
            if rank == 0:
                print(f"[INFO] Steady state reached at t={t:.6f}")
        
# =============================================================
# Done
# =============================================================
if rank == 0:
    print("DONE")