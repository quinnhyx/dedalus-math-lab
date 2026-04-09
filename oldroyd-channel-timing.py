import numpy as np
import dedalus.public as d3
from mpi4py import MPI

# =============================================================
# 2D Oldroyd-B / sPTT-like channel flow in Dedalus v3
#
# Centered channel: y in [-Ly/2, Ly/2]
#
# The conformation tensor uses diffusive interior equations,
# but the wall values are evolved from the constitutive equation
# with kappa_c = 0, following the channel-flow literature.
# =============================================================

# -------------------------
# Parameters
# -------------------------
Ly = 2.0
Lx = 10.0
Ny_list = [256,512,1024]  # try different vertical resolutions to test convergence
Nx = 256
beta = 0.8
dealias = 3 / 2
Reynolds = 0.01
nu = beta / Reynolds
epsilon = 1e-3

kappa_c = 5e-5          # diffusion in conformation equations
wi = 1.0                # Weissenberg number
dt_step = 5e-3
t_end = max(2.0, 10.0 * wi)

alpha_p = (1 - beta) / (Reynolds * wi)
comm = MPI.COMM_WORLD

# Centered wall locations
yL = -Ly / 2
yU =  Ly / 2


for Ny in Ny_list:
    if comm.rank == 0:
        print(f"Running with Ny = {Ny}")

    def mpi_max(val):
        return comm.allreduce(val, op=MPI.MAX)


    def global_int_array(a, dx_phys, dy_phys):
        return comm.allreduce(np.sum(a), op=MPI.SUM) * dx_phys * dy_phys


    # -------------------------
    # Domain
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

    # Tau fields for diffusive scalar equations
    # (one first-order tau + one second-order tau per scalar)
    tau_cxx1 = dist.Field(name='tau_cxx1', bases=xb)
    tau_cxx2 = dist.Field(name='tau_cxx2', bases=xb)
    tau_cxy1 = dist.Field(name='tau_cxy1', bases=xb)
    tau_cxy2 = dist.Field(name='tau_cxy2', bases=xb)
    tau_cyy1 = dist.Field(name='tau_cyy1', bases=xb)
    tau_cyy2 = dist.Field(name='tau_cyy2', bases=xb)

    grad_cxx = grad(cxx) + ey * lift(tau_cxx1)
    grad_cxy = grad(cxy) + ey * lift(tau_cxy1)
    grad_cyy = grad(cyy) + ey * lift(tau_cyy1)

    # Wall-value fields: functions of x and t only
    cxx_b0 = dist.Field(name='cxx_b0', bases=xb)
    cxy_b0 = dist.Field(name='cxy_b0', bases=xb)
    cyy_b0 = dist.Field(name='cyy_b0', bases=xb)

    cxx_bL = dist.Field(name='cxx_bL', bases=xb)
    cxy_bL = dist.Field(name='cxy_bL', bases=xb)
    cyy_bL = dist.Field(name='cyy_bL', bases=xb)

    # Initial condition: identity + centered blob inside the domain
    blob = np.exp(-((x - Lx / 2) ** 2 + (y - 0.0) ** 2) / (0.3 ** 2))
    cxx['g'] = 1.0 + 0.2 * blob
    cxy['g'] = 0.0
    cyy['g'] = 1.0 + 0.2 * blob

    # Initialize wall fields from the initial conformation tensor
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
    # Interior: diffusive PDE
    # Walls: evolve wall values from constitutive equation with kappa_c = 0
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
    csolver.stop_sim_time = t_end


    # -------------------------
    # Coupling / forcing update
    # -------------------------
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


    # -------------------------
    # Diagnostics
    # -------------------------
    def trC_mean():
        cxx.change_scales(1)
        cyy.change_scales(1)
        tr = cxx['g'] + cyy['g']
        return global_int_array(tr, dx_phys, dy_phys) / (Lx * Ly)


    def trC_L2_sq_int():
        cxx.change_scales(1)
        cyy.change_scales(1)
        tr = cxx['g'] + cyy['g']
        return global_int_array(tr * tr, dx_phys, dy_phys)


    def trC_grad_sq_int():
        trF = (cxx + cyy).evaluate()
        trx = dx(trF).evaluate()
        try_ = dy(trF).evaluate()
        trx.change_scales(1)
        try_.change_scales(1)
        return global_int_array(trx['g'] ** 2 + try_['g'] ** 2, dx_phys, dy_phys)


    def any_nonfinite(*fields):
        local_bad = False
        for f in fields:
            f.change_scales(1)
            local_bad = local_bad or (not np.all(np.isfinite(f['g'])))
        return comm.allreduce(local_bad, op=MPI.LOR)


    # -------------------------
    # Output
    # -------------------------
    snap = csolver.evaluator.add_file_handler("snapshots-channel", sim_dt=0.1, max_writes=10)
    snap.add_task(u @ ex, name="ux")
    snap.add_task(u @ ey, name="uy")
    snap.add_task(p, name="p")
    snap.add_task(cxx, name="cxx")
    snap.add_task(cxy, name="cxy")
    snap.add_task(cyy, name="cyy")
    snap.add_task(cxx + cyy, name="trC")


    # -------------------------
    # Time loop
    # -------------------------
    t = 0.0
    it = 0

    comm.Barrier()  # sync before timing
    t_start = MPI.Wtime()
    
    while t < t_end - 1e-14:
        update_forcing_from_C()

        stokes_solver.step(dt_step)
        csolver.step(dt_step)

        t += dt_step
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
                
    comm.Barrier()
    t_end_wall = MPI.Wtime()
    runtime = t_end_wall - t_start
    runtime = comm.allreduce(runtime, op=MPI.MAX)

    if comm.rank == 0:
        print(f"Ny={Ny} runtime: {runtime:.4f} seconds")
        