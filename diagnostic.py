import glob
import re
import h5py
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Resolution / decay diagnostic
#
# Goal:
#   Check spectral decay for:
#       cxx, cxy, cyy, ux, uy
#
# Produces:
#   1. Real-space slices
#   2. Fourier decay in x
#   3. Chebyshev-like decay in y
#
# Recommended:
#   Run separately for Wi=1 and Wi=26
# ============================================================

# ============================================================
# USER SETTINGS
# ============================================================
folder = "snapshots-wi40"

Lx = 10
Ly = 2

# use final snapshot
frame_index = -1

# random locations
x_phys = 5.0 + np.random.uniform(-0.5, 0.5)
y_phys = 0.0 + np.random.uniform(-0.2, 0.2)

# ============================================================
# Find latest snapshot file
# ============================================================
files = sorted(
    glob.glob(f"{folder}/*.h5"),
    key=lambda x: int(re.search(r"_s(\d+)", x).group(1))
)

if len(files) == 0:
    raise RuntimeError("No snapshot files found!")

fname = files[-1]

print("Using file:", fname)

# ============================================================
# Load data
# ============================================================
with h5py.File(fname, "r") as f:

    ux  = f["tasks"]["ux"][frame_index]
    uy  = f["tasks"]["uy"][frame_index]

    cxx = f["tasks"]["cxx"][frame_index]
    cxy = f["tasks"]["cxy"][frame_index]
    cyy = f["tasks"]["cyy"][frame_index]

    t = f["scales"]["sim_time"][frame_index]

print("Loaded snapshot at t =", t)

# ============================================================
# Grid
# ============================================================
Nx, Ny = cxx.shape

x = np.linspace(0, Lx, Nx, endpoint=False)
y = np.linspace(-Ly/2, Ly/2, Ny)

# nearest indices
ix = np.argmin(np.abs(x - x_phys))
iy = np.argmin(np.abs(y - y_phys))

print("x index =", ix, "x =", x[ix])
print("y index =", iy, "y =", y[iy])

# ============================================================
# Helper:
# Fourier decay in x
# ============================================================
def fourier_decay(field_2d):

    # fixed y slice
    signal = field_2d[:, iy]
    
    #signal = signal - np.mean(signal)

    fft_vals = np.fft.rfft(signal)

    amp = np.abs(fft_vals)

    amp /= amp[0]

    return amp

# ============================================================
# Helper:
# Chebyshev-like decay in y
#
# Use DCT as proxy for Chebyshev coefficients
# ============================================================
def cheb_decay(field_2d):

    # fixed x slice
    signal = field_2d[ix, :]
    
    #signal = signal - np.mean(signal)

    coeffs = np.abs(np.fft.rfft(signal))

    coeffs /= coeffs[0]

    return coeffs

def add_contour(ax, field, title, cmap="viridis", levels=40):

    im = ax.contourf(
        x,
        y,
        field.T,
        levels=levels,
        cmap=cmap
    )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    plt.colorbar(im, ax=ax)

    return im

# ============================================================
# Compute spectra
# ============================================================
specs_x = {
    "cxx": fourier_decay(cxx),
    "cxy": fourier_decay(cxy),
    "cyy": fourier_decay(cyy),
    "ux":  fourier_decay(ux),
    "uy":  fourier_decay(uy),
}

specs_y = {
    "cxx": cheb_decay(cxx),
    "cxy": cheb_decay(cxy),
    "cyy": cheb_decay(cyy),
    "ux":  cheb_decay(ux),
    "uy":  cheb_decay(uy),
}

# ============================================================
# Plot
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(22, 10))

# ============================================================
# REAL SPACE: x-slice
# ============================================================
axes[0,0].plot(x, cxx[:, iy], label="cxx")
axes[0,0].plot(x, cxy[:, iy], label="cxy")
axes[0,0].plot(x, cyy[:, iy], label="cyy")
axes[0,0].set_title("Real space at y=0")
axes[0,0].legend()

# ============================================================
# REAL SPACE: y-slice
# ============================================================
axes[0,1].plot(y, cxx[ix, :], label="cxx")
axes[0,1].plot(y, cxy[ix, :], label="cxy")
axes[0,1].plot(y, cyy[ix, :], label="cyy")
axes[0,1].set_title("Real space at x=5")
axes[0,1].legend()

# ============================================================
# Velocity slice
# ============================================================
axes[0,2].plot(x, ux[:, iy], label="ux")
axes[0,2].plot(x, uy[:, iy], label="uy")
axes[0,2].set_title("Velocity slices")
axes[0,2].legend()

# ============================================================
# tr(C)
# ============================================================
trC = cxx + cyy

im0 = axes[0,3].imshow(
    trC.T,
    origin="lower",
    extent=[0, Lx, -Ly/2, Ly/2],
    aspect="auto",
    cmap="viridis"
)

axes[0,3].set_title("tr(C)")
plt.colorbar(im0, ax=axes[0,3])

# ============================================================
# Fourier decay (x)
# ============================================================
for k, v in specs_x.items():
    axes[1,0].plot(np.log10(v + 1e-30), label=k)

axes[1,0].set_title("Fourier decay in x")
axes[1,0].legend()

# ============================================================
# Chebyshev-like decay (y)
# ============================================================
for k, v in specs_y.items():
    axes[1,1].plot(np.log10(v + 1e-30), label=k)

axes[1,1].set_title("Decay in y")
axes[1,1].legend()

# ============================================================
# Contours
# ============================================================
add_contour(
    axes[1,2],
    cxy,
    "cxy contour",
    cmap="RdBu_r"
)

add_contour(
    axes[1,3],
    uy,
    "uy contour",
    cmap="coolwarm"
)

# ============================================================
# Title + save
# ============================================================
fig.suptitle(
    f"Resolution diagnostic: {folder}\n t = {t:.4f}",
    fontsize=16
)

plt.tight_layout()
plt.savefig(
    f"resolution_decay_wi80.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()