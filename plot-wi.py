import glob
import re
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ============================================================
# Parameters
# ============================================================
Lx = 10
Ly = 2

folder = "snapshots-wi40"
output_name = "flow_animation_wi40.mp4"

# ============================================================
# Find snapshot files
# ============================================================
files = sorted(
    glob.glob(f"{folder}/*.h5"),
    key=lambda x: int(re.search(r"_s(\d+)", x).group(1))
)

print("Found files:")
for f in files:
    print(" ", f)

if len(files) == 0:
    raise RuntimeError("No snapshot files found!")

# ============================================================
# Cache file handles (MUCH faster)
# ============================================================
print("\nOpening HDF5 files...")

file_handles = {
    fname: h5py.File(fname, "r")
    for fname in files
}

# ============================================================
# Build frame map
# frame_map[k] = (filename, local_index)
# ============================================================
frame_map = []
times = []

print("\nIndexing frames...")

for fname in files:

    f = file_handles[fname]

    nt = f["tasks"]["ux"].shape[0]
    sim_times = f["scales"]["sim_time"][:]

    for i in range(nt):
        frame_map.append((fname, i))
        times.append(sim_times[i])

times = np.array(times)

print("Total frames:", len(frame_map))

# ============================================================
# Read one frame to determine shape
# ============================================================
sample = file_handles[files[0]]["tasks"]["trC"][0]

Nx, Ny = sample.shape

x0_index = Nx // 2

y = np.linspace(-Ly/2, Ly/2, Ny)

print("Grid shape:", Nx, Ny)

# ============================================================
# Estimate color limits
# ============================================================
print("\nEstimating color limits...")

cmin = np.inf
cmax = -np.inf

umin = np.inf
umax = -np.inf

sample_frames = min(50, len(frame_map))

for k in range(sample_frames):

    fname, i = frame_map[k]

    f = file_handles[fname]

    ux = f["tasks"]["ux"][i]
    uy = f["tasks"]["uy"][i]

    trC = f["tasks"]["trC"][i]

    u_mag = np.sqrt(ux**2 + uy**2)

    cmin = min(cmin, np.min(trC))
    cmax = max(cmax, np.max(trC))

    umin = min(umin, np.min(u_mag))
    umax = max(umax, np.max(u_mag))

print("tr(C) limits:", cmin, cmax)
print("|u| limits:", umin, umax)

# ============================================================
# Figure setup
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax1, ax2, ax3 = axes

dummy = np.zeros((Nx, Ny))

# ============================================================
# tr(C)
# ============================================================
im1 = ax1.imshow(
    dummy.T,
    origin="lower",
    extent=[0, Lx, -Ly/2, Ly/2],
    aspect="auto",
    vmin=cmin,
    vmax=cmax,
    cmap="viridis",
)

ax1.set_title("Polymer Stretch tr(C)")
ax1.set_xlabel("x")
ax1.set_ylabel("y")

# ============================================================
# |u|
# ============================================================
im2 = ax2.imshow(
    dummy.T,
    origin="lower",
    extent=[0, Lx, -Ly/2, Ly/2],
    aspect="auto",
    vmin=umin,
    vmax=umax,
    cmap="inferno",
)

ax2.set_title("Velocity Magnitude |u|")
ax2.set_xlabel("x")
ax2.set_ylabel("y")

# ============================================================
# Slice plot
# ============================================================
line, = ax3.plot(
    y,
    np.zeros(Ny),
    linewidth=2
)

ax3.set_title(f"tr(C) Slice at x index {x0_index}")
ax3.set_xlabel("y")
ax3.set_ylabel("tr(C)")

ax3.set_xlim(-Ly/2, Ly/2)
ax3.set_ylim(cmin, cmax)

ax3.grid(True)

# ============================================================
# Colorbars
# ============================================================
plt.colorbar(im1, ax=ax1)
plt.colorbar(im2, ax=ax2)

# ============================================================
# Global title
# ============================================================
title = fig.suptitle("")

plt.tight_layout(rect=[0, 0, 1, 0.95])

# ============================================================
# Animation update function
# ============================================================
def update(frame_id):

    fname, i = frame_map[frame_id]

    f = file_handles[fname]

    ux = f["tasks"]["ux"][i]
    uy = f["tasks"]["uy"][i]

    trC = f["tasks"]["trC"][i]

    u_mag = np.sqrt(ux**2 + uy**2)

    # --------------------------------------------------------
    # Update images
    # --------------------------------------------------------
    im1.set_data(trC.T)
    im2.set_data(u_mag.T)

    # --------------------------------------------------------
    # Update slice
    # --------------------------------------------------------
    line.set_ydata(trC[x0_index, :])

    # --------------------------------------------------------
    # Update title
    # --------------------------------------------------------
    title.set_text(f"t = {times[frame_id]:.4f}")

    print(
        f"Rendering frame {frame_id+1}/{len(frame_map)}",
        end="\r"
    )

    return im1, im2, line

# ============================================================
# Create animation
# ============================================================
stride = 20

ani = FuncAnimation(
    fig,
    update,
    frames=range(0, len(frame_map), stride),
    interval=50,
    blit=False,
    cache_frame_data=False,
)

# ============================================================
# Save animation
# ============================================================
writer = FFMpegWriter(
    fps=30,
    codec="libx264",
    bitrate=1500
)

print("\nSaving animation...")
print("Output:", output_name)

ani.save(
    output_name,
    writer=writer,
    dpi=120
)

# ============================================================
# Cleanup
# ============================================================
plt.close(fig)

for f in file_handles.values():
    f.close()

print("\nDone.")