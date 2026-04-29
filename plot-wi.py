import glob
import os
import re
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

Lx = 10
Ly = 2

# =============================
# Target folder (wi = 26)
# =============================
folder = "snapshots-wi26.0"

# 自动找所有 h5（并排序）
files = sorted(
    glob.glob(f"{folder}/*.h5"),
    key=lambda x: int(re.search(r"_s(\d+)", x).group(1))
)

print("Found files:", files)

if len(files) == 0:
    raise RuntimeError("No snapshot files found!")

# =============================
# Load data
# =============================
trC = []
ux = []
uy = []
times = []

for fname in files:
    print("Loading:", fname)
    with h5py.File(fname, "r") as f:

        ux.append(f["tasks"]["ux"][:])
        uy.append(f["tasks"]["uy"][:])

        cxx = f["tasks"]["cxx"][:]
        cyy = f["tasks"]["cyy"][:]
        trC.append(cxx + cyy)

        times.append(f["scales"]["sim_time"][:])

# 拼接时间序列
ux = np.concatenate(ux)
uy = np.concatenate(uy)
trC = np.concatenate(trC)
times = np.concatenate(times)

u_mag = np.sqrt(ux**2 + uy**2)

print("First 10 times BEFORE sort:", times[:10])
# =============================
# FIX: sort by time
# =============================
idx = np.argsort(times)

times = times[idx]
ux = ux[idx]
uy = uy[idx]
trC = trC[idx]
u_mag = u_mag[idx]

print("Total frames:", len(times))
print("Data shape:", trC.shape)

# =============================
# Grid
# =============================
Nt, Nx, Ny = trC.shape

x0_index = Nx // 2
y = np.linspace(-Ly/2, Ly/2, Ny)

# =============================
# Color limits
# =============================
cmin, cmax = trC.min(), trC.max()
umin, umax = u_mag.min(), u_mag.max()

# =============================
# Plot
# =============================
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
ax1, ax2, ax3 = axes

# trC
im1 = ax1.imshow(
    trC[0].T,
    origin="lower",
    vmin=cmin,
    vmax=cmax,
    extent=[0, Lx, -Ly/2, Ly/2],
    aspect='auto'
)
ax1.set_title("Polymer Stretch tr(C)")

# velocity
im2 = ax2.imshow(
    u_mag[0].T,
    origin="lower",
    vmin=umin,
    vmax=umax,
    extent=[0, Lx, -Ly/2, Ly/2],
    aspect='auto'
)
ax2.set_title("Velocity |u|")

# slice
line, = ax3.plot(y, trC[0, x0_index, :], linewidth=2)
ax3.set_ylim(cmin, cmax)
ax3.set_title(f"tr(C) slice at x index = {x0_index}")
ax3.set_xlabel("y")

plt.colorbar(im1, ax=ax1)
plt.colorbar(im2, ax=ax2)

title = fig.suptitle("")

# =============================
# Animation
# =============================
def update(i):
    im1.set_data(trC[i].T)
    im2.set_data(u_mag[i].T)
    line.set_ydata(trC[i, x0_index, :])
    title.set_text(f"t = {times[i]:.3f}")
    return im1, im2, line

ani = FuncAnimation(fig, update, frames=len(times), interval=50)

plt.tight_layout()
plt.show()

# =============================
# Save
# =============================
output_name = "flow_animation_wi26.mp4"
print("Saving:", output_name)

ani.save(output_name, dpi=200)

plt.close(fig)

print("Done")