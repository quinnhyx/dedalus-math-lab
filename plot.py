import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------
# Load files
# -----------------------------
files = sorted(glob.glob("snapshots/snapshots_s*.h5"))

trC = []
ux = []
uy = []
times = []

for fname in files:
    with h5py.File(fname, "r") as f:

        ux.append(f["tasks"]["ux"][:])
        uy.append(f["tasks"]["uy"][:])

        cxx = f["tasks"]["cxx"][:]
        cyy = f["tasks"]["cyy"][:]
        trC.append(cxx + cyy)

        times.append(f["scales"]["sim_time"][:])

ux = np.concatenate(ux)
uy = np.concatenate(uy)
trC = np.concatenate(trC)
times = np.concatenate(times)

u_mag = np.sqrt(ux**2 + uy**2)

print("Total frames:", len(times))
print("Data shape:", trC.shape)

# -----------------------------
# Grid info
# -----------------------------
Nt, Nx, Ny = trC.shape

x0_index = Nx // 2   # 固定 x0 在中间
y = np.arange(Ny)

# -----------------------------
# Color limits
# -----------------------------
cmin, cmax = trC.min(), trC.max()
umin, umax = u_mag.min(), u_mag.max()

# -----------------------------
# Plot setup
# -----------------------------
fig, axes = plt.subplots(1,3, figsize=(18,4))

ax1, ax2, ax3 = axes

# trC field
im1 = ax1.imshow(trC[0].T, origin="lower",
                 cmap="viridis", vmin=cmin, vmax=cmax)
ax1.set_title("Polymer Stretch tr(C)")

# velocity magnitude
im2 = ax2.imshow(u_mag[0].T, origin="lower",
                 cmap="plasma", vmin=umin, vmax=umax)
ax2.set_title("Velocity |u|")

# slice plot
line, = ax3.plot(y, trC[0, x0_index, :], linewidth=2)
ax3.set_ylim(cmin, cmax)
ax3.set_title(f"tr(C) slice at x index = {x0_index}")
ax3.set_xlabel("y")
ax3.set_ylabel("tr(C)")

plt.colorbar(im1, ax=ax1)
plt.colorbar(im2, ax=ax2)

title = fig.suptitle("")

# -----------------------------
# Animation update
# -----------------------------
def update(i):

    im1.set_data(trC[i].T)
    im2.set_data(u_mag[i].T)

    line.set_ydata(trC[i, x0_index, :])

    title.set_text(f"t = {times[i]:.3f}")

    return im1, im2, line

ani = FuncAnimation(fig, update, frames=len(times), interval=60)

plt.tight_layout()
plt.show()

# 保存动画
ani.save("flow_slice_animation.mp4", dpi=200)