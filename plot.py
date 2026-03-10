import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------
# Load files
# -----------------------------
files = sorted(glob.glob("snapshots/snapshots_s*.h5"))

# p = []
trC = []
ux = []
uy = []
times = []

for fname in files:
    with h5py.File(fname, "r") as f:

        # p.append(f["tasks"]["p"][:])
        ux.append(f["tasks"]["ux"][:])
        uy.append(f["tasks"]["uy"][:])

        cxx = f["tasks"]["cxx"][:]
        cyy = f["tasks"]["cyy"][:]
        trC.append(cxx + cyy)

        times.append(f["scales"]["sim_time"][:])

# p = np.concatenate(p)
ux = np.concatenate(ux)
uy = np.concatenate(uy)
trC = np.concatenate(trC)
times = np.concatenate(times)

u_mag = np.sqrt(ux**2 + uy**2)

print("Total frames:", len(times))

# -----------------------------
# Color limits (avoid flicker)
# -----------------------------
# pmin, pmax = p.min(), p.max()
cmin, cmax = trC.min(), trC.max()
umin, umax = u_mag.min(), u_mag.max()

# -----------------------------
# Plot setup
# -----------------------------
fig, axes = plt.subplots(1,2, figsize=(15,4))

ax2, ax3 = axes

# im1 = ax1.imshow(p[0].T, origin="lower", cmap="coolwarm", vmin=pmin, vmax=pmax)
# ax1.set_title("Pressure p")

im2 = ax2.imshow(trC[0].T, origin="lower", cmap="viridis", vmin=cmin, vmax=cmax)
ax2.set_title("Polymer Stretch tr(C)")

im3 = ax3.imshow(u_mag[0].T, origin="lower", cmap="plasma", vmin=umin, vmax=umax)
ax3.set_title("Velocity |u|")

# plt.colorbar(im1, ax=ax1)
plt.colorbar(im2, ax=ax2)
plt.colorbar(im3, ax=ax3)

title = fig.suptitle("")

# -----------------------------
# Animation update
# -----------------------------
def update(i):

    # im1.set_data(p[i].T)
    im2.set_data(trC[i].T)
    im3.set_data(u_mag[i].T)

    title.set_text(f"t = {times[i]:.3f}")

    return im2, im3

ani = FuncAnimation(fig, update, frames=len(times), interval=60)

plt.tight_layout()
plt.show()

# save if needed
# ani.save("three_panel_flow.mp4", dpi=200)