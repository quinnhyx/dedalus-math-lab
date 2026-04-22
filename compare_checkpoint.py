import h5py
import numpy as np

def load_snapshot(file, field_name, idx=-1):
    with h5py.File(file, 'r') as f:
        data = f['tasks'][field_name][idx]
        time = f['scales/sim_time'][idx]
    return data, time


def compare_fields(file_full, file_restart, field_name, idx_full=-1, idx_restart=-1):
    f1, t1 = load_snapshot(file_full, field_name, idx_full)
    f2, t2 = load_snapshot(file_restart, field_name, idx_restart)

    print(f"\n=== Comparing {field_name} ===")
    print(f"time_full    = {t1}")
    print(f"time_restart = {t2}")

    if abs(t1 - t2) > 1e-10:
        print("Time mismatch! You are NOT comparing same timestep")
        return

    diff = f1 - f2
    linf = np.max(np.abs(diff))
    l2 = np.sqrt(np.mean(diff**2))

    print(f"Linf error = {linf:.6e}")
    print(f"L2 error   = {l2:.6e}")

    if linf < 1e-10:
        print("PASS (restart is consistent)")
    else:
        print("FAIL (restart not consistent)")


file_full = "snapshots-full/snapshots-full_s1.h5"
file_restart = "snapshots-restart/snapshots-restart_s1.h5"

fields = ["ux", "uy", "p", "cxx", "cxy", "cyy", "trC"]

for field in fields:
    compare_fields(file_full, file_restart, field)
