"""Diagnostic only (NOT used for the reported results): fit a separate recovery time per event to see how much it varies."""

import glob, numpy as np
from scipy.optimize import least_squares
from tyre_fit2 import events

files = sorted(glob.glob("tyreN_sig_*.npz"))
ALL = [events(f, i + 1) for i, f in enumerate(files)]
print("clip  t_abs  depth(1-z0)  tau_event  rmse   held")
for A in ALL:
    for e in A:
        f = lambda q: (0.978 - (0.978 - e["z0"]) * np.exp(-e["t"] / q[0])) - e["z"]
        r = least_squares(f, [0.12], bounds=([0.005], [3]), loss="soft_l1", f_scale=0.05)
        print(
            e["clip"],
            "%5.1f" % e["t_abs"],
            "%.2f" % (1 - e["z0"]),
            "%.3f" % r.x[0],
            "%.3f" % np.sqrt(np.mean(f(r.x) ** 2)),
            e["held"],
        )
