"""Fit effective stiffness/damping of the tyre from release recoveries. Model: x'' + 2*zeta*wn*x' + wn^2*x = 0, x=1-z (deficit), x(0)=x0 measured, x'(0)=0.
Train on tyre_1,tyre_2 (10 events); predict tyre_3,tyre_4 (10 events) with NO refit. Baselines: no-recovery, 1st-order exp (fit on train).
"""

import glob, sys, numpy as np
from scipy.signal import medfilt, find_peaks
from scipy.optimize import least_squares
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def events(p):
    d = np.load(p)
    t = d["t"]
    y = d["minor"]
    ok = np.isfinite(y)
    y = np.interp(t, t[ok], y[ok])
    y = medfilt(y, 13)
    z = y / np.percentile(y, 85)
    pk, _ = find_peaks(-z, prominence=0.25, distance=240, height=-0.8)
    E = []
    for k in pk:
        zm = z[k]
        seg = z[k : k + 240]
        on = k + np.max(np.where(seg <= zm + 0.03)[0])  # last frame still at the held minimum
        w = slice(on, min(on + 240, len(z)))  # 1 s recovery window
        if w.stop - w.start < 120:
            continue
        E.append(dict(t=t[w] - t[on], z=z[w], z0=z[on]))
    return E


def x2(t, wn, zeta, x0):
    s = np.sqrt(complex(zeta**2 - 1 + 0j))
    r1 = wn * (-zeta + s)
    r2 = wn * (-zeta - s)
    if abs(r2 - r1) < 1e-9:
        return x0 * (1 + wn * t) * np.exp(-wn * t)
    return x0 * np.real((r2 * np.exp(r1 * t) - r1 * np.exp(r2 * t)) / (r2 - r1))


def sim2(E, p):
    return 1 - x2(E["t"], p[0], p[1], 1 - E["z0"])


def sim1(E, p):
    return 1 - (1 - E["z0"]) * np.exp(-E["t"] / p[0])


def fit(Es, sim, p0, lb, ub):
    f = lambda p: np.concatenate([sim(E, p) - E["z"] for E in Es])
    r = least_squares(f, p0, bounds=(lb, ub), loss="soft_l1", f_scale=0.05)
    return r.x


rm = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
files = sorted(glob.glob("tyreN_sig_*.npz"))
tr = [e for f in files[:2] for e in events(f)]
te = [e for f in files[2:] for e in events(f)]
print("train events", len(tr), "test events", len(te))
p2 = fit(tr, sim2, [15, 1.5], [1, 0.2], [300, 10])
p1 = fit(tr, sim1, [0.1], [0.005], [2])
print(
    "2nd-order: wn=%.1f rad/s zeta=%.2f  (k/m=%.0f /s^2, c/m=%.1f /s)" % (p2[0], p2[1], p2[0] ** 2, 2 * p2[1] * p2[0]),
    "| 1st-order tau=%.3f s" % p1[0],
)
R = {"2nd": [], "1st": [], "none": []}
fig, ax = plt.subplots(2, 5, figsize=(18, 6))
for a, E in zip(ax.flat, te):
    R["2nd"].append(rm(sim2(E, p2), E["z"]))
    R["1st"].append(rm(sim1(E, p1), E["z"]))
    R["none"].append(rm(E["z0"], E["z"]))
    a.plot(E["t"], E["z"], "k.", ms=2, label="real")
    a.plot(E["t"], sim2(E, p2), "r", label="fitted model")
    a.grid()
    a.set_title("rmse %.3f" % R["2nd"][-1])
ax[0, 0].legend()
plt.tight_layout()
plt.savefig(sys.argv[1], dpi=60)
for k, v in R.items():
    print("held-out RMSE %-5s mean %.3f median %.3f" % (k, np.mean(v), np.median(v)))
print("train RMSE 2nd", np.mean([rm(sim2(E, p2), E["z"]) for E in tr]).round(3))
