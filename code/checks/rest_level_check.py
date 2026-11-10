"""Phase 1: is the rest-level scatter the missing piece?  First-order model, same events/protocol as experiment_1 (S2, train clips1-2, test 3-4, LOCO).
A global r (baseline) | B per-event r = rho * own PRE-squeeze level (legit, known before release) | C per-event r = own settled level (ORACLE, diagnostic only)
"""

import glob, numpy as np
from scipy.signal import medfilt, find_peaks
from scipy.optimize import least_squares


def events(p, clip):
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
        on = k + np.max(np.where(z[k : k + 240] <= zm + 0.03)[0])
        w = slice(on, min(on + 288, len(z)))
        if w.stop - w.start < 72:
            continue
        tt = t[w] - t[on]
        zz = z[w]
        i5 = min(len(zz) - 1, 120)
        held = (zz[i5] - zz[0]) < 0.5 * (1 - zz[0]) and len(zz) > 120
        i0 = k
        while i0 > 0 and z[i0] < 0.85:
            i0 -= 1
        pre = np.median(z[max(0, i0 - 60) : max(1, i0 - 6)]) if i0 > 66 else np.nan
        E.append(
            dict(
                dur=float(t[on] - t[i0]),
                clip=clip,
                t=tt,
                z=zz,
                z0=zz[0],
                held=bool(held),
                pre=pre,
                settle=float(np.median(zz[-30:])),
            )
        )
    return E


files = sorted(glob.glob("tyreN_sig_*.npz"))
ALL = [[e for e in events(f, i + 1) if (not e["held"]) and e["dur"] > 0.3] for i, f in enumerate(files)]
print("events/clip", [len(A) for A in ALL])
EV = [e for A in ALL for e in A]
print("pre-level per event:", np.round([e["pre"] for e in EV], 3))
print("settled per event  :", np.round([e["settle"] for e in EV], 3))
print(
    "corr(pre,settled)=%.2f"
    % np.corrcoef([e["pre"] for e in EV if np.isfinite(e["pre"])], [e["settle"] for e in EV if np.isfinite(e["pre"])])[
        0, 1
    ]
)


def rest(e, q, mode):
    if mode == "A":
        return q[1]
    pre = e["pre"] if np.isfinite(e["pre"]) else 1.0
    if mode == "B":
        return q[1] * pre
    return e["settle"]


def pred(e, q, mode):
    r = rest(e, q, mode)
    return r - (r - e["z0"]) * np.exp(-e["t"] / q[0])


def fit(Es, mode):
    f = lambda q: np.concatenate([pred(e, q, mode) - e["z"] for e in Es])
    return least_squares(f, [0.14, 0.98], bounds=([0.005, 0.85], [3, 1.1]), loss="soft_l1", f_scale=0.05).x


rm = lambda Es, q, mode: float(np.mean([np.sqrt(np.mean((pred(e, q, mode) - e["z"]) ** 2)) for e in Es]))
tail = lambda Es, q, mode: float(
    np.mean([np.sqrt(np.mean((pred(e, q, mode) - e["z"])[e["t"] > 0.6] ** 2)) for e in Es])
)
tr = ALL[0] + ALL[1]
te = ALL[2] + ALL[3]
for mode, name in (
    ("A", "global rest (baseline)"),
    ("B", "per-event PRE-squeeze rest (legit)"),
    ("C", "per-event settled rest (ORACLE)"),
):
    q = fit(tr, mode)
    lo = [rm(ALL[h], fit([e for k in range(4) if k != h for e in ALL[k]], mode), mode) for h in range(4)]
    print(
        "%-38s tau %.3f rho %.3f | train %.4f | held-out %.4f | tail(t>.6) %.4f | LOCO %.4f"
        % (name, q[0], q[1], rm(tr, q, mode), rm(te, q, mode), tail(te, q, mode), np.mean(lo))
    )
