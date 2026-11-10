"""Strain-dependent viscosity test. rate ~ (deficit/0.5)^p; p=0 is the previous model. Same events/split/protocol as tyre_ring2.py.
C = ring with p (kb, release speed, rest, p) ; D = nonlinear first-order (tau, rest, p) ; baselines from tyre_ring_fit.json.
"""

import glob, json, sys, numpy as np
from scipy.optimize import least_squares
from tyre_fit2 import events
import tyre_ring as T

files = sorted(glob.glob("tyreN_sig_*.npz"))
ALL = [events(f, i + 1) for i, f in enumerate(files)]
ok = lambda e: (not e["held"]) and e["dur"] > 0.3
R = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))


def pC(e, q):
    kb, v, r, p = np.exp(q[0]), q[1], q[2], q[3]
    t, y, _ = T.simulate(min(0.99, e["z0"] / r), kb, v, T=float(e["t"][-1]) + 0.02, p=p)
    return r * np.interp(e["t"], t, y)


def pD(e, q):
    tau, r, p = q
    x0 = max(r - e["z0"], 1e-3)
    if p < 1e-4:
        x = x0 * np.exp(-e["t"] / tau)
    else:
        x = (x0 ** (-p) + p * e["t"] / (tau * 0.5**p)) ** (-1 / p)
    return r - x


def fit(Es, mode):
    f = lambda q: np.concatenate([(pC if mode == "C" else pD)(e, q) - e["z"] for e in Es])
    if mode == "C":
        S = [[np.log(1.74), 2.7, 0.98, 0.5], [np.log(3.0), 2.7, 0.98, 1.5]]
        lb = [np.log(0.3), 1, 0.85, 0]
        ub = [np.log(60), 300, 1.1, 4]
    else:
        S = [[0.14, 0.98, 0.3], [0.1, 0.98, 1.5]]
        lb = [0.005, 0.85, 0]
        ub = [3, 1.1, 4]
    best = None
    for q0 in S:
        r = least_squares(f, q0, bounds=(lb, ub), loss="soft_l1", f_scale=0.05, diff_step=0.05, max_nfev=35)
        if best is None or r.cost < best.cost:
            best = r
    return best.x


tr = [e for e in ALL[0] + ALL[1] if ok(e)]
te = [e for e in ALL[2] + ALL[3] if ok(e)]
res = {}
for mode in ("D", "C"):
    P = pC if mode == "C" else pD
    q = fit(tr, mode)
    tests = [R(P(e, q), e["z"]) for e in te]
    trs = [R(P(e, q), e["z"]) for e in tr]
    lo = []
    for h in range(4):
        trn = [e for k in range(4) if k != h for e in ALL[k] if ok(e)]
        qq = fit(trn, mode)
        lo += [R(P(e, qq), e["z"]) for e in ALL[h] if ok(e)]
    res[mode] = dict(
        train=float(np.mean(trs)),
        test=float(np.mean(tests)),
        test_med=float(np.median(tests)),
        loco=float(np.mean(lo)),
        params=[float(x) for x in q],
        per_event_test=[float(x) for x in tests],
    )
    print(mode, res[mode], flush=True)
    json.dump(res, open("tyre_ring3.json", "w"), indent=1)
