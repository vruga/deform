"""Ring model v2: (A) shared release speed, (B) release speed depends on squeeze depth: vh = exp(a + b*depth). Physical-unit mapping via static weight calibration."""

import glob, json, sys, numpy as np
from scipy.optimize import least_squares
from tyre_fit2 import events, fit, pred
import tyre_ring as T

files = sorted(glob.glob("tyreN_sig_*.npz"))
ALL = [events(f, i + 1) for i, f in enumerate(files)]
ok = lambda e: (not e["held"]) and e["dur"] > 0.3


def vh_of(e, q, mode):
    return q[1] if mode == "A" else np.exp(q[1] + q[2] * (1 - e["z0"]))


def ring_pred(e, q, mode):
    kb = np.exp(q[0])
    r = q[-1]
    v = vh_of(e, q, mode)
    t, y, _ = T.simulate(min(0.99, e["z0"] / r), kb, v, T=float(e["t"][-1]) + 0.02)
    return r * np.interp(e["t"], t, y)


def ring_fit(Es, mode):
    f = lambda q: np.concatenate([ring_pred(e, q, mode) - e["z"] for e in Es])
    best = None
    if mode == "A":
        starts = [[np.log(1.0), 10, 0.98], [np.log(4.0), 40, 0.98]]
        lb = [np.log(0.3), 1, 0.85]
        ub = [np.log(30), 300, 1.1]
    else:
        starts = [[np.log(1.5), 1.0, 0.0, 0.98], [np.log(1.5), 0.0, 6.0, 0.98]]
        lb = [np.log(0.3), -2, -10, 0.85]
        ub = [np.log(30), 6, 30, 1.1]
    for q0 in starts:
        r = least_squares(f, q0, bounds=(lb, ub), loss="soft_l1", f_scale=0.05, diff_step=0.05, max_nfev=35)
        if best is None or r.cost < best.cost:
            best = r
    return best.x


R = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
tr = [e for e in ALL[0] + ALL[1] if ok(e)]
te = [e for e in ALL[2] + ALL[3] if ok(e)]
m1 = "1st-order (1 tau)"
q1 = fit(tr, m1, False)
res = {"n_train": len(tr), "n_test": len(te)}
res["null"] = dict(test=float(np.mean([R(e["z0"], e["z"]) for e in te])))
res["first"] = dict(
    train=float(np.mean([R(pred(e, m1, q1[:1], q1[1], False), e["z"]) for e in tr])),
    test=float(np.mean([R(pred(e, m1, q1[:1], q1[1], False), e["z"]) for e in te])),
    params=[float(x) for x in q1],
)
Q = {}
for mode in "AB":
    q = ring_fit(tr, mode)
    Q[mode] = q
    tests = [R(ring_pred(e, q, mode), e["z"]) for e in te]
    trs = [R(ring_pred(e, q, mode), e["z"]) for e in tr]
    lo = []
    for h in range(4):
        trn = [e for k in range(4) if k != h for e in ALL[k] if ok(e)]
        qq = ring_fit(trn, mode)
        lo += [R(ring_pred(e, qq, mode), e["z"]) for e in ALL[h] if ok(e)]
    lo1 = []
    for h in range(4):
        trn = [e for k in range(4) if k != h for e in ALL[k] if ok(e)]
        qa = fit(trn, m1, False)
        lo1 += [R(pred(e, m1, qa[:1], qa[1], False), e["z"]) for e in ALL[h] if ok(e)]
    res["ring" + mode] = dict(
        train=float(np.mean(trs)),
        test=float(np.mean(tests)),
        test_med=float(np.median(tests)),
        loco=float(np.mean(lo)),
        params=[float(x) for x in q],
        per_event_test=[float(x) for x in tests],
    )
    res["first"]["loco"] = float(np.mean(lo1))
    print(mode, res["ring" + mode], flush=True)
json.dump(res, open("tyre_ring2.json", "w"), indent=1)


# ---- physical calibration (static weight test)
def static_force(sag, KB=1.0):
    X = T.X0.copy()
    g = 1 - sag
    dt = 0.4 / (4 * T.KS + 110 * KB + 100)
    dt = min(dt, 1e-3)
    for s in range(int(2.0 / dt)):
        X = X + dt * T.forces(X, KB, g)
    top = X[:, 1] - g / 2
    return float(100.0 * np.maximum(top, 0).sum())


L = 0.0758
ds = 2 * np.pi * 0.5 / T.N
F = 0.333 * 9.81
cal = []
for sag in (0.08, 0.10, 0.12):
    F1 = static_force(sag)
    EI = F * ds * L**2 / F1
    cal.append((sag, F1, EI, F / (sag * L)))
print("static calib (sag, F_model, EI N m^2, k N/m):", cal)
kbm = float(np.exp(Q["B"][0] if "B" in Q else 0))
kbA = float(np.exp(Q["A"][0]))
phys = {"L_m": L, "F_N": F, "cal": [dict(sag=a, EI=c, k_Nmm=d / 1000) for a, b, c, d in cal]}
EI = cal[1][2]
phys["EI"] = EI
for m, qk in (("A", Q["A"]), ("B", Q["B"])):
    kbm = float(np.exp(qk[0]))
    eta = EI / (kbm * ds**2 * L**4)
    phys["eta_" + m] = eta
    phys["c_node_" + m] = eta * ds * L
json.dump(dict(res=res, phys=phys), open("tyre_ring2.json", "w"), indent=1)
print(json.dumps(phys, indent=1))
