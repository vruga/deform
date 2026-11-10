"""Residual-set / memory test for the pouch from hand-free frames only.
Episodes: hand-free intervals (foreign-pixel fraction below a per-clip threshold, lasting >= MIN_FREE s) separated by hand-present intervals (>= 0.5 s).
For each pair of consecutive hand-free intervals: set = median aspect after / median aspect before - 1 (aspect = body width / length, scale-robust).
Also: drift inside each hand-free interval (slope of aspect, %/s) = creep or slow recovery. Noise = spread of aspect inside an interval.
usage: python pouch_memory.py SIGNAL_DIR OUT_JSON"""

import numpy as np, sys, json
from scipy.signal import medfilt

D, OUT = sys.argv[1], sys.argv[2]
MIN_FREE = 1.0
res = {}
for n, thr_mode in [("red_9810", None), ("grey", None), ("red_slowmo_aruco", None), ("red_slow_noaruco", None)]:
    r = np.load(f"{D}/pouch_{n}.npy")
    t = r[:, 0]
    fps = 1 / np.median(np.diff(t))
    asp = r[:, 3]
    fo = r[:, 4]
    k = int(fps * 0.1) | 1
    ok = np.isfinite(asp)
    asp = np.interp(t, t[ok], asp[ok])
    asp = medfilt(asp, k)
    fo = medfilt(np.nan_to_num(fo), k)
    thr = 1.6 * np.percentile(fo, 10) + 0.015
    free = fo < thr
    # intervals
    ivs = []
    i = 0
    while i < len(t):
        if free[i]:
            j = i
            while j < len(t) and free[j]:
                j += 1
            if t[j - 1] - t[i] >= MIN_FREE:
                ivs.append((i, j))
            i = j
        else:
            i += 1
    info = []
    for a, b in ivs:
        seg = asp[a:b]
        tt = t[a:b]
        info.append(
            dict(
                t0=float(tt[0]),
                t1=float(tt[-1]),
                aspect=float(np.median(seg)),
                spread=float(np.std(seg)),
                drift_pct_per_s=float(100 * np.polyfit(tt - tt[0], seg, 1)[0] / np.median(seg)),
            )
        )
    sets = [
        dict(
            between=[round(info[q]["t1"], 1), round(info[q + 1]["t0"], 1)],
            set_pct=float(100 * (info[q + 1]["aspect"] / info[q]["aspect"] - 1)),
        )
        for q in range(len(info) - 1)
    ]
    res[n] = dict(threshold=float(thr), n_free_intervals=len(info), intervals=info, sets=sets)
    print(n, "free thr %.3f, %d hand-free intervals >= %.1fs" % (thr, len(info), MIN_FREE))
    for x in info:
        print(
            "   %.1f-%.1f s  aspect %.3f  spread %.3f  drift %+.2f %%/s"
            % (x["t0"], x["t1"], x["aspect"], x["spread"], x["drift_pct_per_s"])
        )
    for x in sets:
        print("   set between %s: %+.1f %%" % (x["between"], x["set_pct"]))
json.dump(res, open(OUT, "w"), indent=1)
