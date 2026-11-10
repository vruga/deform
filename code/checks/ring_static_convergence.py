import numpy as np, sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from numba import njit
from ring_mesh_convergence import forces


@njit(cache=True)
def relax(n, sag, kb, KS, KC, c, dt, steps):
    R = 0.5
    th = 2 * np.pi * np.arange(n) / n
    X = np.empty((n, 2))
    X[:, 0] = R * np.cos(th)
    X[:, 1] = R * np.sin(th)
    L0 = 2 * R * np.sin(np.pi / n)
    F = np.zeros((n, 2))
    g = 1 - sag
    for s in range(steps):
        forces(X, F, kb, g, L0, KS, KC)
        X += dt * F / c
    tot = 0.0
    for i in range(n):
        if X[i, 1] - g / 2 > 0:
            tot += KC * (X[i, 1] - g / 2)
    return tot


for variant in ("k", "f"):
    print("variant", variant)
    for sag in (0.08, 0.10, 0.12):
        row = []
        for N in (16, 24, 32, 48, 96):
            s = N / 16
            kb = N / 16
            c = 16 / N
            KS = 100 * (s if variant == "k" else 1)
            KC = 100 * (s if variant == "k" else 1)
            stiff = KS * 4 + (kb / (2 * np.sin(np.pi / N) * 0.5) ** 2) * 8 + KC * 2
            dt = 0.25 * c / stiff
            steps = int(60 * c / dt / (1) * 0.02) + 200000
            row.append((N, relax(N, sag, kb, KS, KC, c, dt, min(steps, 6000000))))
        print(
            " sag %.2f  F(N)/F(16):" % sag,
            " ".join("N=%d %.3f" % (n, f / row[0][1]) for n, f in row),
            " F16=%.4f" % row[0][1],
        )
