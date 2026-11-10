"""Numba-compiled version of tyre_ring.simulate (same physics, same constants, ~100x faster).
Ring of N nodes, stretch + turning-angle bending, overdamped, penalty finger plates that close then retract; optional strain-dependent drag p.
"""

import numpy as np
from numba import njit

N = 16
R = 0.5
KS = 100.0
KC = 100.0
_th = np.linspace(0, 2 * np.pi, N, endpoint=False)
X0 = np.stack([R * np.cos(_th), R * np.sin(_th)], 1)
L0 = np.sqrt(((np.roll(X0, -1, 0) - X0) ** 2).sum(1))


@njit(cache=True, fastmath=False)
def _forces(X, F, kb, g, L0):
    n = X.shape[0]
    th0 = 2 * np.pi / n
    for i in range(n):
        F[i, 0] = 0.0
        F[i, 1] = 0.0
    ex = np.empty(n)
    ey = np.empty(n)
    for i in range(n):
        j = (i + 1) % n
        ex[i] = X[j, 0] - X[i, 0]
        ey[i] = X[j, 1] - X[i, 1]
        l = np.sqrt(ex[i] * ex[i] + ey[i] * ey[i])
        f = KS * (l - L0[i]) / l
        F[i, 0] += f * ex[i]
        F[i, 1] += f * ey[i]
        F[j, 0] -= f * ex[i]
        F[j, 1] -= f * ey[i]
    for i in range(n):
        p = (i - 1) % n
        q = (i + 1) % n  # edge p: x_p->x_i (e1), edge i: x_i->x_q (e2)
        e1x = ex[p]
        e1y = ey[p]
        e2x = ex[i]
        e2y = ey[i]
        th = np.arctan2(e1x * e2y - e1y * e2x, e1x * e2x + e1y * e2y)
        ng = kb * (th - th0)
        l1 = max(e1x * e1x + e1y * e1y, 1e-9)
        l2 = max(e2x * e2x + e2y * e2y, 1e-9)
        g1x = -e1y / l1
        g1y = e1x / l1
        g2x = -e2y / l2
        g2y = e2x / l2  # dth/dx_{i-1}=g1, dth/dx_{i+1}=g2, dth/dx_i=-(g1+g2)
        F[i, 0] += ng * (g1x + g2x)
        F[i, 1] += ng * (g1y + g2y)
        F[p, 0] -= ng * g1x
        F[p, 1] -= ng * g1y
        F[q, 0] -= ng * g2x
        F[q, 1] -= ng * g2y
    for i in range(n):
        top = X[i, 1] - g / 2
        bot = -g / 2 - X[i, 1]
        if top > 0:
            F[i, 1] -= KC * top
        if bot > 0:
            F[i, 1] += KC * bot


@njit(cache=True)
def simulate_nb(g0, kb, vh, T, t_close, p, ta, X0, L0):
    n = X0.shape[0]
    X = X0.copy()
    F = np.zeros((n, 2))
    dt = min(0.4 / (4 * KS + 110 * kb + 100), 1e-3)
    n_close = int(t_close / dt)
    ns = int(T / dt)
    every = max(1, int(1 / 240 / dt))
    for s in range(n_close):
        g = 1.0 + (g0 - 1.0) * min(1.0, (s * dt) / (0.6 * t_close))
        _forces(X, F, kb, g, L0)
        for i in range(n):
            X[i, 0] += dt * F[i, 0]
            X[i, 1] += dt * F[i, 1]
    cnt = ns // every + 1
    ts = np.empty(cnt)
    ys = np.empty(cnt)
    k = 0
    for s in range(ns):
        tt = s * dt
        ymax = -1e9
        ymin = 1e9
        # ta>0: the fingers accelerate to the release speed with time constant ta (finger speed vh*(1-exp(-t/ta))); ta=0: constant speed
        g = g0 + vh * tt if ta <= 0.0 else g0 + vh * (tt - ta * (1.0 - np.exp(-tt / ta)))
        for i in range(n):
            if X[i, 1] > ymax:
                ymax = X[i, 1]
            if X[i, 1] < ymin:
                ymin = X[i, 1]
        f = 1.0
        if p > 0.0:
            f = (max(1.0 - (ymax - ymin), 0.03) / 0.5) ** p
        _forces(X, F, kb, g, L0)
        for i in range(n):
            X[i, 0] += dt * f * F[i, 0]
            X[i, 1] += dt * f * F[i, 1]
        if s % every == 0 and k < cnt:  # record AFTER the update, like tyre_ring.simulate
            y1 = -1e9
            y0 = 1e9
            for i in range(n):
                if X[i, 1] > y1:
                    y1 = X[i, 1]
                if X[i, 1] < y0:
                    y0 = X[i, 1]
            ts[k] = s * dt
            ys[k] = y1 - y0
            k += 1
    return ts[:k], ys[:k], X


def simulate(g0, kb, vh, T=1.2, t_close=0.3, p=0.0, ta=0.0):
    return simulate_nb(float(g0), float(kb), float(vh), float(T), float(t_close), float(p), float(ta), X0, L0)


if __name__ == "__main__":
    import time, sys

    sys.path.insert(0, ".")
    import tyre_ring as T0

    for args in [(0.5, 1.74, 2.7), (0.7, 3.0, 10.0), (0.45, 0.5, 1.0)]:
        a = T0.simulate(*args, T=1.2)
        b = simulate(*args, T=1.2)
        m = min(len(a[0]), len(b[0]))
        print(args, "max |dy| %.2e  len %d/%d" % (np.abs(a[1][:m] - b[1][:m]).max(), len(a[0]), len(b[0])))
    a = T0.simulate(0.5, 1.74, 2.7, p=0.5)
    b = simulate(0.5, 1.74, 2.7, p=0.5)
    m = min(len(a[0]), len(b[0]))
    print("with p=0.5: max |dy| %.2e" % np.abs(a[1][:m] - b[1][:m]).max())
    simulate(0.5, 1.0, 5.0)
    t = time.time()
    for _ in range(20):
        simulate(0.5, 1.74, 2.7, T=1.2)
    print("numba %.1f ms per sim" % ((time.time() - t) / 20 * 1e3))
    t = time.time()
    T0.simulate(0.5, 1.74, 2.7, T=1.2)
    print("numpy %.1f ms per sim" % ((time.time() - t) * 1e3))
