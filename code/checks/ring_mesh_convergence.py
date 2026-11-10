"""Mesh convergence of the ring. Same physics as tyre_ring_fast (stretch + turning-angle bending, overdamped, penalty plates), N-flexible.
Continuum-consistent scaling: bending B = kb*l0 fixed -> kb_N = kb16*16/N ; drag per length fixed -> node drag c = l0/l0_16 ; KS,KC scaled by N/16 (variant 'k') or fixed (variant 'f').
Compare z(t) for the fitted parameters (kb16=1.74, v=2.72, rest=.979) at several start widths."""

import numpy as np
from numba import njit


@njit(cache=True)
def forces(X, F, kb, g, L0, KS, KC):
    n = X.shape[0]
    th0 = 2 * np.pi / n
    F[:] = 0.0
    ex = np.empty(n)
    ey = np.empty(n)
    for i in range(n):
        j = (i + 1) % n
        ex[i] = X[j, 0] - X[i, 0]
        ey[i] = X[j, 1] - X[i, 1]
        l = np.sqrt(ex[i] ** 2 + ey[i] ** 2)
        f = KS * (l - L0) / l
        F[i, 0] += f * ex[i]
        F[i, 1] += f * ey[i]
        F[j, 0] -= f * ex[i]
        F[j, 1] -= f * ey[i]
    for i in range(n):
        p = (i - 1) % n
        q = (i + 1) % n
        e1x = ex[p]
        e1y = ey[p]
        e2x = ex[i]
        e2y = ey[i]
        th = np.arctan2(e1x * e2y - e1y * e2x, e1x * e2x + e1y * e2y)
        ng = kb * (th - th0)
        l1 = max(e1x**2 + e1y**2, 1e-9)
        l2 = max(e2x**2 + e2y**2, 1e-9)
        g1x = -e1y / l1
        g1y = e1x / l1
        g2x = -e2y / l2
        g2y = e2x / l2
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
def sim(n, g0, kb, vh, T, KS, KC, c, dt):
    R = 0.5
    th = 2 * np.pi * np.arange(n) / n
    X = np.empty((n, 2))
    X[:, 0] = R * np.cos(th)
    X[:, 1] = R * np.sin(th)
    L0 = 2 * R * np.sin(np.pi / n)
    F = np.zeros((n, 2))
    tc = 0.3
    nc = int(tc / dt)
    ns = int(T / dt)
    every = max(1, int(1 / 240 / dt))
    out = np.empty(ns // every + 2)
    k = 0
    for s in range(nc):
        g = 1.0 + (g0 - 1.0) * min(1.0, (s * dt) / (0.6 * tc))
        forces(X, F, kb, g, L0, KS, KC)
        X += dt * F / c
    for s in range(ns):
        g = g0 + vh * s * dt
        forces(X, F, kb, g, L0, KS, KC)
        X += dt * F / c
        if s % every == 0:
            out[k] = X[:, 1].max() - X[:, 1].min()
            k += 1
    return out[:k]


def run(N, z0, variant, kb16=1.74, v=2.72, r=0.979, T=1.2):
    s = N / 16
    kb = kb16 * N / 16
    c = (1.0 / s) * 1.0
    KS = 100.0 * (s if variant == "k" else 1)
    KC = 100.0 * (s if variant == "k" else 1)
    stiff = KS * 4 + (kb / (2 * np.sin(np.pi / N) * 0.5) ** 2) * 8 + KC * 2
    dt = 0.25 * c / stiff
    return r * sim(N, min(0.99, z0 / r), kb, v, T, KS, KC, c, dt)


if __name__ == "__main__":
    tm = lambda y, t: 0
    for variant in ("k", "f"):
        print("variant", variant, "(k: KS,KC scaled with N; f: fixed)")
        for z0 in (0.35, 0.5, 0.65):
            ref = None
            row = []
            for N in (16, 24, 32, 48, 96):
                y = run(N, z0, variant)
                if N == 16:
                    ref = y
                m = min(len(y), len(ref))
                d = np.sqrt(np.mean((y[:m] - ref[:m]) ** 2))
                t = np.arange(len(y)) / 240.0
                tau = np.nan
                u = 0.979 - y
                u0 = u[0]
                i = np.where(u < u0 / np.e)[0]
                tau = t[i[0]] if len(i) else np.nan
                row.append(
                    "N=%d y0=%.3f y(.2s)=%.3f y(.6s)=%.3f  1/e=%.3fs  rmse vs N16=%.4f"
                    % (N, y[0], y[min(48, len(y) - 1)], y[min(144, len(y) - 1)], tau, d)
                )
            print(" z0=%.2f" % z0)
            [print("   " + r) for r in row]
