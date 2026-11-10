"""Refit the ring at N nodes (16 and 96) with continuum-consistent scaling, same events and protocol as tyre_models_fast.py.

Scaling that keeps the physics fixed as N grows (B = bending stiffness, l0 = node spacing):
    bending  k_b(N) = k_b16 * N / 16      (so B = k_b * l0 is unchanged)
    stretch and contact springs  K(N) = K16 * N / 16
    node drag  c(N) = 16 / N              (drag per unit length unchanged)
The fitted bending parameter is reported as k_b16, i.e. rescaled back to the 16-node ring, so numbers are comparable across N.
Release is instant (no release-speed parameter), so only k_b16 and the rest level are fitted.

Run from ../../data:  PYTHONPATH=../code:../code/checks python ../code/checks/fit_n_nodes.py [16 96]
"""

import glob
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from numba import njit
from scipy.optimize import least_squares

from ring_mesh_convergence import forces
from tyre_fit2 import events

T_CLOSE = 0.3
RELEASE_SPEED = 300.0  # diameters/s: effectively instant


@njit(cache=True)
def simulate(n, g0, kb, vh, duration, ks, kc, c, dt):
    """Ring of n nodes: plates close over T_CLOSE, then open at speed vh. Returns (times, width) sampled every ~1/240 s."""
    r = 0.5
    theta = 2 * np.pi * np.arange(n) / n
    x = np.empty((n, 2))
    x[:, 0] = r * np.cos(theta)
    x[:, 1] = r * np.sin(theta)
    l0 = 2 * r * np.sin(np.pi / n)
    f = np.zeros((n, 2))
    n_close, n_steps = int(T_CLOSE / dt), int(duration / dt)
    every = max(1, int(round(1.0 / 240 / dt)))
    times = np.empty(n_steps // every + 2)
    width = np.empty(n_steps // every + 2)
    for s in range(n_close):
        gap = 1.0 + (g0 - 1.0) * min(1.0, (s * dt) / (0.6 * T_CLOSE))
        forces(x, f, kb, gap, l0, ks, kc)
        x += dt * f / c
    k = 0
    for s in range(n_steps):
        forces(x, f, kb, g0 + vh * s * dt, l0, ks, kc)
        x += dt * f / c
        if s % every == 0:
            times[k] = s * dt
            width[k] = x[:, 1].max() - x[:, 1].min()
            k += 1
    return times[:k], width[:k]


def ring_width(n, z0, kb16, rest, duration):
    s = n / 16
    kb, ks, c = kb16 * s, 100.0 * s, 1.0 / s
    l0 = 2 * 0.5 * np.sin(np.pi / n)
    stiffness = 4 * ks + 8 * kb / l0**2 + 2 * ks
    dt = 0.25 * c / stiffness
    return simulate(n, min(0.99, z0 / rest), kb, RELEASE_SPEED, duration, ks, ks, c, dt)


def event_prediction(args):
    n, event, kb16, rest = args
    t, w = ring_width(n, event["z0"], kb16, rest, float(event["t"][-1]) + 0.02)
    return rest * np.interp(event["t"], t, w)


def fit(pool, n, train, start):
    def residuals(q):
        preds = list(pool.map(event_prediction, [(n, e, q[0], q[1]) for e in train]))
        return np.concatenate([p - e["z"] for p, e in zip(preds, train)])

    result = least_squares(
        residuals, start, bounds=([0.3, 0.85], [20, 1.1]), loss="soft_l1", f_scale=0.05, diff_step=0.03, max_nfev=14
    )
    return result.x


def score(pool, n, q, test):
    preds = list(pool.map(event_prediction, [(n, e, q[0], q[1]) for e in test]))
    return [float(np.sqrt(np.mean((p - e["z"]) ** 2))) for p, e in zip(preds, test)]


if __name__ == "__main__":
    files = sorted(glob.glob("tyreN_sig_*.npz"))
    by_clip = [[e for e in events(f, i + 1) if (not e["held"]) and e["dur"] > 0.3] for i, f in enumerate(files)]
    train, test = by_clip[0] + by_clip[1], by_clip[2] + by_clip[3]
    out = {}
    with ProcessPoolExecutor() as pool:
        for n in [int(a) for a in sys.argv[1:]] or [16, 96]:
            q = fit(pool, n, train, [1.5, 0.98])
            held_out = score(pool, n, q, test)
            loco = []
            for k in range(4):
                tr = [e for j in range(4) if j != k for e in by_clip[j]]
                loco += score(pool, n, fit(pool, n, tr, q), by_clip[k])
            out[str(n)] = dict(
                kb16=float(q[0]), rest=float(q[1]), held_out=float(np.mean(held_out)), loco=float(np.mean(loco))
            )
            print(
                "N=%d  kb16 %.3f  rest %.3f  held-out %.4f  LOCO %.4f"
                % (n, q[0], q[1], np.mean(held_out), np.mean(loco)),
                flush=True,
            )
            json.dump(out, open("../results/check_fit_n_nodes.json", "w"), indent=1)
