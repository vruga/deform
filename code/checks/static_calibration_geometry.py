"""Static weight calibration: how the bending stiffness B depends on (a) the number of ring nodes and (b) which radius the ring represents.

The ring is solved in units of its own diameter L; a flat plate squeezes it by a fraction `sag` and the plate force is F_u (ring units, k_b16 = 1).
Dimensional analysis: B = F * L^2 * B_u / F_u(sag), with B_u = k_b * l0 the ring's bending stiffness in ring units and F = 3.27 N, sag = 7.6 mm / L.
Case A (experiment 1): the ring is the OUTER surface, L = 75.8 mm.   Case B: the ring is the wall's centreline, L = 75.8 - 11.8 = 64.0 mm.
Prints B for N = 16 and 96 in both cases. Run: python static_calibration_geometry.py
"""

import numpy as np
from numba import njit
from ring_mesh_convergence import forces

F_LOAD, SAG_MM = 3.27, 7.6


@njit(cache=True)
def plate_force(n, sag, kb, ks, kc, c, dt, steps):
    r = 0.5
    theta = 2 * np.pi * np.arange(n) / n
    x = np.empty((n, 2))
    x[:, 0] = r * np.cos(theta)
    x[:, 1] = r * np.sin(theta)
    l0 = 2 * r * np.sin(np.pi / n)
    f = np.zeros((n, 2))
    gap = 1 - sag
    for _ in range(steps):
        forces(x, f, kb, gap, l0, ks, kc)
        x += dt * f / c
    total = 0.0
    for i in range(n):
        if x[i, 1] - gap / 2 > 0:
            total += kc * (x[i, 1] - gap / 2)
    return total


def bending_stiffness(n, diameter_mm):
    s = n / 16
    kb, ks, c = 1.0 * s, 100.0 * s, 16.0 / n
    l0 = 2 * 0.5 * np.sin(np.pi / n)
    dt = 0.25 * c / (4 * ks + 8 * kb / l0**2 + 2 * ks)
    sag = SAG_MM / diameter_mm
    steps = int(60 * c / dt * 0.02) + 200000
    f_u = plate_force(n, sag, kb, ks, ks, c, dt, steps)
    b_u = kb * l0
    return F_LOAD * (diameter_mm * 1e-3) ** 2 * b_u / f_u


if __name__ == "__main__":
    print("B (N m^2) from the 3.27 N / 7.6 mm weight test")
    for label, d in (("A: outer surface, L = 75.8 mm", 75.8), ("B: wall centreline, L = 64.0 mm", 64.0)):
        print(" ", label, "  ".join("N=%d: %.2e" % (n, bending_stiffness(n, d)) for n in (16, 96)))
