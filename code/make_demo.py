"""Build figures/demo.mp4 and figures/demo.gif: real held-out squeeze next to the fitted 3D tyre model.

The model is the ring fitted ONLY on clips 1-2 (results/tyre_ring_fit.json). For each test event it receives one number, the squeezed width z0.
Panels:  real video (top view) | 3D tyre rebuilt from the simulated ring | width over time, real vs predicted.

How the 3D picture is made (drawing only, the physics is the ring): the 16 ring nodes are the outer surface of the tyre. A smooth periodic spline gives the
outline, the tread is extruded 37 mm, two rows of lugs and a groove are painted on the outer wall (fixed to the material, so they move with it), and the
inner wall has the lip step seen in the videos (11.8 mm wall on the lip side, a thinner assumed wall on the other). The finger plates are drawn where the model puts them.

Run from ../data:  PYTHONPATH=../code python ../code/make_demo.py      (needs ffmpeg; videos from $TYRE_VIDEOS, default ~/Documents)
"""

import glob
import json
import os
import shutil
import subprocess

import cv2
import matplotlib
import numpy as np
from scipy.interpolate import CubicSpline

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import tyre_ring_fast as RF
from tyre_fit2 import events

# ---- fitted model and geometry (units: tyre diameters) -------------------------------------------------------------
KB, SPEED, REST = json.load(open("../results/tyre_ring_fit.json"))["params"]["ring"]
TREAD = 37.0 / 75.8  # tread width
WALL_LIP, WALL_MAIN = 11.8 / 75.8, 9.5 / 75.8  # radial wall; the main-zone value is an assumption (see README)
LIP_FRACTION = 0.4  # share of the tread width that has the thick lip (estimated from the images)
SIM_FPS, SLOW = 60, 3  # frames per second of real time that we render; playback slowdown
M_THETA, N_Z = 120, 9  # drawing resolution


def simulate_frames(z0, duration=1.2, t_close=0.3):
    """Run the ring (same physics as tyre_ring_fast) and return [(t, nodes, plate_gap)] every 1/SIM_FPS s after the plates start to open."""
    g0 = min(0.99, z0 / REST)
    x, f = RF.X0.copy(), np.zeros((RF.N, 2))
    dt = min(0.4 / (4 * RF.KS + 110 * KB + 100), 1e-3)
    for s in range(int(t_close / dt)):  # plates close
        gap = 1.0 + (g0 - 1.0) * min(1.0, (s * dt) / (0.6 * t_close))
        RF._forces(x, f, KB, gap, RF.L0)
        x += dt * f
    frames, next_t, s = [], 0.0, 0
    while s * dt <= duration:  # plates open at a constant speed
        t = s * dt
        if t >= next_t - 1e-12:
            frames.append((t, x.copy(), g0 + SPEED * t))
            next_t += 1.0 / SIM_FPS
        RF._forces(x, f, KB, g0 + SPEED * t, RF.L0)
        x += dt * f
        s += 1
    return frames


def outline(nodes):
    """Smooth closed outline through the ring nodes: points, outward normals and the material coordinate u in [0, 1)."""
    n = len(nodes)
    p = np.vstack([nodes, nodes[:1]])
    cs = CubicSpline(np.arange(n + 1), p, bc_type="periodic")
    u = np.linspace(0, n, M_THETA, endpoint=False)
    pts, d = cs(u), cs(u, 1)
    normals = np.stack([d[:, 1], -d[:, 0]], 1)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    if np.mean(np.sum(normals * (pts - pts.mean(0)), 1)) < 0:  # make them point outward
        normals = -normals
    return pts, normals, u / n


def tyre_polygons(nodes):
    """Quads and colours for the tyre surface (outer wall with lugs, inner wall with the lip step, two end faces)."""
    pts, nrm, u = outline(nodes)
    z = np.linspace(-TREAD / 2, TREAD / 2, N_Z)
    lip_z = -TREAD / 2 + LIP_FRACTION * TREAD
    uu, zz = np.meshgrid(u, z, indexing="ij")
    row = (zz > 0).astype(float)
    lug = np.clip(np.cos(2 * np.pi * 28 * (uu + 0.5 * row / 28)) * 2, 0, 1)  # two rows, offset by half a pitch
    height = 0.014 * lug * (np.abs(zz) > 0.035) - 0.018 * np.exp(-((zz / 0.03) ** 2))  # lugs outside the central groove
    wall = WALL_MAIN + (WALL_LIP - WALL_MAIN) / (1 + np.exp((zz - lip_z) / 0.01))  # thick lip below lip_z
    base = pts[:, None, :]
    outer = np.concatenate([base + height[..., None] * nrm[:, None, :], zz[..., None]], -1)
    inner = np.concatenate([base - wall[..., None] * nrm[:, None, :], zz[..., None]], -1)
    surfaces = [(outer, 1.0), (inner, -1.0)]
    for side, zc in ((0, -TREAD / 2), (1, TREAD / 2)):  # annular end faces
        rho = np.linspace(0, 1, 5)[None, :, None]
        w = WALL_LIP if side == 0 else WALL_MAIN
        face = base - rho * w * nrm[:, None, :]
        surfaces.append((np.concatenate([face, np.full(face.shape[:2] + (1,), zc)], -1), 1.0))
    polys, cols = [], []
    light = np.array([-0.4, -0.6, 0.7])
    light /= np.linalg.norm(light)
    for grid, _ in surfaces:
        a, b, c, d = grid[:-1, :-1], grid[1:, :-1], grid[1:, 1:], grid[:-1, 1:]
        n3 = np.cross(c - a, d - b)
        n3 /= np.linalg.norm(n3, axis=-1, keepdims=True) + 1e-12
        shade = 0.25 + 0.75 * np.abs(n3 @ light)  # two-sided diffuse
        base_col = np.array([0.17, 0.18, 0.20])
        rgb = np.clip(base_col[None, None] * (0.6 + 1.5 * shade[..., None]), 0, 1)
        polys.append(np.stack([a, b, c, d], -2).reshape(-1, 4, 3))
        cols.append(rgb.reshape(-1, 3))
    return np.concatenate(polys), np.concatenate(cols)


def plate_polygons(gap):
    """Two thin steel plates at y = +-gap/2 (the squeezing fingers)."""
    quads = []
    for s in (-1, 1):
        y0, y1 = s * gap / 2, s * (gap / 2 + 0.03)
        x0, x1, z0, z1 = -0.22, 0.22, -TREAD / 2 - 0.03, TREAD / 2 + 0.03
        for q in (
            [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
            [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
            [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
        ):
            quads.append(q)
    return np.array(quads, float)


def render_event(event, cap, fig, axes, label, bg):
    ax_real, ax_3d, ax_trace = axes
    frames = simulate_frames(event["z0"])
    ts = np.array([t for t, _, _ in frames])
    width = np.array([REST * (n[:, 1].max() - n[:, 1].min()) for _, n, _ in frames])
    rmse = np.sqrt(np.mean((np.interp(event["t"], ts, width) - event["z"]) ** 2))
    cap.set(1, int(event["t_abs"] * 240))
    _, f0 = cap.read()
    gray = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
    mask = (gray < 0.3 * np.percentile(gray, 80)).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    m = cv2.moments(max(contours, key=cv2.contourArea))
    cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
    out, pre = [], int(0.25 * SIM_FPS)
    for k in range(-pre, len(frames)):
        t = k / SIM_FPS
        cap.set(1, int(round((event["t_abs"] + t) * 240)))
        ok, frame = cap.read()
        if not ok:
            continue
        _, nodes, gap = frames[max(k, 0)]
        ax_real.clear()
        ax_real.axis("off")
        ax_real.imshow(cv2.cvtColor(frame[max(cy - 190, 0) : cy + 190, max(cx - 190, 0) : cx + 190], cv2.COLOR_BGR2RGB))
        ax_real.set_title("real  (%s, held out)" % label, color="w", fontsize=10)
        ax_3d.clear()
        ax_3d.set_axis_off()
        ax_3d.set_facecolor(bg)
        polys, cols = tyre_polygons(nodes)
        ax_3d.add_collection3d(Poly3DCollection(polys, facecolors=cols, linewidths=0, antialiaseds=False))
        ax_3d.add_collection3d(
            Poly3DCollection(
                plate_polygons(gap if k >= 0 else frames[0][2]),
                facecolors=(0.7, 0.78, 0.9, 0.28),
                linewidths=0.4,
                edgecolors=(0.9, 0.95, 1, 0.6),
            )
        )
        ax_3d.set_xlim(-0.55, 0.55)
        ax_3d.set_ylim(-0.55, 0.55)
        ax_3d.set_zlim(-0.27, 0.27)
        ax_3d.set_box_aspect((1, 1, 0.5), zoom=1.18)
        ax_3d.view_init(elev=33, azim=-62 + 10 * (k + pre) / (len(frames) + pre))
        ax_3d.set_title("model  (fitted on clips 1-2)", color="w", fontsize=10)
        ax_trace.clear()
        ax_trace.set_facecolor(bg)
        ax_trace.plot(event["t"], event["z"], color="#e6edf3", lw=1.6, label="real")
        ax_trace.plot(ts, width, color="#ff6b6b", lw=2.0, label="model")
        ax_trace.axvline(max(t, 0), color="#7d8590", lw=1)
        ax_trace.set_ylim(0.35, 1.05)
        ax_trace.set_xlim(-0.05, 1.2)
        ax_trace.set_xlabel("time after release (s)", color="w")
        ax_trace.set_ylabel("width / resting width", color="w")
        ax_trace.tick_params(colors="w")
        ax_trace.grid(alpha=0.2)
        for sp in ax_trace.spines.values():
            sp.set_color("#7d8590")
        ax_trace.legend(loc="lower right", fontsize=8, facecolor=bg, labelcolor="w", edgecolor="none")
        ax_trace.set_title("held-out error %.3f" % rmse, color="w", fontsize=10)
        fig.canvas.draw()
        out.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    return out


def main():
    bg = "#0d1117"
    files = sorted(glob.glob("tyreN_sig_*.npz"))
    all_events = [[e for e in events(f, i + 1) if (not e["held"]) and e["dur"] > 0.3] for i, f in enumerate(files)]
    picks = [(3, all_events[2][0], "tyre_3, t=%.1f s"), (4, all_events[3][-1], "tyre_4, t=%.1f s")]
    names = {3: "tyre-3.mov", 4: "tyre_4.mov"}
    videos = os.path.expanduser(os.environ.get("TYRE_VIDEOS", "~/Documents"))
    fig = plt.figure(figsize=(12.5, 4.4), dpi=80, facecolor=bg)
    axes = [fig.add_subplot(1, 3, 1), fig.add_subplot(1, 3, 2, projection="3d"), fig.add_subplot(1, 3, 3)]
    fig.subplots_adjust(left=0.03, right=0.98, wspace=0.12, bottom=0.17, top=0.9)
    frames = []
    for clip, event, label in picks:
        cap = cv2.VideoCapture(os.path.join(videos, names[clip]))
        frames += render_event(event, cap, fig, axes, label % event["t_abs"], bg)
        print("clip", clip, len(frames), "frames", flush=True)
    os.makedirs("../figures/demo_frames", exist_ok=True)
    for i, fr in enumerate(frames):
        cv2.imwrite("../figures/demo_frames/f%04d.png" % i, cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    fps = SIM_FPS / SLOW
    common = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", "../figures/demo_frames/f%04d.png"]
    subprocess.run(common + ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "../figures/demo.mp4"], check=True)
    subprocess.run(
        common
        + [
            "-vf",
            "scale=800:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
            "../figures/demo.gif",
        ],
        check=True,
    )
    shutil.rmtree("../figures/demo_frames")
    print("done:", len(frames), "frames")


if __name__ == "__main__":
    main()
