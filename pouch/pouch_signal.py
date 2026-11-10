"""Pouch (red and grey silicone pencil pouch) 'memory' signal. Per frame: segment the pouch (red by hue; grey by Lab distance from a seed colour on the free pouch),
take the largest blob, PCA axis, body aspect = median cross-section width over 25-80 % of the length / length (scale- and perspective-robust),
and a hand-free flag = fraction of 'foreign' pixels (neither pouch nor the background colour model from frame 0) in a band around the pouch.
usage: python pouch_signal.py OUT_DIR   (videos from ~/Documents and ~/Downloads)"""

import cv2, numpy as np, os, sys

OUT = sys.argv[1]
CLIPS = [
    ("red_slowmo_aruco", "~/Documents/red_slowmo_aruco.mov", 240.0, "red"),
    ("red_slow_noaruco", "~/Documents/red_slow_noaruco.mov", 240.0, "red"),
    ("red_9810", "~/Downloads/IMG_9810.MOV", 30.0, "red"),
    ("grey", "~/Downloads/pencil_pouch.mp4", 30.0, "grey"),
]


def prep(f):
    h, w = f.shape[:2]
    sc = 300.0 / min(h, w)
    return cv2.resize(f, (int(w * sc), int(h * sc)))


def pouch_mask(f, kind, seed):
    if kind == "red":
        h = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        H, S, V = [h[..., i].astype(int) for i in range(3)]
        m = ((H >= 165) | (H <= 8)) & (S > 110) & (V > 60)
    else:
        lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(float)
        m = np.linalg.norm(lab - seed, axis=2) < 16
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return m


def run(name, path, fps, kind):
    c = cv2.VideoCapture(os.path.expanduser(path))
    rows = []
    bgm = None
    seed = None
    i = 0
    while True:
        ok, f = c.read()
        if not ok:
            break
        f = prep(f)
        if seed is None:
            lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(float)
            h, w = lab.shape[:2]
            seed = (
                np.median(lab[int(0.40 * h) : int(0.50 * h), int(0.45 * w) : int(0.55 * w)].reshape(-1, 3), 0)
                if kind == "grey"
                else None
            )
        m = pouch_mask(f, kind, seed)
        n, l, st, _ = cv2.connectedComponentsWithStats(m)
        if n < 2 or st[1:, 4].max() < 600:
            rows.append([np.nan] * 4 + [0])
            i += 1
            continue
        m = (l == 1 + np.argmax(st[1:, 4])).astype(np.uint8)
        if bgm is None:  # background colour model from frame 0, outside the pouch
            lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(float)
            b = lab[cv2.dilate(m, np.ones((15, 15), np.uint8)) == 0]
            bgm = (b.mean(0), np.linalg.inv(np.cov(b.T) + np.eye(3)))
        ys, xs = np.nonzero(m)
        P = np.c_[xs, ys].astype(float)
        mu = P.mean(0)
        w_, v = np.linalg.eigh(np.cov((P - mu).T))
        ax = v[:, 1]
        pe = v[:, 0]
        a = (P - mu) @ ax
        b = (P - mu) @ pe
        L = a.max() - a.min()
        bi = np.floor((a - a.min()) / 2).astype(int)
        nb = bi.max() + 1
        wid = np.array(
            [(b[bi == k].max() - b[bi == k].min() + 1) if (bi == k).sum() > 2 else np.nan for k in range(nb)]
        )
        z = np.arange(nb) * 2 / L
        body = np.nanmedian(wid[(z > 0.25) & (z < 0.8)])
        lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(float)
        band = (cv2.dilate(m, np.ones((31, 31), np.uint8)) > 0) & (m == 0)
        d = lab[band] - bgm[0]
        md = np.sqrt(np.einsum("ij,jk,ik->i", d, bgm[1], d))
        foreign = float(np.mean(md > 6.0)) if band.sum() else 0.0
        rows.append([body, L, body / L, foreign, m.sum()])
        i += 1
    r = np.array(rows)
    np.save(os.path.join(OUT, f"pouch_{name}.npy"), np.c_[np.arange(len(r)) / fps, r])
    print(name, len(r), "frames", fps, "fps", flush=True)


if __name__ == "__main__":
    for cl in CLIPS:
        run(*cl)
