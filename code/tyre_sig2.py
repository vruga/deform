"""Step 1: video -> width signal.

For every frame of each tyre clip: threshold the black rubber *relative to that frame's own brightness* (0.3 x the 80th-percentile brightness, which
cancels the 100 Hz light flicker), take the largest blob and fit a rotated rectangle. The short side is the tyre width, the long side its length.
Output (run from ../data):  tyreN_sig_<clip>.npz with time t (s, 240 fps), minor (width), major (length), area (px).

Videos are read from $TYRE_VIDEOS (default ~/Documents), files tyre*.mov.
"""

import glob
import os

import cv2
import numpy as np

FPS = 240.0  # verified from the 100 Hz room-light flicker (see README)
SIZE = (480, 270)  # frames are downscaled for speed
BRIGHTNESS_FRACTION = 0.3  # a pixel is rubber if V < 0.3 x (80th percentile of V in that frame)
MIN_BLOB_AREA = 400


def frame_measurements(frame):
    """(width, length, area, x_centre) of the tyre blob in one frame, or NaNs if none is found."""
    small = cv2.resize(frame, SIZE)
    value = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[..., 2]
    mask = (value < BRIGHTNESS_FRACTION * np.percentile(value, 80)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > MIN_BLOB_AREA]
    if not contours:
        return [np.nan] * 4
    blob = max(contours, key=cv2.contourArea)
    (cx, _), (w, h), _ = cv2.minAreaRect(blob)
    return [min(w, h), max(w, h), cv2.contourArea(blob), cx]


def extract(path):
    """Write the width signal of one clip."""
    name = os.path.basename(path)[:-4].replace("-", "_")
    capture = cv2.VideoCapture(path)
    rows = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        rows.append(frame_measurements(frame))
    r = np.array(rows)
    t = np.arange(len(r)) / FPS
    np.savez(f"tyreN_sig_{name}.npz", t=t, minor=r[:, 0], major=r[:, 1], area=r[:, 2], fps=FPS)
    print(name, len(r), "frames", flush=True)


if __name__ == "__main__":
    videos = os.path.expanduser(os.environ.get("TYRE_VIDEOS", "~/Documents"))
    for p in sorted(glob.glob(os.path.join(videos, "tyre*.mov"))):
        extract(p)
