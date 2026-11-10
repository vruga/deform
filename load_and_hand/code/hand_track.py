import cv2, numpy as np, sys, os, glob
import mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision

MODEL = sys.argv[1]
opts = vision.HandLandmarkerOptions(
    base_options=mpp.BaseOptions(model_asset_path=MODEL),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
)
det = vision.HandLandmarker.create_from_options(opts)
for p in sorted(glob.glob(os.path.expanduser(os.environ.get("TYRE_VIDEOS", "~/Documents") + "/tyre*.mov"))):
    name = os.path.basename(p)[:-4].replace("-", "_")
    c = cv2.VideoCapture(p)
    rows = []
    i = 0
    while True:
        ok, f = c.read()
        if not ok:
            break
        if i % 2 == 0:  # every 2nd frame (120 Hz) for speed
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            tips = []
            for h in r.hand_landmarks:
                tips.append(
                    [(h[k].x * f.shape[1], h[k].y * f.shape[0]) for k in (4, 8, 12)]
                )  # thumb, index, middle tips
            rows.append((i, tips))
        i += 1
    import pickle

    pickle.dump(rows, open(f"hand_tips_{name}.pkl", "wb"))
    det_n = sum(1 for _, t in rows if t)
    print(name, "frames", len(rows), "with hand", det_n, flush=True)
