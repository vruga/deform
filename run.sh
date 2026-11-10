#!/bin/sh
# Reproduce everything. Needs python with numpy, scipy, opencv-python, matplotlib, numba (and ffmpeg for the demo).
# Videos are read from $TYRE_VIDEOS (default ~/Documents): tyre_1.mov, tyre_2.mov, tyre-3.mov, tyre_4.mov (the load/geometry clips and pouch clips are only needed for load_and_hand/ and pouch/).
# usage: sh run.sh [extract] [demo] [extras]   extras = length validation, bootstrap CIs, fractional fit, depth-dependence test (about 5 minutes)      extract = re-read the videos (slow); demo = rebuild figures/demo.mp4 and demo.gif
set -e
HERE=$(cd "$(dirname "$0")" && pwd); export PYTHONPATH="$HERE/code"
cd "$HERE/data"
for a in "$@"; do [ "$a" = "extract" ] && python "$HERE/code/tyre_sig2.py"; done      # step 1 (slow): width signals -> tyreN_sig_*.npz
python "$HERE/code/tyre_fit2.py"                                        # step 2: linear / 2nd-order / 2-time-constant comparison
python "$HERE/code/tyre_ring_fit.py" "$HERE/figures/heldout_ring_vs_first_order.png"   # step 3: ring vs first-order, held-out + LOCO
python "$HERE/code/tyre_models_fast.py"                                 # step 4: all models + bootstrap (about 15 s)
python "$HERE/code/tyre_no_release_param.py"                            # step 5: release-speed ablation
python "$HERE/code/tyre_ring3d_fit.py"                                  # step 6: 3D stack with a stepped wall
python "$HERE/code/identifiability.py"                                  # step 7: profile likelihood figure
mv -f tyre_models_fast.json tyre_fit2.json tyre_ring_fit.json "$HERE/results/" 2>/dev/null || true
for a in "$@"; do [ "$a" = "demo" ] && python "$HERE/code/make_demo.py"; done          # optional: demo video + gif
for a in "$@"; do [ "$a" = "extras" ] && python "$HERE/code/extra_tests.py"; done        # optional: extra checks (section 3 of the README)
