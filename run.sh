#!/bin/sh
# Reproduce experiment 1. Needs python with numpy, scipy, opencv-python, matplotlib. Videos are read from ~/Documents (tyre_1.mov, tyre_2.mov, tyre-3.mov, tyre_4.mov).
set -e
HERE=$(cd "$(dirname "$0")" && pwd); export PYTHONPATH="$HERE/code"
cd "$HERE/data"
[ "$1" = "extract" ] && python "$HERE/code/tyre_sig2.py"            # step 1 (slow, reads the videos): writes tyreN_sig_*.npz
python "$HERE/code/tyre_fit2.py"                                      # step 2: linear / 2nd-order / 2-time-constant comparison -> tyre_fit2.json
python "$HERE/code/tyre_ring_fit.py" "$HERE/figures/heldout_ring_vs_first_order.png"   # step 3: ring vs first-order, held-out + leave-one-clip-out
python "$HERE/code/tyre_models_fast.py"                              # step 4: all models + bootstrap, about 15 s (needs numba)
mv -f tyre_models_fast.json "$HERE/results/"
mv -f tyre_fit2.json tyre_ring_fit.json "$HERE/results/"
