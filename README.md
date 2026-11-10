# What can one video of a squeeze-and-release tell you about a soft object?

**Object:** a toy rubber tyre (75.8 mm across, 37 mm tread). **Interaction:** pinched between finger and thumb to about half its width, held, released. **Capture:** top-down, 240 fps, a 10 mm ArUco marker for scale.

![held-out squeeze: real video, fitted model, width traces](figures/demo.gif)

*Two squeezes from clips the model never saw (3x slow motion). Left: real. Middle: the fitted 3D tyre model. Right: width over time, real (black) vs predicted (red). The model is given only each event's own squeezed width.*

## The question

The assignment asks for a deformation model whose stiffness and damping are fitted from video. The useful question underneath is: **what can a single video actually determine, and what can it not?** We built the model, fitted it, predicted fresh squeezes, and then tested which parameters the video really pins down.

## Answer in four lines

1. **Video pins down the relaxation time, not stiffness and damping separately.** The tyre is overdamped (no overshoot), so only the stiffness-to-damping ratio is observable: a recovery time of **about 0.14 s**. The profile below shows it is determined to roughly +-30 %.
2. **A known force supplies the absolute scale.** A 333 g weight on the standing tyre (also filmed) gives **0.43 N/mm** (0.36 to 0.54) and, with the ratio above, a damping of order **60 N s/m**.
3. **The hand's release barely matters.** A model with an **instant release and no release parameter** predicts held-out squeezes within 0.001 of one with a fitted release speed. The rubber's own relaxation sets the recovery.
4. **The remaining error is a floor, not missing physics.** Fitting each event separately reaches **0.020**; any single shared model sits at **0.036 to 0.040**, and every model family tried (first-order, second-order, two time constants, ring, strain-dependent viscosity, ODE discovery) lands on the same number.

## 1. Model

* **Representation.** A thin ring of 16 nodes (the tyre cross-section) with stretch springs, bending energy using true large rotation angles (so the ring stiffens as it flattens), overdamped dynamics, and two penalty-contact finger plates. The 27 g mass drops out.
* **3D.** A stack of 5 such rings along the 37 mm tread, joined by 3D axial springs (`code/tyre_ring3d.py`). The videos show the real tyre has a stepped inside (a lip) and two rows of lugs, so the wall is not uniform; `figures/geometry/` shows the views. With full-width contact the stack reproduces the 2D ring (checked to 0.09 mm), and the squeezes reach about 0.5 of the width, which only full-width contact can produce.
* **Solver.** Explicit overdamped integration in Numba, 0.5 ms per simulation.
* **Free parameters:** bending stiffness / drag `kb`, plate release speed, rest level. Nothing is fitted per event.

## 2. Fit and fresh prediction

Protocol: fit on `tyre_1` and `tyre_2` (6 events), freeze, then predict `tyre_3` and `tyre_4` (6 events). Each test event supplies only its own measured squeezed width. Also leave-one-clip-out (LOCO, all 12 events as test events).

RMSE of width / resting width (0.04 is about 3 mm). Null = "stays squeezed".

| event set | null | first-order (tau 0.143 s) | ring model | LOCO first-order / ring |
|---|---|---|---|---|
| all squeezes, flicks included (7 test events) | 0.358 | 0.047 | not run | 0.045 / n.a. |
| **flicks excluded, final (6 test events)** | **0.343** | **0.036** | **0.040** | **0.032 / 0.034** |

The flick rule (a squeeze held under 0.3 s is a flick, not press-hold-release) was decided from the interaction before comparing models. **Most of the 0.047 to 0.036 drop comes from that exclusion, not from a better model.** The physical ring is not more accurate than the one-line exponential; its value is parameters in real units and a 3D shape.

## 3. What the video can and cannot identify

![identifiability profiles](figures/identifiability.png)

*Profile likelihood on all 12 events: fix one parameter, re-fit the others. Grey = noise (bootstrap s.d. 0.0029).*

* **Left, stiffness : damping is pinned** (within noise for kb of 1.5 to 2.8, best 1.74). Scaling stiffness and damping together leaves an overdamped trajectory unchanged, so the video fixes only their ratio.
* **Right, release speed is not identified above about 2.5 diameters/s.** Every faster release is within noise, up to instant. Only slower releases (under 1.5) hurt.
* **Noise floor.** Per-event fits (tau and rest free for each event, 12 events) give 0.0196; shared models give 0.036 to 0.040. More model terms did not help (second-order, second time constant, stretched exponential, strain-dependent viscosity, ODE discovery): see `docs/full_log.md`.
* **Units are bounded by geometry, not by the video.** From the weight test, bending stiffness is 2.1e-3 to 5.6e-3 N m^2 depending on the ring model used. With the tread width (37 mm, caliper), the lip wall (11.8 mm) and an unmeasured thinner main wall (7 to 11.8 mm), Young's modulus is **a few tenths to about 2 MPa**.

Release-speed table (fixed, not fitted; same events and protocol):

| release speed (diam/s) | instant | 20 | 8 | 4 | 2.72 (fitted) | 1.5 |
|---|---|---|---|---|---|---|
| held-out error | 0.0407 | 0.0407 | 0.0405 | 0.0400 | 0.0398 | 0.0487 |

## 4. What the model can and cannot capture

**Can:** the overdamped single-timescale recovery (no overshoot), the geometric stiffening as the ring flattens, the depth of squeeze, stiffness in real units from one known force, a 3D shape with the measured tread width.
**Cannot:** shallow squeezes recover more slowly than predicted (unexplained; a strain-dependent rubber or a gentler hand would look the same); tread lugs, friction and out-of-plane bending; a thin-wall proxy stands in for the solid rubber; creep and permanent set were not separable in these clips; only one tyre, one person, 12 usable events.

## 5. Assumptions

Thin-ring cross-section (squeeze is in-plane); overdamped (verified: damping ratio ran to its upper bound, 5 to 12); frame rate 240 fps measured from the 100 Hz room-light flicker, not assumed; scale from the marker; the lip fraction (40 %) and main-zone wall are estimates, shown as sensitivity, not results; one load point for the absolute stiffness.

## 6. Not worked: exploratory attempts

Hand tracking (MediaPipe loses the fingertips as the hand curls round the black tyre), a tongs clip (its jaws could not be separated from the background), a pencil pouch (held and rotated, so the tyre pipeline does not transfer), and the lift-off recovery in the weight test (only shows recovery is faster than the 0.33 s lift). Details in `docs/full_log.md` and `load_and_hand/`.

## 7. Reproduce

Python with numpy, scipy, opencv, matplotlib, numba; videos are read from `~/Documents` (`tyre_1.mov`, `tyre_2.mov`, `tyre-3.mov`, `tyre_4.mov`).

```
sh run.sh extract                    # width signals from the videos, then fits and tables
cd data && PYTHONPATH=../code python ../code/identifiability.py        # figure in section 3
cd data && PYTHONPATH=../code python ../code/tyre_no_release_param.py  # release-speed table
cd data && PYTHONPATH=../code python ../code/tyre_ring3d_fit.py       # 3D stack with a stepped wall
cd data && PYTHONPATH=../code python ../code/make_demo.py              # demo video and gif
```

```
code/        tyre_sig2.py (video -> width), tyre_fit2.py (events, linear models), tyre_ring_fast.py (ring solver), tyre_models_fast.py (all models, bootstrap),
             tyre_ring3d.py + tyre_ring3d_fit.py (3D stack), identifiability.py, tyre_no_release_param.py, make_demo.py
results/     json behind every table      figures/   demo, identifiability, geometry, held-out overlays
load_and_hand/   weight test, Young's modulus, hand-release attempts      docs/full_log.md   the full history, including what did not work
```
