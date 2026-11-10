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

Protocol: fit on `tyre_1` and `tyre_2`, freeze, then predict `tyre_3` and `tyre_4`. Each test event supplies only its own measured squeezed width. Also leave-one-clip-out (LOCO, every clip held out in turn).

RMSE of width / resting width (0.04 is about 3 mm). Null = "stays squeezed". A flick is a squeeze held under 0.3 s (not press-hold-release); the rule was written down from the interaction before comparing models.

| | null | first-order (tau 0.143 s) | ring / 3D model | LOCO first-order / ring |
|---|---|---|---|---|
| flicks included | 0.358 | 0.047 | not run | 0.045 / n.a. |
| flicks excluded | 0.343 | 0.036 | 0.040 | 0.032 / 0.034 |

Most of the drop from 0.047 to 0.036 comes from excluding flicks, not from a better model. The ring is not more accurate than the one-line exponential; its value is parameters in real units and a 3D shape. The best ring variant (release speed depending on squeeze depth) reaches LOCO 0.029, but its interval against first-order touches zero, so it is not a proven gain.

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

## 6. A second object: the pencil pouch (`pouch/`)

The pouch (a stiff silicone cylinder, red and grey versions of the same object) is the case the tyre method cannot handle: the hand covers it while it is pressed, so a width-versus-time recovery cannot be fitted. What can be measured is its shape **when the hand is off**.
On the red pouch with a fixed camera (`IMG_9810`) the body aspect in three hand-free stretches, separated by 10 to 20 s of handling that includes squeezes to 0.2 to 0.3, is **0.414, 0.412, 0.411**: a residual set of **-0.5 % and -0.2 %** (scatter about 0.5 %) and no creep (+0.06 %/s). It is already at its rest shape when the hand leaves, so it has **no detectable memory**; the release itself is hidden by the hand and the clip is 30 fps, so this is an end-state measurement, not a fitted stiffness or damping.
The grey pouch could not be segmented from the table by colour (the two have almost the same chroma), so there is no grey result; the 240 fps red clips are too noisy to analyse. Details, figure and what a real fit would need: `pouch/README.md`.

## 6b. Not worked: other exploratory attempts

Hand tracking (MediaPipe loses the fingertips as the hand curls round the black tyre), a tongs clip (its jaws could not be separated from the background), and the lift-off recovery in the weight test (it only shows recovery is faster than the 0.33 s lift). Details in `docs/full_log.md` and `load_and_hand/`.

## 7. Reproduce

Python with numpy, scipy, opencv, matplotlib, numba (and ffmpeg for the demo). Videos are read from `$TYRE_VIDEOS` (default `~/Documents`): `tyre_1.mov`, `tyre_2.mov`, `tyre-3.mov`, `tyre_4.mov`.

```
sh run.sh            # steps 2-7 on the saved width signals (about 11 minutes on 8 cores; the slow NumPy ring fit dominates)
sh run.sh extract    # first re-read the videos (slow)
sh run.sh demo       # also rebuild figures/demo.mp4 and demo.gif
```

```
code/        tyre_sig2.py (video -> width), tyre_fit2.py (events, linear models), tyre_ring_fast.py (ring solver), tyre_models_fast.py (all models, bootstrap),
             tyre_ring3d.py + tyre_ring3d_fit.py (3D stack), identifiability.py, tyre_no_release_param.py, make_demo.py
results/     json behind every table      figures/   demo, identifiability, geometry, held-out overlays
load_and_hand/   weight test, Young's modulus, hand-release attempts      docs/full_log.md   the full history, including what did not work
```
