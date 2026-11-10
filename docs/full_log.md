> Full lab log (every experiment, including the ones that did not help). Paths are relative to the repository root. The short version is in the top-level README.

# Experiment 1: learning a squeezed rubber tyre's deformation and recovery from video

Task (problem statement): film yourself pressing or squeezing one soft object and letting go, build a simple 3D model of its
deformation and recovery, fit its behaviour (effective stiffness and damping) from the video, then show the **fitted** model
predicting a **fresh** instance of the same interaction and compare it with what really happened.

Object: a toy rubber tyre, about **75.8 mm** across (scale from a **10 mm ArUco marker**, `DICT_4X4_50` id 13, in every clip).
Interaction: the tyre lies flat, is pinched between finger and thumb to roughly half its width, held, then released.

This README covers the video-fitted deformation model. The weight/load test, Young's modulus and the attempts to measure the
hand's release speed are in the sub-folder `load_and_hand/` (its own README, same experiment).

![held-out squeeze: real video, fitted model, width traces](figures/demo.gif)

*Two held-out squeezes (clips 3 and 4, never used in fitting), 3x slow motion. Left: real video. Middle: the fitted tyre model (the 2D ring extruded over the
37 mm tread; with full-width contact the 3D stack in `code/tyre_ring3d.py` moves identically). Right: width over time, real (black) vs model (red).
The model is given only each event's own measured squeezed width. Full video: `figures/demo.mp4`. Rebuild: `code/make_demo.py`.*


---------------------------------------------------------------------------------------------------

## 1. Data

| clip (in `~/Documents`) | capture | length (real time) | squeezes found | role |
|---|---|---|---|---|
| `tyre_1.mov` | 240 fps slow-mo, top-down, white table | 11.8 s | 5 | train |
| `tyre_2.mov` | 240 fps | 11.3 s | 5 | train |
| `tyre-3.mov` | 240 fps | 12.9 s | 5 | held-out test |
| `tyre_4.mov` | 240 fps | 12.8 s | 5 | held-out test |

Frame rate was checked, not assumed: the room light flickers at 100 Hz and shows up at 0.416 cycles per frame in all four
clips, which means 240 fps (playback is 30 fps, so real time = playback / 8).

Not used for fitting: `tyre_5/6/7.mov` (oblique side views) and `tyre_deformation.mov` (weight test). `tyre_6` and
`tyre_deformation` were used to **measure the tyre's geometry**, see `load_and_hand/`. Those four newer clips are also 240 fps
captures (flicker at 0.417 cycles/frame, peak 80 to 210 times the median), stored in 30 fps containers like the others, so
real time = playback time / 8. `ffprobe` reports 30 fps for every clip; the flicker is the evidence for 240.

## 2. Where we started, and what we deliberately did not do

An earlier pouch pipeline (`fit_slo*.py`, `fit_v1.py`, `fem.py`, still in the parent folder) refit the **hand release time for every
event** (`t_up` was a free per-event number). That quietly lets the model tune itself to the answer it is meant to predict.

In this experiment:

* **No per-episode fitting.** Every parameter is shared by all events. Each event contributes exactly one thing, its own
  *measured* squeezed width at the moment of release (`z0`). Nothing else is tuned per event.
* **No refitting on the test clips.** Parameters are fitted on the training clips and then frozen.
* A per-event fit was run **once, as a diagnostic only** (`code/diag_per_event_tau.py`) to see how much the recovery time varies.
  It is never used for any reported result.

## 3. Pipeline

1. **Width signal** (`code/tyre_sig2.py`). Per frame: threshold the black rubber, take the largest blob, fit a rotated rectangle,
   keep the short side. The threshold is **relative to each frame's own brightness** (`0.3 x 80th-percentile brightness`), then a
   50 ms median filter (12 frames, about 5 flicker periods). Divide by the resting width (85th percentile), so
   `z(t) = width / resting width`.
   *Why:* with a fixed threshold the 100 Hz flicker changed the mask size every other frame (a comb pattern in the signal).
2. **Events.** Squeezes are the minima of `z` (prominence 0.25, at least 1 s apart). For each one:
   * release onset = the last frame still within 0.03 of the held minimum,
   * recovery window = the next 1.2 s (at least 0.3 s required).
3. **Exclusions** (decided from the *interaction*, written down before comparing models):
   * a squeeze held for **< 0.3 s** is a flick, not press-hold-release, so it is dropped;
   * an event where the tyre has not risen halfway after 0.5 s means the hand is still gripping, so it is dropped.
   Event set S2 below uses both rules (12 events: 6 train, 6 test).
4. **Fit** least squares on the recovery windows, soft-L1 loss (robust to the stair-step pixel noise).
5. **Evaluate** on the held-out clips (section 5).

## 4. Training and evaluation

* **Main split: train on `tyre_1` and `tyre_2` (6 events), test on `tyre_3` and `tyre_4` (6 events).** Test clips are never seen in
  fitting. The model is *given* only the test event's own measured starting width and must predict the next 1.2 s of recovery.
* **Leave-one-clip-out (LOCO):** train on three clips, evaluate on the fourth, rotate through all four clips. This is the
  "three clips for training, one for evaluation" setting, applied to each clip in turn, and it uses all 12 events as test events.
  LOCO is the more reliable number because the main split has only 6 test events.
* Metric: **RMSE of `z`** (fraction of the resting width; 0.04 is about 3 mm on a 76 mm tyre). Baseline: **null** = "the tyre
  stays at its squeezed width".

## 5. What we tried, what helped, what did not

### 5a. Event set S1 (flicks included, hand-gripped event flagged and excluded): 8 train / 7 test (`results/tyre_fit2.json`)

| model | free params | train | held-out test | LOCO |
|---|---|---|---|---|
| null (no recovery) | 0 | n/a | 0.358 | n/a |
| 1st-order | tau | 0.045 | 0.053 | 0.049 |
| **1st-order + fitted rest level** | tau, rest | 0.041 | **0.047** | **0.045** |
| 2nd-order spring-damper | wn, zeta | 0.045 | 0.053 | 0.049 |
| 2nd-order + fitted rest | wn, zeta, rest | 0.041 | 0.047 | 0.045 |
| two time constants (Prony / SLS) | tau1, tau2, weight | 0.039 | 0.049 | 0.045 |
| two time constants + fitted rest | + rest | 0.039 | 0.059 | 0.045 |

* **Helped:** fitting the rest level (the tyre settles at 0.978 of the 85th-percentile width, not 1.0): 0.053 to 0.047.
* **Did not help:** a 2nd-order spring-damper. The damping ratio ran to its upper bound (12 in one fit, 5.3 in the other), so the
  tyre is **overdamped**. It has no overshoot, inertia cannot be identified, and the model collapses to the 1st-order one.
* **Did not help:** a second relaxation time. Slightly better on training data, not on held-out data.

### 5b. Event set S2 (also drops flicks < 0.3 s): 6 train / 6 test (`results/tyre_ring_fit.json`, `results/tyre_ring2.json`)

| model | free params | train | held-out test | test median | LOCO |
|---|---|---|---|---|---|
| null (no recovery) | 0 | 0.440 | 0.343 | 0.330 | n/a |
| **1st-order + rest (linear baseline)** | tau = 0.143 s, rest = 0.978 | 0.025 | **0.036** | 0.039 | **0.032** |
| **Ring model (nonlinear)** | kb = 1.74, release speed = 2.72 diam/s, rest = 0.979 | 0.025 | 0.040 | 0.042 | 0.034 |
| Ring + depth-dependent release speed | kb, rest, a, b in `speed = exp(a + b*depth)` | 0.025 | 0.039 | 0.042 | 0.031 |

Fitted depth law: `speed = exp(0.635 + 0.694 * depth)`. For depths 0.3 to 0.55 that is 2.3 to 2.8 diameters/s, a very weak
dependence.

**Reading the numbers honestly**

* The best held-out error went from 0.096 (first attempt, 8 test events) to 0.047 (S1) to 0.036 (S2). **Most of the 0.047 to
  0.036 drop is from excluding flicks, not from a better model.** On the same event set, the nonlinear ring is **not more
  accurate** than the linear baseline (0.040 vs 0.036 held-out, 0.034 vs 0.032 LOCO). With 6 test events the differences between
  the three S2 models are within noise.
* Training error is about 0.025 for every model. The real signal is stair-stepped by pixel quantisation, so a floor around
  0.02 to 0.03 is plausible and more model complexity cannot buy much.
* What we did gain: a **physical** model (below) at the same accuracy, with parameters that map to units (`load_and_hand/`), plus
  honest diagnostics.

### 5c. Diagnostics

* **Recovery time varies 14x between events** (`code/diag_per_event_tau.py`): flicks about 0.02 s, deep squeezes (0.45 to 0.55)
  0.10 to 0.18 s, shallow squeezes (0.30 to 0.41) 0.23 to 0.29 s. A single time constant cannot cover that.
* Fitting a release speed separately per event, with the stiffness fixed, nearly halves the error on the 3 shallowest squeezes
  (see `load_and_hand/`). This suggests the hand lets go more gently on shallow squeezes, but a fitted explanation is not proof and a
  strain-dependent rubber would look the same.

### 5d. Faster code, more model variants, and a significance check (`code/tyre_models_fast.py`, `results/tyre_models_fast.json`)

The ring simulator was rewritten with Numba (`code/tyre_ring_fast.py`): **0.5 ms per simulation instead of 91 ms, about 180x faster,
and identical results** (max difference 4e-16 against the NumPy version `tyre_ring.py`). One script now fits every model with
6 to 12 random starting points, runs all four leave-one-clip-out folds in parallel, and finishes in **about 15 s** (the
NumPy version would take an estimated hour or more: about 10 to 15 minutes per fit, 30 fits). The numbers for the first-order and ring models reproduce those in 5b exactly.

Models added in this round (all on the same 6 train / 6 test / 12 LOCO events):

* **nlfirst:** first-order with strain-dependent viscosity, rate ~ (deficit/0.5)^p.
* **ringC:** the ring with the same strain-dependent drag.
* **ringE:** the finger plates accelerate to the release speed with time constant `ta`, instead of moving at constant speed.
* **ringB:** release speed = exp(a + b * depth) (re-fitted with more starting points).

| model | free params | train | held-out | test median | LOCO | LOCO error minus first-order, 95% CI |
|---|---|---|---|---|---|---|
| first-order (reference) | 2 | 0.0249 | **0.0363** | 0.0392 | 0.0319 | n/a |
| first-order + strain-dependent viscosity | 3 (p = 0.12) | 0.0248 | 0.0356 | 0.0390 | 0.0315 | -0.0004 [-0.0016, +0.0008] |
| ring A (shared release speed) | 3 | 0.0250 | 0.0398 | 0.0416 | 0.0341 | **+0.0021 [+0.0008, +0.0036]** |
| ring B (release speed depends on depth) | 4 | 0.0250 | 0.0391 | 0.0415 | **0.0294** | -0.0026 [-0.0053, +0.0001] |
| ring C (strain-dependent drag, p = 0.17) | 4 | 0.0248 | 0.0394 | 0.0417 | 0.0338 | +0.0018 [+0.0004, +0.0034] |
| ring E (accelerating fingers) | 4 | 0.0245 | 0.0385 | 0.0405 | 0.0331 | +0.0012 [-0.0000, +0.0026] |

The confidence intervals are a paired bootstrap over the 12 leave-one-clip-out events (4000 resamples).

**What this says**

* **Nothing clearly beats the linear first-order model.** The best LOCO number is ring B (0.0294), but its interval touches zero and
  its held-out error (0.0391) is no better. Ring A and ring C are **significantly worse** than the first-order model on LOCO (the
  interval excludes zero), by about 0.002, which is 0.15 mm on a 76 mm tyre.
* Training errors of every model are 0.0245 to 0.0250, so the remaining error looks like a measurement floor (stair-stepped pixel
  signal) and not a missing physical effect.
* **The strain-dependent viscosity idea was checked for implementation errors and profiled over p** (first-order version; the ring
  version behaves the same). The formulas and both implementations are correct. Fixing p and refitting the other parameters:

  | p | train | held-out | LOCO | error on the 4 shallow events | tail error (t > 0.6 s) |
  |---|---|---|---|---|---|
  | 0 | 0.0249 | 0.0363 | 0.0319 | 0.0337 | 0.0174 |
  | 0.3 | 0.0251 | 0.0351 | 0.0314 | 0.0314 | 0.0171 |
  | 1.0 | 0.0280 | 0.0355 | 0.0329 | 0.0301 | 0.0185 |
  | 2.5 | 0.0395 | 0.0382 | 0.0392 | 0.0310 | 0.0180 |

  Larger p helps the shallow events a little (0.0337 to 0.0301) but worsens the deep ones and the training fit, so the total gets
  worse. The tail does **not** improve with p. The tail error is mostly a per-event offset in the settled width: across the
  12 events the settled level differs from the fitted rest level by -1.9 to +1.7 mm (pose, shadow and normalisation scatter), and
  no dynamics can fit that. So 'slow tails' was the wrong explanation for the late-time misses.
* An earlier README said ring B gave 0.031 LOCO. That fit used only two starting points. With more starts it reaches 0.0294 and the
  fitted release-speed law changes, a sign that its parameters are weakly determined.
* The ring's advantage is not accuracy. It is a physical model with parameters in real units (experiment part B).

### 5e. Do we need the fitted release speed? (`code/tyre_no_release_param.py`, `results/tyre_no_release_param.json`)

The ring's finger plates retract at a fitted speed, a stand-in for the unmeasured hand. Hand tracking could not replace it (MediaPipe loses the fingertips,
see `load_and_hand/`; a tongs clip was tried and its jaws could not be separated from the background). So the question was answered the other way:
**fix the speed instead of fitting it** and refit only stiffness and rest level (same events and protocol as 5d):

| plate release speed (diameters/s, fixed) | train | held-out | LOCO | kb |
|---|---|---|---|---|
| instant (300), no release parameter | 0.0266 | 0.0407 | 0.0352 | 1.51 |
| 20 | 0.0265 | 0.0407 | 0.0351 | 1.52 |
| 8 | 0.0263 | 0.0405 | 0.0349 | 1.54 |
| 4 | 0.0257 | 0.0400 | 0.0344 | 1.60 |
| 2.72 (the fitted value) | 0.0250 | 0.0398 | 0.0339 | 1.74 |
| 1.5 | 0.0344 | 0.0487 | 0.0438 | 5.73 |

An **instant release with no release parameter** is within 0.001 of the fitted-speed ring (0.0407 vs 0.0398 held-out, 0.0352 vs 0.0339 LOCO). Only a release slower than
about 1.5 diameters/s hurts. So the rubber's own relaxation explains almost all of the recovery in these clips, and the hand matters mainly for the shallow squeezes (5c).
The linear first-order model remains slightly better than every ring (0.0363 held-out).

## 6. The nonlinear model (ring + finger plates), `code/tyre_ring.py`

| item | what we assumed |
|---|---|
| geometry | 2D cross-section of the tyre as a closed ring of **16 nodes**, rest diameter = 1 unit = 75.8 mm (R = 0.5). Wall thickness is not in the dynamics (a thin-ring model); it is only drawn |
| stretch | stiff springs between neighbouring nodes, constant 100 (fixed, nearly inextensible) |
| bending | energy 1/2 * kb * (turning angle - rest angle)^2 per node, true large-rotation angles, so the ring **stiffens geometrically** as it flattens |
| dynamics | overdamped, drag c = 1 per node, no inertia (the 27 g mass drops out) |
| fingers | two plates close over 0.3 s to the measured squeezed width, then retract at a constant speed |
| contact | penalty force, constant 100 (fixed) |
| output | vertical extent of the ring, times the rest level, compared with the video's short axis |
| not modelled | table friction, out-of-plane motion, tread ridges, true 3D solid, a fingertip force |

The bending force was checked against a numerical gradient of the energy (error 1e-8). 3D view: the ring extruded to a tube with the **measured** geometry:
outer diameter 75.8 mm, **tread width 37 mm** (caliper, measured by the user), radial wall **11.8 mm** (measured from the
standing tyre in `tyre_deformation.mov`, `load_and_hand/`). These dimensions are used for drawing and for converting stiffness to a
modulus, not for the video fit. An earlier version of this README and figure assumed a 25 mm tread width; that was wrong.

### Free parameters (they differ between models)

| model | free parameters | fixed constants |
|---|---|---|
| 1st-order baseline (linear) | tau, rest level (2) | none |
| ring (nonlinear) | **kb** (bending stiffness / viscosity), **release speed**, **rest level** (3) | 16 nodes, stretch 100, contact 100, 0.3 s closing time, R = 0.5 |
| ring + depth law | kb, rest, a, b (4) | same |

The ring has **no recovery-time parameter**. Its recovery time comes from the geometry and depends on squeeze depth, while the
linear model's tau is one constant for every event.

## 7. Limits

* No force measurement in these clips. With an overdamped tyre only the ratio of stiffness to damping is identifiable from them.
  Absolute units come from the weight test (`load_and_hand/`).
* The hand's release is one fitted shared speed. It could not be measured (`load_and_hand/`).
* Shallow squeezes recover more slowly than either model predicts.
* Small data: 12 usable events from 4 clips by one person with one tyre.
* Event exclusions (flick < 0.3 s, hand still gripping) are judgement calls. Numbers with and without them are both shown.
* Width is the minor axis of a rotated rectangle around the dark blob, which can include some shadow and finger.

## 8. Files

```
code/tyre_sig2.py        video -> width signal -> tyreN_sig_*.npz (steps 1 and 2)
code/tyre_fit2.py        events, linear / 2nd-order / 2-tau models, train/test/LOCO (table 5a)
code/tyre_ring.py        the ring simulator
code/tyre_ring_fit.py    ring vs first-order, S2 events, plot of held-out events (table 5b)
code/tyre_ring2.py       adds the depth law and the weight calibration (results/tyre_ring2.json)
code/tyre_ring_fast.py   Numba ring simulator (identical to tyre_ring.py, 180x faster)
code/tyre_models_fast.py all models, one protocol, parallel LOCO + bootstrap (results/tyre_models_fast.json)
code/tyre_ring3d.py       coarse 3D stack of rings with the stepped wall (section 9); tyre_ring3d_fit.py fits it
code/tyre_no_release_param.py   release-speed ablation (5e)
code/make_demo.py        builds figures/demo.mp4 and demo.gif
code/make_3d_figure.py   held-out overlay and 3D tube with the measured geometry
code/tyre_fit.py         FIRST attempt (2nd-order only, 8 test events, error 0.096), kept for the history
code/diag_per_event_tau.py   diagnostic only
code/superseded/         original extraction (tyre_sig.py) and the slow NumPy strain-viscosity run (tyre_ring3.py)
load_and_hand/           load test, Young's modulus, hand-release attempts (own README, code, data, figures)
data/tyreN_sig_*.npz     per-frame width signals (time, minor, major, area)
results/*.json           numbers behind the tables
figures/                 width signals with squeezes, held-out overlays, 3D ring frames
run.sh                   reproduce (python with numpy, scipy, opencv, matplotlib)
```

Earlier tyre work from an older clip set (`../tyre_*.py`, `../tyre_*.json`, `../tyre_videos/`) is left where it was. Its classical
fits scored about 0.06 and are not comparable.

## 9. Coarse 3D model with the real cross-section (`code/tyre_ring3d.py`, `code/tyre_ring3d_fit.py`, `results/tyre_ring3d.json`)

**What the videos show about the shape** (`figures/geometry/`): the tyre is not a uniform ring. Top view (`tyre_1`), oblique (`tyre_5`) and face-on (`tyre_deformation`)
show a **step inside** (an inner ledge, so the wall is thicker on one side of the tread), and the side views (`tyre_6`, `tyre_7`) show **two rows of lugs**, 37 mm wide in total.
The through-hole is 0.65 to 0.70 of the outer diameter, which is the thick-lip wall, 11.8 mm. The thinner wall of the main zone could not be measured at this resolution, so it is a sensitivity value.

**Model.** A stack of 5 rings along the 37 mm tread joined by 3D axial springs; two slices have the lip wall, three the main wall; bending stiffness per slice scales with wall^3.
Plates press the full width: the squeezes reach about 0.5 of the width, and with partial contact the top-view silhouette cannot narrow below 0.81, so the contact spans the tread.
Not modelled: out-of-plane bending, lugs, friction, a true solid wall.

| main-zone wall | train | held-out | LOCO | kb | speed | rest |
|---|---|---|---|---|---|---|
| 2D ring A (reference) | 0.0250 | 0.0398 | 0.0341 | 1.74 | 2.72 | 0.979 |
| 3D stack, uniform 11.8 mm (must equal the 2D ring) | 0.0250 | 0.0398 | 0.0340 | 1.75 | 2.72 | 0.979 |
| 3D stack, main wall 9.5 mm | 0.0250 | 0.0397 | 0.0341 | 1.75 | 2.72 | 0.979 |
| 3D stack, main wall 7.0 mm | 0.0251 | 0.0392 | 0.0337 | 1.80 | 2.74 | 0.974 |

* With a uniform wall the 3D code reproduces the 2D ring, so the implementation is verified.
* **The wall profile cannot be identified from the squeeze clips** (0.0392 to 0.0398).
* **It changes the units.** E = 12 B / sum(b_i t_i^3), lip zone 40 % of the width (an estimate): main wall 11.8 mm gives 0.49 MPa (B = 2.5e-3 N m^2) or 1.11 MPa (B = 5.6e-3); 9.5 mm gives 0.69 or 1.55 MPa; 7.0 mm gives 0.94 or 2.10 MPa.
  So the modulus is "a few tenths to about 2 MPa": 2.2x from the unresolved bending stiffness (the 16-node and 48-node rings disagree), 1.9x from the unmeasured wall.
  A caliper or cut section on the thin wall and a second weight in the load test would close it.

## 10. Exploratory work that did not give a result
* **Tongs clip** (`tyre_with_obj.mov`, 240 fps): rapid repeated pinches with holds under 0.1 s, and the tongs, shadow and tyre merge in one mask. Jaw tracking by background subtraction failed
  (the camera and lighting shift, so the whole marble lights up). The jaws are thin, so their position would be the plate position; it needs a stabilised or higher-contrast recording.
* **Red pencil pouch** (several clips): the pouch is held, rotates and is covered by the hand, and the tyre fits do not transfer (RMSE 0.05 to 0.3). Two clean events returned to 1.02 to 1.03 times the starting width, consistent with no residual set but not a measurement.
* **Weight-lift recovery** (`load_and_hand/`): the tyre rises with the lifted bottle over about 0.33 s and is at full height at separation, so it only shows recovery is faster than that ramp.

## 11. Final results (the flick-excluded set is the final result)

All numbers are RMSE of width / resting width on held-out clips; null = "stays squeezed". The flick rule (a squeeze held under 0.3 s is a flick, not press-hold-release) was written down from the interaction before comparing models;
most of the drop between the two rows is from that exclusion, not from a better model.

| event set | null | first-order + rest (tau 0.143 s) | ring (nonlinear) | LOCO, first-order / ring |
|---|---|---|---|---|
| S1: flicks included (8 train / 7 test) | 0.358 | 0.047 | not run on S1 | 0.045 / n.a. |
| **S2: flicks excluded, final (6 train / 6 test)** | **0.343** | **0.036** | **0.040** | **0.032 / 0.034** |
