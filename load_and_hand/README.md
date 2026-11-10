# Experiment 1, part B: absolute stiffness, Young's modulus and the hand's release speed

Part A of this experiment (`../README.md`) fits the tyre's recovery from squeeze videos. Two things it cannot give:

1. **Absolute stiffness.** A video shows displacement, not force. With an overdamped tyre only the ratio of stiffness to
   damping can be read from the recovery curve.
2. **The hand's release.** The tyre cannot recover faster than the fingers let go, and the finger motion was not measured.

This folder records what we tried for each, including what failed. The model, data and split are described in `../README.md`.

---------------------------------------------------------------------------------------------------

## A. Load test: stiffness and Young's modulus (worked, with caveats)

**Clip:** `~/Documents/tyre_deformation.mov`: a **240 fps capture** in a 30 fps container (1872 frames = **7.8 s of real time**,
not 62 s; the flicker is at 0.4165 cycles/frame, 150 times the median level). An earlier version of this README treated it as
30 fps. All times below are real time (playback / 8). The tyre stands on its edge, filmed side-on. A **333 g bottle** (never held,
as stated) is placed on top at about 1.1 s, rests there for **about 6 s**, until about 7.1 s, then is lifted off. The tyre weighs
**27 g**. The 10 mm marker is on a tray at a different depth, so it was **not** used for scale; the tyre's own 75.8 mm diameter
(from the top-view clips) is.

**Measurement** (`code/def_track.py`): the bottle is found by its blue colour; its lowest edge is the top of the tyre while
loaded. The floor row is taken from the marker (to follow camera drift) plus a constant measured in the unloaded frames. Height
is normalised by the unloaded height (120.7 px at 360 px width).

| time | tyre height / unloaded |
|---|---|
| 0 to 0.75 s, unloaded | 1.00 |
| 1.75 to 2.5 s, loaded | 0.92 |
| 3.1 to 3.75 s | 0.88 |
| 5.0 to 5.6 s | 0.92 |
| 6.25 to 6.9 s | 0.91 |
| 7.1 to 7.5 s, unloaded again | 1.02 |

Steady value **0.90 +/- 0.02** (the +/- is measurement noise on the bottle edge). Sag is about 7.6 mm (range 6 to 9 mm).

**Derived numbers**

| quantity | value | how |
|---|---|---|
| load | 3.27 N | 0.333 kg x 9.81 (tyre's own weight adds under 0.2 N) |
| static stiffness | **0.43 N/mm** (0.36 to 0.54) | load / sag |
| effective damping of the squeezing mode | about **60 N s/m** | stiffness x recovery time (430 N/m x 0.14 s). Order of magnitude only: squeeze recoveries are at about 50% strain, the load test at about 10% |
| bending stiffness EI, thin-ring formula | 2.1e-3 N m^2 | sag = 0.149 F R^3 / EI with R = 32 mm (centre-line radius = (75.8 - 11.8) / 2) |
| bending stiffness EI, **from our ring model** | **5.6e-3 N m^2** (4.7 to 7.0e-3) | the ring model squeezed with 3.27 N must sag 7.6 mm (`code/tyre_ring2.py`) |
| Young's modulus (ring-model EI) | **about 1.1 MPa** (0.9 to 1.4) | E = 12 EI / (b t^3) with **b = 37 mm** (caliper, user-measured) and **t = 11.8 mm** (measured, section A2). I = b t^3 / 12 = 5.07e-9 m^4 |
| Young's modulus (thin-ring EI) | about 0.4 MPa | same formula with EI = 2.1e-3 |

The two EI estimates differ by a factor of about 2.7. The thin-ring formula assumes a thin wall and a point load, while our tyre
is thick (centre-line radius about 2.7 times the wall) and was loaded by a flat bottle base. The ring model includes the flat
contact and the large deflection (10% sag), so it is the one to quote: **a modulus of about 1 MPa, good to a factor of 2**. It is
an *effective* modulus, because the tread lugs add stiffness and the wall is not a uniform solid. (An earlier version of this table
said 1 to 5 MPa because it assumed b = 25 mm and t = 8 to 14 mm; with the measured b and t the range is much narrower.)

**Physical units for the ring model.** With EI fixed from the static calibration, the fitted `kb` fixes the viscosity:
bending stiffness / viscosity = 2.2e-6 m^4/s, i.e. drag about 2.5e3 N s/m^2 per unit length (37.7 N s/m per node). The release
speed 2.72 diameters/s is **about 206 mm/s** (75.8 mm x 2.72). The recovery time constant is about 0.14 s.

**Sanity check:** a linear estimate of the finger force at the deepest squeeze (about 38 mm of travel at 430 N/m) is roughly
10 to 20 N, an ordinary fingertip pinch. Geometric stiffening means the true figure is probably a little higher.

**Limits of the load test**

* **One load point.** No force-deflection curve, so linearity is untested.
* The held height wanders between 0.88 and 0.92. With no hand on the bottle this is most likely measurement noise plus a little
  creep and rocking. A clean creep curve (a second time constant under constant load) cannot be extracted.
* Tread width (37 mm, caliper) and wall thickness (11.8 mm, from video) are now measured, but the tread lugs make the real wall
  irregular, so the modulus is an effective value.
* After the bottle leaves, the tyre reads 1.02, probably because it settled or rotated slightly, not a real over-recovery.
* The scale comes from the top-view clips (75.8 mm), not from this clip's marker, which sits on a tray at a different depth.

**What would tighten it:** the same test with 2 or 3 more known masses, which would also show whether the tyre is linear.

### A2. Geometry measured from the side-view clips

| quantity | value | how |
|---|---|---|
| true frame rate of `tyre_deformation`, `tyre_5`, `tyre_6`, `tyre_7` | **240 fps** | 100 Hz light flicker at 0.4165 to 0.4192 cycles/frame, peak 80 to 210 times the median level |
| tyre width in `tyre_6` (marker check) | **76.6 mm** | marker side 53.8 px = 10 mm gives 5.38 px/mm; the tyre is 412 px wide. Agrees with 75.8 mm from the top-view clips, so the two scales are consistent (tyre and marker are at a similar depth in this clip) |
| tread width in `tyre_6` | about 40 mm in the image | front-face height 224 px / 5.38 px/mm, which includes shadow and a slight viewing angle. Consistent with the caliper value of **37 mm**, which is what we use |
| radial wall thickness | **11.8 mm** | brightness profile across the standing tyre in `tyre_deformation.mov` (23 unloaded frames): outer 246 px, hole 170 px, walls 38 px and 39 px, so wall = 0.156 x outer diameter = 11.8 mm; hole diameter about 52 mm |

The wall measurement includes the tread lugs, so the structural thickness may be a little smaller. A first attempt to find the
hole by segmenting the dark rubber failed (it latched onto the shadow), so the profile method above was used instead.

## B. Hand release speed (could not be measured; fitted instead)

**1. Skin-colour mask** (`code/tyre_hand.py`, `figures/skin_mask_vs_tyre_width.png`). The skin fraction of the frame tracks the
hand entering and leaving, but the hand takes seconds to leave the frame while the tyre recovers in a fraction of a second, so it
says nothing about when the fingertips let go. **Not usable.**

**2. MediaPipe hand landmarks** (`code/hand_track.py`, Hand Landmarker model, every 2nd frame). Coverage inside the release
windows (+/- 0.4 s around each release):

| clip | events | frames with a hand detected in the window |
|---|---|---|
| tyre_1 | 4 | 13/71, 20/96, 0/96, 8/96 |
| tyre_2 | 4 | 8/77, 37/96, 75/96, 76/97 |
| tyre_3 | 4 | 28/88, 14/96, 22/96, 1/96 |
| tyre_4 | 4 | 84/96, 93/97, 42/97, 0/96 |

Coverage ranges from 0 to 97%, and the thumb-to-index distance jumps from about 265 px to 16 px within a few frames. Some values
exceed the tyre's own width while it is being pinched. A hand curled around the rubber hides the fingertips, and the landmark
labels become unreliable. **Not usable as a measured input.**

**3. Fit the release speed instead** (the ring model's finger plates retract at a fitted speed):

* Shared speed, fitted on `tyre_1` and `tyre_2`: **2.72 diameters/s, about 206 mm/s**.
* **Per-event release speed**, with the ring's stiffness fixed (`code/per_event_release_speed.py`):

| depth of squeeze | fitted release speed (diam/s) | error with shared speed -> own speed |
|---|---|---|
| deep, 0.45 to 0.55 (8 events) | 1.9 to 3.3 for five events; three more are very fast (8.4, 9.8 and 264, the last pegged at the fit bound) | similar |
| **shallow, 0.30 to 0.41 (4 events)** | **0.56, 0.78, 0.85, 2.06** | **0.048 -> 0.024, 0.036 -> 0.015, 0.046 -> 0.029, 0.026 -> 0.025** |

Mean error across the 12 events 0.032 -> 0.025; correlation between depth and log release speed 0.54. This hints that **the hand
lets go more gently on shallow squeezes**, which would explain why they recover more slowly. It is a fitted explanation from 12
events, not a measurement, and a strain-dependent rubber would look the same.

* **Release speed as a function of depth** (`speed = exp(a + b*depth)`, shared across events): held-out 0.039 vs 0.040 for a
  single shared speed, LOCO 0.031 vs 0.034. **No meaningful gain**: the fitted slope is weak (b = 0.69), because it is dominated
  by the deep events.

Takeaway: knowing the true release speed would reduce the error on shallow squeezes a lot (to about 0.02), but we have no
reliable way to measure it from this video. A depth-vs-speed law does not capture it. A direct measurement (a marker on the
fingernail, a second camera, or a force plate) would be needed.

## C. Related work checked

* **PhysDreamer** ([arXiv 2404.13026](https://arxiv.org/abs/2404.13026)): estimates a spatially varying Young's modulus from a
  *generated* video, using a video-diffusion prior and a differentiable simulation. Different route from ours (we fit real video
  and calibrate with a weight). I read the summary pages, not the full paper.
* **Hybrid Particle-Voxel Dynamics for Multimodal 3D World Simulation** ([arXiv 2611.04492](https://arxiv.org/abs/2611.04492)):
  an action-conditioned learned world model with tactile signals. Read the abstract only (full PDF too large to fetch). It needs
  robot data, known actions and tactile sensing, none of which we have, so it was not applied. Relevant to us as framing: its
  known **action** input is our missing hand release, and its **tactile** channel is our missing force.

## D. Files

```
code/def_track.py                 load-test video -> height curve (bottle edge vs floor)
code/tyre_hand.py                 skin-colour mask per frame
code/hand_track.py                MediaPipe Hand Landmarker (needs the .task model file, downloaded separately, python 3.12 env)
code/per_event_release_speed.py   per-event release speed with ring stiffness fixed (run from ../code)
data/def_track.npy, def_height.npy   load-test measurements
data/hand_tyre_*.npy, hand_tips_*.pkl   skin fraction and landmark detections
figures/                          load-test signals and height, skin mask vs tyre width
```

The static calibration and the physical-unit conversion are computed in `../code/tyre_ring2.py` and stored in
`../results/tyre_ring2.json` (`phys` block).
