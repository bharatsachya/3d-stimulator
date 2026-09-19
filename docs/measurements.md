# Measurements

Numbers as they are taken, so the README's table is transcribed rather than
reconstructed. Every row records what was measured, on what, with which settings.

The brief requires timings from the demonstrated test environment, which is the
EC2 `m7i-flex.large`. Laptop figures are kept alongside, clearly labelled, because
the comparison between them turned out to be informative in its own right.

---

## Probe: the per-frame cost floor

`tools/probe.py` measures the three stages that run on every frame and that no
design choice removes: decode, ORB detect+compute, and brute-force matching with
the ratio test. It excludes PnP, triangulation and bundle adjustment.

### 17 Sep 2026 — dev laptop (NOT the target environment)

Apple Silicon, macOS 26.4.1, OpenCV 4.14.0, Python 3.13.
Clip: `samples/synth_dolly.mp4`, 300 frames @ 30fps → 100 processed @ 10fps,
640px working width, 1000 features, Lowe ratio 0.75.

| stage | total ms | ms/frame | % of wall |
|---|---|---|---|
| decode | 113.9 | 1.14 | 6.3 |
| orb | 1550.9 | 15.51 | 85.6 |
| match | 144.7 | 1.45 | 8.0 |
| **wall** | **1812.2** | **18.12** | 100 |

Budget is 100 ms/frame at 10fps processing. Floor is 18.1 ms/frame, leaving
~82 ms/frame for tracking, triangulation and BA.

**ORB is 85% of the floor.** Any tuning of the per-frame cost is a conversation
about ORB — feature count and working resolution — not about decode or matching.

### Thread scaling — the finding that matters for a 2 vCPU target

Same clip and settings, varying OpenCV's thread count:

| threads | orb ms/frame | wall ms/frame |
|---|---|---|
| 1 | 20.64 | 29.85 |
| 2 | 15.21 | 17.67 |
| 4 | 15.70 | 18.26 |
| 8 | 16.22 | 19.16 |

**This workload stops scaling after 2 threads.** Going from 1 to 2 buys 40%;
going from 2 to 8 buys nothing and is very slightly worse, presumably scheduling
overhead on work too small to divide.

That is a useful result for the assignment's framing. The 2 vCPU constraint is
not a handicap this pipeline is straining against — it is already at the point
where more cores stop helping. Which is the client's own thesis about efficient
hardware, arrived at by measurement rather than assertion.

### Caveat on thread reporting

`cv2.setNumThreads()` changes execution time on this build while
`cv2.getNumThreads()` keeps returning 8 regardless. The timings above are real —
1 thread is measurably slower — but the reported thread count is not. The probe
therefore records `threads_requested` separately, and that is the field to quote.

---

## Probe on the TARGET instance — the submission numbers

EC2 `m7i-flex.large` in eu-north-1a: Intel Xeon Platinum 8488C (Sapphire
Rapids), 2 vCPU (**1 physical core, 2 threads**), 7.8 GB RAM, Ubuntu 24.04,
Linux 6.17, OpenCV 4.14.0, Python 3.12.3.

Clip: `tum_fr1_xyz.mp4`, 798 frames @ 31.15 fps → 266 processed @ 10 fps,
640px working width, 1000 features, Lowe ratio 0.75.

| stage | total ms | ms/frame | % of wall |
|---|---|---|---|
| decode | 455.1 | 1.71 | 13.3 |
| orb | 1452.6 | 5.46 | 42.5 |
| match | 1504.2 | 5.66 | 44.0 |
| **wall** | **3420.3** | **12.86** | 100 |

**12.9 ms/frame against a 100 ms budget — 87 ms/frame of headroom** for PnP,
triangulation and bundle adjustment.

### Stability under sustained load

`m7i-flex` is a flex instance, so sustained full-rate CPU was worth checking
before trusting any of the above. Six consecutive probe runs back to back:

| run | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| ms/frame | 13.2 | 13.1 | 13.2 | 13.1 | 13.1 | 13.0 |

Flat, with 0% CPU steal throughout. Timings from this instance are trustworthy
for a job of this length. Worth re-checking during the longer parameter sweep,
where sustained demand is higher.

### The finding that changes a previous conclusion

Compare the same clip on both machines:

| stage | laptop (M-series, 8 threads) | EC2 (Xeon 8488C, 2 threads) |
|---|---|---|
| decode | 0.51 ms/frame (4%) | 1.71 ms/frame (13%) |
| orb | **9.50 ms/frame (82%)** | **5.46 ms/frame (43%)** |
| match | **1.49 ms/frame (13%)** | **5.66 ms/frame (44%)** |
| wall | 11.52 ms/frame | 12.86 ms/frame |

The totals are within 12% of each other, which is a coincidence — the
composition is completely different. **ORB is faster on the server** (5.46 vs
9.50) and **matching is nearly 4x slower** (5.66 vs 1.49).

This retracts a conclusion drawn earlier from laptop numbers alone: *"ORB is 85%
of the floor, so tuning per-frame cost is a conversation about ORB."* That is
true on Apple Silicon and false on the deployment target, where matching is
co-equal with ORB. Any tuning work has to treat the matcher as a first-class
cost — reducing the feature count helps twice over, since brute-force matching
is quadratic in descriptor count while ORB is roughly linear.

It is also the clearest possible argument for CLAUDE.md's rule that timings come
from the target instance and never from a dev laptop. The headline number would
have been roughly right; the engineering conclusion drawn from it was wrong.

### End-to-end through nginx on the instance

| check | result |
|---|---|
| POST /api/jobs, 17 MB upload | **202 in 47 ms** |
| poll → done | 3 polls, progress 0.03 → 0.55 → 1.00 |
| 27 MB upload | 413 rejected |
| unknown job id | 404 |
| vendored three.module.js | 200, 603 KB |

(Processing is still the stage-0 stub; what this verifies is the transport path.)

### Public round trip, laptop to Stockholm

| check | result |
|---|---|
| POST 17 MB over the public internet | **43.2 s** |
| 202 returned once bytes arrived | immediate |
| poll -> done | 2 polls |
| GET / | 200 in 0.43 s |

That 43 seconds is **upload time, not processing time** — roughly 3.2 Mbit/s
upstream from this connection. The "202 in milliseconds" property holds for the
server: it responds the moment the body has arrived. But a reviewer watching the
browser experiences the whole 43 seconds.

Two consequences worth acting on:

1. **It vindicates the 25 MB cap and the "short clips" hint.** A 25 MB upload on
   this connection is over a minute of waiting before any work starts. The cap is
   a UX decision at least as much as a memory one.

2. **The upload needs a real progress indicator.** `fetch()` cannot report upload
   progress; only `XMLHttpRequest` exposes `upload.onprogress`. Right now the bar
   sits at "uploading" with no movement for the entire transfer, which is the
   worst possible moment to look frozen. Fix this when the viewer is built.

---

## TUM RGB-D: real footage with real ground truth

`tools/tum_to_video.py` converts a TUM sequence into an mp4 plus ground truth in
our own format. It encodes to video deliberately rather than reading the PNG
folder directly, so the benchmark travels the identical code path as a reviewer
dragging a file onto the page — same decode cost, same container overhead.

TUM supplies what neither phone footage nor the synthetic scene can supply
together: real imagery with externally measured 100 Hz motion-capture poses.

### `freiburg3_nostructure_notexture_far` — the low-texture failure mode

This sequence is from TUM's *Testing and Debugging* category. It is a blank
wall, deliberately constructed to defeat SLAM; ORB-SLAM2's paper reports failure
on it. It is used here as the low-texture failure case CLAUDE.md requires, and
**not** as a reconstruction benchmark.

474 frames, 15.9s, 31.1 fps, 640x480.

| measurement | value |
|---|---|
| ORB keypoints (1000 requested) | median **12** |
| matches surviving the ratio test | median **4** |
| frames 0 vs 30, after ratio test | **2** |
| image standard deviation | 20.2 (near-uniform grey) |

Initialization needs 50 essential-matrix inliers. There are four correspondences
to work with. This cannot and must not produce a reconstruction — the required
behaviour is a clear diagnosis, which is what `INITIALIZATION_FAILED` says.

### A timing trap this sequence exposes

The probe on this clip reports a **4.6 ms/frame** floor, against 18.1 ms/frame
for the synthetic one. It is not faster in any useful sense: ORB costs 4.05
ms/frame here versus 15.51 ms because there is nothing in the image to detect
or describe.

So per-frame cost is a function of scene content, not only of resolution and
feature budget. **A single headline timing number would be misleading**, and the
README has to quote the settings and the clip together, with the richly textured
case as the honest worst case.

### The intrinsics experiment this enables

TUM publishes measured intrinsics: fr3 is fx=535.4, fy=539.2, cx=320.1, cy=247.6.
Our heuristic guesses 0.9 x width = 576 for these 640x480 frames — **+7.6%**.

Because ground truth exists, that guess can be costed rather than merely
disclosed: run the same clip with measured K and with heuristic K, and report
the difference in ATE. That turns "focal length is estimated, not measured" from
a limitation paragraph into a measured quantity.

### `freiburg1_xyz` — the reconstruction benchmark

The counterpart to the blank wall: a textured desk with the camera translating
along each axis in turn. TUM's standard easy case, chosen as the smallest
download (427 MB) that actually reconstructs.

798 frames, 25.6 s, 31.2 fps, 640x480 → 16.6 MB as mp4, which fits under the
25 MB upload cap and the 30 s duration limit. It can be dropped onto the
deployed page exactly as a reviewer would.

| clip | ORB keypoints | matches after ratio test | floor ms/frame |
|---|---|---|---|
| `synth_dolly` (synthetic) | 1000 | 628 | 18.1 |
| `tum_fr1_xyz` (real) | 1000 | 546 | **11.5** |
| `tum_nostructure_notexture` (blank wall) | 12 | 4 | 4.6 |

Real footage sits between the synthetic scene and the degenerate one, which is
the ordering you would want: the synthetic texture is denser than reality, and
the blank wall has nothing to find.

### Two-view geometry on real data, measured vs heuristic intrinsics

Frame 0 against increasing gaps, essential matrix + `recoverPose`, compared to
100 Hz motion capture. Baselines are the true metric distance between camera
centres. Both intrinsic sets were run on identical correspondences.

| gap | baseline (m) | inliers | measured K rot° / t-dir° | heuristic K rot° / t-dir° |
|---|---|---|---|---|
| 3 | 0.037 | 531 | 0.80 / 12.4 | 0.89 / 11.2 |
| 6 | 0.072 | 456 | 0.44 / 8.8 | 0.44 / 8.5 |
| 15 | 0.181 | 344 | 6.01 / 32.1 | 0.69 / 9.4 |
| **30** | **0.354** | 156 | **1.47 / 1.75** | 2.09 / 2.80 |
| 60 | 0.103 | 348 | 1.52 / 21.7 | 1.12 / 21.5 |
| 90 | 0.048 | 137 | 1.73 / 35.2 | 1.96 / 63.9 |
| 150 | 0.096 | 317 | 0.58 / 9.4 | 3.83 / 39.6 |
| 240 | 0.157 | 162 | 0.69 / 5.5 | 1.88 / 6.2 |

Two things to take from this, and one thing not to.

**Baseline dominates everything.** The best result by a wide margin is the row
with the longest baseline (0.354 m → 1.75° translation error); the worst are the
rows where the camera happened to return near its starting point, so a large
frame gap still means a tiny baseline. That is the whole justification for the
minimum-parallax gate at initialization: frame distance is not baseline, and
choosing an initialization pair by frame index alone would frequently pick a
degenerate pair.

**An 11.3% focal error does not dominate at this stage.** The heuristic
(0.9 x width = 576) against fr1's measured fx = 517.3 wins some rows and loses
others, all inside the two-view noise. That is worth knowing but not yet worth
concluding from — two-view estimates on short baselines are noisy, and the
question that matters is what the error does to a full trajectory after
map-based tracking and bundle adjustment. Measure it there.

**Do not read these as pipeline accuracy.** They are raw two-view estimates
with no map, no PnP and no BA — the floor the real pipeline should beat, not a
result.

---

## Still to measure

- [ ] **ATE for the full pipeline on `tum_fr1_xyz`**, Sim(3)-aligned. The
      machinery is built and unit-tested (`vslam/align.py`); it needs a
      trajectory to evaluate.
- [ ] **Cost of the focal-length heuristic**, measured as the ATE difference
      between measured K and 0.9 x width on the same clip.
- [ ] Bundle adjustment cost per keyframe — the one stage with real risk of
      eating the budget.
- [ ] The five-parameter sweep: processed fps, working resolution, features per
      frame, BA window size, BA frequency.
- [ ] End-to-end wall clock for a 10s clip, which is what the brief actually asks.

---

# Stage A — diagnosing the frame-186 tracking loss

Tracking died at frame 186 of 798 on TUM fr1_xyz, capping every accuracy figure
at 23% coverage. Four hypotheses were proposed and the pipeline was instrumented
per frame (`tools/diagnose.py`) rather than fixed speculatively.

| hypothesis | verdict | evidence |
|---|---|---|
| H1 cull outpacing triangulation | **refuted** | **0 points culled in the entire run**; 1934 triangulated, 2121 in the map at the end |
| H2 keyframes stop, camera exits the map | **refuted** | keyframe inserted 3 frames before the failure; **1596 points still projecting into view** |
| H3 viewpoint change breaks matching | **supported** | map healthy and in view, match rate collapses from 20-25% of visible points to **5.0%** |
| H4 min-30-inliers too strict | contributory only | inliers fall 246 → 87 → 52 → 44 → 24; relaxing the threshold buys about one frame of a cliff |

Not motion blur: Laplacian variance at the failing frames (196-236) is *higher*
than forty frames earlier (159-220), and ORB still returns its full 1000
features. Ground truth shows the camera accelerating from 0.008 m per step
(frames 162-171) to 0.038-0.041 m per step from 174 onward, and the frames show
a second monitor entering from the right. The scene genuinely changes.

## What was tried and did not work

**Projection-guided matching: implemented, measured, rejected.**

In isolation it looked decisive. At the exact failing frame, with the TRUE pose:

| search | matches |
|---|---|
| global brute force over 1464 map descriptors | 90 |
| projection-guided, 8 px radius | **203** |
| projection-guided, 40 px radius | 180 |
| projection-guided, 80 px radius | 162 |

End to end in the pipeline, using a *predicted* pose, it was consistently worse:

| config | poses | last frame | ATE cm | ms/frame |
|---|---|---|---|---|
| brute force | **59** | **180** | 1.67 | 43.7 |
| projection r=30, ratio 0.75 | 8 | 27 | 0.79 | 21.6 |
| projection r=30, no ratio | 23 | 72 | 0.99 | 25.4 |
| projection r=60, no ratio | 23 | 72 | 1.05 | 26.9 |
| projection r=100, no ratio | 23 | 72 | 1.03 | 25.2 |

Three things this exposed, each worth more than the feature would have been:

1. **The isolated test used the true pose; the pipeline only has a predicted
   one.** Constant-velocity prediction error was measured at a median of 8-23 px,
   p90 up to 31 px, and a maximum of 165 px, so correct matches routinely fell
   outside the search disc.
2. **Radius was never the live variable.** r=30, 60 and 100 give *identical*
   results, because the `k=8` nearest-neighbour cap binds before the radius does.
3. **The raw match counts were never comparable.** The local window held 217 map
   points while brute force reported 238 matches — it assigns several frame
   keypoints to the same map point, while the projection matcher enforces
   one-to-one.

Kept in the code behind a flag, defaulted off, because the mechanism is sound
and would likely win with a better motion model. Shipping it on this evidence
would not be.

**Denser keyframes and a larger local map: measured, no material change.**

| config | poses | last frame | ATE cm |
|---|---|---|---|
| baseline (3 frames / 10% depth / 0.70 ratio, window 8) | 59 | 180 | 1.67 |
| translation trigger 10% → 5% | 59 | 180 | 1.48 |
| translation trigger 10% → 3% | 61 | 186 | 2.00 |
| tracked-ratio trigger 0.70 → 0.85 | 59 | 180 | 1.87 |
| minimum keyframe gap 3 → 2 | 62 | 189 | 1.76 |
| local window 8 → 16 | 59 | 180 | 1.59 |

Every configuration still dies between frames 180 and 189. The loss is not a
tuning problem.

---

# Stage B — the cull

`cull` was the third-largest cost in the pipeline at 4.71 ms/frame, 17% of the
budget. Sweeping its threshold showed something better than a speed problem:

| max error px | culled | map points | ATE cm | ms/frame |
|---|---|---|---|---|
| **5.0 (shipped default)** | **0** | 1827 | 1.67 | 33.3 |
| disabled entirely | 0 | 1827 | 1.67 | 32.2 |
| 3.0 | 8 | 1880 | 1.67 | 22.4 |
| 1.0 | 1028 | 1601 | 1.79 | 24.8 |
| 0.5 | 2329 | 547 | 1.65 | 17.8 |
| 5.0, min observations 3 | 3104 | 93 | 2.56 | 17.9 |

**At its shipped threshold the cull removed zero points and was byte-identical
to disabling it.** The reason is structural rather than accidental: triangulation
only admits points whose reprojection error is already below 4 px, so a 5 px cull
cannot fire on anything triangulation let through. It becomes meaningful only
once bundle adjustment starts moving points after creation.

The scan was also O(whole map) on every keyframe. Restricting it to points
observed by recent keyframes:

| cull scope | ATE cm | cull ms/frame | total ms/frame |
|---|---|---|---|
| whole map | 1.67 | 7.26 | 45.9 |
| local window of 8 keyframes | 1.67 | **5.32** | **25.9** |

Identical accuracy, 27% less cull cost. The threshold itself is deliberately
**not** retuned here: 0.5 px looks best on this clip, and fitting a threshold to
one sequence is the error this project has avoided elsewhere. It is revisited
across the full benchmark once bundle adjustment gives it something to do.

---

# Stage D1 — surviving tracking loss

Stage A established that the loss is the camera exploring, not a defect to be
prevented. So the pipeline stopped terminating on a failed frame: it skips the
frame without emitting a pose, keeps the map, and tries the next one. A full
relocalization against the entire map runs as a second line of defence.

| variant | poses | coverage | reloc | skipped | path m | ATE cm | % of path | ms/frame |
|---|---|---|---|---|---|---|---|---|
| terminate on first loss | 59 | 22.7% | 0 | 0 | 1.941 | 1.67 | 0.86 | 31.9 |
| **skip + relocalize** | **234** | **99.7%** | 0 | 30 | **7.069** | 3.59 | **0.51** | 24.4 |
| skip only, relocalize ablated | 234 | 99.7% | 0 | 30 | 7.069 | 3.59 | 0.51 | 24.1 |

Coverage goes from 23% of the sequence to **99.7%**, over 3.6x the ground-truth
path. Absolute ATE rises from 1.67 cm to 3.59 cm, which is expected over a much
longer trajectory — but error **as a fraction of path length improves**, from
0.86% to 0.51%.

**An honest note on what did the work.** `relocalize()` never once succeeded on
this sequence; the ablation row proves it, being identical to the row above it.
The entire gain comes from the far simpler change of not giving up after a
single failed frame. Relocalization is retained because it should matter on
sequences where the camera is lost for longer, but **it is unvalidated and is
not what produced this result.**

---

# Stage C — bundle adjustment

Sliding window over the last 5 keyframes, optimising poses **and** points
jointly, `scipy.optimize.least_squares` with `method="trf"`, a sparse Jacobian
supplied via `jac_sparsity`, Huber loss at 2 px, and the first keyframe in the
window held fixed to remove gauge freedom.

## The first version was a no-op, and nearly shipped as one

It reduced reprojection error by 0.2% and recovered 0.1% of a deliberately
injected perturbation. It terminated after **two function evaluations** with
status 3 (`xtol` satisfied) while the gradient norm was still **3143** — nowhere
near a minimum.

The cause was variable scaling. The parameter vector mixes rotation vectors in
radians, translations, and 3D point coordinates, which respond at completely
different magnitudes. Across 3549 parameters the *relative* step looked
negligible even when the absolute step was not, so `trf` concluded it had
converged before it had started. Recovering a known 0.05-unit perturbation of
the map points:

| tolerances | x_scale | recovered |
|---|---|---|
| xtol 1e-4 | 1.0 | **0.1%** (the shipped configuration) |
| xtol 1e-10 | 1.0 | 1.7% |
| xtol 1e-10 | **"jac"** | **86.6%** |

`x_scale="jac"` rescales each variable by its Jacobian column, and is the
standard remedy for a badly-scaled bundle adjustment.

### How nearly this was missed

Three earlier experiments varying `diff_step`, tolerances and `x_scale` all
returned **byte-identical** results, which read as strong evidence that none of
them mattered. They were no-ops: `ba.py` does
`from scipy.optimize import least_squares`, so patching
`scipy.optimize.least_squares` never touched the imported name.

The lesson is worth more than the bug. *Identical numbers across varied inputs
are evidence of a broken experiment, not of an insensitive system.* A parameter
sweep that returns the same answer for every setting should be disbelieved
before it is reported.

## What BA is worth, measured end to end

| config | poses | path m | ATE cm | % of path | ms/frame | BA ms/frame |
|---|---|---|---|---|---|---|
| no BA | 234 | 7.069 | 3.59 | 0.51 | 24.8 | 0 |
| nfev 20, every keyframe | 238 | 7.161 | 3.43 | 0.48 | 43.6 | 20.1 |
| **nfev 50, every 2nd keyframe** | 235 | 7.142 | **3.22** | **0.45** | 70.8 | 46.0 |
| nfev 50, every keyframe | 243 | 7.415 | 4.19 | 0.57 | **106.0** | 83.9 |

Every second keyframe is the shipped default. Running BA on *every* keyframe
costs 106 ms/frame — over the 100 ms budget — and scores **worse**. More
optimisation is not monotonically better when it competes for time with the
frames that feed it.

## An honest limit: BA does not fix drift

Applied as a single global pass over all 52 keyframes on a fixed trajectory,
before the scaling bug was found, BA moved ATE by **nothing** (3.77 cm before and
after) while reprojection error fell 0.14%. Even working correctly the effect is
modest, and the reason is structural rather than a tuning failure:

**BA minimises reprojection error, and drift is nearly invisible to reprojection
error.** A slowly accumulating scale or pose error remains perfectly consistent
with every observation that produced it — the reconstruction can bend or shrink
and the images still explain it. Only an observation linking distant parts of the
trajectory introduces a constraint that drift violates, which is loop closure.

## Result on the target instance

EC2 `m7i-flex.large`, TUM fr1_xyz, with relocalization and BA:

| metric | value |
|---|---|
| coverage | **99.7%** of the sequence, tracking never lost |
| poses | 241 evaluated |
| keyframes / map points | 54 / 5366 |
| mean reprojection | 0.897 px |
| ground-truth path | 7.239 m |
| **ATE RMSE** | **2.17 cm** |
| ATE median / max | 1.63 / 7.55 cm |
| **RMSE / path length** | **0.30%** |
| wall | 66.5 ms/frame against a 100 ms budget |

For comparison with where Stage A began — 1.56 cm over 1.980 m at 23.4% coverage,
0.79% of path — this is **4.3x the coverage at 2.6x better relative accuracy**.

| stage | ms/frame | % of wall |
|---|---|---|
| bundle_adjustment | 39.47 | 59.4 |
| match_map | 8.50 | 12.8 |
| orb | 6.48 | 9.7 |
| cull | 3.86 | 5.8 |
| decode | 1.98 | 3.0 |
| relocalize | 1.62 | 2.4 |
| pnp | 1.42 | 2.1 |
| triangulate | 1.28 | 1.9 |

---

# Stage G (first pass) — the benchmark that exposed the real limitation

Running across five TUM sequences instead of one changed the picture completely.
With skip-and-retry but no re-initialization:

| sequence | frames | coverage | ATE cm | % of path |
|---|---|---|---|---|
| fr1_xyz | 798 | **99.7%** | 2.17 | 0.30 |
| fr1_desk | 613 | **6.0%** | 3.26 | 7.99 |
| fr1_desk2 | 640 | **5.8%** | 3.44 | 17.11 |
| fr1_room | 1362 | **7.6%** | 3.31 | 5.66 |
| fr2_desk | 2965 | **11.0%** | 8.68 | 3.51 |

The 99.7% result was a property of **one sequence**, not of the system. Every
other sequence terminated after hitting the same 60-frame lost cap.

## Why, and why no parameter fixed it

A death spiral: the map can only grow from tracked frames, so once tracking is
lost the map freezes. A camera that explores *away* from its initial map can
never re-acquire it, and waiting is futile. fr1_xyz survives only because its
camera oscillates inside a volume roughly 0.94 m across and keeps returning.

Confirmed by sweeping the obvious knobs across three sequences:

| config | fr1_desk coverage | fr1_room coverage | fr1_xyz % of path |
|---|---|---|---|
| baseline (min 30 inliers) | 6.0% | 7.6% | 0.30 |
| min 20 inliers | 6.5% | 7.6% | 0.27 |
| min 15 inliers | 6.5% | 7.6% | 0.36 |
| min 20 + keyframes at 5% depth | 6.5% | 7.6% | 0.37 |
| min 15 + 5% + ratio 0.85 | 6.5% | 7.6% | 0.37 |

Nothing moves. This is not a tuning problem.

## The fix: re-initialize instead of waiting

After a short wait, start a fresh map from the current frames and carry on.

| sequence | before | after |
|---|---|---|
| fr1_desk coverage | 6.0% | **100.0%** |
| fr1_desk poses | 13 | 85 |
| fr1_desk error | 7.99% of path | **0.88%** |

Each new segment carries its **own arbitrary scale and origin** — monocular
scale is unobservable and nothing links a new segment's units to the old ones.
Measured directly on fr1_xyz, the recovered scale factor per segment was 0.107,
0.055, 0.026 and 0.067: four segments, four different units.

Evaluation therefore aligns **each segment separately** and reports the
pose-weighted RMS across them, with the segment count alongside. A trajectory in
twenty pieces is a worse result than the same error in one piece, and the reader
must be able to see that. Aligning the whole thing with one similarity transform
would charge the system for a discontinuity it cannot observe.

The trade is real: on fr1_xyz, where waiting *does* work, re-initialization
costs accuracy (0.30% → 0.42%) by fragmenting a trajectory that would have
recovered intact. It is on by default because the benchmark, not one sequence,
decides.

---

# Stage E — loop closure

## Detection works

Bag-of-words over a 256-word vocabulary built from the sequence itself with
`cv2.kmeans`, keyframes indexed by tf-idf-weighted histogram, queried by cosine
similarity, every candidate geometrically verified by PnP before acceptance.

On fr1_xyz: **102 candidates, 78 verified**, in 2.8 s. Rejected candidates
failed on inlier count (24-30 against the required 40) and are logged rather
than silently dropped.

## The pose graph folded the trajectory, and why

The first version asserted that the two ends of a closure occupy the **same
position**. That is not what revisiting a place means — the camera returns
*near* somewhere it has been, from a different spot and angle.

The optimiser satisfied all 78 constraints exactly, drove the residual to
**0.0000**, and collapsed the map: ATE went from 3.22 cm to 9.82 cm.

*A residual of exactly zero against 78 over-determined constraints is evidence
that the constraints are vacuous, not evidence of convergence.*

The correct constraint was already being computed and thrown away. Geometric
verification localises the query keyframe by PnP against the match keyframe's
**old** map points, which predate the drift — so its recovered centre is a
drift-corrected estimate of where that keyframe belongs. Each closure now
contributes a unary pull toward that position. ATE 3.22 → 3.60 cm: no longer
destructive, and no longer an improvement either.

## Honest status

Loop closure is implemented and **not yet demonstrated to help**. fr1_xyz is the
wrong sequence to judge it on — it oscillates inside a 0.94 m box with error
already at 0.45% of path, so there is no accumulated drift for a closure to
correct. The claim in requirement 4 that loop closure corrects drift is
well-founded in the literature and is *not* evidenced by any measurement here.

---

# Stage G (final) — the benchmark

EC2 `m7i-flex.large`, sole job on the instance, `frames_before_reinitialization=8`,
BA every 2nd keyframe at `max_nfev=50`. Each segment Sim(3)-aligned separately;
ATE is the pose-weighted RMS across segments.

| sequence | frames | coverage | segments | path m | ATE cm | % of path | keyframes | points | ms/frame |
|---|---|---|---|---|---|---|---|---|---|
| fr1_xyz | 798 | 99.7% | 5 | 4.93 | 1.57 | **0.32** | 43 | 5200 | 61.4 |
| fr1_desk | 613 | 100.0% | 14 | 3.67 | 2.82 | **0.77** | 47 | 5751 | 44.1 |
| fr1_desk2 | 640 | 92.0% | 13 | 1.98 | 3.10 | **1.57** | 31 | 3015 | 60.3 |
| fr1_room | 1362 | 99.9% | 27 | 5.92 | 3.26 | **0.55** | 89 | 9147 | 55.0 |
| fr2_desk | 2965 | 100.0% | 13 | 14.71 | 11.75 | **0.80** | 234 | 32492 | 59.3 |

Every sequence stays inside the 100 ms/frame budget, on two vCPUs, with no GPU.

Reproduced independently after the fact: a fresh run of fr1_desk returned
100.0% coverage, 14 segments, 2.82 cm, 0.77%, 44.5 ms/frame against the
benchmark's 44.1 — deterministic to the last decimal on accuracy.

**Excluded, and stated rather than quietly omitted:** `fr1_360`
(rotation-dominated) and `fr1_floor` (low texture). The ORB-SLAM authors note
these are unsuitable for monocular systems, since a monocular system cannot
initialise without parallax — a property of the sensor, not the implementation
(Mur-Artal, Montiel & Tardós, *ORB-SLAM: A Versatile and Accurate Monocular SLAM
System*, IEEE T-RO 2015).

---

# The fourth failure mode: a static camera watching moving objects

A reviewer uploaded `cars.mp4` — a camera fixed to a highway overpass, 1280x720,
984 frames at 30 fps. The camera never moves. The only motion in the sequence is
traffic passing beneath it.

**It returned a plausible-looking result instead of a diagnosis**: a few camera
frusta with long rays fanning out, rendered as though the reconstruction had
worked. That is the worst failure this system can have, because a reviewer who
has not seen a correct point cloud has no way to know.

## What the clip actually looks like

Matched-feature displacement, frame 0 against later frames, at the working
resolution:

| frame gap | matches | median | p90 | max | under 1 px |
|---|---|---|---|---|---|
| 3 | 358 | 0.40 px | 6.12 px | 493.99 px | 52% |
| 30 | 202 | 0.00 px | 2.07 px | 261.44 px | 80% |
| 60 | 69 | 0.00 px | 45.51 px | 488.59 px | 70% |

Frame differencing, frame 0 against frame 30: mean absolute difference 4.61, with
only **4.6% of pixels** changing by more than 20 levels. That 4.6% is the cars.

The signature is the bimodality — a median near **zero** beside a max near
**490 px**. A translating camera moves everything by varying amounts; a fixed
camera moves nothing except whatever happens to be driving past.

## Why the existing parallax gate could not catch it

This is a genuine limitation of the check rather than a bug in it, and it is
worth stating that way.

The gate requires median parallax above 1.0°. It measures **apparent** feature
motion, and apparent motion has two possible causes: a camera moving through a
static scene, or a static camera with independently moving objects. Nothing in a
displacement distribution distinguishes those two, and the cars supplied ample
apparent motion to read as camera translation.

Underneath sits a deeper assumption. **Monocular SLAM rests on a rigid, static
world.** Features attached to moving cars violate that assumption outright, so
`findEssentialMat` can fit the *cars'* motion and report it as the *camera's* —
which is where the fanning rays came from. Handling this properly requires
motion segmentation, which is out of scope here.

## The statistic that does separate, measured before any gate

Fraction of matched features displaced less than one pixel, across all 30
candidate initialization pairs, measured on the target instance:

| clip | min | median |
|---|---|---|
| **cars** | **49%** | **77%** |
| synth_dolly | 0% | 0% |
| synth_rotate | 0% | 0% |
| fr1_xyz | 0% | 0% |
| fr1_desk | 0% | 0% |
| fr1_desk2 | 0% | 0% |
| fr1_room | 0% | 0% |
| fr2_desk | 0% | 1% |
| nostructure | — (too few matches on every pair; caught by the texture gate) |

Every valid clip sits at zero. The static-camera clip sits at 49% in its *most
favourable* pair. The threshold is placed at 25%, the middle of a 48-point gap —
the opposite situation to the homography ratio below, and the reason one is
gated on and the other is not.

## The moving-object check that was measured and NOT shipped

The proposal was to examine where the essential matrix's inliers sit: if they
cluster in a small image region, the estimate is fitting object motion rather
than camera motion. Measured as convex-hull area of the inliers, as a fraction
of the image:

| clip | min | median | max |
|---|---|---|---|
| **cars** | 27.0% | **41.7%** | 51.5% |
| synth_dolly | 26.5% | 48.8% | 62.1% |
| fr1_xyz | 13.3% | 22.6% | 47.1% |
| fr1_desk | 8.0% | 15.2% | 51.5% |
| fr1_desk2 | 3.9% | 13.1% | 27.6% |
| fr1_room | 2.6% | **9.2%** | 43.1% |
| fr2_desk | 5.0% | 22.1% | 39.6% |

**It does not separate, and the sign is backwards.** `cars` has a *larger* inlier
hull (41.7%) than every valid sequence; `fr1_room` sits at 9.2%. A gate on "small
hull means moving objects" would reject the good sequences and pass the bad one.

The reason is instructive: on a static camera the stationary background *is* the
inlier set. A zero-displacement correspondence is perfectly consistent with zero
camera motion, so RANSAC keeps those matches, and they span the entire frame.
The inliers are spread widely *because* the scene is static.

## Re-measuring the homography ratio, with three times the data

The degeneracy ratio was measured earlier on three clips — rotation 0.43-0.45
against translation 0.29-0.34, a 0.09 margin — and deliberately not branched on.
Re-measured across nine clips and 226 candidate pairs:

| | min | median | max |
|---|---|---|---|
| pure rotation | 0.400 | 0.446 | 0.458 |
| static camera | 0.396 | 0.471 | 0.485 |
| valid clips | 0.000 | 0.314 | **0.429** |

**The margin is now negative.** Rotation's minimum (0.400) falls below the valid
maximum (0.429): the distributions overlap and no per-pair threshold separates
them. More data made the case for gating *worse*, which is the usual direction
when a margin was thin to begin with.

There is a second, independent reason never to gate on it: a homography explains
correspondences under pure rotation **or** when the scene is planar, and a planar
scene filmed by a translating camera — a desk filling the frame, a road surface —
is a perfectly valid input that this would reject for being flat.

It is therefore surfaced as a **confidence warning** on the result, never a
rejection.

## The silent-output path, audited

`cars.mp4` reached a `done` status with:

| | |
|---|---|
| poses | 35 |
| segments | **11** |
| map points | **0** |
| keyframes | 22 |
| mean reprojection error | **nan** |
| flags | "the map is sparse" (cosmetic) |

Nothing in the pipeline checked that the run had produced a reconstruction.
Initialization was checked and tracking was checked, but the **final result**
never was. The frusta the reviewer saw were bare keyframe poses; the "long rays"
were frustum wireframes scaled from a bounding box spanning eleven disjoint
segments at unrelated arbitrary scales.

Three changes followed: a run finishing with fewer than 25 map points or no
finite reprojection error now raises `DEGENERATE_RECONSTRUCTION` rather than
rendering; segment thrashing raises a `LOW_CONFIDENCE` flag; and the results page
shows segment count, initialization parallax, median tracked points and mean
reprojection error, with concerning values marked.

### The fix's own regression, caught by the guard

The first version of the degeneracy check read the reprojection error from
`world_map` — the **last** segment's map. That map is legitimately empty whenever
a video ends mid-initialization, so the check condemned **fr1_desk2**, a sequence
with 3015 points across 13 healthy segments, as a degenerate reconstruction. The
error is now averaged across the segments that actually hold points.

## A platform-dependent weakness in the rotation gate, found while testing this

`synth_rotate` — a clip whose camera centre provably never moves — is correctly
rejected on the development laptop (ARM) and **accepted on the EC2 target**
(x86), with the same code and the same file. Initialization reports 1.247° of
parallax on a sequence that has none.

The mechanism is fundamental, not a coding error. Under pure rotation the
essential matrix is degenerate and `recoverPose` must still return a **unit-norm**
translation — it has no way to express "zero baseline". The reconstruction
therefore acquires a fake baseline, and triangulated points subtend a genuine
ray angle between the two fake centres. Whether that fake parallax lands above or
below the 1.0° threshold comes down to floating-point and SIMD differences
between architectures.

The clip now carries the `degenerate_geometry` warning (dominance 0.45) on the
target, so it is no longer silent — but **the pure-rotation gate is weaker than
previously documented**, and that is stated rather than papered over with a
threshold the evidence above shows cannot be drawn.
