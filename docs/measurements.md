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
