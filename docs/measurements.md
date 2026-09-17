# Measurements

Numbers as they are taken, so the README's table is transcribed rather than
reconstructed. Every row records what was measured, on what, with which settings.

**Nothing here is a submission number yet.** The brief requires timings from the
demonstrated test environment, which is the EC2 `m7i-flex.large`. Laptop figures
below exist only to establish shape and to catch surprises early.

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

---

## Still to measure

- [ ] **The same probe on the EC2 `m7i-flex.large`.** This is the number that
      counts; everything above is shape-finding. Expect single-core speed, not
      core count, to drive the difference.
- [ ] Bundle adjustment cost per keyframe — the one stage with real risk of
      eating the budget.
- [ ] The five-parameter sweep: processed fps, working resolution, features per
      frame, BA window size, BA frequency.
- [ ] End-to-end wall clock for a 10s clip, which is what the brief actually asks.
