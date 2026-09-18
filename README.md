# Monocular sparse SLAM

Upload a video from a single ordinary camera and get back the estimated camera
trajectory and a sparse 3D point cloud, rendered in an orbitable 3D view with a
measured per-stage timing breakdown. It runs on two free-tier vCPUs with no GPU.

**Live:** http://13.63.181.231/

---

## Results

Measured on the deployment instance across five sequences of the
[TUM RGB-D benchmark](https://cvg.cit.tum.de/data/datasets/rgbd-dataset), whose
ground-truth trajectories come from a 100 Hz motion-capture system.

```bash
python tools/benchmark.py --dataset-root ~/tum --out out/benchmark.json
```

| sequence | frames | coverage | segments | path | ATE RMSE | **% of path** | ms/frame |
|---|---|---|---|---|---|---|---|
| fr1_xyz | 798 | 99.7% | 5 | 4.93 m | 1.57 cm | **0.32%** | 61.4 |
| fr1_desk | 613 | 100.0% | 14 | 3.67 m | 2.82 cm | **0.77%** | 44.1 |
| fr1_desk2 | 640 | 92.0% | 13 | 1.98 m | 3.10 cm | **1.57%** | 60.3 |
| fr1_room | 1362 | 99.9% | 27 | 5.92 m | 3.26 cm | **0.55%** | 55.0 |
| fr2_desk | 2965 | 100.0% | 13 | 14.71 m | 11.75 cm | **0.80%** | 59.3 |

Every sequence completes inside the 100 ms/frame budget on two vCPUs with no GPU.
A fresh run of fr1_desk afterwards returned 100.0% coverage, 14 segments,
2.82 cm and 0.77% against the benchmark's identical figures — accuracy is
deterministic on a given machine.

### Read the coverage and the segment count, not just the error

Three numbers are needed to judge any row, and quoting the error alone would be
misleading in two separate directions.

**Coverage** is the fraction of the sequence that produced poses at all. A system
that gives up early reports a *better* ATE than one that keeps going, for the
worst possible reason. An earlier version of this system reported 1.56 cm on
fr1_xyz over 23.4% of the sequence; the honest comparison is against the 0.32%
of path it now achieves over 99.7%.

**Segments** is how many times the map had to be restarted. When tracking cannot
be recovered the pipeline begins a fresh map rather than stopping — and a new map
has a new origin, a new orientation and, because monocular scale is unobservable,
a **new arbitrary unit**. Measured directly on fr1_xyz, the recovered scale factor
for its four segments was 0.107, 0.055, 0.026 and 0.067: four segments, four
different units, nothing in the images relating them.

So each segment is Sim(3)-aligned **separately** and the reported ATE is the
pose-weighted RMS across them. Aligning the whole trajectory with a single
transform would charge the system for a discontinuity it cannot observe. But a
trajectory in 27 pieces is plainly a worse result than the same error in one
piece, and the count is printed so the reader can see it. fr1_room's 0.55% is
across 27 restarts; fr1_xyz's 0.32% is across 5.

### Excluded sequences, and why

`fr1_360` and `fr1_floor` are excluded. The ORB-SLAM authors note that sequences
dominated by rotation without translation, or with no texture, are unsuitable for
monocular systems, because a monocular system cannot initialise without parallax
— a property of the sensor rather than of the implementation (Mur-Artal, Montiel
and Tardós, *ORB-SLAM: A Versatile and Accurate Monocular SLAM System*, IEEE
T-RO 2015). They are named here rather than quietly omitted.

The low-texture case is demonstrated separately on fr3
`nostructure_notexture_far`, where the required behaviour is a clear diagnosis
rather than a reconstruction — see *Failure modes* below.

### On the alignment, and why scale is solved for

ATE is computed after aligning estimated and true trajectories with a
**similarity transform — rotation, translation and one scale factor, seven
degrees of freedom**, using Umeyama's closed-form solution.

The scale term is not a convenience. A monocular reconstruction is defined only
up to scale: a small scene viewed closely and a large one viewed from far away
produce identical images. Comparing against metric ground truth without solving
for scale would measure that arbitrary choice rather than any error the system
made. Seven-DOF alignment is the standard procedure for monocular evaluation and
is what the ORB-SLAM papers report.

The alignment deliberately **refuses to absorb reflections** — its rotation is
constrained to determinant +1 — which caught a real class of bug, described under
*Measurement discipline*.

### Failure modes produce different advice

| clip | diagnosis | what the user is told |
|---|---|---|
| pure rotation | median **484** matches, **0.00°** parallax | move sideways through the scene |
| TUM `nostructure_notexture_far` | median **3** matches | the scene needs texture |

Making these differ was deliberate work. Both look identical from outside —
initialization fails — but *"try walking sideways"* is actively wrong advice for
someone filming a blank wall, and *"find more texture"* is useless to someone
standing still and turning. The pipeline separates them by how far the failure
got: plenty of matches with no parallax means the camera did not translate;
almost no matches means there was nothing to match.

## Timing

The measurement environment is the deliverable as much as the number is.

| | |
|---|---|
| instance | **AWS EC2 `m7i-flex.large`**, free-tier eligible |
| CPU | 2 vCPU (1 physical core, 2 threads), Intel Xeon Platinum 8488C |
| memory | 7.8 GB |
| GPU | **none** |
| OS | Ubuntu 24.04.4 LTS, Linux 6.17.0-1017-aws |

At 10 fps processing the budget is **100 ms per frame**. Measured on the
instance, fr1_xyz, 640 px working width, 1000 features, with bundle adjustment
and re-initialization enabled:

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
| local_map | 0.97 | 1.5 |
| unaccounted | — | 1.3 |
| **total** | **66.50** | 100 |

Across the five benchmark sequences the range is **44–61 ms/frame**, all inside
the budget, on two free-tier-eligible vCPUs with no GPU. The modest hardware is
the point of the result, not an apology for it: the brief asks for an algorithm
that performs under constrained resources, and the constraint is met with room
to spare on the smallest sensible instance.

`unaccounted` is wall clock minus the sum of measured stages. It is reported so
the breakdown cannot silently omit time.

### The workload does not want more cores

Measured with `tools/probe.py`, varying OpenCV's thread count **on the target
instance**, which has 2 vCPUs:

| threads | ms/frame (per-frame floor) |
|---|---|
| 1 | 13.3 |
| 2 | 13.1 |

Essentially flat — the second thread buys about 2%. `lscpu` explains it: the two
vCPUs are two hardware threads on a **single physical core**, so there is no
second execution unit to spread across. This pipeline is effectively
single-core-bound here.

The development laptop, with eight real cores, behaves completely differently on
the same clip and settings: 29.85 ms/frame at one thread, 17.67 at two, 18.26 at
four, 19.16 at eight — the second thread is worth 40% and scaling stops after it.

Together the two tables are the argument for the small instance: a larger one
would add cores this workload demonstrably cannot use. Note also that the laptop
table *alone* would have supported a wrong conclusion, namely that thread count
matters a great deal here.

Six consecutive runs on the instance held between 13.2 and 13.0 ms/frame for the
per-frame floor with 0% CPU steal, so `m7i-flex` burst behaviour is not
distorting these numbers at this job length.

---

## Measurement discipline

This section exists because it is the part of the work that could not have been
generated on demand. ORB, an essential matrix, PnP and a point cloud are a
handful of OpenCV calls. Knowing which of them costs anything, on the machine
that will actually run them, is not.

**Matching was separated from the PnP solve.** The two were originally reported
as one `pnp` stage at 34% of wall time. That misattributes the cost badly:
brute-force Hamming matching is O(n·m) in descriptor counts, while PnP is a small
iterative solve on a few hundred points. Anyone tuning one would have been
looking at the other. Split, they are 27.4% and 6.0% — and the conclusion
inverts, because the expensive half scales quadratically with the feature budget
and the cheap half barely scales at all.

**The map cull was found consuming 17% of frame time, by measuring.** Nothing
about `cull` suggests expense; it recomputes reprojection error across the whole
map on every keyframe, which is invisible in the source and obvious in the
profile. It is the third-largest cost in the pipeline and was never a suspect.

**The profile inverts between development machine and target.** On an Apple
Silicon laptop ORB is 48.7% of wall time and matching 7.2%; on the Xeon target
they are 27.1% and 27.4%. The totals agree within a few percent — 22.6 ms against
24.1 — purely by coincidence. An engineering conclusion drawn from the laptop
("tuning is a conversation about ORB") is simply false on the machine that runs
the service. Every timing claim in this document therefore comes from the
instance.

**The export coordinate convention was verified by round-trip, not by
inspection.** Export inverts world-to-camera into camera-to-world and flips Y for
Three.js. The Y flip is a *reflection*, determinant −1, and the Sim(3) alignment
is constrained to proper rotations precisely so it cannot quietly absorb one.
Running the evaluator both ways: undoing the flip reproduces the pre-export ATE
exactly, leaving it in place produces a worse one. A silently mirrored
trajectory is a bug that looks like drift, and this is how it reveals itself.

**Bundle adjustment shipped as a no-op, and a parameter sweep hid it.** The
first working BA reduced reprojection error by 0.2% and recovered 0.1% of a
deliberately injected perturbation. It terminated after *two* function
evaluations reporting `xtol` convergence, while the gradient norm was 3143. The
cause was variable scaling: the parameter vector mixes rotation vectors in
radians, translations and 3D coordinates, and across 3549 parameters the
relative step looked negligible even when the absolute step was not. With
`x_scale="jac"`, recovery of the same perturbation went from 0.1% to **86.6%**.

Worse — and more instructive — three earlier experiments varying `diff_step`,
tolerances and `x_scale` all returned **byte-identical** results, which read as
strong evidence that none of them mattered. They were no-ops: `ba.py` does
`from scipy.optimize import least_squares`, so patching
`scipy.optimize.least_squares` never touched the imported name. *Identical
numbers across varied inputs are evidence of a broken experiment, not of an
insensitive system.* There is now a regression test that demands BA recover a
known perturbation, because every structural check — it runs, it returns, its
Jacobian sparsity is provably correct — passed against the broken version.

**A loop-closure constraint folded the trajectory, and a residual of zero gave
it away.** The first pose graph asserted that the two ends of a closure occupy
the same position. Revisiting a place does not mean that — the camera returns
*near* somewhere it has been. The optimiser satisfied all 78 constraints exactly,
drove the residual to **0.0000**, and collapsed the map: ATE 3.22 cm → 9.82 cm.
A residual of exactly zero against 78 over-determined constraints is evidence
the constraints are vacuous, not evidence of convergence. The correct constraint
was already being computed and discarded — PnP localises the query keyframe
against *old* map points that predate the drift, so its recovered centre is a
drift-corrected estimate of where that keyframe belongs.

**One good sequence is not a result.** Before the benchmark existed, this system
reported 99.7% coverage on fr1_xyz and that looked like a general claim. Run
across five sequences, coverage on the other four was **6–11%**. The 99.7% was a
property of one clip whose camera oscillates inside a 0.94 m box and keeps
returning to mapped territory. Everything downstream of that discovery — the
re-initialization design, the segment-aware evaluation, the threshold chosen by
cross-sequence sweep — exists because a second sequence was tried.

**The map cull ran for 17% of frame time and removed nothing.** Sweeping its
threshold showed zero points culled across an entire sequence at the shipped
5 px, making it byte-identical to disabling it. The cause is structural:
triangulation already admits only points below 4 px reprojection error, so a
5 px cull cannot fire on anything triangulation let through. Nothing in the
source suggests this; only the profile and the sweep do.

**A threshold was measured and deliberately not shipped.** During initialization
the ratio of homography inliers to homography-plus-fundamental inliers is a
recognised signal for degenerate geometry: a homography explains correspondences
exactly when the camera rotates about its own centre or the scene is planar, both
of which make the essential matrix undetermined. It separates the cases here:

| motion | homography dominance |
|---|---|
| pure rotation | 0.43 – 0.45 |
| translation | 0.29 – 0.34 |

The signal is real, and it is **reported in the failure message as supporting
evidence but never branched on.** The margin is about 0.09, measured on three
clips, and it does not approach the 1.0 that theory suggests because
`findFundamentalMat` still finds plenty of degenerate inliers on rotation-only
correspondences. A threshold placed inside a margin that thin, fitted on that
little data, is intuition-tuning dressed up as adaptivity — it would work on the
clips it was fitted to and fail silently elsewhere. The diagnosis instead rests
on something far more robust: whether there were any matches at all. This is a
case where the right engineering decision was to measure something, find it
insufficiently separated, and ship the measurement rather than the threshold.

---

## Architecture

The repository is split into a pure algorithm package and a thin web wrapper,
and the separation is load-bearing rather than decorative.

```
vslam/          the algorithm. Imports numpy, OpenCV, scipy. Never FastAPI.
  timing.py     StageTimer, written before anything it measures
  video.py      decode, 10 fps subsampling, 640 px downscale, grayscale
  camera.py     intrinsics, the focal-length heuristic and its override
  features.py   ORB detection, brute-force Hamming matching, ratio test
  initialize.py essential matrix, parallax gate, first triangulation
  mapping.py    Keyframe / MapPoint / Map, triangulation, culling
  tracking.py   solvePnPRansac against map points, keyframe triggers
  pipeline.py   orchestration; owns the timer
  align.py      Sim(3) alignment and ATE
  export.py     world->camera inversion and Y-up conversion, once, at the edge

app/            the web layer. Imports FastAPI. Never imports OpenCV at module scope.
  main.py       routes and the static mount
  jobs.py       background worker, concurrency semaphore
  store.py      in-memory job registry
  uploads.py    streaming upload with the size cap enforced mid-stream
  schema.py     typed failure vocabulary
  static/       vanilla JS, no build step; Three.js vendored

tools/          probe.py, evaluate.py, make_synthetic.py, tum_to_video.py
```

`vslam/` never importing the web framework is what let stages 1 to 5 be
developed and tested from a command line before any of it was wired to HTTP, and
it is what will let the parameter sweep be a loop over a Python function instead
of a loop over HTTP requests. `app/jobs.py` imports the pipeline *inside* the
worker function rather than at module scope, so starting the API does not pay for
loading OpenCV, NumPy and SciPy.

### Request handling: 202 then poll, never synchronous

`POST /api/jobs` spools the upload to disk, returns **202 Accepted with a job id
in milliseconds**, and processes in a detached background task. The client polls
`GET /api/jobs/{id}`.

Processing inside the request would be a mistake with a specific failure mode:
nginx's default `proxy_read_timeout` and most browsers' idle timeouts are both
60 seconds. A long request does not merely fail — it dies taking every completed
frame with it, because the work only ever existed in that request's memory. The
202-then-poll shape was ported from Assignment 1, where the same reasoning
applies to a slower worker.

Two details that matter more than they look:

The pipeline runs via `asyncio.to_thread`, not directly in the coroutine. Unlike
Assignment 1's I/O-bound worker, this work is CPU-bound with no await points
anywhere, so running it on the event loop would block every poll from every
client for the entire job. It works because OpenCV and NumPy release the GIL
around native code.

Upload bytes are read **inside** the concurrency semaphore, so peak memory is
bounded by concurrency rather than by how many people upload at once.

### Job addressing without authentication

There is no login: a reviewer must be able to open the URL and upload a video.
Job ids are 32 characters from `secrets.token_urlsafe`, so the id is the
capability, and an unknown id returns **404, not 403** — a 403 would confirm
which ids exist, which is the fact worth withholding.

---

## Technical decisions

### Input: 10 fps, 640 px, grayscale

A 30 fps source is subsampled to every third frame. Consecutive frames at 30 fps
have almost no baseline between them, so the extra frames cost a full ORB
detection and a full match while contributing nearly no parallax — they are
pure cost. Frames are downscaled to a 640 px long edge with `INTER_AREA`, chosen
specifically because it averages over the source region rather than sampling it:
aliased edges produce spurious FAST corners that do not repeat between frames,
which is the worst possible input to a matcher. Colour is discarded because ORB
is computed on intensity.

Skipped frames use `grab()` without `retrieve()`, avoiding the colour conversion
and copy. This is cheaper but not free — inter-frame compression means a skipped
frame must still be decoded to reconstruct its successors — and decode still
costs 1.96 ms/frame, 8% of the budget.

### ORB rather than SIFT

ORB produces 256-bit binary descriptors compared with a Hamming distance, which
is an XOR and a popcount. SIFT's 128 floats and L2 distance are roughly 5–10×
slower to both compute and match, and the 100 ms budget does not have room. ORB
was also unencumbered by patents throughout the period SIFT was not. The cost is
robustness: ORB is rotation invariant and, through its 8-level 1.2-scale pyramid,
scale invariant, but it is not affine invariant and it degrades faster under
motion blur. Both appear in the limitations.

### Lowe's ratio test rather than cross-checking

`BFMatcher(crossCheck=True)` keeps a match only when each descriptor is the
other's best match, which rejects *non-mutual* matches. The ratio test compares
the best match against the **second best**, and discards the match when they are
close — meaning whatever the region looks like, it looks like that in at least
two places, so the correspondence is ambiguous regardless of how small its
absolute distance is. Repetitive texture — brick, carpet, foliage, a row of
windows — produces exactly this, and it is where mismatches actually come from.
The two are mutually exclusive in OpenCV anyway, since `crossCheck=True` forbids
the `k=2` search the ratio test needs.

### Tracking against the map, not frame to frame

This is the single largest drift reduction available to a monocular system and
the reason this is SLAM rather than visual odometry.

Frame-to-frame odometry estimates each pose relative to the previous frame and
chains the results. Every estimate carries error and chaining composes those
errors, so drift compounds with the number of frames, without bound. Tracking
against a persistent map instead solves each pose from 3D points triangulated by
earlier keyframes: the reference is the map, so an error in one frame does not
become part of the next frame's reference. Drift still exists, because the map
accumulates its own error, but it grows far more slowly.

`solvePnPRansac` is used rather than plain PnP because descriptor matching
produces wrong matches even after the ratio test, and PnP is least squares — a
single gross outlier drags the solution a long way. RANSAC excludes outliers
rather than averaging them in.

### Choosing the initialization pair by testing, not by index

Initialization needs two frames with enough baseline, and the video does not say
which. Candidate pairs are tested rather than assumed, because **frame distance
is not baseline**. Measured on this same clip:

| pair | true baseline | translation direction error |
|---|---|---|
| frames 0–30 | 0.354 m | 1.75° |
| frames 0–90 | 0.048 m | 35.23° |

Frame 90 is three times further away in time and has a seven times *smaller*
baseline, because `fr1_xyz` oscillates and the camera had wandered back near
where it started. Choosing by frame index would have picked the far worse pair,
confidently.

### Parallax as a ray angle, not pixel disparity

Pixel disparity is the tempting proxy and it is wrong, because it conflates
rotation with translation. A camera rotating on the spot sweeps every feature
right across the image while producing exactly zero depth information. The angle
between the two viewing rays at a triangulated point is immune to this: under
pure rotation both rays point the same way and the angle is zero, which is the
truth. This is what makes the rotation clip fail loudly instead of silently
emitting a garbage map, and it is why the minimum-parallax gate is specified in
degrees of ray angle rather than pixels.

### Keyframe triggers are scale-relative

A keyframe is inserted when translation since the last one exceeds 10% of the
median scene depth, **or** when the tracked-point ratio falls below 0.7, with a
minimum of 3 frames between keyframes.

The translation threshold is a fraction of scene depth rather than an absolute
distance because a monocular map has no units — "the camera moved 10 cm" is not
a sentence this system can form. Moving 10% of the distance to what you are
looking at produces roughly the same parallax whether the scene is a desk or a
street, and parallax is the quantity that matters for triangulating new
structure. The tracked-ratio trigger exists because translation is not the only
way to run out of map: turning a corner leaves the existing points behind with
very little translation, and waiting for the translation trigger would lose
tracking first.

### Bundle adjustment

A **sliding window over the last 5 keyframes**, optimising poses *and* points
jointly — not pose-only, which would take the points as truth and force all the
error into the cameras, exactly the wrong place when the points are themselves
noisy triangulations. `scipy.optimize.least_squares`, `method="trf"`, Huber loss
at 2 px so a surviving mismatch cannot dominate the sum, the first keyframe held
fixed to remove gauge freedom, and a hard `max_nfev` ceiling so the optimiser
cannot eat the time budget.

The sparse Jacobian matters more than anything else here. With 5 keyframes and
~2000 points the parameter vector has ~6000 entries; a dense Jacobian would be
~10^8 entries estimated by finite differences, one column per parameter. Almost
every entry is structurally zero — a residual for point P in keyframe K depends
only on P's three coordinates and K's six pose parameters — and passing that
structure as `jac_sparsity` is the difference between feasible and impossible.

Sliding window, not full history: full-history cost grows with video length, so a
longer clip would blow the budget for reasons unrelated to how hard the
reconstruction is. The trade is real — a window cannot correct drift that
accumulated before it.

SciPy rather than g2o or Ceres is deliberate and genuinely costly: SciPy installs
from a wheel with no build step and every line of the residual function is
readable, which matters for a system whose author must be able to explain it. It
is meaningfully slower than Ceres. BA is 59% of frame time here, and a Ceres
implementation would reclaim most of that.

**What BA is worth, measured.** Running it every *second* keyframe at
`max_nfev=50` improves ATE from 0.51% to 0.45% of path on fr1_xyz. Running it on
*every* keyframe costs 106 ms/frame — over budget — and scores **worse** (0.57%).
More optimisation is not monotonically better when it competes for time with the
frames that feed it.

**And an honest limit: BA does not fix drift.** Applied as a single global pass
over all 52 keyframes of a fixed trajectory, it moved ATE by *nothing*. The
reason is structural: BA minimises reprojection error, and drift is nearly
invisible to reprojection error — a slowly accumulating scale or pose error stays
perfectly consistent with every image that produced it. Only an observation
linking distant parts of the trajectory violates it, which is loop closure.

### Surviving tracking loss, and re-initializing when that fails

Tracking loss is not a defect to be prevented; it is what happens when a camera
explores. Two mechanisms, both chosen on measurement:

**Skip and retry.** A failed frame no longer ends the run — it is skipped, no
pose emitted, and the next frame tried. On fr1_xyz this alone took coverage from
22.7% to **99.7%**, and error as a fraction of path from 0.86% to 0.51%.
Relocalization against the whole map was also implemented as a second line of
defence; ablating it changed the result by **nothing**, because it never once
succeeded. It is retained for harder cases and reported as unvalidated.

**Re-initialization.** Skip-and-retry generalised poorly: 6–11% coverage on the
other four benchmark sequences. The map can only grow from tracked frames, so
once tracking is lost the map freezes, and a camera exploring *away* from its
initial map can never re-acquire it. After 8 lost frames the pipeline therefore
starts a fresh map and carries on. The wait was swept across three sequences:

| wait | fr1_xyz % of path | fr1_desk coverage / error | fr1_room coverage / error |
|---|---|---|---|
| off | 0.30 | 6.0% / 7.99% | 7.6% / 5.66% |
| **8** | **0.32** | **100.0% / 0.77%** | **99.9% / 0.55%** |
| 20 | 0.31 | 100.0% / 1.21% | 99.9% / 0.84% |
| 35 | 0.30 | 83.8% / 2.06% | 95.7% / 1.18% |

Waiting longer preserves a single coordinate frame where tracking *will* recover,
which is why the easiest sequence mildly prefers it. Eight costs fr1_xyz 0.02
percentage points and takes the others from single-digit coverage to complete.

### Loop closure

Bag-of-words place recognition over a 256-word vocabulary built from the sequence
itself, keyframes indexed by tf-idf histogram and queried by cosine similarity,
with every candidate geometrically verified by PnP before acceptance. Detection
works: 102 candidates and **78 verified closures** on fr1_xyz in 2.8 s. A Sim(3)
pose graph — 7 DOF per node, because monocular scale *drifts* and a rigid SE(3)
graph has no parameter able to absorb that — distributes the error backwards.

`cv2.kmeans` rather than scikit-learn's `MiniBatchKMeans`: OpenCV is already a
dependency, and adding a large one for a single function call sits badly with a
project whose argument is about constrained compute. No accuracy claim is made
either way. A caveat is documented in the source: ORB descriptors are binary, and
k-means on binary data treated as floats approximates DBoW2's k-majority rather
than matching it.

**Status: implemented, not demonstrated to help.** On fr1_xyz it moves ATE from
3.22 cm to 3.60 cm — no longer destructive after the constraint fix, and no
improvement. That sequence oscillates inside a 0.94 m box with error already at
0.45% of path, so there is no accumulated drift for a closure to correct. The
literature's claim that loop closure corrects drift is well-founded; **it is not
evidenced by any measurement in this repository**, and is not claimed to be.

### Rejections, on measurement

Four things were built or specified, measured, and not shipped. They are listed
because a decision rejected on evidence says more than one adopted on taste.

**Projection-guided matching — implemented and rejected.** Predicting the pose,
projecting each map point, and searching only nearby keypoints is what ORB-SLAM
does, and in isolation the evidence was strong: at the exact frame where tracking
died, 203 matches against brute force's 90. End to end it was consistently worse
— **23 poses against 59**, at every radius and ratio tested.

The investigation was worth more than the feature. The isolated test used the
*true* pose; the pipeline only has a *predicted* one, and constant-velocity
prediction error measured a median of 8–23 px with a maximum of 165 px, so
correct matches routinely fell outside the search disc. Two further errors
surfaced: the radius was never the live variable (a `k=8` neighbour cap binds
first, which is why r=30, 60 and 100 gave *identical* results), and the raw
counts were never comparable (brute force assigns several keypoints to one map
point; the projection matcher enforces one-to-one). Kept behind a flag, off by
default, because the mechanism is sound and would likely win with a better
motion model.

**Lucas–Kanade inter-frame tracking — rejected on profiling.** Replacing
descriptor matching with optical flow is a known way to cut per-frame cost. On
the target instance `match_map` costs 8.50 ms of a 66.5 ms frame, so the ceiling
on the saving is about 8.5 ms against 33 ms of unused budget.

The number matters and it is worth being precise about which machine produced it.
The laptop profile would have supported a much stronger version of this
conclusion — there matching is 1.61 ms and the ceiling looks like 1.5 ms. The
target says 8.50 ms, five times larger. The conclusion survives the correction
only because the headroom is large; a tighter budget would have made LK worth
doing. Its remaining genuine case is robustness through gradual appearance
change, which is a different argument and not one about speed.

**A degeneracy threshold — measured and deliberately not branched on.** See
*Measurement discipline*: 0.43–0.45 under rotation against 0.29–0.34 under
translation, a 0.09 margin on three clips. Reported as evidence in the failure
message; the diagnosis rests on something far more robust.

**A C++ rewrite — considered and rejected.** ORB detection, matching and PnP are
already compiled OpenCV, called on the order of a hundred times per frame, and
interpreter dispatch is microseconds against milliseconds of native work. The
measured Python overhead is the 1.3% `unaccounted` row, an upper bound that also
includes genuine untimed work. The real target, if this needed to be faster,
is bundle adjustment at 59% of frame time — and the fix there is Ceres or a
better-conditioned problem, not a different language for the glue.

**Retuning the cull threshold on one sequence — declined.** A sweep showed
0.5 px gives the best ATE *and* the lowest cost on fr1_xyz. Not adopted: it is
one clip, and fitting a threshold to one clip is the failure this project has
avoided elsewhere.

## The two things that cannot be fixed

**Scale is unobservable from a single lens.** The reconstruction has the correct
*shape* and an unknown *size*. This is a property of the sensor, not a
shortcoming of the implementation: a small scene photographed closely and a large
one photographed from far away produce identical images, and no amount of
processing can separate them. It is the reason a geologist puts a rock hammer in
the frame. Every distance this system reports is in arbitrary units, the UI says
so, and metres are never printed. Map-based tracking, the parallax gate and
bundle adjustment all limit how much the scale factor *drifts* within a
sequence, but none of them can fix it globally — that requires loop closure with
Sim(3) pose-graph optimization, which is out of scope.

**Focal length is estimated, not measured.** Unlike scale, focal length is
knowable in principle; we simply are not told it, because the system accepts
uploads from unknown cameras. The heuristic is `fx = fy = 0.9 × width`, a
horizontal field of view of about 58°. Measured against real cameras it is
mediocre but not absurd: TUM `freiburg1` publishes fx = 517.3 where the heuristic
gives 576 (+11.3%), and `freiburg3` publishes 535.4 against the same 576 (+7.6%).
The accuracy figures in this document were produced **with the heuristic, not
with the published intrinsics**, so they include whatever that error costs. A
user override is available; what the error costs in ATE has not yet been measured
as a controlled comparison.

---

## Limitations

**Trajectories fragment into segments.** This is the largest one. When tracking
cannot be recovered the pipeline restarts the map, and each segment carries its
own arbitrary scale and origin — fr1_room needs 27 restarts to cover its
sequence. Accuracy *within* a segment is good; the relationship *between*
segments is unknown and unknowable from images alone. A system with working
relocalization would rejoin them; see below.

**Relocalization is implemented but has never succeeded.** Ablating it changes
no measured result. The skip-and-retry path recovers first on easy sequences, and
on hard ones the map has frozen before relocalization could help. It is reported
as unvalidated rather than as a feature.

**Loop closure is implemented but unproven.** Detection works (78 verified
closures on fr1_xyz); the correction has not been shown to improve any measured
trajectory, because no benchmarked sequence here has enough uncorrected drift for
it to fix. Not claimed as a result.

**Bundle adjustment cannot correct drift**, only local inconsistency — measured,
not assumed: a global pass over all keyframes moved ATE by nothing.

**No inertial fusion.** Phone IMU data would constrain rotation between frames
and supply metric scale, addressing two problems listed here. Out of scope.

**No lens distortion correction.** Correcting it needs calibration we do not
have. Uncorrected radial distortion bends straight lines near the frame edge,
biasing exactly the features with the most parallax information. The fr1
sequences used for evaluation are **not** pre-undistorted, so the reported
accuracy includes this effect.

**Sparse, not dense.** A point cloud at ORB keypoints, not a surface.

**In-memory job state, single worker.** Jobs do not survive a restart, and state
is process-local, so the service cannot scale to a second replica without
replacing `app/store.py`. Concurrency is fixed at one because the workload
already stops scaling past two threads: a second concurrent job would not finish
sooner, it would make both miss the budget.

**Uploads capped at 25 MB and 60 seconds**, enforced mid-stream and by nginx.

## Setup

Requires Python 3.11+.

```bash
git clone <repo> && cd slam
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt

# Sample clips are generated, not committed — they are reproducible from a seed.
./.venv/bin/python tools/make_synthetic.py --motion dolly
./.venv/bin/python tools/make_synthetic.py --motion rotate     # the failure case

./.venv/bin/uvicorn app.main:app --reload     # http://127.0.0.1:8000
```

Reproducing the benchmark needs the TUM sequence:

```bash
curl -O https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz
tar xzf rgbd_dataset_freiburg1_xyz.tgz
./.venv/bin/python tools/tum_to_video.py rgbd_dataset_freiburg1_xyz --out samples/tum_fr1_xyz.mp4
./.venv/bin/python tools/evaluate.py samples/tum_fr1_xyz.mp4 samples/tum_fr1_xyz.truth.json --runs 3
```

Tests: `./.venv/bin/python -m pytest`

### Deployment

Full instructions in [`deploy/AWS.md`](deploy/AWS.md). In outline: an
`m7i-flex.large` running Ubuntu 24.04 with an elastic IP, the app under
`/opt/slam` as a systemd unit bound to 127.0.0.1, and nginx in front.

Two details that cost real time and are documented there. nginx defaults
`client_max_body_size` to 1 MB, which rejects every video upload with a 413
generated before the application ever sees the request — so the application's own
limit and its friendly message never run, and the failure looks like an
application bug. And a virtualenv must never be copied between directories: its
console scripts hardcode an absolute interpreter path in their shebang, so a
copied venv fails with `203/EXEC Permission denied` on a file that `ls -l` and
`test -x` both report as executable, because the denied permission belongs to the
interpreter named in the shebang rather than to the script.

---

## Libraries and versions

No pretrained models are used. This is a classical geometry pipeline; there is no
neural network anywhere in it.

| package | version | role |
|---|---|---|
| opencv-python-headless | 4.14.0.94 | ORB, matching, essential matrix, PnP, triangulation, video decode |
| numpy | 2.5.3 | array arithmetic throughout |
| scipy | 1.18.1 | bundle adjustment solver (designed; not yet wired in) |
| fastapi | 0.141.1 | HTTP layer |
| uvicorn | 0.53.0 | ASGI server |
| python-multipart | 0.0.32 | multipart upload parsing |
| Three.js | r180 | 3D viewer, vendored into `app/static/vendor/` |

`opencv-python-headless` rather than `opencv-python` is deliberate: the standard
wheel links GTK/Qt/X11 for `imshow`, which a server has no display for. The
headless wheel drops them, which is why the entire dependency install on the
instance takes 10.6 s from wheels with no compile step and no `libgl1` to chase.

OpenCV is pinned to the 4.x line rather than 5.0. OpenCV 5 is a major version
with signature and default changes across exactly the calls this pipeline relies
on, and essentially every reference describing `findEssentialMat` and
`solvePnPRansac` behaviour documents 4.x.

Three.js is **vendored, not loaded from a CDN**, so the page has no external
dependency at review time. Vendoring means vendoring the whole import graph:
from r167 the build ships as `three.module.js` plus `three.core.js`, and shipping
only the first produces a 404 on the second that the browser reports against the
entry point rather than the missing file. `tests/test_static_assets.py` now walks
every relative import and every bare specifier against the import map so this
cannot recur.

---

## AI usage

This project was built with AI assistance (Claude). A running record was kept in
[`docs/ai-usage.md`](docs/ai-usage.md) as the work happened, rather than
reconstructed afterwards.

**What was generated.** Essentially all of the code was drafted by the assistant,
including the geometry modules, the web layer, the viewer, the synthetic scene
renderer and the measurement tooling. Also the bulk of the explanatory comments
and this document.

**What that means, and does not.** The pipeline itself — ORB, essential matrix,
`recoverPose`, PnP, triangulation — is a small number of OpenCV calls that any
current assistant will produce on request. It is not where the work was. The work
was in deciding what to measure, running it on the machine that matters,
and acting on results that contradicted the plan.

**Reversals and rejections, which are the load-bearing part of this record:**

*A Next.js frontend on Vercel was recommended, adopted into a written plan, and
then reversed.* The argument for it was a conventional modern stack and a static
export that could be hosted on both Vercel and EC2. The argument against, which
won, was that it manufactures a cross-origin problem that does not otherwise
exist, forces HTTPS onto the critical path (a browser blocks `fetch` from an
HTTPS page to a plain-HTTP endpoint as mixed content), adds a second deployable
and a build step, and weakens a requirement that says "deploy to AWS" by putting
the page a reviewer opens somewhere else. The original decision — FastAPI serving
static files, vanilla JS, no build step — was restored.

*Clerk authentication was recommended, adopted, and then dropped entirely.* The
enumeration requirement it was meant to satisfy is met instead by unguessable
random job ids and a 404 on unknown ones. A sign-up wall in front of a demo URL
described as publicly accessible was the wrong trade, and three dependencies left
with it.

*Lucas–Kanade tracking was recommended for speed and rejected on profiling data*,
as documented above — including correcting the argument when the target-instance
numbers turned out four times larger than the laptop's.

*A homography-based degeneracy threshold was recommended, measured, and not
shipped* — see *Measurement discipline*.

**Errors the assistant made that measurement caught**, recorded because they
calibrate how much to trust the rest: an incomplete Three.js vendoring that
worked in no browser; a synthetic test scene built from isolated point sprites
that was unsolvable for reasons that took a real investigation to find (ORB's
31 px descriptor window around a 7 px dot describes its *neighbours*, which move
differentially under parallax); a `uint8` wraparound that turned gentle sensor
noise into violent speckle; a file-upload control that could never open because
a click handler recursed with the input it contained; and a path-length figure
that was wrong in a throwaway diagnostic script because `np.linalg.norm` was
given the wrong axis — caught only because subsampling an ordered path appeared
to make it longer, which is geometrically impossible.
