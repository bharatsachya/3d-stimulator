# Monocular sparse SLAM

Upload a video from a single ordinary camera and get back the estimated camera
trajectory and a sparse 3D point cloud, rendered in an orbitable 3D view with a
measured per-stage timing breakdown. It runs on two free-tier vCPUs with no GPU.

**Live:** http://13.63.181.231/

---

## Results

Accuracy is measured against the [TUM RGB-D benchmark](https://cvg.cit.tum.de/data/datasets/rgbd-dataset)
sequence `freiburg1_xyz`, whose ground-truth trajectory comes from a 100 Hz
motion-capture system. Every figure below was produced on the deployment
instance by:

```bash
python tools/evaluate.py samples/tum_fr1_xyz.mp4 samples/tum_fr1_xyz.truth.json --runs 3
```

| metric | value |
|---|---|
| ATE RMSE | **1.56 cm** |
| ATE median | 1.28 cm |
| ATE max | 4.24 cm |
| ground-truth path over the evaluated segment | 1.980 m |
| **RMSE as a fraction of path length** | **0.79%** |
| poses evaluated | 62 |
| keyframes | 17 |
| map points | 2121 |
| mean reprojection error | 0.813 px |
| run-to-run standard deviation over 3 runs | 0.000 cm |

### Read the coverage before the accuracy

**That 1.56 cm covers 62 poses spanning source frames 0–183 of 798 — 23.4% of
the sequence. Tracking is lost at frame 186 and there is no relocalization, so
the run ends there.** (62 poses is the expected count for that span, since the
pipeline processes every third frame; 62/798 is not the coverage figure and
quoting it that way would understate coverage as 7.8%.) An ATE quoted without its coverage is close to meaningless, because a
system that gives up early reports a better number than one that keeps going,
for the worst possible reason. The honest one-line statement is:

> ATE RMSE 1.56 cm over the tracked segment — 62 poses covering frames 0–183 of
> 798, 23.4% of the sequence, 1.980 m of ground-truth path — with tracking lost
> at frame 186 and no relocalization implemented.

**Bundle adjustment is not implemented yet.** This figure is what map-based
tracking, the parallax gate and reprojection culling achieve on their own.

### What happens to error as the path grows

Rather than assert how drift would extrapolate to the full sequence, it was
measured over growing prefixes of the tracked segment:

| poses | ground-truth path | ATE RMSE | % of path |
|---|---|---|---|
| 10 | 0.379 m | 0.84 cm | 2.23% |
| 24 | 0.840 m | 1.14 cm | 1.36% |
| 38 | 1.335 m | 1.61 cm | 1.20% |
| 52 | 1.820 m | 1.49 cm | 0.82% |
| 59 | 1.941 m | 1.67 cm | 0.86% |

A power-law fit gives `ATE ~ path^0.46` — sublinear, with the error *fraction*
falling as the path lengthens. That is not a general claim about the system and
should not be read as one. `fr1_xyz` confines the camera to a box roughly 0.94 m
across and repeatedly revisits it, so the camera keeps re-observing map points it
has already triangulated instead of extending into new territory. That is
map-based tracking working as intended, and it is also exactly the condition
under which drift is least visible. **A sequence that explores new space would
behave differently, and that has not been measured here.** No extrapolation to
the full 798 frames is offered in either direction.

### On the alignment, and why scale is solved for

ATE is computed after aligning the estimated and true trajectories with a
**similarity transform — rotation, translation and one scale factor, seven
degrees of freedom**, using Umeyama's closed-form solution.

The scale term is not a convenience. A monocular reconstruction is defined only
up to scale, because a small scene viewed closely and a large one viewed from
far away produce identical images. Comparing against metric ground truth without
solving for scale would measure that arbitrary choice rather than any error the
system made. Seven-DOF alignment is the standard procedure for monocular
evaluation and is what the ORB-SLAM papers report for their monocular results.
The recovered factor on this run is 0.0388, which is meaningful only as "the
arbitrary units are about 1/26 of a metre each".

The alignment deliberately **refuses to absorb reflections**: its rotation is
constrained to determinant +1. That constraint caught a real class of bug, as
described under *Measurement discipline* below.

### Path length, and why it is stated with its interval

Path length depends on which interval and which sampling rate you measure, so
quoting a percentage without them is not reproducible. Measured from
`groundtruth.txt` directly:

| interval | samples | path |
|---|---|---|
| full motion-capture span, 100 Hz | 3000 | 9.159 m |
| RGB frame span, every frame | 796 | 8.011 m |
| RGB frame span, at the 10 Hz processing rate | 266 | 7.985 m |
| **evaluated (tracked) segment** | **62** | **1.980 m** |

The 8.011 m against 7.985 m difference shows the figure is not inflated by
motion-capture jitter — changing the sampling rate moves it by 0.3%, so the path
is genuine motion rather than accumulated noise.

Published summaries of this sequence quote a trajectory length that differs from
the 9.159 m measured here from the distributed `groundtruth.txt`. That
discrepancy was noticed and deliberately not resolved by guessing: the convention
behind the published figure has not been established, so no claim is made about
it. Every path length in this document is computed from the distributed file by
the command shown above, with its interval and sampling rate stated, so a reader
can reproduce each one exactly.

### Failure modes produce different advice

Two clips that must fail, and do:

| clip | diagnosis | what the user is told |
|---|---|---|
| synthetic pure rotation | median **484** matches, **0.00°** parallax | move sideways through the scene |
| TUM `nostructure_notexture_far` | median **3** matches | the scene needs texture |

Making these differ was deliberate work, not a formatting exercise. Both look
identical from the outside — initialization fails — but *"try walking sideways
past the subject"* is actively wrong advice for someone filming a blank wall, and
*"find more texture"* is useless to someone who is standing still and turning.
The pipeline distinguishes them by how far the failure got: plenty of matches
with no parallax means the camera did not translate; almost no matches means
there was nothing to match.

---

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
instance, `tum_fr1_xyz.mp4`, 640 px working width, 1000 features:

| stage | ms/frame | % of wall |
|---|---|---|
| match_map | 6.62 | 27.4 |
| orb | 6.55 | 27.1 |
| cull | 4.14 | 17.2 |
| decode | 1.96 | 8.1 |
| triangulate | 1.49 | 6.2 |
| pnp | 1.45 | 6.0 |
| local_map | 0.70 | 2.9 |
| initialize | 0.36 | 1.5 |
| unaccounted | — | 3.5 |
| **total** | **24.14** | 100 |

**24 ms per frame against a 100 ms budget, on two free-tier vCPUs with no GPU.**
The modest hardware is the point of the result, not an apology for it: the brief
asks for an algorithm that performs under constrained resources, and meeting the
budget four times over on the smallest sensible instance is the claim.

`unaccounted` is wall clock minus the sum of the measured stages. It is reported
so that the breakdown cannot silently omit time; if it ever grows, something real
is happening outside the instrumented stages.

### The workload does not want more cores

Measured with `tools/probe.py`, varying OpenCV's thread count. **On the target
instance**, which has 2 vCPUs:

| threads | ms/frame (floor) |
|---|---|
| 1 | 13.3 |
| 2 | 13.1 |

Essentially flat — the second thread buys about 2%. That is consistent with what
`lscpu` reports: the instance's two vCPUs are two hardware threads on a *single
physical core*, so there is no second execution unit for the work to spread
across. This pipeline is effectively single-core-bound on this machine.

The development laptop, which has eight real cores, behaves completely
differently on the same clip and settings — 29.85 ms/frame at one thread, 17.67
at two, 18.26 at four, 19.16 at eight. There the second thread is worth 40% and
scaling stops after that.

The two tables together are the argument for the small instance. A larger one
would add cores this workload demonstrably cannot use, and the target already
meets the budget four times over on a single physical core. That is a measured
conclusion rather than a rationalisation — and note that the laptop table alone
would have supported a *wrong* one, namely that two threads matter a great deal.

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

### Bundle adjustment — designed, not yet implemented

**Status: not implemented.** The accuracy figures above are pre-BA. The design
is settled and the pipeline has a place for it at the keyframe step:

A **sliding window over the last 5 keyframes**, not full history. Full-history
cost grows with video length, so a longer clip would blow the time budget for
reasons unrelated to how hard the reconstruction is. `scipy.optimize.least_squares`
with method `trf` and a sparse `jac_sparsity` mask, Huber loss at δ = 2 px, a
hard `max_nfev = 50` ceiling so the optimizer cannot eat the budget, and the
first keyframe's pose held fixed to fix the gauge.

SciPy rather than g2o or Ceres is deliberate and is a real tradeoff, not a
free choice: SciPy installs from a wheel with no build step and every line of the
residual function is readable, which matters for a system whose author must be
able to explain it. It is meaningfully slower than Ceres. That is the cost, and
it is accepted knowingly.

### Rejections, on measurement

**Lucas–Kanade inter-frame tracking was rejected on profiling data.** Replacing
descriptor matching with optical-flow tracking between frames is a well-known way
to cut per-frame cost, and it is what several reference implementations do. On
the target instance `match_map` costs 6.62 ms of a 24.14 ms frame, so the
absolute ceiling on the saving is about 6.6 ms — against 76 ms of unused budget.
Removing it would not change whether the system meets its target.

It is worth being precise about how this number was arrived at, because the
laptop profile would have supported a much stronger version of the same
conclusion: there, matching is 1.61 ms and the ceiling looks like 1.5 ms. The
target says 6.62 ms — four times larger. The conclusion survives the correction,
but only because the headroom is large; a tighter budget would have made this
worth doing. LK's remaining genuine case is robustness through gradual appearance
change, which is a different argument and not one about speed.

**A C++ rewrite was considered and rejected.** ORB detection, brute-force
matching and PnP are already compiled OpenCV. Python calls into them on the order
of a hundred times per frame, and interpreter dispatch is microseconds against
milliseconds of native work. The measured Python overhead is the 3.5%
`unaccounted` row, and that is an upper bound including genuine untimed work. A
rewrite would target a few percent while discarding readability entirely.

**Dual-model homography/essential initialization was rejected**, as described
under *Measurement discipline*: the signal separating the cases was measured,
found to have a 0.09 margin on three clips, and shipped as reported evidence
rather than as a branch.

---

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

**Tracking loss with no relocalization.** On `fr1_xyz` tracking is lost at frame
186 of 798 and the job returns a flagged partial trajectory. This is specified
behaviour — stop, return what was reconstructed, flag it — but it is also the
largest single gap between this system and a complete one, and it is why the
headline accuracy figure covers 23.4% of a sequence rather than all of it.

**No loop closure.** The system cannot recognise that it has returned to a place
it has already mapped, so accumulated drift is never corrected globally.

**No inertial fusion.** Phone IMU data would constrain rotation between frames
and give metric scale, both of which address problems listed above. Out of scope.

**No lens distortion correction.** Correcting it requires calibration parameters
we do not have. Uncorrected radial distortion bends straight lines near the frame
edge, which biases exactly the features with the most parallax information. The
TUM `freiburg1` footage used for evaluation is *not* pre-undistorted, so the
reported accuracy includes this effect.

**Sparse, not dense.** The output is a point cloud at ORB keypoints, not a
surface or a mesh.

**In-memory job state, single worker.** Jobs do not survive a restart, and state
is process-local so the service cannot be scaled to a second replica without
replacing `app/store.py`. Concurrency is fixed at one because the measured
workload already stops scaling past two threads: a second concurrent job would
not finish sooner, it would make both miss the budget.

**Duration is enforced server-side, size both.** Uploads are capped at 25 MB
(enforced mid-stream and by nginx) and 60 seconds.

---

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
