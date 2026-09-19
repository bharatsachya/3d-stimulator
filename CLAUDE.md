# CLAUDE.md

Project instructions for this repository. Read this before writing code.

## What this is

A monocular RGB sparse point-cloud SLAM web application. Upload a video, get back
the estimated camera trajectory and a sparse 3D point cloud, rendered in an
orbitable 3D view, with measured per-stage timings.

Built as a take-home assessment for O-HIVE (LuxPM Co., Ltd., Seoul). Their
product is real-time on-device 3D spatial intelligence using cost-efficient
cameras. Their CEO stated explicitly that the hardware limits in this assignment
are deliberate: *"we are also power efficient, so the assignments are designed
in such a way that challenges candidates to come up with algorithms that can
perform with limited hardware resources."*

**That reframes the whole task. Efficiency under constraint IS the deliverable,
not a caveat.** A beautiful point cloud with no timing data scores worse than an
uglier one with a proper measurement table.

## Hard requirements from the brief

1. Single-lens RGB sparse point-cloud SLAM
2. User uploads a video
3. Estimate camera pose and trajectory
4. Implement an approach to minimize accumulated pose error and drift
5. Generate and visualize the sparse 3D point cloud and camera trajectory
6. 10-second input video → target processing time ≤ 10 seconds, under the
   demonstrated test environment
7. Deploy to AWS, publicly accessible URL

Submission also requires: git repo, README with setup/deployment instructions,
architecture and major technical decisions, libraries and pretrained models used,
known limitations, **measured processing time and test environment**, and an
"AI Usage" section.

Note requirement 7: Assignment 2 says "deploy to AWS" with no "or equivalent"
escape hatch. AWS, not Azure, not Modal.

## Target environment

AWS EC2 `m7i-flex.large` — 2 vCPU, 8 GB RAM, **no GPU**, Ubuntu 24.04.
Free-tier eligible. All timing claims must be measured on this instance, never
on a dev laptop.

Sparse SLAM is CPU work, so no GPU is needed. Do not propose GPU inference,
serverless GPU, or CUDA anything.

## Locked design decisions

These are settled. Do not re-litigate them mid-build; if you think one is wrong,
say so before writing code.

### Input
| Parameter | Value |
|---|---|
| Processed frame rate | 10 fps (~100 frames for a 10s video, 100ms/frame budget) |
| Working resolution | 640px long edge |
| Colour | Grayscale (ORB ignores colour) |
| Limits | Reject > 60s or > 25MB at upload |

### Features and matching
| Parameter | Value |
|---|---|
| Detector | ORB — free, fast, binary descriptors. SIFT is 5-10x slower and the time budget rules it out |
| Features per frame | 1000 (tunable) |
| Pyramid | 8 levels, scale 1.2 (OpenCV defaults; scale invariance matters when approaching objects) |
| Matcher | BFMatcher Hamming + Lowe ratio test at 0.75 — rejects ambiguous matches, not merely non-mutual ones |

### Intrinsics
| Parameter | Value |
|---|---|
| Focal length | `0.9 * width`, user-overridable |
| Principal point | Image centre |
| Lens distortion | Ignored — correcting it needs calibration we don't have. Document as a limitation |

### Initialization
| Parameter | Value |
|---|---|
| Search window | Frames 0-30, advance until parallax sufficient |
| Min median parallax | 1.0 degrees |
| Min essential-matrix inliers | 50 |
| RANSAC threshold | 1.0 px |
| On failure | Fail the job with "insufficient parallax — camera may be rotating in place". **Fail loudly. Never silently emit garbage.** |

### Tracking
| Parameter | Value |
|---|---|
| Method | `solvePnPRansac` against **map points**, not frame-to-frame |
| Reprojection threshold | 3.0 px |
| Min inliers to accept a pose | 30 |
| On tracking lost | Stop, return the partial trajectory, flag it. No relocalization |

Tracking against a persistent map rather than chaining frame-to-frame is the
single largest drift reduction available. Do not replace it with frame-to-frame
odometry for simplicity.

### Keyframes and map
| Parameter | Value |
|---|---|
| Keyframe trigger | Translation > 10% of median scene depth, OR tracked-point ratio < 0.7 |
| Min frames between keyframes | 3 |
| Triangulate only if parallax | > 1.0 degrees |
| Cull map points with reprojection error | > 5 px after BA |
| Min observations to retain a point | 2 |

Keyframe triggers are scale-relative, not absolute, because there are no units.

### Bundle adjustment
| Parameter | Value |
|---|---|
| Scope | Sliding window over the last 5 keyframes |
| Frequency | On every new keyframe (fall back to every 2nd if too slow) |
| Solver | `scipy.optimize.least_squares`, method `trf`, sparse Jacobian via `jac_sparsity` |
| Loss | Huber, delta = 2px |
| Iteration cap | `max_nfev = 50` — a hard ceiling so BA cannot eat the time budget |
| Gauge | First keyframe pose held fixed |

Sliding window, not full history: full-history cost grows with video length and
would blow the budget immediately.

SciPy over g2o/Ceres is deliberate — it installs from a wheel and every line is
readable. Slower than Ceres, and that tradeoff belongs in the README.

### Output
| Parameter | Value |
|---|---|
| Units | Arbitrary, labelled as such. **Never print metres** |
| Coordinates | Convert OpenCV Y-down to Three.js Y-up once, at export |
| Format | JSON — poses as 4x4 matrices, points as xyz + observation count |

## Tunable by measurement, not by guess

Five parameters are starting points to be swept and recorded: processed frame
rate, working resolution, features per frame, BA window size, BA frequency.

For each, record time and quality impact in a table. That table is the answer to
the CEO's email about power efficiency. Do not tune these by intuition.

## Instrumentation — build it in from the first commit

Every stage times itself and reports ms/frame: decode, ORB detect+compute,
matching, PnP, triangulation, bundle adjustment. Return totals plus the
per-stage breakdown with the job, and render it on the results page.

Retrofitting timing is annoying and it is an explicit submission requirement.
Build it in first.

## Explicitly out of scope

Loop closure, relocalization, IMU fusion, lens distortion correction, metric
scale, dense reconstruction, multi-video sessions.

Each of these gets a line in the README limitations section. Do not silently
omit them and do not start implementing them without asking.

## Architecture

Reuse the pattern from Assignment 1 (the business-card extractor) — same shape,
different worker:

- FastAPI, `POST /api/jobs` returns **202 with a job ID in milliseconds**
- Processing in a background task, client polls `GET /api/jobs/{id}`
- Never process synchronously in the request: nginx and browser timeouts are
  both 60s, and a long request dies and loses all completed work
- Bytes spooled to disk, read inside the concurrency semaphore so peak memory is
  bounded by concurrency rather than by input size
- 404 (not 403) on another user's job id, so ids can't be enumerated
- Typed failure statuses on every stage; a failure becomes a flagged result,
  never a 500

Visualization: Three.js — `Points` for the cloud, `Line` for the trajectory,
`OrbitControls` for navigation, small frusta at keyframes if time allows. Plain
vanilla JS, no framework, no build step.

## The two things that cannot be fixed

Say these plainly in the README; do not paper over them.

**Scale is unobservable.** A single lens cannot distinguish a small near object
from a large far one — like a close-up photo of a rock formation with no hammer
in frame for scale. The reconstruction is correct up to an unknown scale factor,
and that factor drifts over the sequence. This is a property of the sensor, not a
limitation of the implementation. Mitigated by map-based tracking, the
minimum-parallax check, and bundle adjustment. Corrected globally only by loop
closure with Sim(3) pose-graph optimization, which is out of scope.

**Focal length is estimated, not measured.** It is *knowable* in principle —
unlike scale — but we accept arbitrary uploads from unknown cameras. Hence the
heuristic plus a user override.

## Known failure modes

Test these deliberately and document them:
- **Pure rotation** — no parallax, initialization never succeeds
- **Motion blur** — smears the corners ORB depends on
- **Low texture** — blank walls, clean floors yield too few matchable keypoints
- **Tracking loss mid-sequence** — returns a partial trajectory

## How to work with me on this

- I am new to SLAM. Explain the reasoning as you go, and stop to check I'm
  following before moving to the next major piece.
- **I will be interviewed about this code and asked to modify it live.** I must
  be able to explain every line. Prefer code I can read over code that is
  clever.
- Build in stages and let me test each one before continuing. Suggested order:
  input/features → initialization → tracking → keyframes+BA → export → viewer →
  app wrapper → measurement sweep.
- Ask before adding any dependency.
- Ask before changing anything in the "Locked design decisions" section.
- Keep a running note of significant suggestions I rejected or modified — that
  is the README's AI Usage section and it is much easier to write as we go.

## Deadline

21 September 2026. Assignment 1 is substantially built; this one is the priority.

Priority order if time runs short: a deployed, measured, honestly-documented
core pipeline beats an undeployed sophisticated one. Deploy early, then improve.

---

## Decisions settled during planning (17 Sep 2026)

These join the locked set above. Same rule: don't re-litigate mid-build.

### No authentication
A reviewer opening the public URL must be able to upload a video immediately.
No sign-up, no Clerk, no bearer tokens, no per-user scoping.

The enumeration requirement is still met, by a different mechanism: a job id is
an unguessable random token (`secrets.token_urlsafe`), and an id that is not in
the store returns **404, not 403**. 403 would confirm that an id exists, which is
the leak the requirement is about.

### One deployable — the frontend is not split out
FastAPI serves `app/static/` directly. Plain vanilla JS, no framework, no build
step, as written in the Architecture section above. Everything is same-origin,
so there is no CORS middleware, no API base URL to configure, and every `fetch`
uses a relative path.

Three.js is **vendored** into `app/static/vendor/` rather than loaded from a CDN,
so the page has no external dependency at review time. Plain Three.js, not
react-three-fiber.

Frontend files: `index.html`, `app.js` (upload + poll + render), `viewer.js`
(the Three.js scene), `style.css`.

Polling uses `setTimeout`, scheduled after each response lands — never
`setInterval`. Under `setInterval` a slow response stacks further requests on an
already-saturated 2-vCPU box, which is precisely when extra load hurts most.

### Upload cap: 25MB
Not 100MB. A 10-second clip at any sane phone setting is a few MB, and a 100MB
4K file is downsampled to 640px long edge before a single feature is detected --
so the extra bytes buy nothing and cost upload time and disk. The cap is stated
in the UI, with short clips recommended.

Duration > 30s is still rejected, checked after decode opens the file.

### Test data -- both sources
Real phone clips *and* a synthetic sequence.
- Phone: 2-3 short clips -- a textured desk or bookshelf, a sideways walk past
  objects, and one deliberate pure-rotation clip to demonstrate the documented
  failure mode.
- Synthetic: `tools/make_synthetic.py` renders a known 3D point set along a
  known camera path. Ground truth exists, so drift is a measured number (ATE
  against the known path) rather than an adjective. Requirement 4 says
  "minimize accumulated pose error" -- that claim needs a number behind it.

### Dependencies -- approved list
`numpy`, `opencv-python-headless`, `scipy`, `fastapi`, `uvicorn[standard]`,
`python-multipart`. That is the whole list. Nothing here makes outbound HTTP
calls, so there is no HTTP client dependency. Anything beyond this list needs a
fresh ask.

### Build order -- one deviation from the suggested order
The *empty shell* of the app wrapper deploys at stage 0, not stage 7: FastAPI +
202 + poll + nginx + systemd on EC2, with a stub worker returning canned JSON.
Deployment fails for reasons unrelated to SLAM (security groups, wheel builds,
firewall rules) and that risk is worth retiring on day one. The real worker drops
into the same seam at stage 7.

### HTTPS -- done, after the pipeline
Plain HTTP was sufficient for the deliverable, since a same-origin frontend has
no mixed-content problem, so this was deliberately left until the pipeline
existed. It is now served over TLS at https://13.63.181.231.sslip.io/.

A publicly-trusted certificate cannot be issued for a bare IP by the ordinary
ACME path, so HTTPS needs a hostname. sslip.io resolves 13.63.181.231.sslip.io
to 13.63.181.231 by construction with no registration, which satisfies Let's
Encrypt's HTTP-01 challenge. The nginx server blocks live in deploy/nginx.conf
rather than being written by `certbot --nginx`: the config is redeployed from
the repository, so a hand-edit on the box would be silently reverted and TLS
would break with no obvious cause.

### Instance isolation
SLAM gets its **own** `m7i-flex.large`. It does not share 2 vCPUs with any other
workload, because a co-tenant process makes the 10-second budget unreachable and
every measurement in the README meaningless.
