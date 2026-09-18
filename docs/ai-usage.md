# AI usage — running note

Kept as we go, per CLAUDE.md, so the README's AI Usage section summarizes a real
record rather than a reconstruction written at the end.

Format: what was suggested, what was decided, and why the difference. The
rejected recommendations are the interesting half — they are where a human
judgement call overrode a plausible-sounding one.

---

## 17 Sep 2026 — planning

### Rejected: split the frontend into a Next.js app on Vercel

**Suggested.** Move the UI into a separate Next.js deployable hosted on Vercel,
with the FastAPI SLAM service on EC2 behind it. Reasoning offered at the time:
familiar deployment story, and a static export could be hosted on both Vercel
and EC2 to satisfy the "deploy to AWS" requirement unambiguously.

**Rejected.** The original decision stands: one deployable, FastAPI serves
`app/static/`, vanilla JS, no framework, no build step.

**Why the rejection is right.** The split buys nothing this project needs and
charges real cost for it:

- It manufactures a cross-origin problem that does not otherwise exist — CORS
  middleware, an API base URL to configure per environment, and preflight
  behaviour to debug — on a four-day build.
- It forces HTTPS onto the critical path. A Vercel page is HTTPS, and a browser
  hard-blocks `fetch` from HTTPS to a plain-HTTP EC2 endpoint as mixed content.
  That means a domain and a certificate before the first frame is ever decoded.
- It adds a second deployable, a second dashboard, and a build step to a project
  whose stated goal is code the author can explain line by line in an interview.
- It weakens requirement 7. The brief says "deploy to AWS" with no escape hatch,
  and the page a reviewer actually opens would not have been on AWS.

The suggestion optimized for a conventional modern stack. The assignment
rewards a small, measurable, explainable one.

### Rejected: Clerk authentication, ported from Assignment 1

**Suggested.** Port `app/auth.py` from the business-card extractor — Clerk
session tokens verified RS256 against the JWKS endpoint — so jobs are scoped per
user and another user's job id returns 404.

**Initially accepted, then reversed on reflection. Final decision: no auth at
all.**

**Why.** The deliverable is a demo a reviewer opens once. A sign-up wall between
that reviewer and the upload form is pure friction against the thing being
assessed, and it buys a property nobody evaluating a SLAM assignment is testing.

The underlying requirement — that job ids cannot be enumerated — survives
without auth, via a different mechanism: ids are `secrets.token_urlsafe` tokens,
and an unknown id returns 404 rather than 403, so a probe cannot distinguish
"exists but not yours" from "does not exist". Three dependencies
(`PyJWT[crypto]`, `certifi`, `httpx`) left the requirements file with it.

### Corrected: a factual error about AWS free-tier eligibility

Claimed during planning that `m7i-flex.large` is not free-tier eligible, on the
basis of the older t2/t3.micro free tier. **This was wrong** — the EC2 console
labels it free-tier eligible, and the current AWS Free Plan permits it while
blocking GPU instance types. Corrected before it could reach the README.

### Accepted: measure before building

Write `vslam/timing.py` and `tools/probe.py` first, and run the probe on the
target EC2 instance before any pipeline architecture is built on top of its
assumptions. If decode + ORB + matching alone consumes the 10-second budget, the
locked parameters change before code depends on them, not after.

### Accepted: synthetic ground truth alongside real footage

Phone clips alone cannot support the claim in requirement 4 ("minimize
accumulated pose error and drift") — without a known trajectory, drift can only
be argued visually. `tools/make_synthetic.py` renders a known point set along a
known camera path, which turns drift into an ATE number.

### Accepted: deploy an empty shell on day one

Stand up FastAPI + nginx + systemd on EC2 with a stub worker before the SLAM
pipeline exists, so deployment failures (security groups, wheel builds, firewall
rules) surface on day one instead of on the deadline.

### Accepted: vendor Three.js rather than load it from a CDN

The page then has no external dependency at review time, and works if the
reviewer's network blocks the CDN.

---

## 18 Sep 2026 — hardening pass

### Rejected on measurement: projection-guided matching

**Suggested and implemented.** Replace brute-force descriptor matching against
the local map with a projection-guided search: predict the pose, project each map
point into the image, and compare only against keypoints within a few pixels. It
is what ORB-SLAM does and the isolated evidence was strong — at the exact frame
where tracking died, 203 matches against brute force's 90.

**Rejected.** End to end it was consistently worse: 23 poses against 59, across
every radius and ratio tested.

**Why the isolated measurement misled.** It used the *true* pose. The pipeline
only has a *predicted* one, and constant-velocity prediction error was measured
at a median of 8-23 px with a maximum of 165 px — so correct matches routinely
fell outside the search disc. Two further errors surfaced while investigating:
the radius was never the live variable (a `k=8` neighbour cap binds first, which
is why r=30, 60 and 100 gave identical results), and the raw match counts were
never comparable (brute force assigns several keypoints to one map point, where
the projection matcher enforces one-to-one). Kept behind a flag, defaulted off.

### Rejected: retuning the cull threshold on one sequence

Sweeping it showed 0.5 px gives the best ATE and the lowest cost on fr1_xyz.
**Not adopted.** It is one clip, and fitting a threshold to one clip is the exact
failure this project has avoided elsewhere. What *was* adopted is the structural
finding: at the shipped 5 px the cull removed **zero** points across the whole
sequence, because triangulation already gates at 4 px — so it could never fire.

### Corrected: a scaling bug that made bundle adjustment a no-op

BA reduced reprojection error by 0.2% and recovered 0.1% of a known injected
perturbation. The cause was `x_scale`: with isotropic scaling across a parameter
vector mixing radians, translations and 3D coordinates, `trf` terminated after
two evaluations believing it had converged while the gradient norm was 3143.
With `x_scale="jac"`, recovery of the same perturbation went to 86.6%.

**Recorded because of how nearly it was missed.** Three experiments varying
`diff_step`, tolerances and `x_scale` all returned byte-identical results, which
read as evidence that none of them mattered. They were no-ops — `ba.py` imports
`least_squares` by name, so patching `scipy.optimize.least_squares` never touched
it. *Identical numbers across varied inputs indicate a broken experiment, not an
insensitive system.*

### Modified: scikit-learn was specified for the loop-closure vocabulary

The brief asked for `sklearn.MiniBatchKMeans` to cluster ORB descriptors.
**Used `cv2.kmeans` instead.** OpenCV is already a dependency and this project's
entire argument is about what fits in a constrained compute budget; adding a
large dependency for one function call would sit badly with that. No accuracy
claim is made either way — it is a dependency decision, not a numerical one, and
the binary-descriptor caveat (k-means on binary data treated as floats is an
approximation to DBoW2's k-majority) is documented in `vslam/bow.py`.

### Corrected: figures quoted from the laptop rather than the target

The brief restated per-stage timings that were the development laptop's
(`match_map` 1.61 ms, 7.2%). On the deployment instance the same stage measures
6.62 ms and 27.4% — four times larger. The Lucas-Kanade rejection still holds,
but on a corrected premise, and the README says so.

### Corrected: an assumption about how drift grows

The brief stated drift is superlinear and that full-sequence error would exceed
four times the measured figure. Measured over growing prefixes, ATE fits
`path^0.46` — sublinear, with the error *fraction* falling. The README refuses to
extrapolate in either direction, because 23% of one confined sequence supports
neither claim.

### The honest ablation

Skip-and-retry plus relocalization took coverage on fr1_xyz from 22.7% to 99.7%.
Ablating `relocalize()` entirely changed **nothing** — it had never once
succeeded. The whole gain came from not giving up after a single failed frame.
Reported that way rather than crediting the more sophisticated component.
