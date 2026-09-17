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
