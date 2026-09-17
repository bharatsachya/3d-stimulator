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

## Still to measure

- [ ] **The same probe on the EC2 `m7i-flex.large`.** This is the number that
      counts; everything above is shape-finding. Expect single-core speed, not
      core count, to drive the difference.
- [ ] Bundle adjustment cost per keyframe — the one stage with real risk of
      eating the budget.
- [ ] The five-parameter sweep: processed fps, working resolution, features per
      frame, BA window size, BA frequency.
- [ ] End-to-end wall clock for a 10s clip, which is what the brief actually asks.
