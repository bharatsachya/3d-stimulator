"""
Render a synthetic video with a KNOWN camera trajectory and KNOWN 3D structure.

WHY THIS EXISTS

Requirement 4 asks us to "minimize accumulated pose error and drift". That is a
quantitative claim, and on phone footage there is nothing to check it against --
you can look at a trajectory and say it appears smooth, which is an opinion.
Here the camera path is something we chose, so the estimated path can be
compared against it and drift becomes a number (ATE) that goes in the README.

It also solves a mundane problem: this machine has no ffmpeg, and a test clip is
needed before any pipeline code can be run at all.

WHY TEXTURED SURFACES AND NOT A CLOUD OF DOTS

The first version of this file rendered each 3D landmark as its own small
patch floating on a flat background. It looked plausible and it did not work:
ORB matches collapsed as the baseline grew, roughly four-fold per doubling,
which no amount of retuning density or patch design fixed.

The reason is worth writing down, because it is really a fact about ORB rather
than about the renderer. ORB describes a ~31px neighbourhood around a corner.
Around a 7px isolated dot, that neighbourhood is mostly OTHER dots -- and those
neighbours sit at different depths, so parallax moves them by different amounts.
The descriptor was therefore describing the accidental arrangement of nearby
landmarks, which changes with every camera movement, rather than the landmark
itself.

Real scenes do not behave that way because real landmarks live on surfaces. A
corner on a textured wall has a neighbourhood that is more of that same wall,
and it moves coherently with the corner. So this version renders textured
planes, warped by the exact homography their pose implies.

WHY THE SCENE IS NOT ONE PLANE

A purely planar scene is DEGENERATE for essential-matrix estimation: every
correspondence is explained by a homography, epipolar geometry is not uniquely
determined, and `findEssentialMat` returns a confident wrong answer. So the
scene is a room -- back wall, two side walls, floor, ceiling -- plus free-standing
panels at assorted depths in the middle of the frustum. The panels matter: with
walls alone, almost everything inside the camera's field of view lands on the
back wall and the scene is near-planar in practice even though it is not in
principle.

MOTION MODES

  dolly   forward + lateral translation, gentle yaw. The good case: parallax
          accumulates steadily and initialization should succeed early.
  orbit   circular arc about the scene centre, always looking inward. Strongest
          parallax; the easiest case for triangulation.
  rotate  pure yaw about the camera centre, ZERO translation. The documented
          failure case -- no translation means no parallax means no triangulable
          structure, and initialization must fail loudly rather than emit
          garbage. This clip is what proves that behaviour.

USAGE

  python tools/make_synthetic.py --motion dolly --out samples/synth_dolly.mp4
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Y IS DOWN. This module works entirely in OpenCV's camera convention -- x
# right, y down, z forward into the scene -- because that is what every OpenCV
# call the pipeline makes assumes. The single conversion to a Y-up world for
# Three.js happens once, at export, and nowhere else.
WORLD_UP = np.array([0.0, -1.0, 0.0])

# Every surface starts at z >= 3.0 and the camera never travels past z = 2.7.
# That is deliberate: it guarantees no quad ever crosses the image plane, so the
# renderer needs no near-plane clipping. Clipping a warped quad correctly is
# real work, and this scene simply does not need it.
NEAR_SAFE_Z = 3.0


@dataclass
class Quad:
    """A textured planar surface. Corners are ordered TL, TR, BR, BL."""

    corners: np.ndarray  # (4, 3) world coordinates
    texture: np.ndarray  # (H, W) uint8


def look_at(eye: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a world->camera rotation and translation for a camera at `eye`
    pointing at `target`.

    Returns (R, t) such that  x_camera = R @ x_world + t.
    R's rows are the camera's right / down / forward axes in world coordinates.
    """
    forward = target - eye
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, WORLD_UP)
    right = right / np.linalg.norm(right)

    down = np.cross(forward, right)

    R = np.stack([right, down, forward], axis=0)
    t = -R @ eye
    return R, t


def make_texture(rng: np.random.Generator, size: int = 256, cells: int = 40) -> np.ndarray:
    """
    Blocky random texture: a small random grid upsampled with NEAREST.

    NEAREST, not LINEAR, on purpose. Nearest keeps hard edges between cells, and
    hard edges are what FAST detects as corners. A smoothly interpolated texture
    is all gradient and no corner, so ORB finds far fewer stable keypoints on it.
    """
    small = rng.integers(20, 236, size=(cells, cells), dtype=np.uint8)
    return cv2.resize(small, (size, size), interpolation=cv2.INTER_NEAREST)


def make_scene(rng: np.random.Generator) -> list[Quad]:
    """The room, plus free-standing panels that break its planarity."""
    quads: list[Quad] = []

    def add(corners):
        quads.append(Quad(np.asarray(corners, dtype=float), make_texture(rng)))

    x0, x1 = -5.0, 5.0
    y0, y1 = -3.5, 3.5  # y down: y0 is the ceiling, y1 the floor
    z0, z1 = NEAR_SAFE_Z, 11.0

    # Back wall (z = z1)
    add([[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    # Left wall (x = x0) and right wall (x = x1)
    add([[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]])
    add([[x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]])
    # Ceiling (y = y0) and floor (y = y1)
    add([[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]])
    add([[x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0]])

    # Free-standing panels. These are what put genuinely different depths in the
    # middle of the frame, where most of the matched features actually live.
    for _ in range(9):
        cx = rng.uniform(-2.8, 2.8)
        cy = rng.uniform(-2.0, 2.2)
        cz = rng.uniform(4.0, 9.0)
        half_w = rng.uniform(0.5, 1.1)
        half_h = rng.uniform(0.5, 1.1)
        # Tilt each panel slightly so they are not all parallel to the back wall,
        # which would reintroduce the planarity problem in a subtler form.
        yaw = rng.uniform(-0.6, 0.6)
        pitch = rng.uniform(-0.3, 0.3)
        cy_, sy_ = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        right = np.array([cy_, 0.0, -sy_]) * half_w
        down = np.array([0.0, cp, sp]) * half_h
        centre = np.array([cx, cy, cz])
        add([centre - right - down, centre + right - down,
             centre + right + down, centre - right + down])

    return quads


def camera_path(motion: str, n_frames: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate the ground-truth (R, t) for every frame."""
    poses = []
    centre = np.array([0.0, 0.0, 9.0])

    for i in range(n_frames):
        s = i / max(n_frames - 1, 1)  # 0 -> 1 across the clip

        if motion == "dolly":
            # Lateral motion dominates: for a forward-facing camera it generates
            # far more parallax per metre travelled than forward motion does,
            # because forward motion moves points mostly along the view ray.
            eye = np.array([-2.0 + 4.0 * s, 0.0, 0.3 + 2.2 * s])
            target = centre + np.array([0.6 * s, 0.0, 0.0])
            R, t = look_at(eye, target)

        elif motion == "orbit":
            angle = np.deg2rad(-22.0 + 44.0 * s)
            radius = 7.5
            eye = centre + np.array(
                [radius * np.sin(angle), 0.0, -radius * np.cos(angle)]
            )
            R, t = look_at(eye, centre)

        elif motion == "rotate":
            # PURE ROTATION. The camera centre never moves, so there is zero
            # baseline between any two frames and nothing can be triangulated.
            # Initialization must detect this and fail.
            eye = np.array([0.0, 0.0, 1.0])
            yaw = np.deg2rad(-18.0 + 36.0 * s)
            target = eye + np.array([np.sin(yaw), 0.0, np.cos(yaw)])
            R, t = look_at(eye, target)

        else:
            raise ValueError(f"unknown motion: {motion!r}")

        poses.append((R, t))
    return poses


def render(
    quads: list[Quad],
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Render one frame by perspective-warping every quad into the image."""
    # Mid-grey with mild sensor noise. Note the arithmetic is done in float and
    # clipped BEFORE narrowing to uint8: casting a negative value straight to
    # uint8 wraps it to ~250, which turns gentle noise into violent speckle that
    # ORB will happily spend its whole feature budget on. That was a real bug
    # here, caught because match counts behaved in a way geometry could not explain.
    frame = np.clip(110.0 + rng.normal(0.0, 3.0, size=(height, width)), 0, 255).astype(
        np.uint8
    )

    # Painter's algorithm: far surfaces first, so near ones overwrite them.
    # Correct occlusion for convex, non-intersecting quads, which is all we have.
    def mean_depth(quad: Quad) -> float:
        cam = (R @ quad.corners.T).T + t
        return float(np.mean(cam[:, 2]))

    for quad in sorted(quads, key=mean_depth, reverse=True):
        cam = (R @ quad.corners.T).T + t
        if np.any(cam[:, 2] <= 0.5):
            # Cannot happen with this scene (see NEAR_SAFE_Z) but a renderer that
            # silently divides by a negative depth produces garbage, not an error.
            continue

        projected = (K @ cam.T).T
        image_pts = (projected[:, :2] / projected[:, 2:3]).astype(np.float32)

        # Entirely off-screen: skip the warp rather than pay for it.
        if (
            image_pts[:, 0].max() < 0
            or image_pts[:, 1].max() < 0
            or image_pts[:, 0].min() > width
            or image_pts[:, 1].min() > height
        ):
            continue

        th, tw = quad.texture.shape
        tex_pts = np.float32([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]])
        H = cv2.getPerspectiveTransform(tex_pts, image_pts)

        # Warp the texture and, separately, a solid mask, so the quad's silhouette
        # is composited rather than its black surround being painted over the scene.
        warped = cv2.warpPerspective(quad.texture, H, (width, height))
        mask = cv2.warpPerspective(
            np.full((th, tw), 255, np.uint8), H, (width, height)
        )
        np.copyto(frame, warped, where=mask > 127)

    # A touch of blur: a real lens is never pixel-sharp, and a perfectly sharp
    # synthetic image would flatter ORB relative to the phone clips.
    return cv2.GaussianBlur(frame, (3, 3), 0.6)


def sample_truth_points(quads: list[Quad], rng: np.random.Generator, per_quad: int = 60):
    """
    Sample 3D points across the surfaces, as ground-truth structure.

    Not needed for trajectory ATE -- the poses cover that exactly -- but useful
    for sanity-checking triangulated depth later.
    """
    points = []
    for quad in quads:
        tl, tr, br, bl = quad.corners
        for _ in range(per_quad):
            a, b = rng.random(), rng.random()
            top = tl + (tr - tl) * a
            bottom = bl + (br - bl) * a
            points.append(top + (bottom - top) * b)
    return np.array(points)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a synthetic SLAM test clip.")
    parser.add_argument("--motion", choices=["dolly", "orbit", "rotate"], default="dolly")
    parser.add_argument("--out", default=None, help="output .mp4 path")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=30, help="source fps, as a phone would record")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    out_path = Path(args.out or f"samples/synth_{args.motion}.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    n_frames = int(args.seconds * args.fps)

    quads = make_scene(rng)
    truth_points = sample_truth_points(quads, rng)

    # The SAME focal-length heuristic the pipeline will assume (0.9 * width).
    # Rendering with it means the synthetic test isolates the geometry: if the
    # trajectory comes out wrong, it is not because the intrinsics disagreed.
    # The phone clips are where the heuristic itself gets stressed.
    focal = 0.9 * args.width
    K = np.array(
        [[focal, 0.0, args.width / 2.0], [0.0, focal, args.height / 2.0], [0.0, 0.0, 1.0]]
    )

    poses = camera_path(args.motion, n_frames)

    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height)
    )
    if not writer.isOpened():
        raise SystemExit(f"could not open VideoWriter for {out_path}")

    for i, (R, t) in enumerate(poses):
        gray = render(quads, R, t, K, args.width, args.height, np.random.default_rng(1000 + i))
        # VideoWriter wants 3 channels; the pipeline converts back to gray on
        # read. Encoding a grayscale source as BGR is what a phone does anyway.
        writer.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    writer.release()

    truth = {
        "motion": args.motion,
        "fps": args.fps,
        "width": args.width,
        "height": args.height,
        "K": K.tolist(),
        "note": "poses are 4x4 world->camera; units are arbitrary but CONSISTENT, "
        "unlike the monocular reconstruction which is up to an unknown scale",
        "poses": [],
        "points": truth_points.tolist(),
    }
    for R, t in poses:
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        truth["poses"].append(T.tolist())

    truth_path = out_path.with_suffix(".truth.json")
    truth_path.write_text(json.dumps(truth))

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"wrote {out_path}  ({n_frames} frames, {args.fps} fps, {size_mb:.1f} MB)")
    print(
        f"wrote {truth_path}  ({len(quads)} surfaces, "
        f"{len(truth_points)} sampled landmarks, {n_frames} ground-truth poses)"
    )


if __name__ == "__main__":
    main()
