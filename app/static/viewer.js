/*
 * The 3D view: point cloud, camera trajectory, keyframe frusta.
 *
 * Plain Three.js in a module, not react-three-fiber. Two reasons: there is no
 * build step in this project, and this code has to be explainable line by line.
 *
 * COORDINATES: everything arriving here is already Y-UP. The pipeline works in
 * OpenCV's Y-down convention throughout and converts exactly once, at export.
 * Converting in the viewer instead would mean the exported JSON and the picture
 * disagreed about which way is up, which is the kind of bug that costs an hour.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

let active = null;  // the live scene, so it can be torn down on reset

/** Release GPU resources. Not optional: WebGL contexts are a limited resource,
 *  and creating a new renderer per upload without disposing leaks them until
 *  the browser starts dropping the oldest context. */
export function disposeViewer() {
  if (!active) return;
  const { renderer, controls, container, frameId, onResize, geometries, materials } = active;
  cancelAnimationFrame(frameId);
  window.removeEventListener('resize', onResize);
  controls.dispose();
  geometries.forEach((g) => g.dispose());
  materials.forEach((m) => m.dispose());
  renderer.dispose();
  if (renderer.domElement.parentNode === container) {
    container.removeChild(renderer.domElement);
  }
  active = null;
}

export function renderReconstruction(container, result) {
  disposeViewer();

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x05070a);

  const width = container.clientWidth || 800;
  const height = container.clientHeight || 460;

  const camera = new THREE.PerspectiveCamera(60, width / height, 0.01, 5000);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(width, height);
  // Cap the pixel ratio at 2: beyond that the extra fragments cost real time on
  // a laptop GPU and nobody can see the difference.
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;

  const geometries = [];
  const materials = [];

  const points = result.points ?? [];
  const observations = result.observations ?? [];
  const poses = result.poses ?? [];

  // ---- Map points -------------------------------------------------------
  if (points.length) {
    const positions = new Float32Array(points.length * 3);
    const colors = new Float32Array(points.length * 3);

    // Colour by observation count. A point seen by many keyframes is far better
    // constrained than one seen by the minimum two, and showing that directly is
    // more honest than drawing every point with equal confidence.
    const maxObservations = Math.max(4, ...observations);
    const weak = new THREE.Color(0x2e5d3a);
    const strong = new THREE.Color(0x7ee787);

    points.forEach((point, i) => {
      positions[i * 3 + 0] = point[0];
      positions[i * 3 + 1] = point[1];
      positions[i * 3 + 2] = point[2];

      const confidence = Math.min((observations[i] ?? 2) / maxObservations, 1);
      const colour = weak.clone().lerp(strong, confidence);
      colors[i * 3 + 0] = colour.r;
      colors[i * 3 + 1] = colour.g;
      colors[i * 3 + 2] = colour.b;
    });

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    const material = new THREE.PointsMaterial({ size: 0.035, vertexColors: true });
    scene.add(new THREE.Points(geometry, material));
    geometries.push(geometry);
    materials.push(material);
  }

  // ---- Camera trajectory ------------------------------------------------
  // Each pose is a 4x4 camera-to-world matrix, so its translation column is the
  // camera centre in world coordinates. That column IS the trajectory.
  const centres = poses.map((pose) => new THREE.Vector3(pose[0][3], pose[1][3], pose[2][3]));

  if (centres.length > 1) {
    const geometry = new THREE.BufferGeometry().setFromPoints(centres);
    const material = new THREE.LineBasicMaterial({ color: 0xffb454 });
    scene.add(new THREE.Line(geometry, material));
    geometries.push(geometry);
    materials.push(material);
  }

  // ---- Keyframe frusta --------------------------------------------------
  // A small wireframe pyramid per keyframe, oriented by its pose, so the view
  // direction is visible and not just the path. Without these a trajectory is
  // ambiguous: you cannot tell which way the camera was facing.
  const keyframeIndices = result.keyframe_indices ?? [];
  if (keyframeIndices.length && centres.length) {
    const scale = frustumScale(centres);
    const frustumGeometry = makeFrustumGeometry(scale);
    const frustumMaterial = new THREE.LineBasicMaterial({ color: 0xff6b6b });
    geometries.push(frustumGeometry);
    materials.push(frustumMaterial);

    for (const index of keyframeIndices) {
      const pose = poses[index];
      if (!pose) continue;
      const frustum = new THREE.LineSegments(frustumGeometry, frustumMaterial);
      // Three.js matrices are column-major; the pose arrives as row-major
      // nested arrays. `set` takes row-major arguments, so the elements go in
      // reading order and Three.js transposes internally.
      frustum.matrixAutoUpdate = false;
      frustum.matrix.set(
        pose[0][0], pose[0][1], pose[0][2], pose[0][3],
        pose[1][0], pose[1][1], pose[1][2], pose[1][3],
        pose[2][0], pose[2][1], pose[2][2], pose[2][3],
        pose[3][0], pose[3][1], pose[3][2], pose[3][3],
      );
      scene.add(frustum);
    }
  }

  // ---- Frame the scene --------------------------------------------------
  frameScene(camera, controls, points, centres);

  const onResize = () => {
    const w = container.clientWidth || width;
    const h = container.clientHeight || height;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  };
  window.addEventListener('resize', onResize);

  const state = { renderer, controls, container, onResize, geometries, materials, frameId: 0 };
  active = state;

  (function animate() {
    state.frameId = requestAnimationFrame(animate);
    controls.update();   // required because damping is enabled
    renderer.render(scene, camera);
  })();
}

/** Frustum size relative to how far the camera actually travelled, so the
 *  markers stay legible whether the scene spans 2 units or 200. */
function frustumScale(centres) {
  if (centres.length < 2) return 0.1;
  const box = new THREE.Box3().setFromPoints(centres);
  const span = box.getSize(new THREE.Vector3()).length();
  return Math.max(span * 0.03, 0.02);
}

function makeFrustumGeometry(scale) {
  const d = scale;          // forward depth
  const w = scale * 0.7;    // half width
  const h = scale * 0.5;    // half height

  // +Z is forward. The pipeline's poses use OpenCV's camera axes with the
  // Y flip already applied at export, so forward remains +Z here.
  const apex = [0, 0, 0];
  const corners = [
    [-w, -h, d], [w, -h, d], [w, h, d], [-w, h, d],
  ];

  const vertices = [];
  for (const corner of corners) vertices.push(...apex, ...corner);      // sides
  for (let i = 0; i < 4; i++) {
    vertices.push(...corners[i], ...corners[(i + 1) % 4]);              // rim
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
  return geometry;
}

/** Point the camera at the reconstruction and back off far enough to see it. */
function frameScene(camera, controls, points, centres) {
  const box = new THREE.Box3();
  for (const point of points) box.expandByPoint(new THREE.Vector3(...point));
  for (const centre of centres) box.expandByPoint(centre);

  if (box.isEmpty()) {
    camera.position.set(0, 0, 5);
    controls.target.set(0, 0, 0);
    controls.update();
    return;
  }

  const centre = box.getCenter(new THREE.Vector3());
  const radius = Math.max(box.getSize(new THREE.Vector3()).length() / 2, 0.001);

  // Trigonometric fit rather than a guessed multiplier: half the field of view
  // subtends the bounding sphere, so this distance puts the whole thing on
  // screen for any scene size.
  const halfFov = THREE.MathUtils.degToRad(camera.fov / 2);
  const distance = (radius / Math.sin(halfFov)) * 1.15;

  camera.position.copy(centre).add(new THREE.Vector3(distance * 0.6, distance * 0.45, distance * 0.7));
  camera.near = Math.max(distance / 1000, 0.001);
  camera.far = distance * 20;
  camera.updateProjectionMatrix();

  controls.target.copy(centre);
  controls.update();
}
