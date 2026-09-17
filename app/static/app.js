/*
 * Upload, poll, and hand the result to the viewer.
 *
 * Everything is same-origin -- FastAPI serves this file -- so every URL here is
 * relative. There is no API base to configure and no CORS to think about.
 */

import { renderReconstruction, disposeViewer } from './viewer.js';

const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');

const panels = {
  upload: document.getElementById('upload-panel'),
  progress: document.getElementById('progress-panel'),
  error: document.getElementById('error-panel'),
  result: document.getElementById('result-panel'),
};

/** Show exactly one of the panels; hide the rest. */
function show(name) {
  for (const [key, element] of Object.entries(panels)) {
    element.hidden = key !== name;
  }
}

// ---------------------------------------------------------------- upload

dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault();
    fileInput.click();
  }
});
fileInput.addEventListener('change', () => {
  if (fileInput.files.length) startJob(fileInput.files[0]);
});

// The dragover handler must preventDefault, otherwise the browser's default
// action takes over and navigates away to the dropped file.
for (const type of ['dragenter', 'dragover']) {
  dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    dropzone.classList.add('over');
  });
}
for (const type of ['dragleave', 'drop']) {
  dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    dropzone.classList.remove('over');
  });
}
dropzone.addEventListener('drop', (event) => {
  const file = event.dataTransfer?.files?.[0];
  if (file) startJob(file);
});

for (const button of document.querySelectorAll('[data-action="reset"]')) {
  button.addEventListener('click', () => {
    disposeViewer();
    fileInput.value = '';
    show('upload');
  });
}

async function startJob(file) {
  document.getElementById('progress-file').textContent =
    `${file.name} — ${(file.size / (1024 * 1024)).toFixed(1)} MB`;
  setProgress('uploading', 0, 0);
  show('progress');

  const body = new FormData();
  body.append('video', file);

  let response;
  try {
    response = await fetch('/api/jobs', { method: 'POST', body });
  } catch (networkError) {
    return showError('Upload failed', 'Could not reach the server.', String(networkError));
  }

  if (!response.ok) {
    // The server's typed rejections (413 too large, 400 empty) carry a
    // `detail` that already reads as a sentence, so surface it directly.
    let detail = `HTTP ${response.status}`;
    try {
      detail = (await response.json()).detail ?? detail;
    } catch { /* body was not JSON; keep the status line */ }
    return showError('Upload rejected', detail, '');
  }

  const { job_id: jobId } = await response.json();
  pollJob(jobId);
}

// ---------------------------------------------------------------- polling

/*
 * setTimeout, scheduled AFTER each response lands -- never setInterval.
 *
 * Under setInterval a slow response does not delay the next request, so
 * requests stack up on a server that is already saturated. On a 2 vCPU box
 * running the reconstruction, that is exactly the moment when adding load is
 * most harmful. Chaining timeouts guarantees at most one poll in flight.
 */
const POLL_INTERVAL_MS = 400;

async function pollJob(jobId) {
  let response;
  try {
    response = await fetch(`/api/jobs/${jobId}?summary=true`);
  } catch (networkError) {
    return showError('Lost contact with the server', String(networkError), '');
  }

  if (response.status === 404) {
    return showError('Job not found', 'This job is no longer available.', '');
  }

  const summary = await response.json();
  setProgress(summary.stage, summary.frames_done, summary.frames_total);

  if (summary.status === 'failed') {
    return showError(
      'Could not reconstruct this video',
      summary.failure_message ?? 'Processing failed.',
      summary.failure_detail ?? '',
    );
  }

  if (summary.status === 'done') {
    const full = await (await fetch(`/api/jobs/${jobId}`)).json();
    return showResult(full);
  }

  setTimeout(() => pollJob(jobId), POLL_INTERVAL_MS);
}

function setProgress(stage, done, total) {
  document.getElementById('progress-stage').textContent = stage;
  document.getElementById('progress-count').textContent =
    total ? `${done} / ${total} frames` : '';
  const fraction = total ? done / total : 0;
  document.getElementById('progress-fill').style.width = `${(fraction * 100).toFixed(1)}%`;
}

// ---------------------------------------------------------------- results

function showError(title, message, detail) {
  document.getElementById('error-title').textContent = title;
  document.getElementById('error-message').textContent = message;
  document.getElementById('error-detail').textContent = detail ?? '';
  show('error');
}

function showResult(job) {
  const result = job.result;
  const timing = job.timing ?? {};

  show('result');
  // Render only once the panel is visible: a hidden container has zero width,
  // and a WebGL canvas sized to zero stays zero until something forces a resize.
  renderReconstruction(document.getElementById('viewer'), result);

  const stats = [
    ['camera poses', result.n_poses],
    ['keyframes', result.keyframe_indices.length],
    ['map points', result.n_points],
    ['units', result.units],
  ];
  document.getElementById('stats').innerHTML = stats
    .map(([label, value]) => `<div><dt>${label}</dt><dd>${value}</dd></div>`)
    .join('');

  const flags = document.getElementById('flags');
  flags.innerHTML = (result.flag_messages ?? [])
    .map((message) => `<div class="flag">${escapeHtml(message)}</div>`)
    .join('');

  renderTiming(timing);
}

function renderTiming(timing) {
  const headline = timing.ms_per_frame
    ? `${(timing.wall_ms / 1000).toFixed(2)} s total for ${timing.frames} processed frames `
      + `— ${timing.ms_per_frame.toFixed(1)} ms per frame`
    : 'No timing recorded.';
  document.getElementById('timing-headline').textContent = headline;

  const cell = (value, digits = 1) =>
    value === null || value === undefined ? '—' : Number(value).toFixed(digits);

  const rows = (timing.stages ?? []).map((stage) => `
    <tr>
      <td>${escapeHtml(stage.stage)}</td>
      <td>${cell(stage.total_ms)}</td>
      <td>${stage.calls}</td>
      <td>${cell(stage.mean_ms, 2)}</td>
      <td>${cell(stage.ms_per_frame, 2)}</td>
      <td>${cell(stage.pct_of_wall)}</td>
    </tr>`);

  rows.push(`
    <tr class="unaccounted">
      <td>unaccounted</td>
      <td>${cell(timing.unaccounted_ms)}</td>
      <td>—</td><td>—</td><td>—</td>
      <td>${cell(timing.unaccounted_pct)}</td>
    </tr>`);

  rows.push(`
    <tr class="total">
      <td>wall clock</td>
      <td>${cell(timing.wall_ms)}</td>
      <td>—</td><td>—</td>
      <td>${cell(timing.ms_per_frame, 2)}</td>
      <td>100.0</td>
    </tr>`);

  document.querySelector('#timing-table tbody').innerHTML = rows.join('');
}

/* Flag and failure text originates server-side, but it is still interpolated
   into innerHTML, so escape it rather than relying on that staying true. */
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}
