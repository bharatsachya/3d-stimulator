/*
 * Upload, poll, and hand the result to the viewer.
 *
 * Everything is same-origin -- FastAPI serves this file -- so every URL here is
 * relative. There is no API base to configure and no CORS to think about.
 */

/*
 * The viewer is loaded on demand rather than imported at the top.
 *
 * A static `import` of viewer.js would make the whole module -- upload, polling,
 * error reporting, everything -- fail to execute if Three.js could not be
 * resolved. An ES module that throws during evaluation runs NONE of its code,
 * so a WebGL problem would silently disable the upload button, which is a
 * miserable failure to diagnose.
 *
 * Loading it only when there is a result to draw means a broken viewer costs
 * you the 3D view and nothing else.
 */
let viewer = null;

async function loadViewer() {
  if (!viewer) viewer = await import('./viewer.js');
  return viewer;
}

const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');

/*
 * Limits, fetched from the server rather than hardcoded here.
 *
 * The server enforces them regardless -- a client-side check is a courtesy, not
 * a control. But duplicating the numbers in two places guarantees they drift,
 * and then the page confidently states a limit the server disagrees with.
 *
 * The fallbacks apply only if /health cannot be reached, in which case the
 * upload would fail anyway.
 */
let limits = { max_upload_mb: 25, max_duration_seconds: 30 };

(async function loadLimits() {
  try {
    const health = await (await fetch('/health')).json();
    limits = { ...limits, ...health.settings };
    document.getElementById('limit-hint').textContent =
      `Up to ${limits.max_upload_mb} MB and ${limits.max_duration_seconds} seconds.`;
  } catch {
    // Leave the defaults in place; the upload path reports real failures.
  }
})();

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

/*
 * There is deliberately NO click handler here. #dropzone is a <label for>, so
 * the browser opens the file picker itself, for both mouse and keyboard.
 *
 * The previous version did attach one, to a div that CONTAINED the input, and
 * called fileInput.click() from it. The synthetic click bubbled back to the
 * div, which called fileInput.click() again, recursing until the browser gave
 * up -- and the picker never opened, with nothing logged to explain it.
 */
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
    viewer?.disposeViewer();
    // Clearing the value matters: without it, choosing the SAME file again
    // fires no 'change' event and the page appears frozen.
    fileInput.value = '';
    show('upload');
  });
}

/*
 * Surface anything that would otherwise fail silently.
 *
 * Module-level errors do not reach a try/catch anywhere in this file, and an
 * uncaught promise rejection prints only to a console the user is not looking
 * at. This turns both into something visible on the page.
 */
window.addEventListener('error', (event) => {
  showError('Something went wrong in the page', event.message ?? String(event.error), '');
});
window.addEventListener('unhandledrejection', (event) => {
  showError('Something went wrong in the page', String(event.reason), '');
});

/**
 * Read a video's duration in the browser, without uploading it.
 *
 * Resolves to null if the browser cannot decode the container -- that is not a
 * rejection, since the server may well manage a format the <video> element
 * will not preview.
 */
function readDuration(file) {
  return new Promise((resolve) => {
    const element = document.createElement('video');
    const url = URL.createObjectURL(file);
    const done = (value) => {
      URL.revokeObjectURL(url);   // or the blob leaks for the page's lifetime
      resolve(value);
    };
    element.preload = 'metadata';
    element.onloadedmetadata = () => done(
      Number.isFinite(element.duration) ? element.duration : null
    );
    element.onerror = () => done(null);
    element.src = url;
  });
}

async function startJob(file) {
  const megabytes = file.size / (1024 * 1024);

  /*
   * CHECK BEFORE UPLOADING, NOT AFTER.
   *
   * Measured from this connection, a 17 MB upload takes 43 seconds. Letting an
   * oversized file upload in full just to be rejected at the far end wastes all
   * of that, and the rejection arrives as nginx's own HTML error page -- nginx
   * enforces its body limit before the request ever reaches the application, so
   * the friendly message the server would have sent never runs.
   *
   * Checking here costs nothing and turns a 43-second dead end into instant,
   * actionable feedback.
   */
  if (megabytes > limits.max_upload_mb) {
    return showError(
      'That video is too large',
      `${file.name} is ${megabytes.toFixed(1)} MB, and the limit is `
      + `${limits.max_upload_mb} MB.`,
      'Trimming the clip to a few seconds, or recording at 1080p rather than 4K, '
      + 'usually brings it well under. Only about 100 frames are processed no '
      + 'matter how long the video is, and each is scaled to 640px wide before '
      + 'any feature is detected — so a larger file buys no extra detail.',
    );
  }

  const duration = await readDuration(file);
  if (duration !== null && duration > limits.max_duration_seconds) {
    return showError(
      'That video is too long',
      `${file.name} runs ${duration.toFixed(1)} seconds, and the limit is `
      + `${limits.max_duration_seconds} seconds.`,
      'Around five to ten seconds of steady sideways motion is the sweet spot.',
    );
  }

  document.getElementById('progress-file').textContent =
    `${file.name} — ${megabytes.toFixed(1)} MB`
    + (duration !== null ? `, ${duration.toFixed(1)}s` : '');
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
    return showError(...(await describeFailure(response)));
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

/**
 * Turn a failed response into something a person can act on.
 *
 * The server's own rejections are JSON with a `detail` that already reads as a
 * sentence. But a 413 usually does NOT come from the server at all: nginx
 * enforces its body size limit before the request reaches the application and
 * answers with an HTML error page. Parsing that as JSON throws, and the old
 * code then displayed the bare string "HTTP 413", which tells the user nothing.
 */
async function describeFailure(response) {
  let detail = null;
  try {
    detail = (await response.json()).detail ?? null;
  } catch {
    // Not JSON -- almost certainly an nginx error page.
  }

  if (detail) return ['Upload rejected', detail, ''];

  if (response.status === 413) {
    return [
      'That video is too large',
      `The server refused it as over the ${limits.max_upload_mb} MB limit.`,
      'This was rejected at the web server before reaching the application, '
      + 'which usually means the file is somewhat over the limit rather than '
      + 'just at it.',
    ];
  }

  return [
    'Upload rejected',
    `The server responded ${response.status} ${response.statusText || ''}`.trim(),
    '',
  ];
}

function showError(title, message, detail) {
  document.getElementById('error-title').textContent = title;
  document.getElementById('error-message').textContent = message;
  document.getElementById('error-detail').textContent = detail ?? '';
  show('error');
}

async function showResult(job) {
  const result = job.result;
  const timing = job.timing ?? {};

  show('result');

  // Stats and timings first, so they appear even if the 3D view cannot.
  renderStats(result);
  renderTiming(timing);

  // Render only once the panel is visible: a hidden container has zero width,
  // and a WebGL canvas sized to zero stays zero until something forces a resize.
  try {
    const { renderReconstruction } = await loadViewer();
    renderReconstruction(document.getElementById('viewer'), result);
  } catch (error) {
    // A failed viewer must not discard a successful reconstruction. The numbers
    // are already on screen; say what is missing and why.
    document.getElementById('viewer').innerHTML =
      `<p class="viewer-failed">3D view unavailable: ${escapeHtml(String(error))}<br>`
      + `The reconstruction itself succeeded — see the figures below.</p>`;
  }
}

function renderStats(result) {

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
