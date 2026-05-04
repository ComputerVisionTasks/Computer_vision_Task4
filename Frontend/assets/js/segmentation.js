/**
 * segmentation.js
 * Handles all interactions on the Segmentation page.
 */

'use strict';

// ── State ──────────────────────────────────────────────────
let uploadedFile    = null;
let uploadedDataURL = null;
let currentMethod   = 'kmeans';
let seedPoints      = [];   // array of {x, y} for region-growing (multiple seeds)

// ── DOM refs ───────────────────────────────────────────────
const dropzone          = document.getElementById('seg-dropzone');
const fileInput         = document.getElementById('seg-file-input');
const previewWrap       = document.getElementById('seg-preview-wrap');
const dropzoneWrap      = document.getElementById('seg-dropzone-wrap');
const tabBtns           = document.querySelectorAll('.seg-tab-btn');
const tabPanels         = document.querySelectorAll('.seg-tab-panel');
const runBtn            = document.getElementById('seg-run-btn');
const clearBtn          = document.getElementById('seg-clear-btn');
const originalImg       = document.getElementById('seg-original-img');
const resultImg         = document.getElementById('seg-result-img');
const resultPlaceholder = document.getElementById('seg-result-placeholder');
const loadingBox        = document.getElementById('seg-loading');
const errorBox          = document.getElementById('seg-error');
const errorMsg          = document.getElementById('seg-error-msg');
const infoRow           = document.getElementById('seg-info-row');
const seedWrapper       = document.getElementById('rg-seed-wrapper');
const seedMarkerEl      = document.getElementById('rg-seed-marker');  // first marker (reused for single display)

// Sliders
const kSlider    = document.getElementById('kmeans-k');
const kVal       = document.getElementById('kmeans-k-val');
const thrSlider  = document.getElementById('rg-threshold');
const thrVal     = document.getElementById('rg-threshold-val');
const aggSlider  = document.getElementById('agg-clusters');
const aggVal     = document.getElementById('agg-clusters-val');
const msSlider   = document.getElementById('ms-bandwidth');
const msVal      = document.getElementById('ms-bandwidth-val');

// ── Slider live-update ─────────────────────────────────────
function bindSlider(slider, display) {
  if (!slider || !display) return;
  display.textContent = slider.value;
  slider.addEventListener('input', () => { display.textContent = slider.value; });
}
bindSlider(kSlider, kVal);
bindSlider(thrSlider, thrVal);
bindSlider(aggSlider, aggVal);
bindSlider(msSlider, msVal);

// ── Tab switching ──────────────────────────────────────────
tabBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    const method = btn.dataset.method;
    tabBtns.forEach(b => b.classList.remove('active'));
    tabPanels.forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    const panel = document.getElementById(`tab-${method}`);
    if (panel) panel.classList.add('active');
    currentMethod = method;
    // Toggle crosshair cursor: only Region Growing needs the seed picker
    if (seedWrapper) {
      seedWrapper.classList.toggle('rg-active', method === 'region-growing');
    }

    // Reset and clear seeds whenever switching methods
    seedPoints = [];
    clearSeedMarkers();
    updateSeedStatus();

    // Clear ONLY the result image and info — do NOT show the spinner
    clearResultDisplay();
  });
});

// ── Upload / Drag-and-drop ─────────────────────────────────
dropzone.addEventListener('click', () => fileInput.click());
dropzone.querySelector('.seg-link').addEventListener('click', e => { e.stopPropagation(); fileInput.click(); });

dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('drag-over'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag-over'));
dropzone.addEventListener('drop', e => {
  e.preventDefault();
  dropzone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith('image/')) loadFile(f);
});
fileInput.addEventListener('change', () => { if (fileInput.files[0]) loadFile(fileInput.files[0]); });

function loadFile(file) {
  uploadedFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    uploadedDataURL = e.target.result;

    // Show original image
    originalImg.src = uploadedDataURL;
    originalImg.style.display = 'block';

    // Hide empty state
    const emptyState = document.getElementById('seg-empty-state');
    if (emptyState) emptyState.style.display = 'none';

    // Reset seeds
    seedPoints = [];
    clearSeedMarkers();
    updateSeedStatus();

    dropzoneWrap.classList.add('d-none');
    previewWrap.classList.remove('d-none');
    clearResultDisplay();
  };
  reader.readAsDataURL(file);
}

// ── Clear / Reset ──────────────────────────────────────────
clearBtn.addEventListener('click', () => {
  uploadedFile    = null;
  uploadedDataURL = null;
  seedPoints      = [];
  fileInput.value = '';

  originalImg.src = '';
  originalImg.style.display = 'none';

  const emptyState = document.getElementById('seg-empty-state');
  if (emptyState) emptyState.style.display = '';

  dropzoneWrap.classList.remove('d-none');
  previewWrap.classList.add('d-none');

  clearSeedMarkers();
  clearResultDisplay();
});

/**
 * Clear the result panel — hide result image, hide spinner, clear info.
 * Does NOT show the loading spinner (that is only shown during a real request).
 */
function clearResultDisplay() {
  resultImg.src = '';
  resultImg.style.display = 'none';
  resultPlaceholder.style.display = 'none';  // ← never auto-show the spinner
  loadingBox.classList.add('d-none');
  errorBox.classList.add('d-none');
  infoRow.innerHTML = '';
}

// ── Multiple seed markers ──────────────────────────────────
/**
 * Remove all dynamically created seed markers from the image wrapper.
 * (The original seedMarkerEl is kept in the DOM but hidden.)
 */
function clearSeedMarkers() {
  document.querySelectorAll('.seg-seed-dot').forEach(el => el.remove());
  if (seedMarkerEl) seedMarkerEl.style.display = 'none';
}

/** Add a visible dot marker at fractional position (fracX, fracY) on the wrapper. */
function addSeedMarker(fracX, fracY) {
  const dot = document.createElement('div');
  dot.className = 'seg-seed-marker seg-seed-dot';  // reuse same style
  dot.style.display = 'block';
  dot.style.left = `${fracX * 100}%`;
  dot.style.top  = `${fracY * 100}%`;

  // Find the image wrapper inside #rg-seed-wrapper
  const wrap = document.querySelector('#rg-seed-wrapper .seg-panel-img-wrap');
  if (wrap) wrap.appendChild(dot);
}

function updateSeedStatus() {
  const status = document.getElementById('rg-seed-status');
  if (!status) return;
  if (seedPoints.length === 0) {
    status.textContent = 'No seeds — click the image to add one';
  } else {
    status.textContent = `${seedPoints.length} seed${seedPoints.length > 1 ? 's' : ''} selected`;
  }
}

// ── Seed picker ────────────────────────────────────────────
if (seedWrapper) {
  // "Clear seeds" button
  const clearSeedsBtn = document.getElementById('rg-clear-seeds');
  if (clearSeedsBtn) {
    clearSeedsBtn.addEventListener('click', () => {
      seedPoints = [];
      clearSeedMarkers();
      updateSeedStatus();
    });
  }

  // Click on the original image panel to add a seed point
  // Only active when Region Growing is the selected method
  seedWrapper.addEventListener('click', e => {
    if (currentMethod !== 'region-growing') return;
    if (!uploadedDataURL) return;
    const rect = originalImg.getBoundingClientRect();
    if (rect.width === 0) return;

    const fracX = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const fracY = Math.max(0, Math.min(1, (e.clientY - rect.top)  / rect.height));

    // Show a marker dot at the clicked position
    addSeedMarker(fracX, fracY);

    // Derive actual pixel coordinates from natural image dimensions
    const tmp = new Image();
    tmp.onload = () => {
      const px = Math.round(fracX * tmp.naturalWidth);
      const py = Math.round(fracY * tmp.naturalHeight);
      seedPoints.push({ x: px, y: py });
      updateSeedStatus();
    };
    tmp.src = uploadedDataURL;
  });
}

// ── Run segmentation ───────────────────────────────────────
runBtn.addEventListener('click', runSegmentation);

async function runSegmentation() {
  if (!uploadedDataURL) { showError('Please upload an image first.'); return; }

  if (currentMethod === 'region-growing' && seedPoints.length === 0) {
    showError('Click on the image to pick at least one seed point.');
    return;
  }

  clearResultDisplay();
  // Show loading indicator only now (user explicitly clicked Run)
  loadingBox.classList.remove('d-none');
  runBtn.disabled = true;

  try {
    const formData = new FormData();
    formData.append('image', uploadedFile);

    let endpoint = '';
    switch (currentMethod) {
      case 'kmeans':
        endpoint = '/api/segment/kmeans';
        formData.append('k',        kSlider ? kSlider.value : 3);
        formData.append('max_iter', 20);
        break;
      case 'region-growing':
        endpoint = '/api/segment/region-growing';
        formData.append('threshold', thrSlider ? thrSlider.value : 25);
        // Send all seed points as a JSON array [[x1,y1],[x2,y2],...]
        formData.append('seeds', JSON.stringify(seedPoints.map(s => [s.x, s.y])));
        break;
      case 'agglomerative':
        endpoint = '/api/segment/agglomerative';
        formData.append('n_clusters', aggSlider ? aggSlider.value : 4);
        break;
      case 'meanshift':
        endpoint = '/api/segment/meanshift';
        formData.append('bandwidth', msSlider ? msSlider.value : 20);
        break;
    }

    const response = await fetch(endpoint, { method: 'POST', body: formData });
    const data     = await response.json();
    loadingBox.classList.add('d-none');

    if (!data.success) { showError(data.error || 'Segmentation failed.'); return; }

    // Display result
    resultImg.src = data.result_image;
    resultImg.style.display = 'block';
    buildInfoRow(data);

  } catch (err) {
    loadingBox.classList.add('d-none');
    showError('Network error: ' + err.message);
  } finally {
    runBtn.disabled = false;
  }
}

// ── Info row ───────────────────────────────────────────────
function buildInfoRow(data) {
  infoRow.innerHTML = '';

  if (data.elapsed_ms !== undefined) {
    infoRow.innerHTML += `<span class="seg-timing-badge"><i class="bi bi-clock-history"></i> ${data.elapsed_ms} ms</span>`;
  }

  const pills = [];
  if (data.k          !== undefined) pills.push({ icon: 'bi-grid-3x3-gap', label: `k = ${data.k}` });
  if (data.n_clusters !== undefined) pills.push({ icon: 'bi-diagram-3',    label: `clusters = ${data.n_clusters}` });
  if (data.threshold  !== undefined) pills.push({ icon: 'bi-sliders',      label: `threshold = ${data.threshold}` });
  if (data.seeds      !== undefined) pills.push({ icon: 'bi-geo-alt-fill', label: `${data.seeds.length} seed${data.seeds.length !== 1 ? 's' : ''}` });
  if (data.bandwidth  !== undefined) pills.push({ icon: 'bi-broadcast',    label: `bandwidth = ${data.bandwidth}` });

  pills.forEach(p => {
    infoRow.innerHTML += `<span class="seg-info-pill"><i class="bi ${p.icon}"></i> ${p.label}</span>`;
  });
}

// ── Error helper ───────────────────────────────────────────
function showError(msg) {
  errorBox.classList.remove('d-none');
  errorMsg.textContent = msg;
}
