/**
 * segmentation.js
 * Handles all interactions on the Segmentation page:
 *  – Image upload / drag-and-drop
 *  – Method tab switching
 *  – Parameter controls
 *  – Region-growing seed picker
 *  – API calls to /api/segment/*
 *  – Result display
 */

'use strict';

// ──────────────────────────────────────────────────────────
//  State
// ──────────────────────────────────────────────────────────
let uploadedFile   = null;   // File object from the input
let uploadedDataURL = null;  // base64 data URL of the uploaded image
let currentMethod  = 'kmeans';
let seedX = null, seedY = null;  // for region-growing

// ──────────────────────────────────────────────────────────
//  DOM references
// ──────────────────────────────────────────────────────────
const dropzone     = document.getElementById('seg-dropzone');
const fileInput    = document.getElementById('seg-file-input');
const previewWrap  = document.getElementById('seg-preview-wrap');
const dropzoneWrap = document.getElementById('seg-dropzone-wrap');
const tabBtns      = document.querySelectorAll('.seg-tab-btn');
const tabPanels    = document.querySelectorAll('.seg-tab-panel');
const runBtn       = document.getElementById('seg-run-btn');
const clearBtn     = document.getElementById('seg-clear-btn');
const originalImg  = document.getElementById('seg-original-img');
const resultImg    = document.getElementById('seg-result-img');
const resultPlaceholder = document.getElementById('seg-result-placeholder');
const loadingBox   = document.getElementById('seg-loading');
const errorBox     = document.getElementById('seg-error');
const errorMsg     = document.getElementById('seg-error-msg');
const infoRow      = document.getElementById('seg-info-row');

// Sliders
const kSlider      = document.getElementById('kmeans-k');
const kVal         = document.getElementById('kmeans-k-val');
const thrSlider    = document.getElementById('rg-threshold');
const thrVal       = document.getElementById('rg-threshold-val');
const aggSlider    = document.getElementById('agg-clusters');
const aggVal       = document.getElementById('agg-clusters-val');
const msSlider     = document.getElementById('ms-bandwidth');
const msVal        = document.getElementById('ms-bandwidth-val');

// Region growing seed picker
const seedWrapper  = document.getElementById('rg-seed-wrapper');
const seedImg      = document.getElementById('rg-seed-img');
const seedMarker   = document.getElementById('rg-seed-marker');
const seedStatus   = document.getElementById('rg-seed-status');

// ──────────────────────────────────────────────────────────
//  Slider live update
// ──────────────────────────────────────────────────────────
function bindSlider(slider, display) {
  if (!slider || !display) return;
  display.textContent = slider.value;
  slider.addEventListener('input', () => { display.textContent = slider.value; });
}
bindSlider(kSlider, kVal);
bindSlider(thrSlider, thrVal);
bindSlider(aggSlider, aggVal);
bindSlider(msSlider, msVal);

// ──────────────────────────────────────────────────────────
//  Tab switching
// ──────────────────────────────────────────────────────────
tabBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    const method = btn.dataset.method;
    tabBtns.forEach(b => b.classList.remove('active'));
    tabPanels.forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    const panel = document.getElementById(`tab-${method}`);
    if (panel) panel.classList.add('active');
    currentMethod = method;
    clearResults();
  });
});

// ──────────────────────────────────────────────────────────
//  Upload / Drag-and-drop
// ──────────────────────────────────────────────────────────
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

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) loadFile(fileInput.files[0]);
});

function loadFile(file) {
  uploadedFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    uploadedDataURL = e.target.result;

    // Show the original image
    originalImg.src = uploadedDataURL;
    originalImg.style.display = 'block';

    // Hide empty state
    const emptyState = document.getElementById('seg-empty-state');
    if (emptyState) emptyState.style.display = 'none';

    // Update region growing seed image too
    if (seedImg) seedImg.src = uploadedDataURL;
    if (seedStatus) seedStatus.textContent = 'Click on the image above to pick a seed point';
    seedX = null; seedY = null;
    if (seedMarker) seedMarker.style.display = 'none';

    dropzoneWrap.classList.add('d-none');
    previewWrap.classList.remove('d-none');
    clearResults();
  };
  reader.readAsDataURL(file);
}

// ──────────────────────────────────────────────────────────
//  Clear / reset
// ──────────────────────────────────────────────────────────
clearBtn.addEventListener('click', () => {
  uploadedFile = null;
  uploadedDataURL = null;
  seedX = null; seedY = null;
  fileInput.value = '';
  dropzoneWrap.classList.remove('d-none');
  previewWrap.classList.add('d-none');
  if (seedMarker) seedMarker.style.display = 'none';
  // Reset original image
  originalImg.src = '';
  originalImg.style.display = 'none';
  // Restore empty state
  const emptyState = document.getElementById('seg-empty-state');
  if (emptyState) emptyState.style.display = '';
  clearResults();
});

function clearResults() {
  resultImg.src = '';
  resultImg.style.display = 'none';
  resultPlaceholder.style.display = 'flex';
  loadingBox.classList.add('d-none');
  errorBox.classList.add('d-none');
  infoRow.innerHTML = '';
}

// ──────────────────────────────────────────────────────────
//  Region Growing – seed picker
// ──────────────────────────────────────────────────────────
// Region Growing – seed picker: click on the original image panel
if (seedWrapper) {
  seedWrapper.addEventListener('click', e => {
    if (!uploadedDataURL) return;
    const rect = originalImg.getBoundingClientRect();
    if (rect.width === 0) return;   // image not rendered yet

    const fracX = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const fracY = Math.max(0, Math.min(1, (e.clientY - rect.top)  / rect.height));

    // Show marker on image
    seedMarker.style.display = 'block';
    seedMarker.style.left = `${fracX * 100}%`;
    seedMarker.style.top  = `${fracY * 100}%`;

    // Derive actual pixel coordinates from natural image size
    const tmp = new Image();
    tmp.onload = () => {
      seedX = Math.round(fracX * tmp.naturalWidth);
      seedY = Math.round(fracY * tmp.naturalHeight);
      const status = document.getElementById('rg-seed-status');
      if (status) status.textContent = `Seed set: (${seedX}, ${seedY})`;
    };
    tmp.src = uploadedDataURL;
  });
}

// ──────────────────────────────────────────────────────────
//  Run segmentation
// ──────────────────────────────────────────────────────────
runBtn.addEventListener('click', runSegmentation);

async function runSegmentation() {
  if (!uploadedDataURL) {
    showError('Please upload an image first.');
    return;
  }

  // Region growing needs a seed point
  if (currentMethod === 'region-growing' && (seedX === null || seedY === null)) {
    showError('Please click on the image in the "Seed Point" section to pick a seed pixel first.');
    return;
  }

  clearResults();
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
        formData.append('seed_x',    seedX);
        formData.append('seed_y',    seedY);
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

    if (!data.success) {
      showError(data.error || 'Segmentation failed.');
      return;
    }

    // Show result image
    resultImg.src = data.result_image;
    resultImg.style.display = 'block';
    resultPlaceholder.style.display = 'none';

    // Build info row
    buildInfoRow(data);

  } catch (err) {
    loadingBox.classList.add('d-none');
    showError('Network error: ' + err.message);
  } finally {
    runBtn.disabled = false;
  }
}

// ──────────────────────────────────────────────────────────
//  Info row builder
// ──────────────────────────────────────────────────────────
function buildInfoRow(data) {
  infoRow.innerHTML = '';

  // Timing badge
  if (data.elapsed_ms !== undefined) {
    infoRow.innerHTML += `
      <span class="seg-timing-badge">
        <i class="bi bi-clock-history"></i> ${data.elapsed_ms} ms
      </span>`;
  }

  const pills = [];
  if (data.k            !== undefined) pills.push({ icon: 'bi-grid-3x3-gap', label: `k = ${data.k}` });
  if (data.n_clusters   !== undefined) pills.push({ icon: 'bi-diagram-3',    label: `clusters = ${data.n_clusters}` });
  if (data.threshold    !== undefined) pills.push({ icon: 'bi-sliders',       label: `threshold = ${data.threshold}` });
  if (data.seed         !== undefined) pills.push({ icon: 'bi-geo-alt-fill',  label: `seed (${data.seed[0]}, ${data.seed[1]})` });
  if (data.bandwidth    !== undefined) pills.push({ icon: 'bi-broadcast',     label: `bandwidth = ${data.bandwidth}` });

  pills.forEach(p => {
    infoRow.innerHTML += `
      <span class="seg-info-pill">
        <i class="bi ${p.icon}"></i> ${p.label}
      </span>`;
  });
}

// ──────────────────────────────────────────────────────────
//  Error helper
// ──────────────────────────────────────────────────────────
function showError(msg) {
  errorBox.classList.remove('d-none');
  errorMsg.textContent = msg;
}
