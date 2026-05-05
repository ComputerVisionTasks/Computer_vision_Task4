'use strict';

// ──────────────────────────────────────────────────────────
//  State
// ──────────────────────────────────────────────────────────
let uploadedFile    = null;
let uploadedDataURL = null;
let currentMethod   = 'optimal';

// ──────────────────────────────────────────────────────────
//  DOM references
// ──────────────────────────────────────────────────────────
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

// ──────────────────────────────────────────────────────────
//  Slider live update helper
// ──────────────────────────────────────────────────────────
function bindSlider(sliderId, displayId) {
  const slider  = document.getElementById(sliderId);
  const display = document.getElementById(displayId);
  if (!slider || !display) return;
  display.textContent = slider.value;
  slider.addEventListener('input', () => { display.textContent = slider.value; });
}

bindSlider('optimal-window',   'optimal-window-val');
bindSlider('otsu-window',      'otsu-window-val');
bindSlider('spectral-classes', 'spectral-classes-val');
bindSlider('spectral-sigma',   'spectral-sigma-val');
bindSlider('spectral-window',  'spectral-window-val');

// ──────────────────────────────────────────────────────────
//  Global / Local radio → toggle Window Size slider
// ──────────────────────────────────────────────────────────
function bindScopeRadios(name, windowWrapId) {
  const radios = document.querySelectorAll(`input[name="${name}"]`);
  const wrap   = document.getElementById(windowWrapId);
  if (!wrap) return;
  radios.forEach(r => {
    r.addEventListener('change', () => {
      wrap.style.display = r.value === 'local' && r.checked ? '' : 'none';
    });
  });
}

bindScopeRadios('optimal-scope',  'optimal-window-wrap');
bindScopeRadios('otsu-scope',     'otsu-window-wrap');
bindScopeRadios('spectral-scope', 'spectral-window-wrap');

// ──────────────────────────────────────────────────────────
//  Tab switching
// ──────────────────────────────────────────────────────────
tabBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    const method = btn.dataset.method;
    tabBtns.forEach(b  => b.classList.remove('active'));
    tabPanels.forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    const panel = document.getElementById(`tab-${method}`);
    if (panel) panel.classList.add('active');
    currentMethod = method;
    clearResultDisplay();
  });
});

// ──────────────────────────────────────────────────────────
//  Upload / Drag-and-drop
// ──────────────────────────────────────────────────────────
dropzone.addEventListener('click', () => fileInput.click());
dropzone.querySelector('.seg-link').addEventListener('click', e => { e.stopPropagation(); fileInput.click(); });

dropzone.addEventListener('dragover',  e => { e.preventDefault(); dropzone.classList.add('drag-over'); });
dropzone.addEventListener('dragleave', ()  => dropzone.classList.remove('drag-over'));
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
    originalImg.src = uploadedDataURL;
    originalImg.style.display = 'block';
    const emptyState = document.getElementById('seg-empty-state');
    if (emptyState) emptyState.style.display = 'none';
    dropzoneWrap.classList.add('d-none');
    previewWrap.classList.remove('d-none');
    clearResultDisplay();
  };
  reader.readAsDataURL(file);
}

// ──────────────────────────────────────────────────────────
//  Clear / reset
// ──────────────────────────────────────────────────────────
clearBtn.addEventListener('click', () => {
  uploadedFile = null;
  uploadedDataURL = null;
  fileInput.value = '';
  dropzoneWrap.classList.remove('d-none');
  previewWrap.classList.add('d-none');
  originalImg.src = '';
  originalImg.style.display = 'none';
  const emptyState = document.getElementById('seg-empty-state');
  if (emptyState) emptyState.style.display = '';
  clearResultDisplay();
});

function clearResults() {
  resultImg.src = '';
  resultImg.style.display = 'none';
  resultPlaceholder.style.display = 'flex';
  loadingBox.classList.add('d-none');
  errorBox.classList.add('d-none');
  infoRow.innerHTML = '';
}

// Hide result display without showing spinner (used on tab switch / clear)
function clearResultDisplay() {
  resultImg.src = '';
  resultImg.style.display = 'none';
  resultPlaceholder.style.display = 'none';
  loadingBox.classList.add('d-none');
  errorBox.classList.add('d-none');
  infoRow.innerHTML = '';
}

// ──────────────────────────────────────────────────────────
//  Helpers to read current radio / slider values
// ──────────────────────────────────────────────────────────
function getScopeValue(name) {
  const checked = document.querySelector(`input[name="${name}"]:checked`);
  return checked ? checked.value : 'global';
}
function getSliderValue(id, fallback) {
  const el = document.getElementById(id);
  return el ? el.value : fallback;
}

// ──────────────────────────────────────────────────────────
//  Run thresholding
// ──────────────────────────────────────────────────────────
runBtn.addEventListener('click', runThresholding);

async function runThresholding() {
  if (!uploadedDataURL) { showError('Please upload an image first.'); return; }

  clearResults();
  loadingBox.classList.remove('d-none');
  runBtn.disabled = true;

  try {
    const formData = new FormData();
    formData.append('image', uploadedFile);

    let endpoint = '';

    switch (currentMethod) {
      case 'optimal': {
        endpoint = '/api/threshold/optimal';
        const scope = getScopeValue('optimal-scope');
        formData.append('scope', scope);
        if (scope === 'local') formData.append('window_size', getSliderValue('optimal-window', 15));
        break;
      }
      case 'otsu': {
        endpoint = '/api/threshold/otsu';
        const scope = getScopeValue('otsu-scope');
        formData.append('scope', scope);
        if (scope === 'local') formData.append('window_size', getSliderValue('otsu-window', 15));
        break;
      }
      case 'spectral': {
        endpoint = '/api/threshold/spectral';
        const scope = getScopeValue('spectral-scope');
        formData.append('scope',   scope);
        formData.append('classes', getSliderValue('spectral-classes', 3));
        formData.append('sigma',   getSliderValue('spectral-sigma', 1.0));
        if (scope === 'local') formData.append('window_size', getSliderValue('spectral-window', 15));
        break;
      }
    }

    const response = await fetch(endpoint, { method: 'POST', body: formData });
    const data     = await response.json();

    loadingBox.classList.add('d-none');

    if (!data.success) { showError(data.error || 'Thresholding failed.'); return; }

    resultImg.src = data.result_image;
    resultImg.style.display = 'block';
    resultPlaceholder.style.display = 'none';
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
  if (data.elapsed_ms !== undefined) {
    infoRow.innerHTML += `<span class="seg-timing-badge"><i class="bi bi-clock-history"></i> ${data.elapsed_ms} ms</span>`;
  }
  const pills = [];
  if (data.scope       !== undefined) pills.push({ icon: 'bi-globe2',           label: data.scope });
  if (data.threshold   !== undefined) pills.push({ icon: 'bi-sliders',          label: `threshold = ${data.threshold}` });
  if (data.classes     !== undefined) pills.push({ icon: 'bi-grid-3x3-gap',     label: `classes = ${data.classes}` });
  // if (data.sigma       !== undefined) pills.push({ icon: 'bi-activity',         label: `σ = ${data.sigma}` });
  if (data.window_size !== undefined) pills.push({ icon: 'bi-square',           label: `window = ${data.window_size}` });
  pills.forEach(p => {
    infoRow.innerHTML += `<span class="seg-info-pill"><i class="bi ${p.icon}"></i> ${p.label}</span>`;
  });
}

// ──────────────────────────────────────────────────────────
//  Error helper
// ──────────────────────────────────────────────────────────
function showError(msg) {
  errorBox.classList.remove('d-none');
  errorMsg.textContent = msg;
}