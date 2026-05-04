(function() {
  'use strict';

  // ── DOM Elements ────────────────────────────────
  const tabButtons = document.querySelectorAll('.fr-tab-btn');
  const tabPanels = document.querySelectorAll('.fr-tab-panel');

  // Detection elements
  const detUploadArea = document.getElementById('det-upload-area');
  const detImageInput = document.getElementById('det-image-input');
  const detPreviewWrap = document.getElementById('det-preview-wrap');
  const detPreviewOriginal = document.getElementById('det-preview-original');
  const detPreviewResult = document.getElementById('det-preview-result');
  const detPreviewPlaceholder = document.getElementById('det-preview-placeholder');
  const detClearBtn = document.getElementById('det-clear-btn');
  const detRerunBtn = document.getElementById('det-rerun-btn');
  const detLoading = document.getElementById('det-loading');
  const detError = document.getElementById('det-error');
  const detErrorMsg = document.getElementById('det-error-msg');
  const detResultsEmpty = document.getElementById('det-results-empty');
  const detResults = document.getElementById('det-results');
  const detFaceCount = document.getElementById('det-face-count');
  const detModeBadge = document.getElementById('det-mode-badge');
  const detFaceList = document.getElementById('det-face-list');

  // Camera elements
  const detCameraBtnWrap = document.getElementById('det-camera-btn-wrap');
  const detOpenCameraBtn = document.getElementById('det-open-camera-btn');
  const cameraModal = document.getElementById('camera-modal');
  const cameraVideo = document.getElementById('camera-video');
  const cameraCanvas = document.getElementById('camera-canvas');
  const cameraCloseBtn = document.getElementById('camera-close-btn');
  const cameraCancelBtn = document.getElementById('camera-cancel-btn');
  const cameraCaptureBtn = document.getElementById('camera-capture-btn');

  // Recognition elements
  const recUploadArea = document.getElementById('rec-upload-area');
  const recImageInput = document.getElementById('rec-image-input');
  const recPreviewWrap = document.getElementById('rec-preview-wrap');
  const recPreviewImage = document.getElementById('rec-preview-image');
  const recClearBtn = document.getElementById('rec-clear-btn');
  const recLoading = document.getElementById('rec-loading');
  const recError = document.getElementById('rec-error');
  const recErrorMsg = document.getElementById('rec-error-msg');
  const recResult = document.getElementById('rec-result');
  const recResultAvatar = document.getElementById('rec-result-avatar');
  const recPersonName = document.getElementById('rec-person-name');
  const recConfidencePct = document.getElementById('rec-confidence-pct');
  const recConfBar = document.getElementById('rec-conf-bar');
  const recDistance = document.getElementById('rec-distance');

  // Metrics & People
  const peopleList = document.getElementById('people-list');
  const metricsLoading = document.getElementById('metrics-loading');
  const metricsContent = document.getElementById('metrics-content');
  const metricsError = document.getElementById('metrics-error');
  const metricsErrorText = document.getElementById('metrics-error-text');

  // Camera stream
  let cameraStream = null;

  // ── Tab Switching ──────────────────────────────
  function switchTab(tabId) {
    tabButtons.forEach(btn => btn.classList.remove('active'));
    tabPanels.forEach(panel => panel.classList.remove('active'));
    const activeBtn = document.querySelector(`.fr-tab-btn[data-tab="${tabId}"]`);
    const activePanel = document.getElementById(`tab-${tabId}`);
    if (activeBtn) activeBtn.classList.add('active');
    if (activePanel) activePanel.classList.add('active');
  }

  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.getAttribute('data-tab')));
  });

  // ── Utility Functions ─────────────────────────
  function show(el) { el.classList.remove('d-none'); }
  function hide(el) { el.classList.add('d-none'); }
  function resetError(container, msgEl) {
    hide(container);
    if (msgEl) msgEl.textContent = '';
  }
  function showError(container, msgEl, message) {
    if (msgEl) msgEl.textContent = message;
    show(container);
  }

  // ── Camera Handling ───────────────────────────
  async function openCamera() {
    try {
      cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } });
      cameraVideo.srcObject = cameraStream;
      show(cameraModal);
    } catch (err) {
      alert('Camera access denied or not available.');
      console.error(err);
    }
  }

  function closeCamera() {
    if (cameraStream) {
      cameraStream.getTracks().forEach(track => track.stop());
      cameraStream = null;
      cameraVideo.srcObject = null;
    }
    hide(cameraModal);
  }

  function captureFrame() {
    if (!cameraVideo.srcObject) return;
    const video = cameraVideo;
    const canvas = cameraCanvas;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
      const file = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' });
      closeCamera();
      handleDetectionFile(file);
    }, 'image/jpeg', 0.95);
  }

  // Camera event listeners
  detOpenCameraBtn.addEventListener('click', openCamera);
  cameraCloseBtn.addEventListener('click', closeCamera);
  cameraCancelBtn.addEventListener('click', closeCamera);
  cameraCaptureBtn.addEventListener('click', captureFrame);

  // ── Detection Logic ───────────────────────────
  let lastDetectionFile = null;

  async function runDetection(file) {
    hide(detError);
    hide(detResults);
    hide(detResultsEmpty);
    hide(detPreviewPlaceholder);
    show(detLoading);

    const formData = new FormData();
    formData.append('image', file);

    try {
      const response = await fetch('/api/detect', { method: 'POST', body: formData });
      const data = await response.json();
      if (!data.success) throw new Error(data.error || 'Detection failed');

      // Show result image
      detPreviewResult.src = data.result_image;
      hide(detPreviewPlaceholder);
      hide(detLoading);

      // Populate summary
      detFaceCount.textContent = data.face_count;
      detModeBadge.textContent = data.mode === 'grayscale' ? 'Grayscale' : 'Color';
      show(detResults);

      // List faces
      detFaceList.innerHTML = data.faces.map(face => `
        <div class="fr-face-item">
          <div class="fr-face-num">${face.id}</div>
          <div class="fr-face-coords">
            x: ${face.x}, y: ${face.y}, w: ${face.width}, h: ${face.height}
          </div>
          <div class="fr-face-size">${face.width}×${face.height}</div>
        </div>
      `).join('');
    } catch (err) {
      hide(detLoading);
      showError(detError, detErrorMsg, err.message);
    }
  }

  function handleDetectionFile(file) {
    lastDetectionFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
      detPreviewOriginal.src = e.target.result;
      show(detPreviewWrap);
      hide(detUploadArea);
      hide(detCameraBtnWrap);  // hide camera button after file chosen
      show(detPreviewPlaceholder);
    };
    reader.readAsDataURL(file);

    runDetection(file);
  }

  // Detection upload handlers
  detUploadArea.addEventListener('click', () => detImageInput.click());
  detImageInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleDetectionFile(file);
  });

  detUploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    detUploadArea.classList.add('dragover');
  });
  detUploadArea.addEventListener('dragleave', () => {
    detUploadArea.classList.remove('dragover');
  });
  detUploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    detUploadArea.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      handleDetectionFile(file);
    } else {
      showError(detError, detErrorMsg, 'Please drop a valid image file.');
    }
  });

  detClearBtn.addEventListener('click', () => {
    lastDetectionFile = null;
    detImageInput.value = '';
    hide(detPreviewWrap);
    show(detUploadArea);
    show(detCameraBtnWrap);   // show camera button again
    hide(detResults);
    show(detResultsEmpty);
    hide(detError);
  });

  detRerunBtn.addEventListener('click', () => {
    if (lastDetectionFile) {
      hide(detResults);
      hide(detError);
      show(detPreviewPlaceholder);
      runDetection(lastDetectionFile);
    }
  });

  // ── Recognition Logic ─────────────────────────
  let lastRecognitionFile = null;

  async function runRecognition(file) {
    hide(recError);
    hide(recResult);
    show(recLoading);

    const formData = new FormData();
    formData.append('image', file);

    try {
      const response = await fetch('/api/recognize', { method: 'POST', body: formData });
      const data = await response.json();
      if (!data.success) throw new Error(data.error || 'Recognition failed');

      hide(recLoading);
      show(recResult);

      recPersonName.textContent = data.person;
      recConfidencePct.textContent = `${Math.round(data.confidence)}%`;
      recConfBar.style.width = `${Math.round(data.confidence)}%`;
      recDistance.textContent = data.distance.toFixed(4);
      recResultAvatar.textContent = data.person.charAt(0).toUpperCase();
    } catch (err) {
      hide(recLoading);
      showError(recError, recErrorMsg, err.message);
    }
  }

  function handleRecognitionFile(file) {
    lastRecognitionFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
      recPreviewImage.src = e.target.result;
      show(recPreviewWrap);
      hide(recUploadArea);
    };
    reader.readAsDataURL(file);

    runRecognition(file);
  }

  recUploadArea.addEventListener('click', () => recImageInput.click());
  recImageInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleRecognitionFile(file);
  });

  recUploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    recUploadArea.classList.add('dragover');
  });
  recUploadArea.addEventListener('dragleave', () => {
    recUploadArea.classList.remove('dragover');
  });
  recUploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    recUploadArea.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      handleRecognitionFile(file);
    } else {
      showError(recError, recErrorMsg, 'Please drop a valid face image.');
    }
  });

  recClearBtn.addEventListener('click', () => {
    lastRecognitionFile = null;
    recImageInput.value = '';
    hide(recPreviewWrap);
    show(recUploadArea);
    hide(recResult);
    hide(recError);
  });

  // ── Load Model Metrics ────────────────────────
  async function loadMetrics() {
    metricsLoading.style.display = 'block';
    metricsContent.style.display = 'none';
    if (metricsError) metricsError.style.display = 'none';

    try {
      const response = await fetch('/api/model-metrics');
      const data = await response.json();
      if (!data.success || !data.available) throw new Error(data.error || 'Metrics unavailable');

      document.getElementById('train-accuracy').textContent = (data.train_accuracy * 100).toFixed(2) + '%';
      document.getElementById('test-accuracy').textContent = (data.test_accuracy * 100).toFixed(2) + '%';
      const modelInfoEl = document.getElementById('model-info');
      if (modelInfoEl && data.model_info) {
        modelInfoEl.innerHTML = `<strong>PCA comp:</strong> ${data.model_info.n_components} &nbsp;|&nbsp; <strong>KNN neighbors:</strong> ${data.model_info.n_neighbors}`;
      }

      const tableBody = document.getElementById('class-metrics-table');
      if (tableBody && data.class_metrics) {
        tableBody.innerHTML = '';
        Object.entries(data.class_metrics).forEach(([classId, metric]) => {
          const percent = (metric.accuracy * 100).toFixed(1);
          const colorClass = metric.accuracy >= 0.9 ? 'success' : (metric.accuracy >= 0.7 ? 'warning' : 'danger');
          const row = document.createElement('tr');
          row.innerHTML = `
            <td>${classId}</td>
            <td>${metric.label}</td>
            <td>${metric.samples}</td>
            <td><span class="badge bg-${colorClass}">${percent}%</span></td>
            <td>
              <div class="fr-acc-bar-wrap"><div class="fr-acc-bar" style="width:${percent}%"></div></div>
            </td>
          `;
          tableBody.appendChild(row);
        });
      }

      metricsLoading.style.display = 'none';
      metricsContent.style.display = 'block';
    } catch (err) {
      metricsLoading.style.display = 'none';
      if (metricsError) metricsError.style.display = 'block';
      if (metricsErrorText) metricsErrorText.textContent = err.message;
    }
  }

  // ── Initialize ────────────────────────────────
  window.addEventListener('DOMContentLoaded', () => {
    loadMetrics();
  });

})();