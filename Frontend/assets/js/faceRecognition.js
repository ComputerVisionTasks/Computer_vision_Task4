/**
 * faceRecognition.js
 * Single upload → parallel face detection + face recognition
 * No tabs. Results appear side-by-side automatically.
 */
(function () {
  "use strict";

  /* ─── Helpers ─────────────────────────────── */
  const $       = (id) => document.getElementById(id);
  const show    = (el) => { if (el) el.classList.remove('d-none'); };
  const hide    = (el) => { if (el) el.classList.add('d-none'); };
  const showEl  = (el) => { if (el) el.style.display = ''; };
  const hideEl  = (el) => { if (el) el.style.display = 'none'; };

  function readFile(file) {
    return new Promise((res) => {
      const r = new FileReader();
      r.onload = (e) => res(e.target.result);
      r.readAsDataURL(file);
    });
  }

  /* ─── Badge helpers ───────────────────────── */
  function setBadgeRunning(badgeEl) {
    if (!badgeEl) return;
    badgeEl.className = 'fr-running-badge ms-auto';
    badgeEl.innerHTML = '<span class="fr-spinner-sm"></span> Running…';
  }

  function setBadgeDone(badgeEl) {
    if (!badgeEl) return;
    badgeEl.className = 'fr-running-badge done ms-auto';
    badgeEl.innerHTML = '<i class="bi bi-check-circle-fill"></i> Done';
  }

  function setBadgeError(badgeEl) {
    if (!badgeEl) return;
    badgeEl.className = 'fr-running-badge error ms-auto';
    badgeEl.innerHTML = '<i class="bi bi-x-circle-fill"></i> Error';
  }

  /* ─── Upload wiring ───────────────────────── */
  function initUpload() {
    const area  = $('upload-area');
    const input = $('image-input');
    if (!area || !input) return;

    area.addEventListener('click', () => input.click());

    area.addEventListener('dragover', (e) => {
      e.preventDefault();
      area.classList.add('dragover');
    });
    area.addEventListener('dragleave', () => area.classList.remove('dragover'));

    area.addEventListener('drop', (e) => {
      e.preventDefault();
      area.classList.remove('dragover');
      const file = e.dataTransfer?.files[0];
      if (file && file.type.startsWith('image/')) handleFile(file);
    });

    input.addEventListener('change', (e) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
    });

    $('clear-btn')?.addEventListener('click', resetAll);
  }

  /* ─── Main handler ────────────────────────── */
  async function handleFile(file) {
    const dataUrl = await readFile(file);

    // Show upload thumbnail
    const prevImg = $('preview-image');
    if (prevImg) prevImg.src = dataUrl;
    hide($('upload-area'));
    show($('preview-wrap'));
    show($('clear-btn'));

    // Reveal results section
    const section = $('results-section');
    if (section) section.classList.remove('d-none');

    // Reset both panels to loading state
    resetDetectionPanel();
    resetRecognitionPanel();

    // Fire both requests in parallel — they are independent
    Promise.all([
      runDetection(file),
      runRecognition(file),
    ]);
  }

  /* ─── Reset helpers ───────────────────────── */
  function resetDetectionPanel() {
    setBadgeRunning($('det-status-badge'));
    hide($('det-idle'));
    show($('det-placeholder'));
    hide($('det-img-wrap'));
    hide($('det-summary'));
    hide($('det-error'));
    const img = $('det-result-img');
    if (img) img.src = '';
  }

  function resetRecognitionPanel() {
    setBadgeRunning($('rec-status-badge'));
    hide($('rec-idle'));
    show($('rec-placeholder'));
    hide($('rec-result'));
    hide($('rec-error'));
    hide($('rec-roc-wrap'));
  }

  function resetAll() {
    $('image-input').value = '';
    show($('upload-area'));
    hide($('preview-wrap'));
    hide($('clear-btn'));
    hide($('results-section'));
  }

  /* ─── Face Detection ──────────────────────── */
  async function runDetection(file) {
    const formData = new FormData();
    formData.append('image', file);

    try {
      const res  = await fetch('/api/detect', { method: 'POST', body: formData });
      const data = await res.json();
      if (!res.ok || !data.success) throw new Error(data.error || 'Detection failed');

      // Show annotated image (no labels drawn by backend)
      const img = $('det-result-img');
      if (img && data.result_image) {
        img.src = data.result_image;  // Already includes data:image/jpeg;base64,
      }
      hide($('det-placeholder'));
      show($('det-img-wrap'));

      // Face summary
      const faces = data.faces || [];
      $('det-face-count').textContent = faces.length;

      const modeBadge = $('det-mode-badge');
      if (modeBadge) {
        modeBadge.textContent = data.mode === 'grayscale' ? 'Grayscale' : 'Color';
        modeBadge.style.background = data.mode === 'grayscale' ? '#6c757d' : 'var(--accent-color)';
      }

      const list = $('det-face-list');
      if (list) {
        list.innerHTML = faces.length === 0
          ? '<div class="text-muted" style="font-size:0.82rem;padding:0.5rem 0">No faces detected. Try a clearer or larger image.</div>'
          : faces.map((f, i) => {
              const area = f.w * f.h;
              const sizeLabel = area > 10000 ? 'Large' : area > 3000 ? 'Medium' : 'Small';
              return `
                <div class="fr-face-item" style="animation-delay:${i * 0.06}s">
                  <div class="fr-face-num">${i + 1}</div>
                  <div class="fr-face-coords">
                    <strong>Face ${i + 1}</strong><br>
                    <span style="font-size:0.75rem">x:${f.x} y:${f.y} &nbsp;·&nbsp; ${f.w}×${f.h}px</span>
                  </div>
                  <span class="fr-face-size">${sizeLabel}</span>
                </div>`;
            }).join('');
      }

      show($('det-summary'));
      setBadgeDone($('det-status-badge'));

    } catch (err) {
      hide($('det-placeholder'));
      $('det-error-msg').textContent = err.message;
      show($('det-error'));
      setBadgeError($('det-status-badge'));
    }
  }

  /* ─── Face Recognition ────────────────────── */
  async function runRecognition(file) {
    const formData = new FormData();
    formData.append('image', file);

    try {
      const res  = await fetch('/api/recognize', { method: 'POST', body: formData });
      const data = await res.json();
      if (!res.ok || !data.success) throw new Error(data.error || 'Recognition failed');

      const name    = data.person || 'Unknown';
      const conf    = Math.round(data.confidence || 0);
      const isUnknown = name.toLowerCase() === 'unknown';

      // Avatar
      const avatar = $('rec-avatar');
      if (avatar) {
        avatar.textContent = isUnknown ? '?' : name.charAt(0).toUpperCase();
        avatar.className   = isUnknown ? 'fr-result-avatar unknown' : 'fr-result-avatar';
      }

      // Result card style
      const card = $('rec-result-card');
      if (card) {
        card.className = isUnknown ? 'fr-result-card unknown' : 'fr-result-card';
      }

      $('rec-person-name').textContent    = name;
      $('rec-confidence-pct').textContent = conf + '%';
      $('rec-distance').textContent       = (data.distance || 0).toFixed(4);

      // Confidence bar
      const bar = $('rec-conf-bar');
      if (bar) {
        bar.style.width = '0%';
        bar.className   = conf < 40 ? 'fr-progress-fill low' : 'fr-progress-fill';
        requestAnimationFrame(() => { bar.style.width = conf + '%'; });
      }

      hide($('rec-placeholder'));
      show($('rec-result'));
      show($('rec-roc-wrap'));
      setBadgeDone($('rec-status-badge'));

    } catch (err) {
      hide($('rec-placeholder'));
      $('rec-error-msg').textContent = err.message;
      show($('rec-error'));
      setBadgeError($('rec-status-badge'));
    }
  }

  /* ─── Known people ────────────────────────── */
  async function loadPeople() {
    const container = $('people-list');
    if (!container) return;
    try {
      const res  = await fetch('/api/labels');
      const data = await res.json();
      if (!data.success || !data.labels?.length) throw new Error('No labels');
      container.innerHTML = data.labels.map(label => `
        <div class="fr-people-item">
          <div class="fr-person-avatar">${label.charAt(0).toUpperCase()}</div>
          <div class="fr-person-name">${label}</div>
        </div>`).join('');
    } catch {
      container.innerHTML = '<div class="fr-people-loading text-muted">Connect Flask backend to load database.</div>';
    }
  }

  /* ─── Model metrics ───────────────────────── */
  async function loadMetrics() {
    const loading = $('metrics-loading');
    const content = $('metrics-content');
    const error   = $('metrics-error');
    if (!loading || !content) return;

    try {
      const res  = await fetch('/api/model-metrics');
      const data = await res.json();
      if (!data.success || !data.available) throw new Error(data.error || 'Not available');

      const trainAcc = (data.train_accuracy * 100).toFixed(2) + '%';
      const testAcc  = (data.test_accuracy  * 100).toFixed(2) + '%';

      $('train-accuracy').textContent = trainAcc;
      $('test-accuracy').textContent  = testAcc;

      // Hero strip
      const hAcc = $('hero-test-acc');
      const hComp = $('hero-components');
      const hClass = $('hero-classes');
      if (hAcc)   hAcc.textContent   = testAcc;
      if (hComp)  hComp.textContent  = data.model_info?.n_components  ?? '–';
      if (hClass) hClass.textContent = data.model_info?.total_classes ?? '–';

      // Model info
      const mi = $('model-info');
      if (mi && data.model_info) {
        mi.innerHTML = `PCA: <strong>${data.model_info.n_components}</strong> &nbsp;·&nbsp; KNN k: <strong>${data.model_info.n_neighbors}</strong>`;
      }

      // Per-class table
      const tbody = $('class-metrics-table');
      if (tbody && data.class_metrics) {
        tbody.innerHTML = Object.entries(data.class_metrics).map(([, m], idx) => {
          const pct   = (m.accuracy * 100).toFixed(1);
          const color = m.accuracy >= 0.9 ? '#20c997' : m.accuracy >= 0.7 ? '#ffc107' : '#dc3545';
          return `
            <tr>
              <td><span style="color:var(--accent-color);font-weight:700">${idx + 1}</span></td>
              <td><strong>${m.label}</strong></td>
              <td>${m.samples}</td>
              <td><span class="badge" style="background:${color};font-size:0.75rem">${pct}%</span></td>
              <td style="min-width:90px">
                <div class="fr-acc-bar-wrap">
                  <div class="fr-acc-bar" style="width:${pct}%"></div>
                </div>
              </td>
            </tr>`;
        }).join('');
      }

      hideEl(loading);
      showEl(content);
      if (error) hideEl(error);

    } catch (err) {
      hideEl(loading);
      if (error) {
        showEl(error);
        const et = $('metrics-error-text');
        if (et) et.textContent = err.message || 'Could not load model metrics';
      }
    }
  }

  /* ─── Boot ────────────────────────────────── */
  window.addEventListener('load', () => {
    initUpload();
    loadPeople();
    loadMetrics();
  });

})();