/**
 * scripts.js — FracAssist UI logic
 *
 * Connects the Flask inference backend (inference/app.py) to the HTML.
 * Entry point: python inference/app.py → http://127.0.0.1:5000
 */

const API_URL = 'http://127.0.0.1:5000';

// ─── State ────────────────────────────────────────────────────────────────
let _resultData    = null;
let _currentOverlay = 'box'; // matches the default checked radio

// ─── Tab switching ─────────────────────────────────────────────────────────
document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        document.querySelectorAll('.tab-panel').forEach(p => p.classList.add('hidden'));
        document.getElementById('tab-' + tab.dataset.tab).classList.remove('hidden');
    });
});

// ─── File picker ───────────────────────────────────────────────────────────
document.getElementById('select-btn').addEventListener('click', () => {
    document.getElementById('file-input').click();
});

document.getElementById('file-input').addEventListener('change', e => {
    const file = e.target.files[0];
    if (file) handleFile(file);
    e.target.value = ''; // allow re-selecting the same file
});

// ─── Drag and drop ─────────────────────────────────────────────────────────
const imageDisplay = document.getElementById('image-display');

imageDisplay.addEventListener('dragover', e => {
    e.preventDefault();
    imageDisplay.classList.add('drag-over');
});

imageDisplay.addEventListener('dragleave', () => {
    imageDisplay.classList.remove('drag-over');
});

imageDisplay.addEventListener('drop', e => {
    e.preventDefault();
    imageDisplay.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
});

// ─── Overlay toggle (GradCAM / Bounding Box) ───────────────────────────────
document.querySelectorAll('input[name="view-toggle"]').forEach(radio => {
    radio.addEventListener('change', () => {
        _currentOverlay = radio.value;
        if (_resultData) _updateOverlay(_currentOverlay);
    });
});

// ─── Refresh / reset ───────────────────────────────────────────────────────
document.getElementById('refresh-btn').addEventListener('click', resetUI);

// ─── File handling → POST /predict ─────────────────────────────────────────
function handleFile(file) {
    const allowed = ['image/jpeg', 'image/jpg', 'image/png'];
    if (!allowed.includes(file.type)) {
        showError('Invalid file type. Please use JPG or PNG.');
        return;
    }

    _resultData = null;
    showState('loading');
    resetMetrics();

    const formData = new FormData();
    formData.append('image', file);

    fetch(`${API_URL}/predict`, { method: 'POST', body: formData })
        .then(res => res.json())
        .then(data => {
            if (data.error) { showError(data.error); return; }
            _resultData = data;
            applyPrediction(data);
        })
        .catch(() => {
            showError('Server unreachable — run: python inference/app.py');
        });
}

// ─── Apply prediction response to UI ──────────────────────────────────────
function applyPrediction(data) {
    const isFrac = data.label === 'Fractured';
    const prob   = Math.round((data.fracture_probability || 0) * 100);
    const conf   = data.mode === 'YOLO-LED' ? data.yolo_confidence : data.resnet_probability;
    const confStr = conf != null ? conf.toFixed(2) : '—';

    // Fracture Probability card
    const probVal = document.getElementById('prob-val');
    probVal.textContent = prob + '%';
    probVal.className   = 'card-value ' + (isFrac ? 'alert-red' : 'text-teal');

    const probSub = document.getElementById('prob-sub');
    probSub.textContent = isFrac ? 'HIGH RISK — FRACTURED' : 'LOW RISK — NON-FRACTURED';
    probSub.className   = 'card-subtitle ' + (isFrac ? 'alert-red' : 'text-teal');

    // Model Confidence card
    const confVal = document.getElementById('conf-val');
    confVal.textContent = data.mode === 'YOLO-LED' ? `YOLO: ${confStr}` : `ResNet: ${confStr}`;
    confVal.className   = 'card-value ' + (isFrac ? 'alert-red' : 'text-teal');

    const confSub = document.getElementById('conf-sub');
    if (data.mode === 'YOLO-LED' && data.resnet_probability) {
        confSub.textContent = `ResNet: ${data.resnet_probability.toFixed(2)}`;
    } else {
        confSub.textContent = data.mode === 'YOLO-LED'
            ? 'Primary: YOLO detector'
            : 'Primary: ResNet-18 classifier';
    }

    // Body Part card
    const bodyConf = data.body_part_confidence > 0
        ? `${data.body_part || '—'}: ${(data.body_part_confidence * 100).toFixed(1)}%`
        : (data.body_part || '—');
    document.getElementById('body-val').textContent = bodyConf;
    document.getElementById('body-sub').textContent = 'region classification';

    // Status banner
    const banner = document.getElementById('status-banner');
    banner.className = 'status-banner ' + (isFrac ? 'status-fractured' : 'status-ok');
    document.getElementById('status-dot').className  = 'status-dot ' + (isFrac ? 'dot-red' : 'dot-teal');
    document.getElementById('status-text').textContent =
        data.mode === 'YOLO-LED' ? 'YOLO-LED DETECTION' : 'CLASSIFIER-LED';
    document.getElementById('status-model').textContent =
        data.mode === 'YOLO-LED' ? 'Y1B + E4e' : 'E4e ResNet-18';

    // Show result image with current overlay
    showState('result');
    _updateOverlay(_currentOverlay);
}

// ─── Overlay image swap ─────────────────────────────────────────────────────
function _updateOverlay(mode) {
    if (!_resultData) return;
    const img   = document.getElementById('result-img');
    const badge = document.getElementById('img-badge');

    if (mode === 'grad' && _resultData.gradcam_image) {
        img.src = _resultData.gradcam_image;
        badge.textContent = 'GradCAM · ResNet-18 · layer4';
    } else if (mode === 'box' && _resultData.xray_with_box) {
        img.src = _resultData.xray_with_box;
        const conf = _resultData.yolo_confidence != null
            ? (_resultData.yolo_confidence * 100).toFixed(0) + '%'
            : '—';
        badge.textContent = `YOLO · Y1B · conf ${conf}`;
    } else {
        // Requested overlay not available — fall back gracefully
        img.src = _resultData.gradcam_image || _resultData.xray_with_box || '';
        badge.textContent = mode === 'box' ? 'No YOLO detection' : 'GradCAM unavailable';
    }
}

// ─── UI state helpers ──────────────────────────────────────────────────────
function showState(state) {
    // empty uses .image-placeholder (always visible unless hidden)
    // loading + result use .img-overlay (absolute, sits on top)
    document.getElementById('state-empty').classList.toggle('hidden', state !== 'empty');
    document.getElementById('state-loading').classList.toggle('hidden', state !== 'loading');
    document.getElementById('state-result').classList.toggle('hidden', state !== 'result');
}

function resetMetrics() {
    ['prob-val', 'conf-val', 'body-val'].forEach(id => {
        const el = document.getElementById(id);
        el.textContent = '—';
        el.className   = 'card-value';
    });
    document.getElementById('prob-sub').textContent  = 'awaiting image';
    document.getElementById('prob-sub').className    = 'card-subtitle';
    document.getElementById('conf-sub').textContent  = 'YOLO / ResNet-18 score';
    document.getElementById('body-sub').textContent  = 'region classification';

    const banner = document.getElementById('status-banner');
    banner.className = 'status-banner';
    document.getElementById('status-dot').className  = 'status-dot';
    document.getElementById('status-text').textContent = 'awaiting prediction';
    document.getElementById('status-model').textContent = '';
}

function showError(msg) {
    showState('empty');
    const banner = document.getElementById('status-banner');
    banner.className = 'status-banner status-error';
    document.getElementById('status-dot').className  = 'status-dot dot-amber';
    document.getElementById('status-text').textContent = 'ERROR';
    document.getElementById('status-model').textContent = msg;
}

function resetUI() {
    _resultData = null;
    showState('empty');
    resetMetrics();
    document.getElementById('view-box').checked = true;
    _currentOverlay = 'box';
}

// ─── Health check on load — populate Config device field ──────────────────
fetch(`${API_URL}/health`)
    .then(r => r.json())
    .then(d => {
        if (d.status === 'ok') {
            const el = document.getElementById('cfg-device');
            if (el) el.textContent = d.device || 'connected';
        }
    })
    .catch(() => {
        // Server not running — silent until user tries to predict
    });
