/**
 * scripts.js — FracAssist UI logic
 *
 * Connects the Flask inference backend (inference/app.py) to the HTML.
 * Entry point: python inference/app.py → http://127.0.0.1:5000
 */

const API_URL = 'http://127.0.0.1:5000';

// ─── State ────────────────────────────────────────────────────────────────
let _resultData     = null;
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

// ─── Zoom controls ─────────────────────────────────────────────────────────
document.getElementById('zoom-slider').addEventListener('input', e => {
    document.getElementById('result-img').style.transform = `scale(${e.target.value / 100})`;
    document.getElementById('zoom-label').textContent = e.target.value + '%';
});

document.getElementById('zoom-reset').addEventListener('click', _resetZoom);

function _resetZoom() {
    document.getElementById('zoom-slider').value = 100;
    document.getElementById('result-img').style.transform = 'scale(1)';
    document.getElementById('zoom-label').textContent = '100%';
}

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
    _resetZoom();

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

    // ── Fracture Probability card ──────────────────────────────────────────
    const probVal = document.getElementById('prob-val');
    probVal.textContent = prob + '%';
    probVal.className   = 'card-value ' + (isFrac ? 'alert-red' : 'text-teal');

    const probSub = document.getElementById('prob-sub');
    probSub.textContent = isFrac ? 'HIGH RISK — FRACTURED' : 'LOW RISK — NON-FRACTURED';
    probSub.className   = 'card-subtitle ' + (isFrac ? 'alert-red' : 'text-teal');

    // ── YOLO model card ────────────────────────────────────────────────────
    const yoloVal = document.getElementById('yolo-val');
    const yoloSub = document.getElementById('yolo-sub');
    if (data.yolo_confidence != null) {
        yoloVal.textContent = data.yolo_confidence.toFixed(2);
        yoloVal.className   = 'model-card-value alert-red';
        yoloSub.textContent = 'BOX DETECTED';
    } else {
        yoloVal.textContent = '—';
        yoloVal.className   = 'model-card-value text-teal';
        yoloSub.textContent = 'NO DETECTION';
    }

    // ── ResNet-18 model card ───────────────────────────────────────────────
    const resnetVal = document.getElementById('resnet-val');
    const resnetSub = document.getElementById('resnet-sub');
    if (data.resnet_probability != null) {
        const rp = data.resnet_probability;
        resnetVal.textContent = rp.toFixed(2);
        resnetVal.className   = 'model-card-value ' + (rp >= 0.375 ? 'alert-red' : 'text-teal');
        resnetSub.textContent = (rp >= 0.375 ? 'FRAC' : 'NON-FRAC') + ' · thr 0.375';
    } else {
        resnetVal.textContent = '—';
        resnetVal.className   = 'model-card-value';
        resnetSub.textContent = 'not loaded';
    }

    // ── DenseNet-169 model card ────────────────────────────────────────────
    const densenetVal = document.getElementById('densenet-val');
    const densenetSub = document.getElementById('densenet-sub');
    if (data.densenet_probability != null) {
        const dp = data.densenet_probability;
        densenetVal.textContent = dp.toFixed(2);
        densenetVal.className   = 'model-card-value ' + (isFrac ? 'alert-red' : 'text-teal');
        densenetSub.textContent = 'D1 output';
    } else {
        densenetVal.textContent = '—';
        densenetVal.className   = 'model-card-value';
        densenetSub.textContent = 'D1 not loaded';
    }

    // ── Status banner ──────────────────────────────────────────────────────
    const banner = document.getElementById('status-banner');
    banner.className = 'status-banner ' + (isFrac ? 'status-fractured' : 'status-ok');
    document.getElementById('status-dot').className  = 'status-dot ' + (isFrac ? 'dot-red' : 'dot-teal');
    document.getElementById('status-text').textContent =
        data.mode === 'YOLO-LED' ? 'YOLO-LED DETECTION' : 'CLASSIFIER-LED';
    document.getElementById('status-model').textContent =
        data.mode === 'YOLO-LED' ? 'Y1B + E4a' : 'E4a ResNet-18';

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
        img.src = _resultData.gradcam_image || _resultData.xray_with_box || '';
        badge.textContent = mode === 'box' ? 'No YOLO detection' : 'GradCAM unavailable';
    }
}

// ─── UI state helpers ──────────────────────────────────────────────────────
function showState(state) {
    document.getElementById('state-empty').classList.toggle('hidden', state !== 'empty');
    document.getElementById('state-loading').classList.toggle('hidden', state !== 'loading');
    document.getElementById('state-result').classList.toggle('hidden', state !== 'result');
}

function resetMetrics() {
    // Fracture probability card
    const probVal = document.getElementById('prob-val');
    probVal.textContent = '—';
    probVal.className   = 'card-value';
    const probSub = document.getElementById('prob-sub');
    probSub.textContent = 'awaiting image';
    probSub.className   = 'card-subtitle';

    // Model cards
    [
        ['yolo-val',     'model-card-value', 'yolo-sub',     'detector'],
        ['resnet-val',   'model-card-value', 'resnet-sub',   'classifier'],
        ['densenet-val', 'model-card-value', 'densenet-sub', 'classifier'],
    ].forEach(([valId, valClass, subId, subText]) => {
        const v = document.getElementById(valId);
        v.textContent = '—';
        v.className   = valClass;
        document.getElementById(subId).textContent = subText;
    });

    // Status banner
    const banner = document.getElementById('status-banner');
    banner.className = 'status-banner';
    document.getElementById('status-dot').className   = 'status-dot';
    document.getElementById('status-text').textContent = 'awaiting prediction';
    document.getElementById('status-model').textContent = '';
}

function showError(msg) {
    showState('empty');
    const banner = document.getElementById('status-banner');
    banner.className = 'status-banner status-error';
    document.getElementById('status-dot').className   = 'status-dot dot-amber';
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
