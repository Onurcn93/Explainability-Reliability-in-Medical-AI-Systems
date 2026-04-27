"""
inference/app.py — FracAssist Flask server.

Run from repo root:
    python inference/app.py

Routes:
    GET  /         → serve index.html
    GET  /health   → {"status": "ok", "models_loaded": true}
    POST /predict  → multipart/form-data with 'image' field → inference JSON
"""

import os
import sys
import tempfile

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

# Ensure both inference/ and repo root are on path so all imports resolve
_INFERENCE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT     = os.path.dirname(_INFERENCE_DIR)
sys.path.insert(0, _INFERENCE_DIR)
sys.path.insert(0, _REPO_ROOT)

from config import CONFIG
from predict import load_models, predict as run_predict

_ROOT = _REPO_ROOT

app = Flask(__name__, static_folder=_ROOT, static_url_path='')
CORS(app)  # Required: index.html may be opened as file:// or cross-origin

_ALLOWED = {".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_file(CONFIG["index_html"])


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": True,
        "device": CONFIG["device"],
    })


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    # --- Validate upload ---
    if "image" not in request.files:
        return jsonify({"error": "No 'image' field in request"}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in _ALLOWED:
        return jsonify({"error": "Invalid file type. Accepted: JPG, PNG"}), 400

    tmp_path = None
    try:
        # Save to a temp file; delete after inference
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
            tmp_path = f.name
            file.save(tmp_path)

        inference_mode = request.form.get("inference_mode", "gel")
        if inference_mode not in {"gel", "yolo", "resnet"}:
            inference_mode = "gel"

        result = run_predict(tmp_path, CONFIG, inference_mode=inference_mode)
        return jsonify(result)

    except Exception as e:
        # Strip filesystem paths from error messages before returning
        msg = str(e)
        if os.sep in msg:
            msg = msg.split(os.sep)[-1]
        return jsonify({"error": msg}), 500

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("[INFO] Loading models...")
    try:
        load_models(CONFIG)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print(f"[INFO] Device: {CONFIG['device']}")
    print("[INFO] FracAssist running → http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
