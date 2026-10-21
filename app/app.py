"""
app.py - Flask web application
------------------------------
    python app/app.py        then open http://127.0.0.1:5000

1. The user opens the page and chooses a fruit/vegetable photo (preview shown in the browser).
2. "Detect Freshness" sends the file to /predict.
3. The server checks that the file really is an image, runs the TRAINED model
   (src/predict.py -> predict_image) and shows produce type, freshness and confidence.
"""

import base64
import io
import sys
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_from_directory

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from predict import load_trained_model, predict_image  # noqa: E402
from utils import (  # noqa: E402
    CONFIDENCE_THRESHOLD, SAMPLE_IMAGES_DIR, SUPPORTED_EXTENSIONS, is_valid_image, parse_label,
)

app = Flask(__name__,
            template_folder=str(PROJECT_ROOT / "templates"),
            static_folder=str(PROJECT_ROOT / "static"))
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024   # reject uploads larger than 10 MB

# Load the model ONCE when the server starts (not on every request).
try:
    MODEL, META = load_trained_model()
    MODEL_ERROR = None
except FileNotFoundError as exc:
    MODEL, META, MODEL_ERROR = None, None, str(exc)


def supported_classes():
    """Group the model's classes by produce type for display on the page."""
    if META is None:
        return []
    grouped = {}
    for name in META["class_names"]:
        info = parse_label(name)
        grouped.setdefault(info["produce"], []).append(info["freshness"])
    return sorted(grouped.items())


def sample_images():
    """Test-split images copied by data_preprocessing.py (offered as one-click examples)."""
    if not SAMPLE_IMAGES_DIR.exists():
        return []
    return sorted(p.name for p in SAMPLE_IMAGES_DIR.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)


def page(result=None, image_url=None, error=None):
    return render_template("index.html", classes=supported_classes(), model_error=MODEL_ERROR,
                           threshold=CONFIDENCE_THRESHOLD, samples=sample_images(),
                           result=result, image_url=image_url, error=error)


def validate_upload(file_storage):
    """Return (image_bytes, error_message). Exactly one of them is None."""
    if file_storage is None or file_storage.filename == "":
        return None, "Please choose an image file first."
    ext = Path(file_storage.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return None, f"Unsupported file type '{ext or 'none'}'. Please upload: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
    data = file_storage.read()
    if not data:
        return None, "The uploaded file is empty."
    if not is_valid_image(io.BytesIO(data)):
        return None, "This file is not a valid image (it may be corrupted or renamed)."
    return data, None


def run_prediction(file_storage):
    """Validate the upload and run the model. Returns (result, image_data_url, error)."""
    if MODEL is None:
        return None, None, MODEL_ERROR
    data, error = validate_upload(file_storage)
    if error:
        return None, None, error
    result = predict_image(MODEL, META, io.BytesIO(data))
    mime = file_storage.mimetype or "image/jpeg"
    data_url = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
    return result, data_url, None


@app.route("/", methods=["GET"])
def index():
    return page()


@app.route("/predict", methods=["POST"])
def predict():
    result, image_url, error = run_prediction(request.files.get("image"))
    return page(result=result, image_url=image_url, error=error)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Same as /predict but returns JSON (useful for testing)."""
    result, _, error = run_prediction(request.files.get("image"))
    if error:
        return jsonify({"error": error}), 400
    return jsonify(result)


@app.route("/samples/<path:filename>")
def sample_file(filename):
    """Serve a demo image from sample_images/ so the page can offer it as an example."""
    if filename not in sample_images():
        abort(404)
    return send_from_directory(SAMPLE_IMAGES_DIR, filename)


@app.errorhandler(413)
def too_large(_):
    return page(error="File too large (maximum 10 MB)."), 413


if __name__ == "__main__":
    if MODEL_ERROR:
        print("WARNING:", MODEL_ERROR)
    else:
        print(f"Model loaded: {META['arch']} with {len(META['class_names'])} classes")
    print("Open http://127.0.0.1:5000 in your browser (Ctrl+C to stop)")
    app.run(host="127.0.0.1", port=5000, debug=False)
