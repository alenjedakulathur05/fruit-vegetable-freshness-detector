"""
test_web_app.py
---------------
Automatic check that the Flask app really runs the trained model.

    python tests/test_web_app.py

It uploads every image in sample_images/ (real TEST images the model never
trained on) through the web app's /api/predict endpoint and compares the
answer with the true class in the file name. It also checks that invalid
uploads are rejected.
"""

import csv
import io
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "app"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import app as webapp  # noqa: E402
from utils import SAMPLE_IMAGES_DIR, SUPPORTED_EXTENSIONS  # noqa: E402


def main():
    if webapp.MODEL is None:
        sys.exit(f"Model not loaded: {webapp.MODEL_ERROR}")
    client = webapp.app.test_client()

    # 1. Home page loads and lists the supported classes
    page = client.get("/")
    assert page.status_code == 200 and b"Detect Freshness" in page.data
    print("GET /                      -> 200 OK")

    # 2. Real images -> predictions from the trained model
    # labels.csv (written by data_preprocessing.py) tells us the true class of each sample image
    true_classes = dict(csv.reader(open(SAMPLE_IMAGES_DIR / "labels.csv", encoding="utf-8")))
    images = sorted(p for p in SAMPLE_IMAGES_DIR.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)
    correct = 0
    for path in images:
        true_class = true_classes.get(path.name, "?")
        with open(path, "rb") as f:
            resp = client.post("/api/predict", data={"image": (f, path.name)},
                               content_type="multipart/form-data")
        assert resp.status_code == 200, resp.data
        result = resp.get_json()
        ok = result["class_name"] == true_class
        correct += ok
        flag = "LOW-CONF" if result["low_confidence"] else ""
        print(f"{'OK ' if ok else 'XX '} {path.name[:45]:<45} -> {result['freshness']:<7} {result['produce']:<12} "
              f"{result['confidence']:.2%} {flag}")
    if images:
        print(f"Sample images correct: {correct}/{len(images)}")

    # 3. HTML route shows the result card
    if images:
        with open(images[0], "rb") as f:
            html = client.post("/predict", data={"image": (f, images[0].name)},
                               content_type="multipart/form-data").data.decode()
        assert "Fruit/Vegetable" in html and "Confidence" in html
        print("POST /predict (HTML)       -> result card rendered")

    # 4. Invalid uploads are rejected
    bad = client.post("/api/predict", data={"image": (io.BytesIO(b"hello, not an image"), "notes.txt")},
                      content_type="multipart/form-data")
    assert bad.status_code == 400
    print("text file (.txt)           ->", bad.get_json()["error"])
    fake = client.post("/api/predict", data={"image": (io.BytesIO(b"definitely not a jpeg"), "fake.jpg")},
                       content_type="multipart/form-data")
    assert fake.status_code == 400
    print("renamed text file (.jpg)   ->", fake.get_json()["error"])
    empty = client.post("/api/predict", data={}, content_type="multipart/form-data")
    assert empty.status_code == 400
    print("no file                    ->", empty.get_json()["error"])
    print("\nAll web app checks passed.")


if __name__ == "__main__":
    main()
