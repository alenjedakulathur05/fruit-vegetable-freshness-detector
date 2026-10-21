"""
download_dataset.py
-------------------
Downloads the Kaggle dataset "Fruit and Vegetable Disease (Healthy vs Rotten)"
(public, CC0 licence, ~5.2 GB) and extracts it into dataset/raw/.

Usage (from the project folder):
    python src/download_dataset.py                      # automatic download
    python src/download_dataset.py --from-zip C:/Users/you/Downloads/archive.zip
                                                        # use a zip you downloaded manually

The dataset is public, so Kaggle's download endpoint normally works without an
account. If you have a Kaggle API token (%USERPROFILE%/.kaggle/kaggle.json) it
is used automatically. If the download still fails, see DATASET.md.
"""

import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DATASET_DIR, KAGGLE_DATASET, RAW_DIR, SUPPORTED_EXTENSIONS  # noqa: E402

DOWNLOAD_URL = f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_DATASET}"
ZIP_PATH = DATASET_DIR / "archive.zip"


def count_images(folder: Path) -> int:
    return sum(1 for p in folder.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS)


def kaggle_auth_header() -> dict:
    """Use ~/.kaggle/kaggle.json if it exists (optional - the dataset is public)."""
    token = Path.home() / ".kaggle" / "kaggle.json"
    if not token.exists():
        return {}
    creds = json.loads(token.read_text())
    raw = f"{creds['username']}:{creds['key']}".encode()
    return {"Authorization": "Basic " + base64.b64encode(raw).decode()}


def download_zip(url: str, dest: Path, retries: int = 5) -> None:
    """Stream the zip to disk. If the connection drops, resume where it stopped."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        have = dest.stat().st_size if dest.exists() else 0
        headers = {"User-Agent": "fruit-vegetable-freshness-project", **kaggle_auth_header()}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as resp:
                resumed = resp.status == 206
                total = int(resp.headers.get("Content-Length", 0)) + (have if resumed else 0)
                done = have if resumed else 0
                print(f"Downloading {total / 1e9:.2f} GB" + (f" (resuming at {done / 1e9:.2f} GB)" if resumed else ""))
                start, last = time.time(), 0.0
                with open(dest, "ab" if resumed else "wb") as f:
                    while chunk := resp.read(1024 * 1024):
                        f.write(chunk)
                        done += len(chunk)
                        if time.time() - last > 10:
                            speed = (done - (have if resumed else 0)) / max(time.time() - start, 1e-6) / 1e6
                            print(f"  {done / 1e9:.2f} / {total / 1e9:.2f} GB ({done / max(total, 1):.0%}) at {speed:.1f} MB/s")
                            last = time.time()
            print("Download complete.")
            return
        except urllib.error.HTTPError as exc:
            if exc.code == 416:          # "range not satisfiable" = file already complete
                print("Download already complete.")
                return
            print(f"Attempt {attempt} failed: HTTP {exc.code} {exc.reason}")
            if exc.code in (401, 403):
                break                     # retrying will not help
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            print(f"Attempt {attempt} failed: {exc}")
        time.sleep(10)
    raise RuntimeError("Could not download the dataset automatically.")


def extract_zip(zip_path: Path) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path} -> {RAW_DIR} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW_DIR)


def main():
    parser = argparse.ArgumentParser(description="Download the freshness dataset into dataset/raw/")
    parser.add_argument("--from-zip", type=Path, help="path to a manually downloaded Kaggle zip file")
    parser.add_argument("--keep-zip", action="store_true", help="keep archive.zip after extracting")
    args = parser.parse_args()

    if RAW_DIR.exists() and count_images(RAW_DIR) > 0:
        print(f"dataset/raw already contains {count_images(RAW_DIR)} images - nothing to do.")
        return

    if args.from_zip:
        extract_zip(args.from_zip)
    else:
        try:
            download_zip(DOWNLOAD_URL, ZIP_PATH)
            extract_zip(ZIP_PATH)
        except (RuntimeError, zipfile.BadZipFile) as exc:
            print(f"\n{exc}\nManual procedure:\n"
                  f"  1. Open https://www.kaggle.com/datasets/{KAGGLE_DATASET}\n"
                  "  2. Sign in and click 'Download' (archive.zip, about 5 GB)\n"
                  "  3. Run: python src/download_dataset.py --from-zip <path to archive.zip>")
            sys.exit(1)
        if not args.keep_zip:
            ZIP_PATH.unlink(missing_ok=True)   # free ~5 GB; the images are now in dataset/raw

    print(f"Done. dataset/raw now contains {count_images(RAW_DIR)} image files.")


if __name__ == "__main__":
    main()
