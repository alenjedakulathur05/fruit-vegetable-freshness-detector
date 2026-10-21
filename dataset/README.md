# dataset/

| Folder | What is inside | Created by |
|---|---|---|
| `raw/` | The original Kaggle images, one folder per class (e.g. `Apple__Healthy`, `Banana__Rotten`) | `python src/download_dataset.py` |
| `processed/images/` | The same images after cleaning: RGB, rotation fixed, resized to 224 x 224, duplicates removed | `python src/data_preprocessing.py` |
| `processed/splits.csv` | One row per image: processed path, original path, class name, class index, split (train/val/test), look-alike group | `python src/data_preprocessing.py` |
| `processed/dataset_summary.json` | Counts per class and split, corrupted files, duplicates, image sizes and formats | `python src/data_preprocessing.py` |

The images are **not** stored in Git (about 5 GB). Download them again with
`run.bat download` (or `python src/download_dataset.py`). See `../DATASET.md` for the
source, licence and statistics.
