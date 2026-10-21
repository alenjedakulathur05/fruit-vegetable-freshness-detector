# DATASET.md: Fruit and Vegetable Disease (Healthy vs Rotten)

## 1. Summary

| Item | Value |
|---|---|
| **Dataset name** | Fruit and Vegetable Disease (Healthy vs Rotten) |
| **Source** | Kaggle, uploaded by Muhammad Subhan (`muhammad0subhan`), version 1 |
| **URL** | https://www.kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten |
| **Last updated on Kaggle** | 2024-05-20 |
| **Licence (as stated on Kaggle)** | CC0: Public Domain |
| **Download size** | 5,205,929,743 bytes (about 5.2 GB) |
| **Image files** | 29,291 |
| **Classes** | 28 = 14 produce types × {Healthy, Rotten} |
| **Freshness classes** | Healthy (shown as **Fresh**) and Rotten (shown as **Spoiled**) |
| **File formats** | .jpg 17,384 · .png 11,813 · .jpeg 80 · .webp 14 |
| **Image sizes** | Varying (the uploader recommends resizing) |

The size, file count, format counts and class counts in sections 1 and 3 come from
Kaggle's public file listing API (`/api/v1/datasets/list/...`). I retrieved them on
2026-09-22, before downloading anything. The numbers measured on the downloaded
files, after cleaning, are in section 4.

## 2. Classes

The class list is read automatically from the folder names. Nothing is hard-coded in the model.

| Fruits (9) | Vegetables (5, culinary sense) |
|---|---|
| Apple, Banana, Grape, Guava, Jujube, Mango, Orange, Pomegranate, Strawberry | Bell pepper, Carrot, Cucumber, Potato, Tomato |

Botanically, tomato, cucumber and bell pepper are fruits. In a kitchen or supermarket
they count as vegetables, so this project groups them that way.

Each produce type has two folders, for example `Apple__Healthy` and `Apple__Rotten`,
which gives 28 classes in total.

## 3. Class distribution (Kaggle file listing)

| Produce | Healthy (Fresh) | Rotten (Spoiled) | Total |
|---|---:|---:|---:|
| Apple | 2,438 | 2,930 | 5,368 |
| Banana | 2,000 | 2,800 | 4,800 |
| Bell pepper | 611 | 591 | 1,202 |
| Carrot | 620 | 580 | 1,200 |
| Cucumber | 608 | 593 | 1,201 |
| Grape | 200 | 200 | 400 |
| Guava | 200 | 200 | 400 |
| Jujube | 200 | 200 | 400 |
| Mango | 1,813 | 2,247 | 4,060 |
| Orange | 2,075 | 2,186 | 4,261 |
| Pomegranate | 200 | 200 | 400 |
| Potato | 615 | 585 | 1,200 |
| Strawberry | 1,603 | 1,596 | 3,199 |
| Tomato | 604 | 596 | 1,200 |
| **Total** | **13,787** | **15,504** | **29,291** |

What this table shows:

* **The classes are imbalanced.** The largest class (Apple__Rotten, 2,930 images) is
  about 15 times larger than the smallest (200 images each for grape, guava, jujube
  and pomegranate). Training uses a class-weighted loss, and the evaluation reports
  **macro** averages so that small classes count equally.
* **Fresh and spoiled are roughly balanced overall** (47% fresh, 53% spoiled).
* **Image quality differs between classes.** Some rotten vegetable folders contain
  very small files. Bellpepper__Rotten averages about 8 KB per file and
  Carrot__Rotten about 10 KB, while their healthy counterparts average about 85 KB
  and 117 KB. A model can partly learn "low-resolution photo means rotten" instead
  of the actual spoilage. This is a real dataset bias (see README → Limitations).

## 4. Numbers measured on the downloaded files

<!-- CLEANED-STATS-START -->
These were measured by `src/data_preprocessing.py` while reading all 29,291 downloaded
files, so they do not depend on the split:

| Measurement | Value |
|---|---|
| Image files read | 29,291 |
| Corrupted or unreadable | 0 |
| Formats actually found (by content) | JPEG 17,046 · PNG 12,227 · WEBP 17 · MPO 1 |
| Original width | min 80 px · median 308 px · max 6,183 px |
| Original height | min 95 px · median 258 px · max 4,345 px |
| Size after preprocessing | 224 x 224 RGB |

Note that the file extension does not always match the real format: 11,813 files end in
`.png` but 12,227 are really PNG, and one `.jpg` is actually an MPO (a multi-picture
JPEG from a camera). Pillow reads the real format, so this causes no problem.

**Counts after de-duplication and the split are being re-measured.** The first version
of the duplicate check did not recognise the dataset's own augmented copies
(`rotated_by_45_…`, `saltandpepper_…`, `translation_…`, `vertical_flip_…`,
`…_augmented_7.jpg` — 8,272 files in total) as copies of the same photo, so copies of
one photo could land in different splits. `src/data_preprocessing.py` now groups them
by their original photo and prints a leakage check that must read 0. Re-run
`.\run.bat preprocess` and read the new numbers from
`dataset/processed/dataset_summary.json`.
<!-- CLEANED-STATS-END -->

## 5. Folder structure

After downloading, the dataset looks like this:

```
dataset/raw/
└── Fruit And Vegetable Diseases Dataset/
    ├── Apple__Healthy/        FreshApple (1).jpg, ...
    ├── Apple__Rotten/
    ├── Banana__Healthy/
    ├── ...
    └── Tomato__Rotten/
```

After preprocessing:

```
dataset/processed/
├── images/<class>/<class>_00001.jpg   224 x 224 RGB copies
├── splits.csv                         path, class, class index, split (train/val/test), look-alike group
└── dataset_summary.json               all statistics used in this file
```

## 6. Download procedure

### Automatic (used by this project)

```
run.bat download
```

or `.venv\Scripts\python src\download_dataset.py`. The script:

1. downloads `archive.zip` (about 5.2 GB) from Kaggle's public download endpoint
   `https://www.kaggle.com/api/v1/datasets/download/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten`.
   No Kaggle account is needed because the dataset is public. If the connection
   drops, the download resumes where it stopped.
2. extracts the zip into `dataset/raw/`
3. deletes the zip to free about 5 GB (use `--keep-zip` to keep it)

If `%USERPROFILE%\.kaggle\kaggle.json` (a Kaggle API token) exists, it is sent
along, but it is not required.

### Manual (if the automatic download fails)

1. Open the dataset URL above and sign in to Kaggle.
2. Click **Download** and save `archive.zip`.
3. Run `.venv\Scripts\python src\download_dataset.py --from-zip "C:\path\to\archive.zip"`.

## 7. Why this dataset was selected

I compared the public datasets below. Every licence, size and description here comes
from Kaggle's API or the dataset page.

| Dataset | Produce | Licence | Size | Problem for this project |
|---|---|---|---|---|
| **Fruit and Vegetable Disease (Healthy vs Rotten)** (muhammad0subhan), **selected** | 9 fruits + 5 vegetables, 28 classes | CC0 | 5.2 GB, 29,291 images | Compiled from other datasets: possible duplicates and quality differences, which the preprocessing handles |
| Fresh and Stale Images of Fruits and Vegetables (raghavrpotdar) | apple, banana, bitter gourd, capsicum, orange, tomato | CC0 | 1.5 GB | The description says the images include **video frames** and **about 20 augmentations per original photo**, so near-identical copies would leak between train and test |
| Fruits fresh and rotten for classification (sriramr) | apple, banana, orange only | Unknown | 1.9 GB | Fruits only; licence unknown |
| Fresh and Rotten Classification (swoyam2609) | fruits and vegetables (description) | CDLA-Permissive-1.0 | 3.1 GB | Generic description, class list not documented |
| Fruit and Vegetable (Fresh vs Rotten) (zlatan599) | several | MIT | exactly 5,205,929,743 bytes | Same byte size as the selected dataset, so it appears to be a re-upload of it |
| Fresh and Rotten - Fruits and Vegetables (filipemonteir) | apple, mango, orange, potato, tomato | Unknown | 0.7 GB | Fewer types; licence unknown |
| Mendeley datasets (e.g. *Fresh and Rotten Fruits Dataset for Machine-Based Evaluation of Fruit Quality*, *FruitNet*) | fruits only | CC BY 4.0 | varies | Have no vegetables |

The selected dataset was the best fit for four reasons:

1. **It covers fruits and vegetables**, including every example the project brief lists
   (apple, banana, orange, mango, strawberry, grape, tomato, potato, carrot,
   cucumber, bell pepper).
2. **Its labels map directly onto the task:** each folder is `<Produce>__Healthy` or
   `<Produce>__Rotten`, so one multi-class classifier predicts produce type and
   freshness together.
3. **It is large enough for transfer learning** (about 29k images), and a
   lightweight CNN still trains in minutes on an 8 GB GPU.
4. **CC0 licence and public download:** it can be used for academic work and
   downloaded automatically without an account.

### Suitability for academic use

CC0 means the uploader waived copyright, so use in a college project is allowed.
The uploader also states that the images were compiled from other Kaggle and GitHub
datasets, which may carry their own licences (for example CC BY 4.0). In your
report, cite this Kaggle dataset, and mention that it is a compilation.

**Citation:** Muhammad Subhan, *Fruit and Vegetable Disease (Healthy vs Rotten)*, Kaggle, 2024.
https://www.kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten
