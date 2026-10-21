# Fruit & Vegetable Freshness Detection System

Upload a photo of a fruit or vegetable. A trained deep-learning model tells you what
it is, whether it is **fresh** or **spoiled**, and how confident it is.

The model is trained and included in this repository (`models/fruit_vegetable_freshness_model.pt`),
so you can run the web app straight away without downloading the dataset.

## Quick start for teammates

```
git clone <this repository>
cd fruit-vegetable-freshness-detector
.\run.bat setup
.\run.bat app
```

Then open http://127.0.0.1:5000 and upload a photo, or click one of the images from
`sample_images/`. `run.bat setup` installs Python 3.12 if it is missing, creates a
virtual environment and installs PyTorch; it needs about 4 GB of disk space.
The 5 GB image dataset is **not** in this repository — you only need it to train
again (`.\run.bat download`). Everything needed to run the app is already here.

For the presentation, start from **PRESENTATION_NOTES.md**: a 3-5 minute script plus
30 questions a teacher may ask, with answers.

---

## Problem statement

Every day, shops, warehouses and kitchens throw away fruit and vegetables that have
spoiled, and sometimes spoiled items reach customers. Freshness is usually checked by
a person looking at each item, which is slow, subjective and hard to scale.
Spoilage mostly shows up **visually**: brown or black patches, mould, wrinkled or
shrivelled skin, soft collapsed areas and colour changes. That makes it a good fit
for **computer vision**. A camera and a trained model can check items quickly and
consistently.

## Objective

Build a system that:

1. takes **one image** of a fruit or vegetable,
2. runs it through a **trained convolutional neural network**, and
3. reports the **produce type** (e.g. Apple), the **freshness** (FRESH / SPOILED) and
   the **confidence** (softmax probability),
4. through a **command-line tool** (`src/predict.py`) and a **web application** (Flask),
5. and warns the user when the prediction is uncertain, because the image may be
   outside the 14 supported produce types.

## Machine learning approach

This is a **supervised image classification** problem with 28 classes.

```
Image (JPG/PNG)
   ↓
Preprocessing        fix rotation, RGB, resize to 224×224, normalise with ImageNet mean/std
   ↓
Data augmentation    (training only) flip, ±15° rotation, ±10% shift, ±10% zoom, ±20% brightness
   ↓
CNN / transfer learning   MobileNetV2 pretrained on ImageNet, with a new classification head
   ↓
Classification       softmax over 28 classes → a probability for every class
   ↓
Freshness prediction  most likely class, e.g. "Banana__Rotten" → Fruit/Vegetable: Banana,
                      Freshness: SPOILED, Confidence = its probability
```

**Why one 28-class model instead of two models?** The dataset labels every image with
a combined label such as `Apple__Healthy` or `Apple__Rotten`. One classifier with 28
outputs is the simplest correct design, and the produce type and freshness can be read
straight from the predicted class. The app also shows **P(fresh)**, the sum of the
probabilities of all 14 fresh classes.

## Dataset

**Fruit and Vegetable Disease (Healthy vs Rotten)**, from Kaggle (CC0 licence).
Kaggle lists 29,291 images in 28 classes covering 14 produce types:

* **Fruits:** apple, banana, grape, guava, jujube, mango, orange, pomegranate, strawberry
* **Vegetables:** bell pepper, carrot, cucumber, potato, tomato

See **[DATASET.md](DATASET.md)** for the source, licence, class distribution, how it
was cleaned, how to download it, and why it was chosen.

### Preprocessing pipeline (`src/data_preprocessing.py`)

1. **Load images:** every image in `dataset/raw/`. The class is its folder name.
2. **Detect corrupted images:** every file is fully decoded, and unreadable or
   truncated files are skipped and listed in `dataset_summary.json`.
3. **Fix orientation and colour:** apply the EXIF rotation, put transparent PNGs on a
   white background, and convert to 3-channel RGB.
4. **Resize** to 224×224, the input size MobileNetV2 was pretrained with, and save to
   `dataset/processed/images/`.
5. **Prevent data leakage.** This dataset was compiled from several sources, so the
   same photo can appear more than once. A 72-bit perceptual hash (dHash), computed
   over all 8 flips and 90° rotations of each image, finds:
   * the same picture in **two different classes**: removed (the label is ambiguous)
   * **exact, flipped or rotated copies** inside a class: only one copy kept
   * **look-alikes** (hash distance ≤ 4 bits): kept, but always put in the **same
     split**, so a copy of a test image can never be in the training set
6. **Label encoding:** class names are sorted alphabetically and numbered 0–27, and
   the numbers are saved with the model (`models/class_labels.json`).
7. **Train/validation/test split:** 70% / 15% / 15%, **stratified per class**
   (every class keeps the same proportions) and group-aware (point 5), with a fixed
   random seed (42).
8. **Pixel normalisation** (`src/dataset.py`): pixels 0–255 → 0–1 → `(x − mean) / std`
   with the ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`,
   the same statistics the pretrained backbone was trained with.

### Data augmentation (training images only)

| Augmentation | Setting | Why it is realistic |
|---|---|---|
| Horizontal flip | 50% | A mirrored apple is still an apple |
| Rotation | ±15° | Items are photographed at slight angles |
| Translation | ±10% | The item is not always centred |
| Zoom | 90–110% | The camera is nearer or further away |
| Brightness | ±20% | Lighting changes |

Hue and saturation are **deliberately not changed**, because colour (browning, dark
spots, mould) is exactly what shows spoilage. Changing it could turn a fresh-looking
image into a spoiled-looking one.

## Model

| | |
|---|---|
| Architecture | **MobileNetV2** (torchvision), pretrained on ImageNet-1k (`IMAGENET1K_V2` weights) |
| Transfer learning | The ImageNet convolutional backbone is reused, and a new head is trained on our data |
| Input size | 3 × 224 × 224 (RGB) |
| Backbone output | 1,280 features per image (after global average pooling) |
| New head | Dropout(0.3) → Linear(1,280 → 28) |
| Output layer | 28 scores (logits) → **softmax** → probability of each class |
| Loss function | **Cross-entropy**, class-weighted (rare classes get larger weights) |
| Optimizer | **Adam**: learning rate 1e-3 (phase 1), 1e-4 (phase 2) |
| Batch size | 32 |
| Parameters | 2,259,740 total; 35,868 trainable in phase 1 (head only), 1,717,212 in phase 2 (head + last 5 blocks) |

### Why MobileNetV2?

| Model | Parameters | Compute per image | ImageNet top-1 | Verdict |
|---|---:|---:|---:|---|
| **MobileNetV2** | 3.5 M | 0.3 GFLOPs | ~72% | **Chosen**: small, fast, easy to explain, and enough for 28 visually distinct classes |
| EfficientNet-B0 | 5.3 M | 0.39 GFLOPs | ~77% | A bit more accurate on ImageNet but slower. Available with `--arch efficientnet_b0` |
| ResNet-50 | 25.6 M | 4.1 GFLOPs | ~76–80% | 7× more parameters and 13× more compute, which is unnecessary here and more likely to over-fit about 20k training images |

(The architecture numbers are the published torchvision figures.) MobileNetV2 also
suits a phone or edge device, which is a realistic deployment for this problem.

## Training (`src/train.py`)

1. Load `splits.csv` and inspect the classes and images per split.
2. Build DataLoaders: augmentation for training, plain normalisation for validation.
3. Build MobileNetV2 with ImageNet weights and a new 28-class head.
4. **Phase 1 (feature extraction):** freeze the backbone and train only the head for up to 5 epochs (Adam, lr 1e-3).
5. **Phase 2 (fine-tuning):** reload the best phase-1 weights, unfreeze the last 5
   backbone blocks and train for up to 15 more epochs with a 10× smaller learning
   rate (1e-4), so the high-level features adapt to spoilage textures.
6. Callbacks, written out in plain code so they can be explained:
   * **Model checkpointing:** save the model whenever validation loss improves.
   * **Early stopping:** stop a phase after 4 epochs without improvement.
   * **Learning-rate reduction:** `ReduceLROnPlateau` multiplies the LR by 0.2 after 2 epochs without improvement.
7. Uses the **NVIDIA GPU** with mixed precision when PyTorch can see it, otherwise the CPU.
8. Saves the model, the class labels, the history (CSV/JSON), the plots and `training_summary.json`.

## Evaluation (`src/evaluate.py`)

The model is measured on the **test split**. These images were never used for
training or for choosing the best epoch.

* **Accuracy:** the fraction of test images classified correctly (all 28 classes).
* **Precision** (per class): of the images predicted as class X, how many really are X.
* **Recall** (per class): of the images that really are X, how many the model found.
* **F1-score:** the harmonic mean of precision and recall.
* **Macro average:** every class counts equally, which matters because the classes are imbalanced.
* **Weighted average:** classes weighted by their number of test images.
* **Confusion matrix** (28×28) and a **Fresh vs Spoiled** 2×2 matrix derived from it.
* **Confidence analysis:** how often the model is below the 70% warning threshold, and how accurate it is above and below that threshold.

Outputs go to `results/`: `metrics.json`, `classification_report.txt`, `confusion_matrix.png`,
`confusion_matrix_freshness.png`, `training_accuracy.png`, `training_loss.png`,
`test_predictions.csv` and `misclassified_examples.png`.

## Results

<!-- RESULTS-START -->
**Being re-measured — do not quote any earlier number.**

The model itself is trained and working. The test metrics are currently being measured
again, because an earlier evaluation used a split that was not clean:

* This dataset was compiled from several sources and already contains augmented copies
  of some photos, with names such as `rotated_by_45_Screen Shot ….png` and
  `saltandpepper_Screen Shot ….png`.
* The first version of the duplicate check did not recognise those as copies, so about
  39% of the test images were edited copies of photos that were also in the training
  set. That inflates the score.
* `src/data_preprocessing.py` now keeps every copy of a photo in the same split as its
  original, drops copies whose original cannot be identified, and prints a
  **leakage check** line after splitting, which must read 0.

To produce the final numbers, run the three steps below and read them from
`results/metrics.json` and `results/classification_report.txt`:

```
.\run.bat preprocess
.\run.bat train
.\run.bat evaluate
```

`src/evaluate.py` writes accuracy, macro and weighted precision/recall/F1, the
per-class report, the 28x28 confusion matrix, a Fresh-vs-Spoiled matrix and the
training curves into `results/`.
<!-- RESULTS-END -->

---

## Installation (Windows 11)

Requirements: Windows 10/11, about 20 GB of free disk space, internet. An NVIDIA GPU
is optional but recommended.

**Easiest:** open the project folder in VS Code, open a terminal (**Terminal → New Terminal**) and run:

```
.\run.bat setup
```

This finds Python 3.12 (and installs it with `winget` if it is missing), creates a
virtual environment in `.venv`, installs `requirements.txt` (PyTorch 2.10 with CUDA
12.8, about 3 GB) and runs a GPU check.

**Manual equivalent:**

```
winget install -e --id Python.Python.3.12 --scope user
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python src\check_gpu.py
```

`check_gpu.py` prints whether PyTorch can see the GPU and saves the answer to
`results/environment_check.txt`.

> Why PyTorch rather than TensorFlow? TensorFlow has had no GPU support on
> native Windows since version 2.10, while PyTorch's CUDA 12.8 builds support RTX 50-series
> cards such as the RTX 5050 directly on Windows.

## Commands

All commands are run from the project folder. `run.bat <step>` does the same thing and
also saves the output in `logs\`.

| Step | Command | Shortcut |
|---|---|---|
| Download the dataset | `.venv\Scripts\python src\download_dataset.py` | `.\run.bat download` |
| Preprocess | `.venv\Scripts\python src\data_preprocessing.py` | `.\run.bat preprocess` |
| **Train** | `.venv\Scripts\python src\train.py` | `.\run.bat train` |
| **Evaluate** | `.venv\Scripts\python src\evaluate.py` | `.\run.bat evaluate` |
| **Predict one image** | `.venv\Scripts\python src\predict.py sample_images\<file>.jpg` | `.\run.bat predict sample_images\<file>.jpg` |
| **Web application** | `.venv\Scripts\python app\app.py`, then open http://127.0.0.1:5000 | `.\run.bat app` |
| Web-app self-test | `.venv\Scripts\python tests\test_web_app.py` | `.\run.bat test` |
| Everything (setup → test) | | `.\run.bat all` |

Useful training options: `--head-epochs`, `--finetune-epochs`, `--batch-size`,
`--arch efficientnet_b0`, `--no-class-weights`, `--cpu`, and `--max-per-class 20`
for a 2-minute smoke test.

## Web application

`app/app.py` (Flask) loads the trained model **once** at start-up. The page lets you:

1. choose an image (or click one of the test images under *"Or pick one of the test images"*),
2. see a **preview**,
3. click **Detect Freshness**, which uploads the file to `/predict`,
4. see **Fruit/Vegetable**, **Freshness** (FRESH in green / SPOILED in red),
   **Confidence**, P(fresh) / P(spoiled) and the top-3 classes.

The app also validates uploads:

* no file, wrong extension, empty file → clear error message
* a file that is not really an image (e.g. a renamed `.txt`) → rejected after trying to decode it
* larger than 10 MB → rejected
* top probability below **70%** → yellow warning: *"Low confidence: image may not
  belong to the supported fruit/vegetable categories"*

A JSON endpoint `/api/predict` returns the same result for automated tests.

## Project structure

```
fruit-vegetable-freshness-detector/
├── app/app.py                  Flask web application
├── templates/index.html        web page (upload, preview, result)
├── static/style.css            page styling
├── src/
│   ├── utils.py                shared settings: paths, image size, normalisation, label parsing, plots
│   ├── download_dataset.py     downloads + extracts the Kaggle dataset
│   ├── data_preprocessing.py   validation, resizing, de-duplication, stratified split
│   ├── dataset.py              PyTorch Dataset, augmentation, DataLoaders
│   ├── model.py                MobileNetV2 transfer-learning model
│   ├── train.py                two-phase training with checkpoint / early stop / LR reduction
│   ├── evaluate.py             test-set metrics, confusion matrix, plots
│   ├── predict.py              single-image prediction (also used by the web app)
│   └── check_gpu.py            checks that PyTorch can use the NVIDIA GPU
├── tests/test_web_app.py       uploads real test images through the web app
├── tools/run_logged.py         helper for run.bat (shows output and saves it to logs\)
├── dataset/                    raw/ and processed/ images (not in Git, ~5 GB)
├── models/                     fruit_vegetable_freshness_model.pt, class_labels.json
├── results/                    metrics, reports, plots
├── sample_images/              28 test-split images for the demo (+ labels.csv)
├── requirements.txt
├── run.bat
├── README.md, DATASET.md, PRESENTATION_NOTES.md
└── .gitignore
```

## Limitations

* **Only 14 produce types.** The model always answers with one of its 28 classes. A
  photo of a lemon, onion, pear or a shoe is still forced into one of them. The
  70% confidence warning helps, but softmax confidence is **not a reliable detector**
  of unknown objects: a model can be confidently wrong on something it has never seen.
* **External appearance only.** A photo shows the surface. Internal rot, bruising
  under the skin, bacterial contamination, pesticide residue and ripeness inside the
  fruit cannot be seen, so a "FRESH" answer is not a food-safety guarantee.
* **Two freshness levels only.** Fresh or spoiled, with no "slightly old" or
  "eat today" level, and no estimate of remaining shelf life.
* **Dataset bias.** The images were compiled from several internet and Kaggle
  sources. Some rotten classes consist of much smaller, lower-quality images than
  their fresh counterparts (see DATASET.md), so the model may partly rely on image
  quality or typical backgrounds instead of the spoilage itself.
* **Class imbalance.** Grape, guava, jujube and pomegranate have only 200 images per
  class, versus up to about 2,900 for apple. Rare classes are learned from fewer examples.
* **Lighting, background and camera.** Most images show a single item on a plain
  background. Real photos with several items, clutter, shadows, strong coloured light,
  blur or unusual angles may be classified worse than the test set suggests.
* **Test accuracy is measured on the same kind of images** as the training data (same
  sources). Accuracy on photos taken in a real shop would probably be lower.
* **One item per image.** There is no detection or localisation, so a basket with
  one rotten apple among fresh ones cannot be analysed.

## Future improvements

* **Larger and more varied data:** more produce types, more images per rare class,
  photos from real shops, phones and lighting conditions, and a separate test set
  collected by us.
* **More freshness levels:** fresh / aging / spoiled, or predicted days of shelf life.
* **Object detection** (e.g. YOLO) to find and judge every item in a crate or on a conveyor belt.
* **Open-set recognition:** a separate "not a supported item" class or an
  out-of-distribution detector instead of relying on softmax confidence.
* **Mobile and edge deployment:** export to ONNX / TensorFlow Lite and run on a phone
  or a Raspberry Pi / Jetson camera at a shop shelf.
* **Real-time camera detection** in the web app (webcam stream).
* **Multimodal freshness estimation:** combine images with gas sensors (ethylene,
  CO₂), temperature and humidity history, weight loss or near-infrared / hyperspectral
  imaging, which can reveal internal spoilage.
* **Explainability:** Grad-CAM heat maps showing which region (e.g. a mouldy patch) led to the decision.
