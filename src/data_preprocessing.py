"""
data_preprocessing.py
---------------------
Runs ONCE before training. It turns the raw downloaded dataset into a clean,
leakage-free, ready-to-train dataset.

Steps
  1. Find every class folder in dataset/raw/ (e.g. "Apple__Healthy").
  2. Open every image and detect corrupted / unreadable files.
  3. Fix rotation, convert to RGB and resize to 224 x 224; save to dataset/processed/images/.
  4. Prevent data leakage (the dataset was compiled from several sources and
     already contains augmented copies of some photos):
       - the same picture in two different classes  -> removed (label is ambiguous)
       - exact / flipped / rotated copies inside one class -> only one copy kept
       - near-identical look-alikes (perceptual hash) and files that the dataset
         itself names as augmented copies of one original photo, e.g.
         "rotated_by_45_Screen Shot ..." -> kept, but forced into the SAME split
       - "..._augmented_12.jpg" copies whose original photo is unknown -> removed
  5. Stratified split per class: 70% train, 15% validation, 15% test.
  6. Save dataset/processed/splits.csv, dataset_summary.json and a class
     distribution chart, and copy a few TEST images into sample_images/ for the demo.

Usage:
    python src/data_preprocessing.py
"""

import argparse
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (  # noqa: E402
    DATASET_SUMMARY_JSON, IMG_SIZE, PROCESSED_DIR, PROCESSED_IMAGES_DIR, PROJECT_ROOT,
    RAW_DIR, RESULTS_DIR, SAMPLE_IMAGES_DIR, SEED, SPLITS_CSV, SUPPORTED_EXTENSIONS,
    display_name, load_and_resize, parse_label, save_json,
)

TRAIN_FRACTION, VAL_FRACTION, TEST_FRACTION = 0.70, 0.15, 0.15
NEAR_DUPLICATE_MAX_BITS = 4   # hash distance (out of 72 bits) treated as "look-alike"

# Some source datasets stored augmented copies next to the original photo, e.g.
#   "Screen Shot 2018-06-08 at 5.08.30 PM.png"                (original)
#   "rotated_by_45_Screen Shot 2018-06-08 at 5.08.30 PM.png"  (augmented copy)
AUGMENTED_PREFIX = re.compile(r"^((rotated_by_\d+|translation|saltandpepper|vertical_flip)_)+", re.IGNORECASE)
# "Banana__Healthy_augmented_17.jpg": an augmented copy whose original is not identifiable
UNKNOWN_ORIGINAL = re.compile(r"_augmented_\d+$", re.IGNORECASE)


def source_photo_key(filename: str) -> str:
    """'rotated_by_45_Screen Shot X.png' -> 'Screen Shot X' (all copies of one photo share a key)."""
    stem = filename.rsplit(".", 1)[0]
    return AUGMENTED_PREFIX.sub("", stem)


# ---------------------------------------------------------------------------
# Step 1: find class folders and image files
# ---------------------------------------------------------------------------
def find_images(raw_dir: Path) -> list:
    """Return [(image_path, class_name), ...]. The class is the parent folder name."""
    items = []
    for path in sorted(raw_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            items.append((path, path.parent.name))
    return items


# ---------------------------------------------------------------------------
# Step 4 helper: perceptual hash that ignores flips and 90-degree rotations
# ---------------------------------------------------------------------------
def dihedral_dhash_bits(img: Image.Image) -> np.ndarray:
    """
    "Difference hash": shrink the image to 9x9 grey pixels and record, for each
    row, whether each pixel is brighter than its right-hand neighbour
    (9 rows x 8 comparisons = 72 bits). Similar-looking images give similar bits.

    We compute the hash for all 8 flips/rotations of the image so that a
    flipped or rotated copy of a picture is still recognised as a copy.
    Returns an array of shape (8, 72) with values 0/1.
    """
    grid = np.asarray(img.convert("L").resize((9, 9), Image.Resampling.BILINEAR), dtype=np.int16)
    variants = []
    for flipped in (grid, np.fliplr(grid)):
        for k in range(4):
            g = np.rot90(flipped, k)
            variants.append((g[:, 1:] > g[:, :-1]).flatten())
    return np.array(variants, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Steps 2 + 3: validate, resize and save ONE image (runs in parallel workers)
# ---------------------------------------------------------------------------
def process_one_image(task):
    src, dst = task
    try:
        with Image.open(src) as im:          # read header only: original size/format
            width, height, fmt = im.width, im.height, im.format
        img = load_and_resize(src, IMG_SIZE)  # decode + fix rotation + RGB + resize
        dst.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst, "JPEG", quality=95)
        bits = dihedral_dhash_bits(img)
        return {"ok": True, "width": width, "height": height, "format": fmt,
                "hash_bits": np.packbits(bits, axis=1).tobytes()}
    except Exception as exc:  # corrupted, truncated or not really an image
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# Step 4: duplicate detection
# ---------------------------------------------------------------------------
def unpack_hashes(hash_bytes_list) -> np.ndarray:
    """bytes -> array (n, 8, 72) of 0/1."""
    arr = np.frombuffer(b"".join(hash_bytes_list), dtype=np.uint8).reshape(len(hash_bytes_list), 8, 9)
    return np.unpackbits(arr, axis=2)[:, :, :72]


def pairwise_hash_distance(bits: np.ndarray) -> np.ndarray:
    """
    Hamming distance between every pair of images, taking the minimum over the
    8 flip/rotation variants of the second image. bits: (n, 8, 72).
    """
    a = bits[:, 0, :].astype(np.float32)          # original orientation of image i
    best = None
    for v in range(8):
        b = bits[:, v, :].astype(np.float32)       # variant v of image j
        dist = a @ (1 - b).T + (1 - a) @ b.T       # number of differing bits
        best = dist if best is None else np.minimum(best, dist)
    return best


def connected_groups(n: int, pairs: np.ndarray) -> np.ndarray:
    """Union-find: images linked by any chain of pairs get the same group id."""
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    return np.array([find(i) for i in range(n)])


def remove_duplicates_and_group(df: pd.DataFrame, bits: np.ndarray):
    """
    Returns (clean_df, stats). clean_df gets a 'group' column: images in the same
    group are look-alikes and must stay in the same split.
    """
    stats = {}
    # (0) augmented copies whose original photo cannot be identified -> drop (possible leakage)
    unknown = df["raw_path"].map(lambda p: bool(UNKNOWN_ORIGINAL.search(p.rsplit("/", 1)[-1].rsplit(".", 1)[0])))
    stats["augmented_copies_with_unknown_original_removed"] = int(unknown.sum())
    stats["files_named_as_augmented_copies"] = int(
        df["raw_path"].map(lambda p: bool(AUGMENTED_PREFIX.match(p.rsplit("/", 1)[-1]))).sum())

    # (a) the same picture (up to flip/rotation) in two different classes -> drop all copies
    canonical = [min(bytes(np.packbits(v)) for v in img_bits) for img_bits in bits]
    df = df.assign(canonical_hash=canonical)
    classes_per_hash = df.groupby("canonical_hash")["label"].nunique()
    conflict_hashes = set(classes_per_hash[classes_per_hash > 1].index)
    conflict_mask = df["canonical_hash"].isin(conflict_hashes).to_numpy() & ~unknown.to_numpy()
    stats["cross_class_duplicates_removed"] = int(conflict_mask.sum())
    conflict_mask |= unknown.to_numpy()

    keep_rows, groups = [], {}
    exact_removed = Counter()
    for label, sub in df[~conflict_mask].groupby("label"):
        idx = sub.index.to_numpy()
        dist = pairwise_hash_distance(bits[idx])
        # (b) exact copies (distance 0) inside the class -> keep the first one only
        exact_pairs = np.argwhere(np.triu(dist == 0, k=1))
        exact_group = connected_groups(len(idx), exact_pairs)
        first_of_group = exact_group == np.arange(len(idx))
        exact_removed[label] = int((~first_of_group).sum())
        kept = idx[first_of_group]
        kept_dist = dist[np.ix_(first_of_group, first_of_group)]
        # (c) look-alikes (small hash distance) -> same group (same split later)
        near_pairs = np.argwhere(np.triu(kept_dist <= NEAR_DUPLICATE_MAX_BITS, k=1))
        # (d) files that are named copies of the same original photo -> same group
        first_position, source_pairs = {}, []
        for position, key in enumerate(df.loc[kept, "source"]):
            if key in first_position:
                source_pairs.append((first_position[key], position))
            else:
                first_position[key] = position
        all_pairs = np.concatenate([near_pairs, np.array(source_pairs, dtype=int).reshape(-1, 2)])
        near_group = connected_groups(len(kept), all_pairs)
        for row, g in zip(kept, near_group):
            groups[row] = f"{label}#{g}"
        keep_rows.extend(kept.tolist())

    clean = df.loc[sorted(keep_rows)].copy()
    clean["group"] = [groups[i] for i in clean.index]
    stats["exact_duplicates_removed_within_class"] = int(sum(exact_removed.values()))
    group_sizes = clean.groupby("group").size()
    stats["look_alike_groups_with_more_than_one_image"] = int((group_sizes > 1).sum())
    stats["images_in_look_alike_groups"] = int(group_sizes[group_sizes > 1].sum())
    stats["largest_look_alike_group"] = int(group_sizes.max())

    # delete processed files of removed images so dataset/processed only holds used images
    removed = df.index.difference(clean.index)
    for p in df.loc[removed, "processed_path"]:
        (PROJECT_ROOT / p).unlink(missing_ok=True)
    return clean.drop(columns=["canonical_hash"]), stats


# ---------------------------------------------------------------------------
# Step 5: stratified, group-aware split
# ---------------------------------------------------------------------------
def split_dataset(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """
    For every class separately: shuffle its groups, give ~15% of the images to
    'test', ~15% to 'val' and the rest to 'train'. A whole group always goes to
    one split, so look-alike images never appear in both train and test.
    """
    rng = np.random.default_rng(seed)
    split_of_group = {}
    for label, sub in df.groupby("label"):
        sizes = sub.groupby("group").size()
        order = rng.permutation(sizes.index.to_numpy())
        n_total, n_test, n_val = len(sub), 0, 0
        for g in order:
            if n_test < TEST_FRACTION * n_total:
                split_of_group[g] = "test"
                n_test += sizes[g]
            elif n_val < VAL_FRACTION * n_total:
                split_of_group[g] = "val"
                n_val += sizes[g]
            else:
                split_of_group[g] = "train"
    df = df.copy()
    df["split"] = df["group"].map(split_of_group)
    return df


# ---------------------------------------------------------------------------
# Step 6: summaries, chart and demo samples
# ---------------------------------------------------------------------------
def plot_class_distribution(df: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = df.groupby("label").size().sort_index()
    colors = ["#2e7d32" if parse_label(c)["freshness"] == "Fresh" else "#c62828" for c in counts.index]
    plt.figure(figsize=(12, 6))
    plt.bar([display_name(c) for c in counts.index], counts.values, color=colors)
    plt.xticks(rotation=75, ha="right", fontsize=8)
    plt.ylabel("Images (after cleaning)")
    plt.title("Images per class (green = fresh, red = spoiled)")
    plt.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(RESULTS_DIR / "class_distribution.png", dpi=120)
    plt.close()


def export_demo_samples(df: pd.DataFrame, per_class: int = 1, seed: int = SEED) -> None:
    """Copy original (raw) TEST images - never seen in training - into sample_images/."""
    if SAMPLE_IMAGES_DIR.exists():
        shutil.rmtree(SAMPLE_IMAGES_DIR)
    SAMPLE_IMAGES_DIR.mkdir(parents=True)
    test = df[df["split"] == "test"]
    manifest = []
    for label, sub in test.groupby("label"):
        for _, row in sub.sample(n=min(per_class, len(sub)), random_state=seed).iterrows():
            src = PROJECT_ROOT / row["raw_path"]
            dst = SAMPLE_IMAGES_DIR / f"{label}_{src.stem}{src.suffix.lower()}".replace(" ", "_")
            shutil.copy2(src, dst)
            manifest.append({"file": dst.name, "true_class": label})
    pd.DataFrame(manifest).to_csv(SAMPLE_IMAGES_DIR / "labels.csv", index=False)


def build_summary(items, results, clean, dup_stats, corrupted) -> dict:
    ok = [r for r in results if r["ok"]]
    widths = np.array([r["width"] for r in ok])
    heights = np.array([r["height"] for r in ok])
    raw_counts = Counter(label for _, label in items)
    final_counts = clean.groupby("label").size()
    split_counts = clean.groupby(["label", "split"]).size().unstack(fill_value=0)

    sizes_per_class = defaultdict(list)
    for (path, label), r in zip(items, results):
        if r["ok"]:
            sizes_per_class[label].append((r["width"], r["height"], path.stat().st_size))

    classes = []
    for label in sorted(raw_counts):
        info = parse_label(label)
        sizes = np.array(sizes_per_class[label]) if sizes_per_class[label] else np.zeros((1, 3))
        classes.append({
            "class_name": label,
            "produce": info["produce"],
            "freshness": info["freshness"],
            "raw_images": int(raw_counts[label]),
            "median_original_width": int(np.median(sizes[:, 0])),
            "median_original_height": int(np.median(sizes[:, 1])),
            "median_file_kb": round(float(np.median(sizes[:, 2])) / 1024, 1),
            "images_after_cleaning": int(final_counts.get(label, 0)),
            "train": int(split_counts.loc[label].get("train", 0)) if label in split_counts.index else 0,
            "val": int(split_counts.loc[label].get("val", 0)) if label in split_counts.index else 0,
            "test": int(split_counts.loc[label].get("test", 0)) if label in split_counts.index else 0,
        })
    fresh_mask = clean["label"].map(lambda c: parse_label(c)["freshness"]) == "Fresh"
    return {
        "raw_dir": str(RAW_DIR.relative_to(PROJECT_ROOT)),
        "num_classes": int(clean["label"].nunique()),
        "num_produce_types": len({c["produce"] for c in classes}),
        "raw_image_files": len(items),
        "corrupted_or_unreadable": len(corrupted),
        "corrupted_files": corrupted[:50],
        **dup_stats,
        "images_after_cleaning": int(len(clean)),
        "fresh_images": int(fresh_mask.sum()),
        "spoiled_images": int((~fresh_mask).sum()),
        "split_sizes": {s: int((clean["split"] == s).sum()) for s in ("train", "val", "test")},
        "original_formats": dict(Counter(r["format"] for r in ok)),
        "file_extensions": dict(Counter(p.suffix.lower() for p, _ in items)),
        "original_width": {"min": int(widths.min()), "median": int(np.median(widths)), "max": int(widths.max())},
        "original_height": {"min": int(heights.min()), "median": int(np.median(heights)), "max": int(heights.max())},
        "processed_image_size": [IMG_SIZE, IMG_SIZE],
        "raw_total_bytes": int(sum(p.stat().st_size for p, _ in items)),
        "near_duplicate_max_bits": NEAR_DUPLICATE_MAX_BITS,
        "split_fractions": [TRAIN_FRACTION, VAL_FRACTION, TEST_FRACTION],
        "seed": SEED,
        "classes": classes,
    }


def main():
    parser = argparse.ArgumentParser(description="Clean, resize, de-duplicate and split the dataset")
    parser.add_argument("--workers", type=int, default=None, help="parallel processes (default: all CPU cores)")
    args = parser.parse_args()

    start = time.time()
    items = find_images(RAW_DIR)
    if not items:
        sys.exit(f"No images found in {RAW_DIR}. Run: python src/download_dataset.py")
    labels = sorted({label for _, label in items})
    print(f"Found {len(items)} image files in {len(labels)} class folders:")
    for label, n in sorted(Counter(label for _, label in items).items()):
        print(f"  {label:<25} {n:>6}")

    if PROCESSED_IMAGES_DIR.exists():
        shutil.rmtree(PROCESSED_IMAGES_DIR)
    tasks = [(src, PROCESSED_IMAGES_DIR / label / f"{label}_{i:05d}.jpg") for i, (src, label) in enumerate(items)]

    print("\nValidating, resizing and hashing images ...")
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, res in enumerate(pool.map(process_one_image, tasks, chunksize=32), 1):
            results.append(res)
            if i % 2000 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} images processed")

    corrupted = [str(src.relative_to(PROJECT_ROOT)) + " -> " + r["error"]
                 for (src, _), r in zip(items, results) if not r["ok"]]
    print(f"Corrupted / unreadable images skipped: {len(corrupted)}")

    rows, hashes = [], []
    for (src, label), (_, dst), r in zip(items, tasks, results):
        if r["ok"]:
            rows.append({"raw_path": src.relative_to(PROJECT_ROOT).as_posix(),
                         "processed_path": dst.relative_to(PROJECT_ROOT).as_posix(),
                         "label": label,
                         "source": source_photo_key(src.name)})
            hashes.append(r["hash_bits"])
    df = pd.DataFrame(rows)
    bits = unpack_hashes(hashes)

    print("Detecting duplicates and look-alike images ...")
    clean, dup_stats = remove_duplicates_and_group(df, bits)
    for k, v in dup_stats.items():
        print(f"  {k}: {v}")

    clean = split_dataset(clean)
    # Leakage check: no val/test image may share its original photo or look-alike group with a train image
    train = clean[clean["split"] == "train"]
    train_sources, train_groups = set(zip(train["label"], train["source"])), set(train["group"])
    held_out = clean[clean["split"] != "train"]
    leaks = sum(((l, s) in train_sources) or (g in train_groups)
                for l, s, g in zip(held_out["label"], held_out["source"], held_out["group"]))
    dup_stats["leakage_check_val_test_images_linked_to_train"] = int(leaks)
    print(f"Leakage check: {leaks} validation/test images share a source photo or look-alike group with training images")
    class_names = sorted(clean["label"].unique())
    clean["label_index"] = clean["label"].map({c: i for i, c in enumerate(class_names)})  # label encoding
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    clean[["processed_path", "raw_path", "label", "label_index", "split", "group", "source"]].to_csv(SPLITS_CSV, index=False)

    summary = build_summary(items, results, clean, dup_stats, corrupted)
    summary["class_names"] = class_names
    summary["processing_seconds"] = round(time.time() - start, 1)
    save_json(summary, DATASET_SUMMARY_JSON)
    plot_class_distribution(clean)
    export_demo_samples(clean)

    print(f"\nImages after cleaning: {len(clean)}  (fresh {summary['fresh_images']}, spoiled {summary['spoiled_images']})")
    print("Split sizes:", summary["split_sizes"])
    print(f"Saved {SPLITS_CSV.relative_to(PROJECT_ROOT)} and {DATASET_SUMMARY_JSON.relative_to(PROJECT_ROOT)}")
    print(f"Finished in {summary['processing_seconds']} s")


if __name__ == "__main__":
    main()
