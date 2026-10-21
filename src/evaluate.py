"""
evaluate.py
-----------
Measures the trained model on the TEST split - images the model never saw
during training or model selection.

    python src/evaluate.py

Outputs (results/)
  metrics.json                 accuracy, precision, recall, F1 (macro + weighted), freshness-only metrics
  classification_report.txt    per-class precision / recall / F1 / support
  confusion_matrix.png / .csv  28 x 28 matrix: rows = true class, columns = predicted class
  confusion_matrix_freshness.png   2 x 2 matrix: Fresh vs Spoiled
  test_predictions.csv         prediction + confidence for every test image
  misclassified_examples.png   some test images the model got wrong
  training_accuracy.png, training_loss.png  (re-drawn from training_history.json)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import FreshnessDataset, get_eval_transform, load_splits  # noqa: E402
from predict import load_trained_model  # noqa: E402
from utils import (  # noqa: E402
    CONFIDENCE_THRESHOLD, PROJECT_ROOT, RESULTS_DIR, default_num_workers, display_name,
    get_device, load_json, parse_label, plot_training_history, save_json,
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402


@torch.no_grad()
def predict_test_set(model, loader, device):
    """Run the model on every test image; return true labels, predicted labels, probabilities."""
    model.eval()
    all_probs, all_labels = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        all_labels.append(labels.numpy())
    probs = np.concatenate(all_probs)
    return np.concatenate(all_labels), probs.argmax(axis=1), probs


def plot_confusion_matrix(cm, names, path, title, annotate=True, figsize=(16, 14)):
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=annotate, fmt="d", cmap="Blues", xticklabels=names, yticklabels=names,
                cbar=True, annot_kws={"size": 7 if len(names) > 10 else 12})
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title(title)
    plt.xticks(rotation=75, ha="right", fontsize=8 if len(names) > 10 else 11)
    plt.yticks(rotation=0, fontsize=8 if len(names) > 10 else 11)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_misclassified(frame, y_true, y_pred, probs, class_names, path, max_images=20):
    wrong = np.where(y_true != y_pred)[0][:max_images]
    if len(wrong) == 0:
        return
    cols = 5
    rows = int(np.ceil(len(wrong) / cols))
    plt.figure(figsize=(cols * 3.2, rows * 3.6))
    for k, i in enumerate(wrong, 1):
        from PIL import Image
        img = Image.open(PROJECT_ROOT / frame.iloc[i]["processed_path"])
        plt.subplot(rows, cols, k)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"true: {display_name(class_names[y_true[i]])}\n"
                  f"pred: {display_name(class_names[y_pred[i]])} ({probs[i].max():.0%})", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained model on the test split")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    device = get_device()
    model, meta = load_trained_model(device=device)
    class_names = meta["class_names"]
    print(f"Loaded model ({meta['arch']}, {len(class_names)} classes) on {device}")

    # --- test data (never used for training or for choosing the best epoch) ---
    frame = load_splits()
    test = frame[frame["split"] == "test"].reset_index(drop=True)
    workers = default_num_workers() if args.workers is None else args.workers
    loader = DataLoader(FreshnessDataset(test, get_eval_transform()), batch_size=args.batch_size,
                        shuffle=False, num_workers=workers)
    print(f"Evaluating on {len(test)} test images ...")
    y_true, y_pred, probs = predict_test_set(model, loader, device)
    confidence = probs.max(axis=1)

    # --- overall metrics ---
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    names = [display_name(c) for c in class_names]
    report = classification_report(y_true, y_pred, labels=list(range(len(class_names))),
                                   target_names=names, digits=4, zero_division=0)

    # --- freshness only (Fresh vs Spoiled), derived from the 28-class prediction ---
    fresh_of = np.array([parse_label(c)["freshness"] == "Fresh" for c in class_names])
    produce_of = np.array([parse_label(c)["produce"] for c in class_names])
    true_fresh, pred_fresh = fresh_of[y_true], fresh_of[y_pred]
    fresh_acc = accuracy_score(true_fresh, pred_fresh)
    fp, fr, ff, _ = precision_recall_fscore_support(true_fresh, pred_fresh, average=None, labels=[True, False],
                                                    zero_division=0)
    produce_acc = float((produce_of[y_true] == produce_of[y_pred]).mean())

    # --- confidence analysis (for the low-confidence warning) ---
    confident = confidence >= CONFIDENCE_THRESHOLD
    conf_stats = {
        "threshold": CONFIDENCE_THRESHOLD,
        "fraction_of_test_images_above_threshold": float(confident.mean()),
        "accuracy_above_threshold": float((y_true[confident] == y_pred[confident]).mean()) if confident.any() else None,
        "accuracy_below_threshold": float((y_true[~confident] == y_pred[~confident]).mean()) if (~confident).any() else None,
    }

    metrics = {
        "test_images": int(len(test)),
        "accuracy": float(acc),
        "precision_macro": float(p_macro), "recall_macro": float(r_macro), "f1_macro": float(f1_macro),
        "precision_weighted": float(p_w), "recall_weighted": float(r_w), "f1_weighted": float(f1_w),
        "misclassified": int((y_true != y_pred).sum()),
        "freshness_only": {
            "accuracy": float(fresh_acc),
            "fresh": {"precision": float(fp[0]), "recall": float(fr[0]), "f1": float(ff[0])},
            "spoiled": {"precision": float(fp[1]), "recall": float(fr[1]), "f1": float(ff[1])},
        },
        "produce_type_accuracy": produce_acc,
        "confidence": conf_stats,
        "model_epoch": meta.get("epoch"),
        "device": str(device),
    }

    # --- save everything ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_json(metrics, RESULTS_DIR / "metrics.json")
    with open(RESULTS_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Test set: {len(test)} images, {len(class_names)} classes\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Macro    precision {p_macro:.4f}  recall {r_macro:.4f}  F1 {f1_macro:.4f}\n")
        f.write(f"Weighted precision {p_w:.4f}  recall {r_w:.4f}  F1 {f1_w:.4f}\n\n")
        f.write(report)
        f.write(f"\nFreshness only (Fresh vs Spoiled) accuracy: {fresh_acc:.4f}\n")
        f.write(f"Produce type accuracy (ignoring freshness): {produce_acc:.4f}\n")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    pd.DataFrame(cm, index=names, columns=names).to_csv(RESULTS_DIR / "confusion_matrix.csv")
    plot_confusion_matrix(cm, names, RESULTS_DIR / "confusion_matrix.png",
                          f"Confusion matrix - test set ({len(test)} images, accuracy {acc:.2%})")
    cm_fresh = confusion_matrix(true_fresh, pred_fresh, labels=[True, False])
    plot_confusion_matrix(cm_fresh, ["Fresh", "Spoiled"], RESULTS_DIR / "confusion_matrix_freshness.png",
                          f"Fresh vs Spoiled - test set (accuracy {fresh_acc:.2%})", figsize=(6, 5))

    test = test.assign(true_class=[class_names[i] for i in y_true],
                       predicted_class=[class_names[i] for i in y_pred],
                       confidence=confidence.round(4), correct=y_true == y_pred)
    test[["raw_path", "true_class", "predicted_class", "confidence", "correct"]].to_csv(
        RESULTS_DIR / "test_predictions.csv", index=False)
    plot_misclassified(test, y_true, y_pred, probs, class_names, RESULTS_DIR / "misclassified_examples.png")

    history_path = RESULTS_DIR / "training_history.json"
    if history_path.exists():
        plot_training_history(load_json(history_path))

    # --- print a short summary ---
    print(f"\nTest accuracy:            {acc:.4f}")
    print(f"Macro precision / recall / F1:    {p_macro:.4f} / {r_macro:.4f} / {f1_macro:.4f}")
    print(f"Weighted precision / recall / F1: {p_w:.4f} / {r_w:.4f} / {f1_w:.4f}")
    print(f"Fresh-vs-Spoiled accuracy: {fresh_acc:.4f} | produce-type accuracy: {produce_acc:.4f}")
    print(f"Misclassified: {metrics['misclassified']} of {len(test)}")
    off_diag = [(cm[i, j], names[i], names[j]) for i in range(len(names)) for j in range(len(names)) if i != j and cm[i, j]]
    for n, t, p in sorted(off_diag, reverse=True)[:5]:
        print(f"  most common confusion: {t} -> {p}  ({n} images)")
    print(f"\nSaved metrics and plots to {RESULTS_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
