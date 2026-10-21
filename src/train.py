"""
train.py
--------
Trains the Fruit & Vegetable Freshness classifier with transfer learning.

    python src/train.py                      # full training (uses the GPU if available)
    python src/train.py --max-per-class 20 --head-epochs 1 --finetune-epochs 1   # 2-minute smoke test

Training happens in two phases:
  Phase 1 - "feature extraction": the pretrained MobileNetV2 backbone is frozen,
            only the new classification head learns (fast, stable).
  Phase 2 - "fine-tuning": the last few backbone blocks are unfrozen and trained
            together with the head using a 10x smaller learning rate, so the
            features adapt to fruit/vegetable textures (spots, mould, wrinkles).

Callbacks implemented by hand (so you can see exactly what they do):
  * model checkpointing   - save the weights whenever validation loss improves
  * early stopping        - stop a phase if validation loss has not improved for N epochs
  * learning-rate reduction - ReduceLROnPlateau divides the LR when validation loss stalls

Outputs
  models/fruit_vegetable_freshness_model.pt   best model (weights + class names + settings)
  models/class_labels.json                    class index -> class name
  results/training_history.csv / .json        loss & accuracy per epoch
  results/training_accuracy.png, training_loss.png
  results/training_summary.json, training_log.txt
"""

import argparse
import logging
import platform
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import build_dataloaders  # noqa: E402
from model import (  # noqa: E402
    SUPPORTED_ARCHS, build_model, count_parameters, freeze_backbone,
    keep_frozen_batchnorm_in_eval_mode, unfreeze_last_blocks,
)
from utils import (  # noqa: E402
    CLASS_LABELS_PATH, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE, MODEL_PATH, MODELS_DIR, RESULTS_DIR,
    default_num_workers, display_name, ensure_dirs, get_device, plot_training_history, save_json, set_seed,
)

log = logging.getLogger("train")


def parse_args():
    p = argparse.ArgumentParser(description="Train the freshness classifier")
    p.add_argument("--arch", default="mobilenet_v2", choices=SUPPORTED_ARCHS)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--head-epochs", type=int, default=5, help="max epochs for phase 1 (frozen backbone)")
    p.add_argument("--finetune-epochs", type=int, default=15, help="max epochs for phase 2 (fine-tuning)")
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--finetune-lr", type=float, default=1e-4)
    p.add_argument("--unfreeze-blocks", type=int, default=5, help="backbone blocks to unfreeze in phase 2")
    p.add_argument("--patience", type=int, default=4, help="early-stopping patience (epochs)")
    p.add_argument("--workers", type=int, default=None, help="DataLoader worker processes")
    p.add_argument("--max-per-class", type=int, default=None, help="use only N images per class/split (quick test)")
    p.add_argument("--no-class-weights", action="store_true", help="do not weight the loss by class frequency")
    p.add_argument("--cpu", action="store_true", help="force CPU even if a GPU is available")
    return p.parse_args()


def setup_logging():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(RESULTS_DIR / "training_log.txt", mode="w", encoding="utf-8")],
    )


# ---------------------------------------------------------------------------
# One pass over the data (training OR evaluation)
# ---------------------------------------------------------------------------
def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    """If an optimizer is given the model learns; otherwise we only measure."""
    training = optimizer is not None
    model.train(training)
    if training:
        keep_frozen_batchnorm_in_eval_mode(model)
    use_amp = device.type == "cuda"   # mixed precision = faster on NVIDIA GPUs

    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(training), \
                torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(images)             # forward pass -> one score (logit) per class
            loss = criterion(outputs, labels)   # cross-entropy: how wrong were the scores?

        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()       # backpropagation: gradient of loss w.r.t. every weight
            scaler.step(optimizer)              # optimizer (Adam) updates the weights
            scaler.update()

        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        seen += labels.size(0)
    return total_loss / seen, correct / seen


def save_checkpoint(model, class_names, args, epoch, val_loss, val_acc):
    torch.save({
        "model_state": model.state_dict(),
        "arch": args.arch,
        "num_classes": len(class_names),
        "class_names": class_names,
        "img_size": IMG_SIZE,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "epoch": epoch,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "torch_version": str(torch.__version__),
    }, MODEL_PATH)


def train_phase(phase, model, loaders, criterion, optimizer, device, args, epochs, state, class_names):
    """Train for up to `epochs` epochs with checkpointing, LR reduction and early stopping."""
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=2, min_lr=1e-6)
    epochs_without_improvement = 0

    for _ in range(epochs):
        state["epoch"] += 1
        start = time.time()
        lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = run_epoch(model, loaders["train"], criterion, device, optimizer, scaler)
        val_loss, val_acc = run_epoch(model, loaders["val"], criterion, device)
        scheduler.step(val_loss)            # learning-rate reduction on plateau
        seconds = time.time() - start

        state["history"].append({
            "epoch": state["epoch"], "phase": phase, "lr": lr,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "seconds": round(seconds, 1),
        })
        msg = (f"[{phase:>9}] epoch {state['epoch']:>2} | lr {lr:.1e} | "
               f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
               f"val loss {val_loss:.4f} acc {val_acc:.4f} | {seconds:.0f}s")

        if val_loss < state["best_val_loss"] - 1e-4:   # model checkpointing
            state.update(best_val_loss=val_loss, best_val_acc=val_acc, best_epoch=state["epoch"])
            save_checkpoint(model, class_names, args, state["epoch"], val_loss, val_acc)
            epochs_without_improvement = 0
            msg += "  <- best so far, saved"
        else:
            epochs_without_improvement += 1
        log.info(msg)

        if epochs_without_improvement >= args.patience:  # early stopping
            log.info(f"Early stopping: validation loss did not improve for {args.patience} epochs.")
            break


def main():
    args = parse_args()
    setup_logging()
    ensure_dirs()
    set_seed()

    device = torch.device("cpu") if args.cpu else get_device()
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    log.info(f"PyTorch {torch.__version__} | CUDA build {torch.version.cuda} | device: {device}"
             + (f" ({gpu_name})" if gpu_name else " (no GPU used)"))
    torch.backends.cudnn.benchmark = True

    # 1-4. Load dataset, inspect classes, use the saved split, apply preprocessing/augmentation
    workers = default_num_workers() if args.workers is None else args.workers
    loaders, frame = build_dataloaders(args.batch_size, workers, args.max_per_class,
                                       pin_memory=device.type == "cuda")
    class_names = (frame[["label_index", "label"]].drop_duplicates()
                   .sort_values("label_index")["label"].tolist())
    log.info(f"{len(class_names)} classes: {', '.join(display_name(c) for c in class_names)}")
    log.info("Images per split: " + ", ".join(f"{s}={len(loaders[s].dataset)}" for s in ("train", "val", "test")))

    # 5. Build the model (pretrained backbone + new head)
    model = build_model(len(class_names), args.arch, pretrained=True).to(device)
    # The dataset is imbalanced (some classes have ~10x more images than others).
    # Class weights make a mistake on a rare class cost more, so the model does
    # not simply ignore small classes. weight_c = N_train / (num_classes * n_c)
    counts = frame[frame["split"] == "train"]["label_index"].value_counts().sort_index()
    class_weights = None
    if not args.no_class_weights:
        class_weights = torch.tensor((counts.sum() / (len(counts) * counts)).to_numpy(), dtype=torch.float32)
        log.info("Class weights (rare classes get larger weights): "
                 + ", ".join(f"{display_name(c)}={w:.2f}" for c, w in zip(class_names, class_weights.tolist())))
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    state = {"epoch": 0, "best_val_loss": float("inf"), "best_val_acc": 0.0, "best_epoch": 0, "history": []}
    start = time.time()

    # 6. Phase 1: train only the classification head
    freeze_backbone(model)
    log.info(f"\nPhase 1 - frozen backbone, parameters: {count_parameters(model)}")
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.head_lr)
    train_phase("head", model, loaders, criterion, optimizer, device, args, args.head_epochs, state, class_names)

    # 7. Phase 2: fine-tune the last backbone blocks, starting from the best phase-1 weights
    if args.finetune_epochs > 0:
        best = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        model.load_state_dict(best["model_state"])
        unfreeze_last_blocks(model, args.unfreeze_blocks)
        log.info(f"\nPhase 2 - fine-tuning last {args.unfreeze_blocks} blocks, parameters: {count_parameters(model)}")
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.finetune_lr)
        train_phase("fine_tune", model, loaders, criterion, optimizer, device, args,
                    args.finetune_epochs, state, class_names)

    # 8. Save labels, history, plots and a summary
    minutes = (time.time() - start) / 60
    save_json({str(i): name for i, name in enumerate(class_names)}, CLASS_LABELS_PATH)
    pd.DataFrame(state["history"]).to_csv(RESULTS_DIR / "training_history.csv", index=False)
    save_json(state["history"], RESULTS_DIR / "training_history.json")
    plot_training_history(state["history"])
    save_json({
        "arch": args.arch, "img_size": IMG_SIZE, "num_classes": len(class_names),
        "batch_size": args.batch_size, "optimizer": "Adam",
        "loss": "CrossEntropyLoss" + ("" if args.no_class_weights else " (class-weighted)"),
        "head_lr": args.head_lr, "finetune_lr": args.finetune_lr, "unfreeze_blocks": args.unfreeze_blocks,
        "max_head_epochs": args.head_epochs, "max_finetune_epochs": args.finetune_epochs,
        "epochs_run": state["epoch"], "best_epoch": state["best_epoch"],
        "best_val_loss": state["best_val_loss"], "best_val_acc": state["best_val_acc"],
        "train_images": len(loaders["train"].dataset), "val_images": len(loaders["val"].dataset),
        "test_images": len(loaders["test"].dataset), "max_per_class": args.max_per_class,
        "device": str(device), "gpu_name": gpu_name, "torch_version": str(torch.__version__),
        "cuda_build": torch.version.cuda, "python": platform.python_version(),
        "training_minutes": round(minutes, 1),
    }, RESULTS_DIR / "training_summary.json")

    log.info(f"\nFinished in {minutes:.1f} min. Best epoch {state['best_epoch']} "
             f"(val loss {state['best_val_loss']:.4f}, val acc {state['best_val_acc']:.4f}).")
    log.info(f"Best model saved to {MODEL_PATH.relative_to(MODELS_DIR.parent)}")
    log.info("Next step: python src/evaluate.py")


if __name__ == "__main__":
    main()
