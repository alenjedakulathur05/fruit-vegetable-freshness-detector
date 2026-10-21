"""
predict.py
----------
Predict the produce type and freshness of ONE image with the trained model.

    python src/predict.py path/to/image.jpg
    python src/predict.py sample_images/Banana__Rotten_xxx.jpg --top-k 5

Pipeline (identical to validation/test preprocessing):
  load image -> fix rotation -> RGB -> resize 224x224 -> tensor 0..1
  -> normalise with ImageNet mean/std -> model -> softmax -> most likely class
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import get_eval_transform  # noqa: E402
from model import build_model  # noqa: E402
from utils import (  # noqa: E402
    CONFIDENCE_THRESHOLD, MODEL_PATH, get_device, is_valid_image, load_and_resize, parse_label,
)


def load_trained_model(model_path=MODEL_PATH, device=None):
    """Rebuild the network and load the trained weights + class names saved by train.py."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run: python src/train.py")
    device = device or get_device()
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model = build_model(checkpoint["num_classes"], checkpoint["arch"], pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()   # eval(): turns off dropout, BatchNorm uses stored statistics
    meta = {k: v for k, v in checkpoint.items() if k != "model_state"}
    return model, meta


def preprocess_image(image_source, img_size: int):
    """Exactly the same steps as the validation/test images -> tensor of shape (1, 3, 224, 224)."""
    image = load_and_resize(image_source, img_size)
    return get_eval_transform()(image).unsqueeze(0)   # add a "batch" dimension of 1


@torch.no_grad()
def predict_image(model, meta, image_source, device=None, top_k: int = 3, threshold: float = CONFIDENCE_THRESHOLD):
    """Return a dictionary with the predicted class, produce, freshness and confidence."""
    device = device or next(model.parameters()).device
    tensor = preprocess_image(image_source, meta["img_size"]).to(device)
    probs = torch.softmax(model(tensor), dim=1)[0].cpu()   # softmax: scores -> probabilities (sum = 1)

    class_names = meta["class_names"]
    top_probs, top_idx = probs.topk(min(top_k, len(class_names)))
    best = class_names[int(top_idx[0])]
    info = parse_label(best)

    # probability that the item is fresh = sum of probabilities of all "fresh" classes
    fresh_prob = float(sum(p for p, c in zip(probs.tolist(), class_names) if parse_label(c)["freshness"] == "Fresh"))
    confidence = float(top_probs[0])
    low_confidence = confidence < threshold
    return {
        "class_name": best,
        "produce": info["produce"],
        "freshness": info["freshness"],
        "confidence": confidence,
        "fresh_probability": fresh_prob,
        "spoiled_probability": 1.0 - fresh_prob,
        "low_confidence": low_confidence,
        "message": ("Low confidence: image may not belong to the supported fruit/vegetable categories, "
                    "or the freshness is unclear." if low_confidence else ""),
        "top_k": [{"class_name": class_names[int(i)], **parse_label(class_names[int(i)]), "probability": float(p)}
                  for p, i in zip(top_probs, top_idx)],
    }


def main():
    parser = argparse.ArgumentParser(description="Predict freshness for one image")
    parser.add_argument("image", type=Path, help="path to a .jpg / .png image")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD)
    args = parser.parse_args()

    if not args.image.exists():
        sys.exit(f"File not found: {args.image}")
    if not is_valid_image(args.image):
        sys.exit(f"Not a readable image file: {args.image}")

    model, meta = load_trained_model()
    result = predict_image(model, meta, args.image, top_k=args.top_k, threshold=args.threshold)

    print(f"Image:            {args.image}")
    print(f"Predicted class:  {result['freshness']} {result['produce']}   ({result['class_name']})")
    print(f"Fruit/Vegetable:  {result['produce']}")
    print(f"Freshness:        {result['freshness'].upper()}")
    print(f"Confidence:       {result['confidence']:.2%}")
    print(f"P(fresh) = {result['fresh_probability']:.2%}   P(spoiled) = {result['spoiled_probability']:.2%}")
    if result["low_confidence"]:
        print("WARNING:", result["message"])
    print(f"Top {len(result['top_k'])} predictions:")
    for item in result["top_k"]:
        print(f"  {item['freshness']:<8} {item['produce']:<12} {item['probability']:.2%}")


if __name__ == "__main__":
    main()
