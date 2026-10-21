"""
check_gpu.py
------------
Checks whether PyTorch can use the NVIDIA GPU and writes the answer to
results/environment_check.txt.

    python src/check_gpu.py
"""

import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import RESULTS_DIR  # noqa: E402


def main():
    lines = [f"Python   : {platform.python_version()} ({sys.executable})",
             f"OS       : {platform.platform()}"]
    try:
        import torch
        import torchvision
    except ImportError as exc:
        lines.append(f"PyTorch not installed: {exc}")
        print("\n".join(lines))
        sys.exit(1)

    lines += [f"PyTorch  : {torch.__version__}",
              f"torchvision: {torchvision.__version__}",
              f"CUDA build: {torch.version.cuda}",
              f"CUDA available: {torch.cuda.is_available()}"]

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        lines += [f"GPU      : {torch.cuda.get_device_name(0)}",
                  f"Compute capability: {props.major}.{props.minor}",
                  f"GPU memory: {props.total_memory / 1024**3:.1f} GB",
                  f"Architectures in this PyTorch build: {', '.join(torch.cuda.get_arch_list())}"]
        # A real computation on the GPU proves that the kernels work for this card
        a = torch.randn(4096, 4096, device="cuda")
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            b = a @ a
        torch.cuda.synchronize()
        lines.append(f"GPU test : 10 matrix multiplications (4096x4096) in {time.time() - start:.3f} s -> OK "
                     f"(checksum {float(b[0, 0]):.2f})")
        lines.append("RESULT   : training will use the GPU")
    else:
        lines.append("RESULT   : no usable GPU found - training will run on the CPU (much slower)")

    text = "\n".join(lines)
    print(text)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "environment_check.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
