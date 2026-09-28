from __future__ import annotations

import argparse
import importlib.metadata
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the local experiment runtime")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--min-vram-gb", type=float, default=16.0)
    args = parser.parse_args()
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is missing. Follow the platform-specific install command.") from exc

    torch_version = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
    if torch_version < (2, 1):
        raise SystemExit(
            f"PyTorch {torch.__version__} is too old for the pinned Transformers version; need >=2.1."
        )
    required = ("transformers", "datasets", "peft", "accelerate", "matplotlib")
    missing = []
    for package in required:
        try:
            print(f"{package}: {importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            missing.append(package)
    if missing:
        raise SystemExit(f"Missing packages: {', '.join(missing)}. Install requirements.txt.")

    cuda = torch.cuda.is_available()
    print(f"python: {sys.version.split()[0]}")
    print(f"torch: {torch.__version__}")
    print(f"CUDA available: {cuda}")
    if cuda:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            memory_gb = properties.total_memory / (1024 ** 3)
            print(f"GPU {index}: {properties.name}; VRAM {memory_gb:.1f} GiB")
            if memory_gb < args.min_vram_gb:
                print(
                    f"  Below the {args.min_vram_gb:.0f} GiB target for the full Qwen2.5-Math-7B "
                    "teacher plus three student runs. A 0.5B student-only smoke run may still fit."
                )
    elif args.require_cuda:
        raise SystemExit("No CUDA GPU is visible to PyTorch.")
    print("Environment check passed.")


if __name__ == "__main__":
    main()
