# Triton Framework

A minimal, extensible Triton-based training & inference scaffold. It includes:

- Custom Triton kernels (example: RMSNorm)
- PyTorch modules wrapping Triton kernels with fallbacks
- Simple training and inference engines
- Config-driven scripts and basic tests

## Install

From the repo root:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ./triton_framework
```

Ensure you have a working PyTorch with CUDA and Triton (PyTorch 2.x bundles Triton). Check GPU:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Quickstart

Train a tiny MLP on dummy data (GPU recommended):

```bash
python triton_framework/scripts/train.py
```

Run inference:

```bash
python triton_framework/scripts/infer.py
```

Benchmark the RMSNorm kernel against a PyTorch reference:

```bash
python triton_framework/scripts/bench_kernel.py
```

## Layout

- `src/triton_framework/kernels/` Triton kernels
- `src/triton_framework/modules/` PyTorch modules using the kernels
- `src/triton_framework/engine/` Training/inference utilities
- `src/triton_framework/data/` Example datasets
- `src/triton_framework/configs/` YAML configs
- `scripts/` Entry points
- `tests/` Unit tests

## Notes

- The code falls back to PyTorch ops when Triton/GPU is unavailable.
- Use this as a starting point and add your own kernels and modules.
