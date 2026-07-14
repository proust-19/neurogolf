# NeuroGolf 2026

Kaggle competition: build ONNX models that solve visual pattern tasks with minimal cost.

## Approach

**Two-stage pipeline:**
1. **Analytical solvers** (identity, color map, transpose, flip, rotate, tile, upscale, crop, spatial gather) - zero-cost pattern matching
2. **Conv2d least-squares solver** - fits convolutional kernels to learn pixel-to-pixel mappings when analytical solvers fail

When no exact solver works, falls back to `Conv2d` weight fitting via least-squares on one-hot encoded grids, with `ArgMax → OneHot` output pipeline.

## Score

**7,267.99 / 10,000** on Kaggle leaderboard (top 30%)

## Key Files

| File | Description |
|------|-------------|
| `solver.py` | Main solver pipeline (analytical + conv2d) |
| `neurogolf-2026/` | Task JSON files |
| `*.ipynb` | Kaggle notebook with ONNX graph surgery pipeline |

## ONNX Graph Surgery

Notebook applies stacked optimization passes to reduce model cost:
- onnxoptimizer + onnxsim simplification
- Unused tensor pruning, weight deduplication
- Index/metadata compression
- Broadcast compression
- FP16 conversion

## Author

Purshotam Kumar - [proust-19](https://github.com/proust-19)
