"""
NeuroGolf 2026 baseline generator.

Analyzes ARC-AGI tasks and synthesizes minimal ONNX networks.
"""

import json
import math
import os
import sys
from collections import Counter

import numpy as np

# Add the utils to path
UTILS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "neurogolf-2026", "neurogolf_utils"
)
sys.path.insert(0, UTILS_DIR)
import neurogolf_utils as nu

BATCH, CHANNELS, HEIGHT, WIDTH = 1, 10, 30, 30


def one_hot(grid):
    """Convert grid to one-hot [1, 10, 30, 30] tensor. Grid placed at top-left."""
    arr = np.zeros((1, CHANNELS, HEIGHT, WIDTH), dtype=np.float32)
    for r, row in enumerate(grid):
        if r >= HEIGHT:
            break
        for c, val in enumerate(row):
            if c >= WIDTH:
                break
            arr[0, int(val), r, c] = 1.0
    return arr


def from_one_hot(arr):
    """Convert [1, 10, H, W] back to grid list."""
    _, _, h, w = arr.shape
    grid = []
    for r in range(h):
        row = []
        for c in range(w):
            colors = [ch for ch in range(CHANNELS) if arr[0, ch, r, c] > 0.5]
            row.append(colors[0] if len(colors) == 1 else 10)
        grid.append(row)
    while grid and all(v == 10 for v in grid[-1]):
        grid.pop()
    while grid and not grid[-1]:
        grid.pop()
    return grid


def build_conv1x1(color_map):
    """
    Build a 1x1 Conv that maps input channels to output channels.
    color_map: dict of {in_channel: out_channel} for pixel-wise remapping.
    """
    w = np.zeros((10, 10, 1, 1), dtype=np.float32)
    for in_ch, out_ch in color_map.items():
        w[out_ch, in_ch, 0, 0] = 1.0

    inp = nu.onnx.helper.make_tensor_value_info("input", nu._DATA_TYPE, nu._GRID_SHAPE)
    out = nu.onnx.helper.make_tensor_value_info("output", nu._DATA_TYPE, nu._GRID_SHAPE)
    w_tensor = nu.onnx.helper.make_tensor(
        "W", nu._DATA_TYPE, list(w.shape), w.flatten().tolist()
    )
    node = nu.onnx.helper.make_node(
        "Conv", ["input", "W"], ["output"], kernel_shape=[1, 1], pads=[0, 0, 0, 0]
    )
    graph = nu.onnx.helper.make_graph([node], "graph", [inp], [out], [w_tensor])
    model = nu.onnx.helper.make_model(
        graph, ir_version=nu._IR_VERSION, opset_imports=nu._OPSET_IMPORTS
    )
    return model


def analyze_color_mapping(examples):
    """
    Check if task is a simple per-pixel color mapping.
    Returns color_map dict or None.
    """
    color_map = {}
    for ex in examples:
        inp = ex["input"]
        out = ex["output"]
        if len(inp) != len(out) or len(inp[0]) != len(out[0]):
            return None
        for r in range(len(inp)):
            for c in range(len(inp[0])):
                in_val = inp[r][c]
                out_val = out[r][c]
                if in_val not in color_map:
                    color_map[in_val] = out_val
                elif color_map[in_val] != out_val:
                    return None
    return color_map


def analyze_conv_pattern(examples, kernel_size=3):
    """
    Try to find a Conv2D kernel that maps input→output for all examples.
    Uses least squares to solve for kernel weights.
    Only works when input and output are the same size AND same across examples.
    """
    # Check all examples have same dimensions
    ref_in_h, ref_in_w = len(examples[0]["input"]), len(examples[0]["input"][0])
    ref_out_h, ref_out_w = len(examples[0]["output"]), len(examples[0]["output"][0])
    for ex in examples:
        if len(ex["input"]) != ref_in_h or len(ex["input"][0]) != ref_in_w:
            return None
        if len(ex["output"]) != ref_out_h or len(ex["output"][0]) != ref_out_w:
            return None
        if ref_in_h != ref_out_h or ref_in_w != ref_out_w:
            return None

    h, w = ref_in_h, ref_in_w
    pad = kernel_size // 2

    X_list, Y_list = [], []
    for ex in examples:
        inp_hot = one_hot(ex["input"])
        out_hot = one_hot(ex["output"])
        X_list.append(inp_hot)
        Y_list.append(out_hot)

    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)

    n = X.shape[0]
    X_padded = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")

    # Build design matrix: for each (batch, out_ch, row, col), gather input patches
    A_rows = []
    b_vals = []
    for b in range(n):
        for oc in range(CHANNELS):
            for r in range(pad, pad + h):
                for c in range(pad, pad + w):
                    patch = X_padded[
                        b, :, r - pad : r + pad + 1, c - pad : c + pad + 1
                    ].flatten()
                    A_rows.append(patch)
                    b_vals.append(Y[b, oc, r - pad, c - pad])

    A = np.array(A_rows)
    b = np.array(b_vals)

    # Solve via least squares
    weights, residuals, rank, singular = np.linalg.lstsq(A, b, rcond=None)

    # Check if solution is exact
    pred = A @ weights
    error = np.max(np.abs(pred - b))
    if error > 1e-4:
        return None

    # Reshape weights
    kernel = weights.reshape(CHANNELS, CHANNELS, kernel_size, kernel_size)
    return kernel


def build_conv2d(kernel):
    """Build a Conv2D ONNX model from a kernel tensor."""
    ks = kernel.shape[2]
    pad = ks // 2
    kernel_flat = kernel.flatten().tolist()

    inp = nu.onnx.helper.make_tensor_value_info("input", nu._DATA_TYPE, nu._GRID_SHAPE)
    out = nu.onnx.helper.make_tensor_value_info("output", nu._DATA_TYPE, nu._GRID_SHAPE)
    w_tensor = nu.onnx.helper.make_tensor(
        "W", nu._DATA_TYPE, list(kernel.shape), kernel_flat
    )
    node = nu.onnx.helper.make_node(
        "Conv", ["input", "W"], ["output"], kernel_shape=[ks, ks], pads=[pad] * 4
    )
    graph = nu.onnx.helper.make_graph([node], "graph", [inp], [out], [w_tensor])
    model = nu.onnx.helper.make_model(
        graph, ir_version=nu._IR_VERSION, opset_imports=nu._OPSET_IMPORTS
    )
    return model


def try_color_mapping(task, examples):
    """Try per-pixel color mapping as baseline."""
    all_examples = (
        examples.get("train", [])
        + examples.get("test", [])
        + examples.get("arc-gen", [])
    )
    color_map = analyze_color_mapping(all_examples)
    if color_map is None:
        return None
    return build_conv1x1(color_map)


def try_conv2d(task, examples, kernel_size=3):
    """Try to solve with a single Conv2D layer."""
    all_examples = (
        examples.get("train", [])
        + examples.get("test", [])
        + examples.get("arc-gen", [])
    )
    kernel = analyze_conv_pattern(all_examples, kernel_size=kernel_size)
    if kernel is None:
        return None
    return build_conv2d(kernel)


def generate(task_num):
    """Generate an ONNX network for the given task number."""
    examples = nu.load_examples(task_num)

    approaches = [
        ("color_map", try_color_mapping),
        ("conv3x3", lambda t, e: try_conv2d(t, e, 3)),
        ("conv5x5", lambda t, e: try_conv2d(t, e, 5)),
    ]

    for name, func in approaches:
        model = func(task_num, examples)
        if model is not None:
            print(f"  task{task_num:03d}: solved by {name}")
            return model

    print(f"  task{task_num:03d}: unsolved")
    return None
