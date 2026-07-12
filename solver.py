"""
NeuroGolf 2026 Solver: Analytical solvers + Conv2d one-hot learned solver.
Based on the top-scoring approach: try near-zero-cost analytical solvers first,
then fall back to least-squares Conv2d fitting with ArgMax→OneHot pipeline.
"""

import json
import math
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

# ─── Constants ────────────────────────────────────────────────────────────────
BATCH, CH, GH, GW = 1, 10, 30, 30
GRID_SHAPE = [BATCH, CH, GH, GW]


def to_onehot(grid):
    """Convert integer grid [H,W] to one-hot [1,10,30,30] float32, zero-padded."""
    oh = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            oh[0, grid[r][c], r, c] = 1.0
    return oh


def to_onehot_np(arr):
    """Convert integer numpy array [H,W] to one-hot [1,10,30,30] float32, zero-padded."""
    oh = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for c in range(10):
        oh[0, c, : arr.shape[0], : arr.shape[1]] = (arr == c).astype(np.float32)
    return oh


def from_onehot(oh):
    """Convert one-hot [1,10,H,W] to integer grid [H,W]."""
    return np.argmax(oh[0], axis=0)


def load_task(path):
    with open(path) as f:
        return json.load(f)


def make_model(nodes, inits=None):
    """Build a minimal ONNX model from nodes and initializers."""
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, GRID_SHAPE)
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, GRID_SHAPE)
    g = helper.make_graph(nodes, "g", [x], [y], initializer=inits or [])
    return helper.make_model(
        g, ir_version=10, opset_imports=[helper.make_opsetid("", 10)]
    )


def save_model(model, path):
    onnx.save(model, path)


def validate_model(path, task_data):
    """Validate model against all train+test examples."""
    try:
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    except Exception:
        return False
    for ex in task_data["train"] + task_data.get("test", []):
        inp = to_onehot(ex["input"])
        exp = to_onehot(ex["output"])
        out = sess.run(["output"], {"input": inp})[0]
        out = (out > 0.0).astype(np.float32)
        if not np.array_equal(out, exp):
            return False
    return True


# ─── Helper: ONNX node builder ────────────────────────────────────────────────
class ONNXBuilder:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self._counter = 0

    def _uid(self, prefix="n"):
        self._counter += 1
        return f"{prefix}_{self._counter}"

    def C(self, name, val):
        arr = np.array(val, dtype=np.float32)
        self.inits.append(numpy_helper.from_array(arr, name=name))
        return name

    def CI(self, name, val):
        arr = np.array(val, dtype=np.int64)
        self.inits.append(numpy_helper.from_array(arr, name=name))
        return name

    def n(self, op, inputs, outputs, **kw):
        self.nodes.append(helper.make_node(op, inputs, outputs, **kw))

    def build(self):
        return make_model(self.nodes, self.inits)


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYTICAL SOLVERS
# ═══════════════════════════════════════════════════════════════════════════════


def solve_identity(task_data):
    """Identity: input == output."""
    for ex in task_data["train"]:
        if ex["input"] != ex["output"]:
            return None
    b = ONNXBuilder()
    b.n("Identity", ["input"], ["output"])
    return b.build()


def solve_constant(task_data):
    """Constant: all outputs are the same grid."""
    grids = [ex["output"] for ex in task_data["train"]]
    if len(set(str(g) for g in grids)) != 1:
        return None
    const_grid = np.array(grids[0], dtype=np.float32)
    b = ONNXBuilder()
    b.inits.append(numpy_helper.from_array(const_grid, name="const_grid"))
    b.n("Identity", ["const_grid"], ["output"])
    return b.build()


def solve_color_map(task_data):
    """Color map: per-pixel color permutation. Check consistency across examples."""

    def get_mapping(inp, out):
        m = {}
        for r in range(len(inp)):
            for c in range(len(inp[0])):
                ic, oc = inp[r][c], out[r][c]
                if ic in m:
                    if m[ic] != oc:
                        return None
                else:
                    m[ic] = oc
        return m

    maps = []
    for ex in task_data["train"]:
        m = get_mapping(ex["input"], ex["output"])
        if m is None:
            return None
        maps.append(m)

    # Check all maps are identical
    first = maps[0]
    for m in maps[1:]:
        if m != first:
            return None

    # Build 10x10 permutation matrix
    W = np.zeros((10, 10, 1, 1), dtype=np.float32)
    for src, dst in first.items():
        W[dst, src, 0, 0] = 1.0

    b = ONNXBuilder()
    b.inits.append(numpy_helper.from_array(W, name="W"))
    b.n("Conv", ["input", "W"], ["output"], name="conv", kernel_shape=[1, 1])
    return b.build()


def solve_transpose(task_data):
    """Transpose: output == input.T (swap rows/cols)."""
    for ex in task_data["train"]:
        inp = np.array(ex["input"])
        out = np.array(ex["output"])
        if inp.shape != out.shape:
            return None
        if not np.array_equal(inp.T, out):
            return None
    b = ONNXBuilder()
    b.n("Transpose", ["input"], ["output"], perm=[0, 1, 3, 2])
    return b.build()


def solve_flip(task_data):
    """Flip: output == flipud(input) or fliplr(input)."""
    for flip_type in ["ud", "lr"]:
        ok = True
        for ex in task_data["train"]:
            inp = np.array(ex["input"])
            out = np.array(ex["output"])
            if flip_type == "ud":
                expected = inp[::-1, :]
            else:
                expected = inp[:, ::-1]
            if not np.array_equal(expected, out):
                ok = False
                break
        if ok:
            b = ONNXBuilder()
            if flip_type == "ud":
                idx = np.arange(29, -1, -1, dtype=np.int64)
                b.inits.append(numpy_helper.from_array(idx, name="flip_idx"))
                b.n("Gather", ["input", "flip_idx"], ["output"], name="flip", axis=2)
            else:
                idx = np.arange(29, -1, -1, dtype=np.int64)
                b.inits.append(numpy_helper.from_array(idx, name="flip_idx"))
                b.n("Gather", ["input", "flip_idx"], ["output"], name="flip", axis=3)
            return b.build()
    return None


def solve_rotate(task_data):
    """Rotate: output == rot90(input, k) for k in {1,2,3}."""
    for k in [1, 2, 3]:
        ok = True
        for ex in task_data["train"]:
            inp = np.array(ex["input"])
            out = np.array(ex["output"])
            expected = np.rot90(inp, k, axes=(0, 1))
            # For 3D (channels, H, W), rot90 on axes=(1,2)
            expected_3d = np.rot90(inp, k, axes=(1, 2))
            if not np.array_equal(expected_3d, out):
                ok = False
                break
        if ok:
            b = ONNXBuilder()
            if k == 1:
                # Transpose axes 2,3 then flip axis 3
                b.n("Transpose", ["input"], ["t"], perm=[0, 1, 3, 2])
                idx = np.arange(29, -1, -1, dtype=np.int64)
                b.inits.append(numpy_helper.from_array(idx, name="flip_idx"))
                b.n("Gather", ["t", "flip_idx"], ["output"], name="flip", axis=3)
            elif k == 2:
                idx29 = np.arange(29, -1, -1, dtype=np.int64)
                b.inits.append(numpy_helper.from_array(idx29, name="flip_r"))
                b.inits.append(numpy_helper.from_array(idx29, name="flip_c"))
                b.n("Gather", ["input", "flip_r"], ["tmp"], name="fr", axis=2)
                b.n("Gather", ["tmp", "flip_c"], ["output"], name="fc", axis=3)
            elif k == 3:
                b.n("Transpose", ["input"], ["t"], perm=[0, 1, 3, 2])
                idx = np.arange(29, -1, -1, dtype=np.int64)
                b.inits.append(numpy_helper.from_array(idx, name="flip_idx"))
                b.n("Gather", ["t", "flip_idx"], ["output"], name="flip", axis=2)
            return b.build()
    return None


def solve_tile(task_data):
    """Tile: output == np.tile(input, (rH, rW)) truncated to 30x30."""
    # Find the tiling pattern from first example
    ex = task_data["train"][0]
    inp = np.array(ex["input"])
    out = np.array(ex["output"])
    H, W = inp.shape
    OH, OW = out.shape

    if OH % H != 0 or OW % W != 0:
        return None

    rH, rW = OH // H, OW // W
    tiled = np.tile(inp, (rH, rW))
    if not np.array_equal(tiled, out):
        return None

    # Verify on other examples
    for ex in task_data["train"][1:]:
        inp = np.array(ex["input"])
        out = np.array(ex["output"])
        tiled = np.tile(inp, (rH, rW))[: out.shape[0], : out.shape[1]]
        if not np.array_equal(tiled, out):
            return None

    b = ONNXBuilder()
    b.n(
        "Slice",
        [
            "input",
            b.CI("starts", [0, 0, 0, 0]),
            b.CI("ends", [1, 10, H, W]),
            b.CI("axes", [0, 1, 2, 3]),
        ],
        ["grid"],
    )
    b.n("Tile", ["grid", b.CI("tile_reps", [1, 1, rH, rW])], ["tiled"])

    # Pad to 30x30 if needed
    if OH < 30 or OW < 30:
        b.n(
            "Pad",
            ["tiled", b.CI("pads", [0, 0, 0, 0, 0, 0, 30 - OH, 30 - OW])],
            ["output"],
            name="pad",
        )
    else:
        b.n("Identity", ["tiled"], ["output"])
    return b.build()


def solve_upscale(task_data):
    """Upscale: output == np.repeat(np.repeat(input, sH, 0), sW, 1)."""
    ex = task_data["train"][0]
    inp = np.array(ex["input"])
    out = np.array(ex["output"])
    H, W = inp.shape
    OH, OW = out.shape

    if OH % H != 0 or OW % W != 0:
        return None
    sH, sW = OH // H, OW // W

    upscaled = np.repeat(np.repeat(inp, sH, axis=0), sW, axis=1)
    if not np.array_equal(upscaled, out):
        return None

    b = ONNXBuilder()
    # Build GatherElements index map
    flat_idx = np.zeros((1, 10, OH * OW), dtype=np.int64)
    for r in range(OH):
        for c in range(OW):
            flat_idx[0, :, r * OW + c] = (r // sH) * W + (c // sW)
    b.inits.append(numpy_helper.from_array(flat_idx, name="flat_idx"))

    b.n("Reshape", ["input", b.CI("ish", [1, 10, H * W])], ["inp_flat"])
    b.n("GatherElements", ["inp_flat", "flat_idx"], ["out_flat"], axis=2)
    b.n("Reshape", ["out_flat", b.CI("osh", [1, 10, OH, OW])], ["output"])
    return b.build()


def solve_spatial_gather(task_data):
    """Spatial gather: fixed pixel remapping from input to output."""
    # Build mapping from first example
    ex = task_data["train"][0]
    inp = np.array(ex["input"])
    out = np.array(ex["output"])
    IH, IW = inp.shape
    OH, OW = out.shape

    # For each output pixel, find source in input
    src_map = np.full((OH, OW, 2), -1, dtype=np.int64)
    const_map = np.full((OH, OW), -1, dtype=np.int64)

    for r in range(OH):
        for c in range(OW):
            oval = out[r, c]
            # Check if any input pixel matches
            found = False
            for ir in range(IH):
                for ic in range(IW):
                    if inp[ir, ic] == oval and src_map[r, c, 0] == -1:
                        src_map[r, c] = [ir, ic]
                        found = True
                        break
                if found:
                    break
            if not found:
                const_map[r, c] = oval

    # Verify on other examples
    for ex in task_data["train"][1:]:
        inp = np.array(ex["input"])
        out = np.array(ex["output"])
        for r in range(OH):
            for c in range(OW):
                if const_map[r, c] >= 0:
                    if out[r, c] != const_map[r, c]:
                        return None
                else:
                    sr, sc = src_map[r, c]
                    if sr >= inp.shape[0] or sc >= inp.shape[1]:
                        return None
                    if inp[sr, sc] != out[r, c]:
                        return None

    # Build ONNX model
    b = ONNXBuilder()
    flat_idx = np.zeros((1, 10, OH * OW), dtype=np.int64)
    for r in range(OH):
        for c in range(OW):
            if const_map[r, c] < 0:
                flat_idx[0, :, r * OW + c] = src_map[r, c, 0] * IW + src_map[r, c, 1]
            else:
                flat_idx[0, :, r * OW + c] = 0  # will be masked out
    b.inits.append(numpy_helper.from_array(flat_idx, name="flat_idx"))

    b.n("Reshape", ["input", CI_shape(b, [1, 10, IH * IW])], ["inp_flat"])
    b.n("GatherElements", ["inp_flat", "flat_idx"], ["out_flat"], axis=2)
    b.n("Reshape", ["out_flat", CI_shape(b, [1, 10, OH, OW])], ["gathered"])

    # Build mask for gather vs constant
    mask = np.zeros((1, 1, OH, OW), dtype=np.float32)
    const_oh = np.zeros((1, 10, OH, OW), dtype=np.float32)
    for r in range(OH):
        for c in range(OW):
            if const_map[r, c] >= 0:
                const_oh[0, const_map[r, c], r, c] = 1.0
            else:
                mask[0, 0, r, c] = 1.0
    b.inits.append(numpy_helper.from_array(mask, name="mask"))
    b.inits.append(numpy_helper.from_array(const_oh, name="const_oh"))

    b.n("Mul", ["gathered", "mask"], ["masked"])
    b.n("Add", ["masked", "const_oh"], ["padded"])

    # Pad to 30x30
    if OH < 30 or OW < 30:
        pads = [0, 0, 0, 0, 0, 0, 30 - OH, 30 - OW]
        b.CI("pads", pads)
        b.n("Pad", ["padded", "pads"], ["output"])
    else:
        b.n("Identity", ["padded"], ["output"])
    return b.build()


def solve_crop(task_data):
    """Crop: output is a centered crop of input."""
    ex = task_data["train"][0]
    inp = np.array(ex["input"])
    out = np.array(ex["output"])
    IH, IW = inp.shape
    OH, OW = out.shape

    if OH > IH or OW > IW:
        return None

    dr = (IH - OH) // 2
    dc = (IW - OW) // 2
    cropped = inp[dr : dr + OH, dc : dc + OW]
    if not np.array_equal(cropped, out):
        return None

    # Verify on others
    for ex in task_data["train"][1:]:
        inp = np.array(ex["input"])
        out = np.array(ex["output"])
        cropped = inp[dr : dr + OH, dc : dc + OW]
        if not np.array_equal(cropped, out):
            return None

    b = ONNXBuilder()
    b.CI("starts", [0, 0, dr, dc])
    b.CI("ends", [1, 10, dr + OH, dc + OW])
    b.CI("axes", [0, 1, 2, 3])
    b.n("Slice", ["input", "starts", "ends", "axes"], ["cropped"])

    if OH < 30 or OW < 30:
        pads = [0, 0, 0, 0, 0, 0, 30 - OH, 30 - OW]
        b.CI("pads", pads)
        b.n("Pad", ["cropped", "pads"], ["output"])
    else:
        b.n("Identity", ["cropped"], ["output"])
    return b.build()


def CI_shape(b, shape):
    """Create a shape initializer and return its name."""
    name = f"shape_{b._counter}"
    b.CI(name, shape)
    return name


# ═══════════════════════════════════════════════════════════════════════════════
# CONV2D ONE-HOT SOLVER (least-squares fitting)
# ═══════════════════════════════════════════════════════════════════════════════


def lstsq_conv_weights(exs, ks, use_bias=False):
    """
    Fit a Conv2d kernel via least-squares on one-hot patches.
    Returns (W_conv, verified) where W_conv is [10, 10, ks, ks] float32.
    """
    pad = ks // 2
    feat_dim = 10 * ks * ks + (1 if use_bias else 0)

    patches = []
    targets = []

    for inp_grid, out_grid in exs:
        H, W = inp_grid.shape
        oh = np.zeros((10, H, W), dtype=np.float64)
        for c in range(10):
            oh[c] = (inp_grid == c).astype(np.float64)

        oh_pad = np.pad(oh, ((0, 0), (pad, pad), (pad, pad)))

        for r in range(H):
            for c in range(W):
                p = oh_pad[:, r : r + ks, c : c + ks].flatten()
                if use_bias:
                    p = np.append(p, 1.0)
                patches.append(p)
                targets.append(int(out_grid[r, c]))

    P = np.array(patches, dtype=np.float64)
    T = np.array(targets, dtype=np.int64)

    # One-hot encode targets
    T_oh = np.zeros((len(T), 10), dtype=np.float64)
    for i, t in enumerate(T):
        T_oh[i, t] = 1.0

    # Solve least squares
    WT, _, _, _ = np.linalg.lstsq(P, T_oh, rcond=None)

    # Verify
    pred = np.argmax(P @ WT, axis=1)
    verified = np.array_equal(pred, T)

    if not verified:
        return None, False

    Wconv = WT.T.reshape(10, feat_dim - (1 if use_bias else 0)).astype(np.float32)
    # Reshape to [10, 10, ks, ks] (no bias)
    Wconv = Wconv.reshape(10, 10, ks, ks).astype(np.float32)

    return Wconv, True


def solve_conv_fixed(task_data, ks, use_bias=False, timeout_s=30.0):
    """Fixed shape: Slice to actual grid, Conv, ArgMax, OneHot, Pad back."""
    t0 = time.time()

    # Analyze shapes
    shapes = set()
    for ex in task_data["train"]:
        inp = np.array(ex["input"])
        out = np.array(ex["output"])
        shapes.add((inp.shape[0], inp.shape[1], out.shape[0], out.shape[1]))

    if len(shapes) != 1:
        return None
    IH, IW, OH, OW = shapes.pop()

    if IH != OH or IW != OW:
        return None

    exs = [(np.array(ex["input"]), np.array(ex["output"])) for ex in task_data["train"]]
    W, ok = lstsq_conv_weights(exs, ks, use_bias)
    if not ok:
        return None

    if time.time() - t0 > timeout_s:
        return None

    pad = ks // 2
    b = ONNXBuilder()

    # Slice to actual grid size
    b.CI("starts", [0, 0, 0, 0])
    b.CI("ends", [1, 10, IH, IW])
    b.CI("axes", [0, 1, 2, 3])
    b.n("Slice", ["input", "starts", "ends", "axes"], ["grid"])

    # Conv
    b.inits.append(numpy_helper.from_array(W, name="W"))
    pads = [pad, pad, pad, pad]
    b.CI("conv_pads", pads)
    b.n("Conv", ["grid", "W"], ["co"], name="conv", pads=pads, kernel_shape=[ks, ks])

    # ArgMax -> OneHot
    b.n("ArgMax", ["co"], ["am"], name="am", axis=1, keepdims=0)
    b.n(
        "OneHot",
        ["am", CI_shape(b, 10), C_val(b, "oh_vals", [0.0, 1.0])],
        ["oh"],
        axis=1,
    )

    # Pad back to 30x30
    if OH < 30 or OW < 30:
        pads = [0, 0, 0, 0, 0, 0, 30 - OH, 30 - OW]
        b.CI("out_pads", pads)
        b.n("Pad", ["oh", "out_pads"], ["output"])
    else:
        b.n("Identity", ["oh"], ["output"])

    return b.build()


def C_val(b, name, val):
    arr = np.array(val, dtype=np.float32)
    b.inits.append(numpy_helper.from_array(arr, name=name))
    return name


def solve_conv_variable(task_data, ks, use_bias=False, timeout_s=30.0):
    """Variable shape: Conv on full 30x30 with mask for zero-padded regions."""
    t0 = time.time()

    exs = [(np.array(ex["input"]), np.array(ex["output"])) for ex in task_data["train"]]

    # Check all same shape
    shapes = set()
    for inp, out in exs:
        shapes.add(inp.shape)
    if len(shapes) != 1:
        return None

    IH, IW = shapes.pop()

    W, ok = lstsq_conv_weights(exs, ks, use_bias)
    if not ok:
        return None

    if time.time() - t0 > timeout_s:
        return None

    pad = ks // 2
    b = ONNXBuilder()

    # Compute mask: 1.0 where input has real data
    b.n("ReduceSum", ["input"], ["mask_raw"], name="mask_raw", axes=[1], keepdims=1)
    b.n("Cast", ["mask_raw"], ["mask"], name="mcast", to=TensorProto.FLOAT)

    # Conv
    b.inits.append(numpy_helper.from_array(W, name="W"))
    pads = [pad, pad, pad, pad]
    b.CI("conv_pads", pads)
    b.n("Conv", ["input", "W"], ["co"], name="conv", pads=pads, kernel_shape=[ks, ks])

    # ArgMax -> OneHot
    b.n("ArgMax", ["co"], ["am"], name="am", axis=1, keepdims=0)
    b.n(
        "OneHot",
        ["am", CI_shape(b, 10), C_val(b, "oh_vals", [0.0, 1.0])],
        ["oh"],
        axis=1,
    )

    # Mask out padded regions
    b.n("Mul", ["oh", "mask"], ["output"])

    return b.build()


def solve_conv_diffshape(task_data, ks, use_bias=False, timeout_s=30.0):
    """Different shape: Slice input -> Conv -> Slice crop output."""
    t0 = time.time()

    exs = [(np.array(ex["input"]), np.array(ex["output"])) for ex in task_data["train"]]

    # Check output always smaller than input
    for inp, out in exs:
        if out.shape[0] > inp.shape[0] or out.shape[1] > inp.shape[1]:
            return None

    # Find consistent crop offset
    shapes = set()
    for inp, out in exs:
        shapes.add((inp.shape[0], inp.shape[1], out.shape[0], out.shape[1]))
    if len(shapes) != 1:
        return None
    IH, IW, OH, OW = shapes.pop()

    # Try top-left and centered offsets
    for dr_off, dc_off in [(0, 0), ((IH - OH) // 2, (IW - OW) // 2)]:
        # Verify offset works on all examples
        ok = True
        for inp, out in exs:
            cropped = inp[dr_off : dr_off + OH, dc_off : dc_off + OW]
            if not np.array_equal(cropped, out):
                ok = False
                break
        if ok:
            # This is actually a crop, not a conv
            return None  # handled by analytical crop solver

    W, ok = lstsq_conv_weights(exs, ks, use_bias)
    if not ok:
        return None

    if time.time() - t0 > timeout_s:
        return None

    pad = ks // 2
    b = ONNXBuilder()

    # Slice input
    b.CI("in_starts", [0, 0, 0, 0])
    b.CI("in_ends", [1, 10, IH, IW])
    b.CI("in_axes", [0, 1, 2, 3])
    b.n("Slice", ["input", "in_starts", "in_ends", "in_axes"], ["grid"])

    # Conv
    b.inits.append(numpy_helper.from_array(W, name="W"))
    pads = [pad, pad, pad, pad]
    b.CI("conv_pads", pads)
    b.n("Conv", ["grid", "W"], ["co"], name="conv", pads=pads, kernel_shape=[ks, ks])

    # ArgMax -> OneHot
    b.n("ArgMax", ["co"], ["am"], name="am", axis=1, keepdims=0)
    b.n(
        "OneHot",
        ["am", CI_shape(b, 10), C_val(b, "oh_vals", [0.0, 1.0])],
        ["oh"],
        axis=1,
    )

    # Crop output to match target size
    # Find where the actual output content is in the conv output
    # Use the same offset as the output placement
    b.CI("out_starts", [0, 0, 0, 0])
    b.CI("out_ends", [1, 10, OH, OW])
    b.CI("out_axes", [0, 1, 2, 3])
    b.n("Slice", ["oh", "out_starts", "out_ends", "out_axes"], ["cropped"])

    # Pad to 30x30
    if OH < 30 or OW < 30:
        pads = [0, 0, 0, 0, 0, 0, 30 - OH, 30 - OW]
        b.CI("out_pads", pads)
        b.n("Pad", ["cropped", "out_pads"], ["output"])
    else:
        b.n("Identity", ["cropped"], ["output"])

    return b.build()


def solve_conv_upscale(task_data, ks, use_bias=False, timeout_s=30.0):
    """Upscale: Conv on input padded to output size, then pad to 30x30."""
    t0 = time.time()

    exs = [(np.array(ex["input"]), np.array(ex["output"])) for ex in task_data["train"]]

    # Check output always >= input
    for inp, out in exs:
        if out.shape[0] < inp.shape[0] or out.shape[1] < inp.shape[1]:
            return None

    shapes = set()
    for inp, out in exs:
        shapes.add((inp.shape[0], inp.shape[1], out.shape[0], out.shape[1]))
    if len(shapes) != 1:
        return None
    IH, IW, OH, OW = shapes.pop()

    # Fit conv: pad input to output size, then fit
    pad_h, pad_w = OH - IH, OW - IW
    padded_exs = []
    for inp, out in exs:
        padded_inp = np.pad(inp, ((0, pad_h), (0, pad_w)))
        padded_exs.append((padded_inp, out))

    W, ok = lstsq_conv_weights(padded_exs, ks, use_bias)
    if not ok:
        return None

    if time.time() - t0 > timeout_s:
        return None

    pad = ks // 2
    b = ONNXBuilder()

    # Pad input to output size first
    if pad_h > 0 or pad_w > 0:
        b.n(
            "Pad",
            ["input", b.CI("in_pads", [0, 0, 0, 0, 0, 0, pad_h, pad_w])],
            ["padded"],
        )

    # Conv on padded input (now same size as output)
    b.inits.append(numpy_helper.from_array(W, name="W"))
    conv_pads = [pad, pad, pad, pad]
    b.CI("conv_pads", conv_pads)
    b.n(
        "Conv",
        ["padded", "W"],
        ["co"],
        name="conv",
        pads=conv_pads,
        kernel_shape=[ks, ks],
    )

    # ArgMax -> OneHot
    b.n("ArgMax", ["co"], ["am"], name="am", axis=1, keepdims=0)
    b.n(
        "OneHot",
        ["am", CI_shape(b, 10), C_val(b, "oh_vals", [0.0, 1.0])],
        ["oh"],
        axis=1,
    )

    # Pad to 30x30
    if OH < 30 or OW < 30:
        b.n(
            "Pad",
            ["oh", b.CI("out_pads", [0, 0, 0, 0, 0, 0, 30 - OH, 30 - OW])],
            ["output"],
        )
    else:
        b.n("Identity", ["oh"], ["output"])

    return b.build()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SOLVER PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

ANALYTICAL_SOLVERS = [
    ("identity", solve_identity),
    ("constant", solve_constant),
    ("color_map", solve_color_map),
    ("transpose", solve_transpose),
    ("flip", solve_flip),
    ("rotate", solve_rotate),
    ("tile", solve_tile),
    ("upscale", solve_upscale),
    ("spatial_gather", solve_spatial_gather),
    ("crop", solve_crop),
]

KERNEL_SIZES = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]


def solve_task(task_num, task_data, outdir, conv_budget=30.0):
    """Solve a single task. Returns (solver_name, model_path) or (None, None)."""
    t0 = time.time()

    # Try analytical solvers first
    for name, solver in ANALYTICAL_SOLVERS:
        try:
            model = solver(task_data)
            if model is not None:
                path = os.path.join(outdir, f"task{task_num:03d}.onnx")
                save_model(model, path)
                if validate_model(path, task_data):
                    return name, path
        except Exception:
            pass

    # Determine shape category
    shapes = set()
    for ex in task_data["train"]:
        inp = np.array(ex["input"])
        out = np.array(ex["output"])
        shapes.add((inp.shape[0], inp.shape[1], out.shape[0], out.shape[1]))

    same_shape = all(ih == oh and iw == ow for ih, iw, oh, ow in shapes)
    fixed_in = len(set((ih, iw) for ih, iw, oh, ow in shapes)) == 1
    diff_shape = all(oh <= ih and ow <= iw for ih, iw, oh, ow in shapes)
    upscale_shape = all(oh >= ih and ow >= iw for ih, iw, oh, ow in shapes)

    # Try Conv solvers
    remaining = conv_budget - (time.time() - t0)

    for ks in KERNEL_SIZES:
        if time.time() - t0 > conv_budget:
            break

        feat = 10 * ks * ks
        if feat > 20000:
            break

        for use_bias in [False, True]:
            if time.time() - t0 > conv_budget:
                break

            try:
                if same_shape and fixed_in:
                    model = solve_conv_fixed(task_data, ks, use_bias, remaining)
                elif same_shape:
                    model = solve_conv_variable(task_data, ks, use_bias, remaining)
                elif diff_shape:
                    model = solve_conv_diffshape(task_data, ks, use_bias, remaining)
                elif upscale_shape:
                    model = solve_conv_upscale(task_data, ks, use_bias, remaining)
                else:
                    continue

                if model is not None:
                    path = os.path.join(outdir, f"task{task_num:03d}.onnx")
                    save_model(model, path)
                    if validate_model(path, task_data):
                        return f"conv_ks{ks}_{'bias' if use_bias else 'nobias'}", path
            except Exception:
                pass

    return None, None


def generate_submission(outdir, task_dir, submission_path="submission.zip"):
    """Generate submission.zip from all solved models."""
    with zipfile.ZipFile(submission_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for task_file in sorted(Path(task_dir).glob("task*.json")):
            task_num = int(task_file.stem.replace("task", ""))
            model_path = os.path.join(outdir, f"task{task_num:03d}.onnx")
            if os.path.exists(model_path):
                zf.write(model_path, f"task{task_num:03d}.onnx")
    print(f"Generated {submission_path}")


def main():
    task_dir = "neurogolf-2026"
    outdir = "fixed"
    os.makedirs(outdir, exist_ok=True)

    results = {}
    total_score = 0
    solved = 0
    unsolved = []

    task_files = sorted(Path(task_dir).glob("task*.json"))
    print(f"Processing {len(task_files)} tasks...")

    for i, task_file in enumerate(task_files):
        task_num = int(task_file.stem.replace("task", ""))
        task_data = load_task(task_file)

        t0 = time.time()
        solver_name, path = solve_task(task_num, task_data, outdir, conv_budget=30.0)
        elapsed = time.time() - t0

        if solver_name:
            # Calculate cost
            try:
                sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
                inp = to_onehot(task_data["train"][0]["input"])
                out = sess.run(["output"], {"input": inp})[0]
                params = sum(np.prod(init.dims) for init in sess.get_inputs())
                cost = params  # simplified cost
                score = max(1.0, 25.0 - math.log(max(1.0, cost)))
            except Exception:
                score = 0

            results[task_num] = (solver_name, score, elapsed)
            total_score += score
            solved += 1
            print(
                f"[{i + 1}/{len(task_files)}] task{task_num:03d}: {solver_name} ({score:.2f} pts, {elapsed:.1f}s)"
            )
        else:
            unsolved.append(task_num)
            results[task_num] = (None, 0, elapsed)
            print(
                f"[{i + 1}/{len(task_files)}] task{task_num:03d}: UNSOLVED ({elapsed:.1f}s)"
            )

    print(f"\n{'=' * 60}")
    print(f"Solved: {solved}/{len(task_files)}")
    print(f"Total score: {total_score:.2f}")
    print(f"Unsolved: {len(unsolved)} tasks")
    if unsolved:
        print(f"Unsolved tasks: {unsolved[:20]}{'...' if len(unsolved) > 20 else ''}")

    # Generate submission
    generate_submission(outdir, task_dir)


if __name__ == "__main__":
    main()
