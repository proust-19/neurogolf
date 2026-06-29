"""
Optimize existing ONNX models for NeuroGolf 2026.

Strategies:
1. Remove unused initializers (dummy_cost_scalar)
2. Dilation optimization: increase dilation + shrink kernel for sparse Conv
3. Batch optimize all models and build submission zip
"""

import copy
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
import onnxruntime as ort

# Add neurogolf_utils
UTILS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "neurogolf-2026", "neurogolf_utils"
)
sys.path.insert(0, UTILS_DIR)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "neurogolf-2026"))
import neurogolf_utils as nu

nu._NEUROGOLF_DIR = (
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "neurogolf-2026"))
    + "/"
)

BATCH, CHANNELS, HEIGHT, WIDTH = 1, 10, 30, 30
GRID_SHAPE = [BATCH, CHANNELS, HEIGHT, WIDTH]


def remove_unused_initializers(model):
    """Remove initializers not referenced by any node (e.g., dummy_cost_scalar)."""
    used_names = set()
    for n in model.graph.node:
        for inp in n.input:
            if inp:
                used_names.add(inp)
    # Also check functions
    for fn in model.functions:
        for n in fn.node:
            for inp in n.input:
                if inp:
                    used_names.add(inp)

    removed = []
    new_init = []
    for init in model.graph.initializer:
        if init.name in used_names:
            new_init.append(init)
        else:
            removed.append(init.name)
    model.graph.ClearField("initializer")
    model.graph.initializer.extend(new_init)
    return model, removed


def get_weight_analysis(model):
    """
    Find Conv nodes with sparse weights that can be optimized via dilation.
    Returns list of transformations.
    """
    init_map = {i.name: i for i in model.graph.initializer}
    input_ref_count = Counter(inp for n in model.graph.node for inp in n.input if inp)
    # Also check function nodes
    for fn in model.functions:
        for n in fn.node:
            for inp in n.input:
                if inp:
                    input_ref_count[inp] += 0  # don't add, just checking

    def divisors(n):
        if n <= 0:
            return [1]
        res = set()
        for i in range(1, int(math.sqrt(n)) + 1):
            if n % i == 0:
                res.add(i)
                res.add(n // i)
        return sorted(res)

    transformations = []

    for node_idx, node in enumerate(model.graph.node):
        if node.op_type != "Conv":
            continue
        if len(node.input) < 2:
            continue

        w_name = node.input[1]
        if w_name not in init_map:
            continue
        if input_ref_count.get(w_name, 0) != 1:
            continue

        weight = numpy_helper.to_array(init_map[w_name])
        if weight.ndim != 4:
            continue

        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        auto_pad = attrs.get("auto_pad", b"NOTSET")
        if isinstance(auto_pad, bytes):
            auto_pad = auto_pad.decode()
        if auto_pad not in ("NOTSET", ""):
            continue

        old_dils = list(attrs.get("dilations", [1, 1]))
        old_kh, old_kw = int(weight.shape[2]), int(weight.shape[3])

        if old_kh <= 1 and old_kw <= 1:
            continue

        nonzero = np.argwhere(weight != 0)
        if len(nonzero) == 0:
            continue

        spatial_positions = {(int(idx[2]), int(idx[3])) for idx in nonzero}
        span_h = (old_kh - 1) * old_dils[0]
        span_w = (old_kw - 1) * old_dils[1]

        best = None
        cand_dils_h = divisors(span_h) if span_h > 0 else [old_dils[0]]
        cand_dils_w = divisors(span_w) if span_w > 0 else [old_dils[1]]
        old_param_count = int(weight.size)

        for ndh in cand_dils_h:
            if span_h > 0 and span_h % ndh != 0:
                continue
            nkh = span_h // ndh + 1 if span_h > 0 else 1

            for ndw in cand_dils_w:
                if span_w > 0 and span_w % ndw != 0:
                    continue
                nkw = span_w // ndw + 1 if span_w > 0 else 1

                new_pc = int(weight.shape[0]) * int(weight.shape[1]) * nkh * nkw
                if new_pc >= old_param_count:
                    continue

                valid = True
                for oy, ox in spatial_positions:
                    ay = oy * old_dils[0]
                    ax = ox * old_dils[1]
                    if ay % ndh != 0 or ax % ndw != 0:
                        valid = False
                        break
                    ny = ay // ndh
                    nx = ax // ndw
                    if not (0 <= ny < nkh and 0 <= nx < nkw):
                        valid = False
                        break

                if not valid:
                    continue

                reduction = old_param_count - new_pc
                candidate = {
                    "node_index": node_idx,
                    "node_name": node.name,
                    "weight_name": w_name,
                    "old_kernel": [old_kh, old_kw],
                    "new_kernel": [nkh, nkw],
                    "old_dilations": [old_dils[0], old_dils[1]],
                    "new_dilations": [ndh, ndw],
                    "old_parameters": old_param_count,
                    "new_parameters": new_pc,
                    "parameter_reduction": reduction,
                    "nonzero_count": len(nonzero),
                }
                if best is None or reduction > best["parameter_reduction"]:
                    best = candidate

        if best is not None:
            transformations.append(best)

    return transformations


def apply_dilation(model, transformation):
    """Apply a dilation transformation to the model."""
    candidate = copy.deepcopy(model)
    init_indices = {i.name: idx for idx, i in enumerate(candidate.graph.initializer)}

    w_name = transformation["weight_name"]
    init_idx = init_indices[w_name]
    old_init = candidate.graph.initializer[init_idx]
    old_w = numpy_helper.to_array(old_init)

    ndh, ndw = transformation["new_dilations"]
    nkh, nkw = transformation["new_kernel"]
    odh, odw = transformation["old_dilations"]

    new_w = np.zeros((old_w.shape[0], old_w.shape[1], nkh, nkw), dtype=old_w.dtype)
    nonzero = np.argwhere(old_w != 0)
    for idx in nonzero:
        oc, ic, oy, ox = int(idx[0]), int(idx[1]), int(idx[2]), int(idx[3])
        ay = oy * odh
        ax = ox * odw
        ny = ay // ndh
        nx = ax // ndw
        new_w[oc, ic, ny, nx] = old_w[oc, ic, oy, ox]

    new_init = numpy_helper.from_array(new_w, name=w_name)
    candidate.graph.initializer[init_idx].CopyFrom(new_init)

    # Update node attributes
    node = candidate.graph.node[transformation["node_index"]]
    retained = [
        a for a in node.attribute if a.name not in ("kernel_shape", "dilations")
    ]
    del node.attribute[:]
    node.attribute.extend(retained)
    node.attribute.extend(
        [
            helper.make_attribute("kernel_shape", [nkh, nkw]),
            helper.make_attribute("dilations", [ndh, ndw]),
        ]
    )

    return candidate


def score_model(model_bytes):
    """Quick score estimate using neurogolf_utils scoring."""
    import onnxruntime as ort
    import tempfile, os

    model = onnx.load_from_bytes(model_bytes)

    # Run shape inference
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
        onnx.checker.check_model(inferred)
    except Exception:
        return None

    # Score using the official scorer
    sanitized = nu.sanitize_model(inferred)
    if sanitized is None:
        return None

    try:
        opts = ort.SessionOptions()
        opts.enable_profiling = True
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            trace_path = f.name
        opts.profile_file_prefix = trace_path.replace(".json", "")

        session = ort.InferenceSession(sanitized.SerializeToString(), opts)
        sample = np.zeros(GRID_SHAPE, dtype=np.float32)
        sample[0, 0, :, :] = 1.0
        session.run(["output"], {"input": sample})
        real_trace = session.end_profiling()

        mem, params = nu.score_network(sanitized, real_trace)
        if mem is None or params is None:
            return None
        cost = mem + params
        points = max(1.0, 25.0 - math.log(max(1.0, cost)))
        os.unlink(real_trace)
        return {"memory": mem, "params": params, "cost": cost, "points": points}
    except Exception:
        try:
            os.unlink(real_trace)
        except:
            pass
        return None


def optimize_directory(input_dir, output_dir, max_tasks=20):
    """Optimize all models in input_dir and save to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    results = []

    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith(".onnx") or not fname.startswith("task"):
            continue

        src = os.path.join(input_dir, fname)
        dst = os.path.join(output_dir, fname)

        # Default: copy as-is
        shutil.copy2(src, dst)

        # Load model
        model = onnx.load(src)

        # Optimization 1: Remove unused initializers
        cleaned, removed = remove_unused_initializers(model)
        if removed:
            print(f"  {fname}: removed {removed}")

        # Optimization 2: Find dilation opportunities
        transforms = get_weight_analysis(cleaned)
        if transforms:
            for t in transforms:
                print(
                    f"  {fname}: dilation {t['old_kernel']}@{t['old_dilations']} -> {t['new_kernel']}@{t['new_dilations']} (-{t['parameter_reduction']} params)"
                )
                cleaned = apply_dilation(cleaned, t)

        # Save optimized model
        onnx.save(cleaned, dst)

        if transforms or removed:
            results.append(
                {
                    "task": fname,
                    "removed_init": removed,
                    "transforms": transforms,
                    "old_size": os.path.getsize(src),
                    "new_size": os.path.getsize(dst),
                }
            )

    return results


def build_submission(networks_dir, output_path):
    """Build a submission.zip from ONNX models in networks_dir."""
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for f in sorted(os.listdir(networks_dir)):
            if f.endswith(".onnx") and f.startswith("task"):
                zf.write(os.path.join(networks_dir, f), f)

    with zipfile.ZipFile(output_path, "r") as zf:
        names = sorted(zf.namelist())
    sha = hashlib.sha256(open(output_path, "rb").read()).hexdigest()
    return {
        "path": output_path,
        "tasks": len(names),
        "sha256": sha,
        "size": os.path.getsize(output_path),
    }


if __name__ == "__main__":
    import sys

    input_dir = sys.argv[1] if len(sys.argv) > 1 else "networks"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "optimized"

    print(f"Optimizing models from {input_dir} -> {output_dir}")
    results = optimize_directory(input_dir, output_dir)
    print(f"\nOptimized {len(results)} models")

    print(f"\nBuilding submission...")
    info = build_submission(output_dir, os.path.join(output_dir, "submission.zip"))
    print(f"Submission: {info['path']}")
    print(f"Tasks: {info['tasks']}")
    print(f"Size: {info['size']} bytes")
    print(f"SHA256: {info['sha256']}")
