"""
Fix all ONNX models by inlining Function nodes (banned by competition rules).
Then rebuild submission.zip.
"""

import os
import sys
import math
import shutil
import zipfile
import hashlib
import tempfile

import numpy as np
import onnx
from onnx import helper
import onnxruntime as ort

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "neurogolf-2026"))
import importlib.util

spec = importlib.util.spec_from_file_location(
    "nu", "neurogolf-2026/neurogolf_utils/neurogolf_utils.py"
)
nu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nu)

BATCH, CHANNELS, HEIGHT, WIDTH = 1, 10, 30, 30
GRID_SHAPE = [BATCH, CHANNELS, HEIGHT, WIDTH]


def inline_functions(model):
    """Inline all Function nodes into regular graph nodes."""
    if not model.functions:
        return model, 0

    inlined_count = 0
    op_counters = {}
    for func in model.functions:
        new_nodes = []
        replaced = False
        for graph_node in model.graph.node:
            if graph_node.op_type == func.name and not replaced:
                imap = dict(zip(func.input, graph_node.input))
                omap = dict(zip(func.output, graph_node.output))

                for fn_node in func.node:
                    new_inputs = [imap.get(x, x) for x in fn_node.input]
                    new_outputs = [omap.get(x, x) for x in fn_node.output]

                    attrs = {}
                    for a in fn_node.attribute:
                        attrs[a.name] = onnx.helper.get_attribute_value(a)

                    node_name = fn_node.name
                    if not node_name:
                        op_type = fn_node.op_type
                        cnt = op_counters.get(op_type, 0)
                        node_name = f"{op_type}_{cnt}"
                        op_counters[op_type] = cnt + 1

                    new_node = helper.make_node(
                        fn_node.op_type,
                        inputs=new_inputs,
                        outputs=new_outputs,
                        name=node_name,
                        **attrs,
                    )
                    new_nodes.append(new_node)
                replaced = True
                inlined_count += 1
            else:
                new_nodes.append(graph_node)

        while len(model.graph.node) > 0:
            model.graph.node.pop()
        model.graph.node.extend(new_nodes)

    del model.functions[:]
    return model, inlined_count


def fix_node_names(model):
    """Ensure node.name == node.output[0] for all nodes (required for profiling/scoring)."""
    for node in model.graph.node:
        if node.output and node.output[0]:
            node.name = node.output[0]
    return model


def remove_bad_opsets(model):
    """Remove non-standard opset domains (e.g., 'golf') that cause scoring failures."""
    to_remove = [
        opset for opset in model.opset_import if opset.domain not in ("", "ai.onnx")
    ]
    for opset in to_remove:
        model.opset_import.remove(opset)
    return model, len(to_remove)


def remove_unused_initializers(model):
    """Remove initializers not referenced by any node."""
    used_names = set()
    for n in model.graph.node:
        for inp in n.input:
            if inp:
                used_names.add(inp)
    for init in model.graph.initializer:
        used_names.add(init.name)

    to_remove = [
        init for init in model.graph.initializer if init.name not in used_names
    ]
    for init in to_remove:
        model.graph.initializer.remove(init)
    return model, len(to_remove)


def score_model(model):
    """Score a model, returns dict or None."""
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
        onnx.checker.check_model(inferred, full_check=True)

        # Fix node names to match output tensor names
        fix_node_names(inferred)

        # Sanitize FIRST - this renames all tensors to safe_name_X
        sanitized = nu.sanitize_model(inferred)

        # Profile the SANITIZED model so trace names match
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

        mem = nu.calculate_memory(sanitized, real_trace)
        params = nu.calculate_params(sanitized)
        os.unlink(real_trace)

        if mem is None or params is None:
            return None

        cost = mem + params
        points = max(1.0, 25.0 - math.log(max(1.0, cost)))
        return {"memory": mem, "params": params, "cost": cost, "points": points}
    except Exception as e:
        return None


def main():
    src_dir = "networks"
    out_dir = "fixed"
    os.makedirs(out_dir, exist_ok=True)

    tasks = sorted(
        [f for f in os.listdir(src_dir) if f.endswith(".onnx") and f.startswith("task")]
    )
    print(f"Processing {len(tasks)} tasks...")

    stats = {"inlined": 0, "cleaned": 0, "already_ok": 0, "failed": 0}
    results_before = []
    results_after = []

    for i, fname in enumerate(tasks):
        src_path = os.path.join(src_dir, fname)
        out_path = os.path.join(out_dir, fname)

        model = onnx.load(src_path)

        # Score BEFORE fix
        before = score_model(model)

        # Fix: inline functions
        model, inlined = inline_functions(model)

        # Fix: remove bad opset domains
        model, removed_opsets = remove_bad_opsets(model)

        # Fix: remove unused initializers
        model, removed = remove_unused_initializers(model)

        # Fix: ensure node names match output tensor names
        model = fix_node_names(model)

        # Score AFTER fix
        after = score_model(model)

        if inlined > 0:
            stats["inlined"] += 1
        if removed > 0:
            stats["cleaned"] += 1
        if before is None and after is not None:
            stats["fixed"] = stats.get("fixed", 0) + 1
        if after is None:
            stats["failed"] += 1

        # Save fixed model
        onnx.save(model, out_path)

        before_pts = before["points"] if before else 0
        after_pts = after["points"] if after else 0
        results_before.append({"file": fname, "points": before_pts, **(before or {})})
        results_after.append({"file": fname, "points": after_pts, **(after or {})})

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(tasks)}...")

    # Summary
    total_before = sum(r["points"] for r in results_before)
    total_after = sum(r["points"] for r in results_after)
    valid_before = len([r for r in results_before if r["points"] > 0])
    valid_after = len([r for r in results_after if r["points"] > 0])

    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    print(f"Functions inlined: {stats['inlined']}")
    print(f"Unused inits cleaned: {stats['cleaned']}")
    print(f"Tasks fixed (None -> scored): {stats.get('fixed', 0)}")
    print(f"Still failing: {stats['failed']}")
    print(f"\nBefore: {valid_before} scoring tasks, total {total_before:.2f} pts")
    print(f"After:  {valid_after} scoring tasks, total {total_after:.2f} pts")
    print(
        f"Delta:  +{total_after - total_before:.2f} pts, +{valid_after - valid_before} tasks"
    )

    # Show newly fixed tasks
    newly_fixed = []
    for b, a in zip(results_before, results_after):
        if b["points"] == 0 and a["points"] > 0:
            newly_fixed.append(a)
    if newly_fixed:
        print(f"\nNewly fixed tasks ({len(newly_fixed)}):")
        for r in sorted(newly_fixed, key=lambda x: -x["points"])[:20]:
            print(f"  {r['file']}: {r['points']:.2f} pts (cost={r['cost']:.0f})")
        if len(newly_fixed) > 20:
            print(f"  ... and {len(newly_fixed) - 20} more")

    # Build submission.zip
    print(f"\nBuilding submission.zip...")
    submission_path = os.path.join(out_dir, "submission.zip")
    with zipfile.ZipFile(
        submission_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9
    ) as zf:
        for f in sorted(os.listdir(out_dir)):
            if f.endswith(".onnx") and f.startswith("task"):
                zf.write(os.path.join(out_dir, f), f)

    with zipfile.ZipFile(submission_path, "r") as zf:
        names = sorted(zf.namelist())
    sha = hashlib.sha256(open(submission_path, "rb").read()).hexdigest()
    size = os.path.getsize(submission_path)
    print(f"Submission: {submission_path}")
    print(f"Tasks: {len(names)}")
    print(f"Size: {size} bytes ({size / 1024 / 1024:.2f} MB)")
    print(f"SHA256: {sha}")


if __name__ == "__main__":
    main()
