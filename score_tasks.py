"""
Fixed per-task scoring - profiles BEFORE sanitize to preserve node name mapping.
"""

import json
import math
import os
import sys
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
import importlib.util

spec = importlib.util.spec_from_file_location(
    "nu", "neurogolf-2026/neurogolf_utils/neurogolf_utils.py"
)
nu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nu)

BATCH, CHANNELS, HEIGHT, WIDTH = 1, 10, 30, 30
GRID_SHAPE = [BATCH, CHANNELS, HEIGHT, WIDTH]


def score_single_task(model_path):
    try:
        model = onnx.load(model_path)
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
        onnx.checker.check_model(inferred, full_check=True)

        # Fix node names to match output tensor names (required for scoring)
        for node in inferred.graph.node:
            if node.output and node.output[0]:
                node.name = node.output[0]

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
        return {"error": str(e)}


def main():
    networks_dir = sys.argv[1] if len(sys.argv) > 1 else "networks"
    tasks = sorted(
        [
            f
            for f in os.listdir(networks_dir)
            if f.endswith(".onnx") and f.startswith("task")
        ]
    )

    print(f"Analyzing {len(tasks)} tasks in {networks_dir}/...")
    print("=" * 90)

    results = []
    total_score = 0
    invalid_count = 0

    for i, fname in enumerate(tasks):
        task_id = int(fname.replace("task", "").replace(".onnx", ""))
        model_path = os.path.join(networks_dir, fname)
        file_size = os.path.getsize(model_path)

        score_info = score_single_task(model_path)

        if score_info and "error" not in score_info:
            results.append(
                {
                    "task": task_id,
                    "file": fname,
                    "points": score_info["points"],
                    "cost": score_info["cost"],
                    "params": score_info["params"],
                    "memory": score_info["memory"],
                    "file_size": file_size,
                }
            )
            total_score += score_info["points"]
        else:
            invalid_count += 1
            err = score_info.get("error", "unknown") if score_info else "None"
            results.append(
                {
                    "task": task_id,
                    "file": fname,
                    "points": 0,
                    "cost": float("inf"),
                    "params": 0,
                    "memory": 0,
                    "file_size": file_size,
                    "error": err,
                }
            )

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(tasks)}...", file=sys.stderr)

    results.sort(key=lambda x: x["points"])

    print(
        f"\n{'Task':<14} {'Points':>8} {'Cost':>12} {'Params':>10} {'Memory':>10} {'FileSize':>10}"
    )
    print("-" * 90)

    for r in results:
        if r["points"] == 0:
            err = r.get("error", "")[:40]
            print(
                f"{r['file']:<14} {'FAIL':>8} {'N/A':>12} {'N/A':>10} {'N/A':>10} {r['file_size']:>10}  {err}"
            )
        else:
            print(
                f"{r['file']:<14} {r['points']:>8.2f} {r['cost']:>12.0f} {r['params']:>10} {r['memory']:>10} {r['file_size']:>10}"
            )

    valid = [r for r in results if r["points"] > 0]
    failed = [r for r in results if r["points"] == 0]

    print("-" * 90)
    print(f"\n{'=' * 90}")
    print(f"SUMMARY")
    print(f"{'=' * 90}")
    print(f"  Total tasks:     {len(tasks)}")
    print(f"  Valid (scoring): {len(valid)}")
    print(f"  Failed:          {len(failed)}")
    print(f"  Total score:     {total_score:.2f} / {25 * len(tasks):.0f} max")
    print(f"  Avg per task:    {total_score / max(1, len(valid)):.2f} (valid only)")

    if valid:
        print(f"\nTOP 15 BEST tasks:")
        for r in sorted(valid, key=lambda x: -x["points"])[:15]:
            print(
                f"  {r['file']}: {r['points']:.2f} pts  cost={r['cost']:.0f}  params={r['params']}  mem={r['memory']}"
            )

        print(f"\nTOP 15 WORST tasks (scoring but low points):")
        for r in sorted(valid, key=lambda x: x["points"])[:15]:
            print(
                f"  {r['file']}: {r['points']:.2f} pts  cost={r['cost']:.0f}  params={r['params']}  mem={r['memory']}"
            )

        high_cost = sorted(valid, key=lambda x: -x["cost"])[:15]
        print(f"\nTOP 15 HIGHEST COST (biggest optimization opportunity):")
        for r in high_cost:
            print(
                f"  {r['file']}: cost={r['cost']:.0f}  params={r['params']}  mem={r['memory']}  pts={r['points']:.2f}"
            )

        low_pts = [r for r in valid if r["points"] <= 5]
        print(f"\nTasks with <= 5 points: {len(low_pts)}")
        print(
            f"  These contribute least: if improved to 25 pts each, +{(25 - 5) * len(low_pts):.0f} points"
        )

    if failed:
        print(f"\nFAILED tasks ({len(failed)}):")
        for r in failed[:20]:
            print(f"  {r['file']}: {r.get('error', '?')[:60]}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")


if __name__ == "__main__":
    main()
