"""
Test each model with different input sizes to find which ones break.
Kaggle may test with different grid sizes than what's in the training data.
"""

import json
import os
import sys
import numpy as np
import onnx
import onnxruntime as ort

TASKS_DIR = "neurogolf-2026"
MODELS_DIR = "fixed"
GRID_SHAPE = [1, 10, 30, 30]


def convert_to_numpy(example, target_h=30, target_w=30):
    """Convert a task example to numpy arrays, padding to target size."""
    benchmark = {}
    for mode in ["input", "output"]:
        arr = np.zeros((1, 10, target_h, target_w), dtype=np.float32)
        grid = example[mode]
        for r, row in enumerate(grid):
            for c, color in enumerate(row):
                if r < target_h and c < target_w:
                    arr[0, color, r, c] = 1.0
        benchmark[mode] = arr
    return benchmark


def sanitize_model(model):
    for node in model.graph.node:
        node.name = node.output[0]
        if "kernel_time" in node.output[0]:
            return None

    name_map, counter = {}, 0

    def get_safe_name(old_name):
        nonlocal counter
        if not old_name or old_name in ["input", "output"]:
            return old_name
        if old_name not in name_map:
            name_map[old_name] = f"safe_name_{counter}"
            counter += 1
        return name_map[old_name]

    for inp in model.graph.input:
        inp.name = get_safe_name(inp.name)
    for init in model.graph.initializer:
        init.name = get_safe_name(init.name)
    for node in model.graph.node:
        for i in range(len(node.input)):
            node.input[i] = get_safe_name(node.input[i])
        for i in range(len(node.output)):
            node.output[i] = get_safe_name(node.output[i])
        if len(node.output) > 0 and node.output[0]:
            node.name = node.output[0]
    for out in model.graph.output:
        out.name = get_safe_name(out.name)
    for vi in model.graph.value_info:
        vi.name = get_safe_name(vi.name)
    for node in model.graph.node:
        node.name = node.output[0]
    return model


def test_model_with_sizes(model_path, task_json_path, test_sizes):
    """Test a model on different input sizes."""
    try:
        with open(task_json_path) as f:
            examples = json.load(f)
    except:
        return None

    model = onnx.load(model_path)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    for node in inferred.graph.node:
        if node.output and node.output[0]:
            node.name = node.output[0]

    sanitized = sanitize_model(inferred)
    if sanitized is None:
        return None

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    try:
        session = ort.InferenceSession(sanitized.SerializeToString(), opts)
    except:
        return None

    all_examples = examples.get("train", []) + examples.get("test", [])
    results = {}

    for size in test_sizes:
        h, w = size
        pass_count = 0
        total = 0

        for ex in all_examples:
            inp_grid = ex["input"]
            out_grid = ex["output"]

            # Skip if example is larger than test size
            if len(inp_grid) > h or len(inp_grid[0]) > w:
                continue

            total += 1
            benchmark = convert_to_numpy(ex, h, w)

            try:
                user_output = session.run(["output"], {"input": benchmark["input"]})[0]
                user_output = (user_output > 0.0).astype(float)

                # Check if output matches (only the non-padded area)
                expected = benchmark["output"]
                match = True
                for r in range(len(out_grid)):
                    for c in range(len(out_grid[0])):
                        user_color = np.argmax(user_output[0, :, r, c])
                        if user_color != out_grid[r][c]:
                            match = False
                            break
                    if not match:
                        break

                if match:
                    pass_count += 1
            except:
                pass

        if total > 0:
            results[size] = (pass_count, total)

    return results


def main():
    # Get all tasks
    task_ids = sorted(
        [
            int(f.replace("task", "").replace(".json", ""))
            for f in os.listdir(TASKS_DIR)
            if f.startswith("task") and f.endswith(".json")
        ]
    )

    # Test sizes: original size, +1, +2, -1, -2, etc.
    test_sizes = [
        (30, 30),  # Original
        (25, 25),  # Smaller
        (20, 20),  # Much smaller
        (15, 15),  # Very small
        (28, 28),  # Slightly smaller
        (32, 32),  # Slightly larger (still within 30x30 limit)
    ]

    print(f"Testing {len(task_ids)} tasks with {len(test_sizes)} sizes...")
    print("=" * 80)

    failing_tasks = []

    for i, task_id in enumerate(task_ids):
        model_path = os.path.join(MODELS_DIR, f"task{task_id:03d}.onnx")
        task_json = os.path.join(TASKS_DIR, f"task{task_id:03d}.json")

        if not os.path.exists(model_path) or not os.path.exists(task_json):
            continue

        results = test_model_with_sizes(model_path, task_json, test_sizes)

        if results:
            # Check if model fails on any size
            fails_on_any = False
            for size, (pass_count, total) in results.items():
                if pass_count < total:
                    fails_on_any = True
                    break

            if fails_on_any:
                failing_tasks.append((task_id, results))

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(task_ids)}...", file=sys.stderr)

    print(f"\nTasks that fail on some sizes: {len(failing_tasks)}")
    print("=" * 80)

    for task_id, results in failing_tasks:
        print(f"\ntask{task_id:03d}:")
        for size, (pass_count, total) in sorted(results.items()):
            status = "PASS" if pass_count == total else f"FAIL ({pass_count}/{total})"
            print(f"  {size[0]}x{size[1]}: {status}")


if __name__ == "__main__":
    main()
