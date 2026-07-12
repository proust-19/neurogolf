"""Diagnose task137 and task175 failures."""

import json
import numpy as np
import onnx
import onnxruntime as ort

GRID_SHAPE = [1, 10, 30, 30]


def convert_to_numpy(example):
    benchmark = {}
    example_shape = (1, 10, 30, 30)
    for mode in ["input", "output"]:
        benchmark[mode] = np.zeros(example_shape, dtype=np.float32)
        grid = example[mode]
        if max(len(grid), len(grid[0])) > 30:
            return None
        for r, row in enumerate(grid):
            for c, color in enumerate(row):
                benchmark[mode][0][color][r][c] = 1.0
    return benchmark


def run_network(session, benchmark_input):
    result = session.run(["output"], {"input": benchmark_input})
    return (result[0] > 0.0).astype(float)


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


def diagnose_task(task_id):
    print(f"\n{'=' * 60}")
    print(f"Diagnosing task{task_id:03d}")
    print(f"{'=' * 60}")

    # Load task examples
    with open(f"neurogolf-2026/task{task_id:03d}.json") as f:
        examples = json.load(f)

    print(f"Train examples: {len(examples.get('train', []))}")
    print(f"Test examples: {len(examples.get('test', []))}")

    # Check grid sizes
    for i, ex in enumerate(examples.get("train", [])[:2]):
        inp = ex["input"]
        out = ex["output"]
        print(
            f"  Train[{i}]: input {len(inp)}x{len(inp[0])}, output {len(out)}x{len(out[0])}"
        )

    # Load model
    model = onnx.load(f"fixed/task{task_id:03d}.onnx")
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    for node in inferred.graph.node:
        if node.output and node.output[0]:
            node.name = node.output[0]

    sanitized = sanitize_model(inferred)
    if sanitized is None:
        print("ERROR: sanitize_model returned None")
        return

    # Count nodes and ops
    ops = {}
    for node in sanitized.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    print(f"Nodes: {len(sanitized.graph.node)}")
    print(f"Ops: {ops}")

    # Run model
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(sanitized.SerializeToString(), opts)

    # Test first training example
    ex = examples["train"][0]
    benchmark = convert_to_numpy(ex)
    if benchmark is None:
        print("ERROR: grid > 30")
        return

    user_output = run_network(session, benchmark["input"])
    expected = benchmark["output"]
    match = np.array_equal(user_output, expected)

    print(f"\nFirst train example:")
    print(f"  Input grid: {len(ex['input'])}x{len(ex['input'][0])}")
    print(f"  Expected output grid: {len(ex['output'])}x{len(ex['output'][0])}")
    print(f"  Match: {match}")

    if not match:
        # Show differences
        diff = user_output != expected
        diff_positions = np.argwhere(diff[0])  # Remove batch dim
        print(f"  Number of differing pixels: {len(diff_positions)}")

        # Convert outputs back to grids for comparison
        def output_to_grid(arr):
            grid = []
            for r in range(30):
                row = []
                for c in range(30):
                    colors = [ch for ch in range(10) if arr[0, ch, r, c] == 1.0]
                    row.append(colors[0] if colors else 0)
                grid.append(row)
            return grid

        user_grid = output_to_grid(user_output)
        expected_grid = output_to_grid(expected)

        print(f"\n  Expected (first 10 rows, first 15 cols):")
        for r in range(min(10, len(ex["output"]))):
            print(f"    {expected_grid[r][:15]}")

        print(f"\n  Got (first 10 rows, first 15 cols):")
        for r in range(min(10, len(ex["output"]))):
            print(f"    {user_grid[r][:15]}")


for task_id in [137, 175]:
    diagnose_task(task_id)
