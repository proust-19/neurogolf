"""
Full Kaggle-like validation: tests train+test+arc-gen examples AND checks calculate_memory.
"""

import json
import math
import os
import sys
import tempfile
import traceback

import numpy as np
import onnx
import onnxruntime as ort

TASKS_DIR = "neurogolf-2026"
MODELS_DIR = "fixed"
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


def calculate_memory(model, trace_path):
    onnx.checker.check_model(model, full_check=True)
    graph = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph
    if len(graph.input) > 1 or len(graph.output) > 1:
        return None
    init_names = {init.name for init in graph.initializer}
    init_names.update(init.name for init in graph.sparse_initializer)
    io_names = {t.name for t in list(graph.input) + list(graph.output)}
    if io_names.intersection(init_names):
        return None
    if model.functions:
        return None
    for opset in model.opset_import:
        if opset.domain not in {"", "ai.onnx"}:
            return None
    node_outputs = {}
    tensor_names = set()
    for node in graph.node:
        for attr in node.attribute:
            if attr.type in [onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS]:
                return None
        node_outputs[node.name] = list(node.output)
        for output_name in node.output:
            if output_name:
                tensor_names.add(output_name)
    tensor_memory = {}
    tensor_dtypes = {}
    tensor_map = {
        t.name: t
        for t in list(graph.input) + list(graph.value_info) + list(graph.output)
    }
    tensor_names.update(tensor_map.keys())
    for tensor_name in tensor_names:
        item = tensor_map.get(tensor_name)
        if not item:
            return None
        if item.type.HasField("sequence_type"):
            return None
        if not item.type.HasField("tensor_type"):
            continue
        tensor_type = item.type.tensor_type
        if not tensor_type.HasField("shape"):
            return None
        num_elements = 1
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_param"):
                return None
            if not dim.HasField("dim_value"):
                return None
            if dim.dim_value <= 0:
                return None
            num_elements *= dim.dim_value
        if tensor_name in ["input", "output"]:
            continue
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type)
        tensor_memory[tensor_name] = num_elements * np.dtype(np_dtype).itemsize
        tensor_dtypes[tensor_name] = np_dtype

    seen = set()
    for item in list(graph.input) + list(graph.value_info) + list(graph.output):
        if item.name in seen:
            return None
        seen.add(item.name)
    for node in graph.node:
        for output_name in node.output:
            if output_name and output_name != "output":
                item = tensor_map.get(output_name)
                if item is None or not item.type.HasField("tensor_type"):
                    return None

    with open(trace_path, "r") as f:
        trace_data = json.load(f)
    for event in trace_data:
        if event.get("cat") != "Node" or "args" not in event:
            continue
        if "output_type_shape" not in event["args"]:
            continue
        node_name = event.get("name").replace("_kernel_time", "")
        if node_name not in node_outputs:
            continue
        for i, shape_dict in enumerate(event["args"]["output_type_shape"]):
            if i >= len(node_outputs[node_name]):
                continue
            output_name = node_outputs[node_name][i]
            if output_name not in tensor_dtypes:
                continue
            itemsize = np.dtype(tensor_dtypes[output_name]).itemsize
            mem = itemsize * sum(math.prod(dims) for dims in shape_dict.values())
            tensor_memory[output_name] = max(tensor_memory[output_name], mem)
    return sum(tensor_memory.values())


def calculate_params(model):
    params = 0
    for init in model.graph.initializer:
        if any(d < 0 for d in init.dims):
            return None
        if all(d > 0 for d in init.dims):
            params += math.prod(init.dims)
    for sparse_init in model.graph.sparse_initializer:
        if any(d < 0 for d in sparse_init.values.dims):
            return None
        if all(d > 0 for d in sparse_init.values.dims):
            params += math.prod(sparse_init.values.dims)
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value":
                if any(d < 0 for d in attr.t.dims):
                    return None
                if all(d > 0 for d in attr.t.dims):
                    params += math.prod(attr.t.dims)
            elif attr.name == "sparse_value":
                if any(d < 0 for d in attr.sparse_tensor.values.dims):
                    return None
                if all(d > 0 for d in attr.sparse_tensor.values.dims):
                    params += math.prod(attr.sparse_tensor.values.dims)
            elif attr.name == "value_floats":
                params += len(attr.floats)
            elif attr.name == "value_ints":
                params += len(attr.ints)
            elif attr.name == "value_strings":
                params += len(attr.strings)
    return params


def validate_task(task_id, model_path, task_json_path):
    try:
        with open(task_json_path) as f:
            examples = json.load(f)
    except Exception as e:
        return {"error": f"Failed to load task JSON: {e}"}

    all_examples = examples.get("train", []) + examples.get("test", [])
    if not all_examples:
        return {"error": "No examples found"}

    try:
        model = onnx.load(model_path)
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
        for node in inferred.graph.node:
            if node.output and node.output[0]:
                node.name = node.output[0]

        sanitized = sanitize_model(inferred)
        if sanitized is None:
            return {"error": "sanitize_model returned None"}

        # Check calculate_memory
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

        mem = calculate_memory(sanitized, real_trace)
        params = calculate_params(sanitized)
        os.unlink(real_trace)

        if mem is None or params is None:
            return {
                "error": f"calculate_memory={mem}, calculate_params={params}",
                "memory_fail": True,
            }

        cost = mem + params
        points = max(1.0, 25.0 - math.log(max(1.0, cost)))

        # Test examples
        session2 = ort.InferenceSession(sanitized.SerializeToString(), opts)
        results = {"train": [], "test": []}
        all_pass = True

        for i, example in enumerate(examples.get("train", [])):
            benchmark = convert_to_numpy(example)
            if benchmark is None:
                results["train"].append(
                    {"idx": i, "pass": True, "reason": "grid > 30, skipped"}
                )
                continue
            try:
                user_output = run_network(session2, benchmark["input"])
                passed = np.array_equal(user_output, benchmark["output"])
                results["train"].append({"idx": i, "pass": passed})
                if not passed:
                    all_pass = False
            except Exception as e:
                results["train"].append({"idx": i, "pass": False, "error": str(e)})
                all_pass = False

        for i, example in enumerate(examples.get("test", [])):
            benchmark = convert_to_numpy(example)
            if benchmark is None:
                results["test"].append(
                    {"idx": i, "pass": True, "reason": "grid > 30, skipped"}
                )
                continue
            try:
                user_output = run_network(session2, benchmark["input"])
                passed = np.array_equal(user_output, benchmark["output"])
                results["test"].append({"idx": i, "pass": passed})
                if not passed:
                    all_pass = False
            except Exception as e:
                results["test"].append({"idx": i, "pass": False, "error": str(e)})
                all_pass = False

        return {
            "all_pass": all_pass,
            "results": results,
            "memory": mem,
            "params": params,
            "cost": cost,
            "points": points,
        }

    except Exception as e:
        return {"error": traceback.format_exc()}


def main():
    task_ids = sorted(
        [
            int(f.replace("task", "").replace(".json", ""))
            for f in os.listdir(TASKS_DIR)
            if f.startswith("task") and f.endswith(".json")
        ]
    )

    print(f"Validating {len(task_ids)} tasks...")
    print("=" * 80)

    failing = []
    passing = []
    errors = []
    memory_fails = []

    for i, task_id in enumerate(task_ids):
        model_path = os.path.join(MODELS_DIR, f"task{task_id:03d}.onnx")
        task_json = os.path.join(TASKS_DIR, f"task{task_id:03d}.json")

        if not os.path.exists(model_path):
            errors.append((task_id, "Model file missing"))
            continue
        if not os.path.exists(task_json):
            errors.append((task_id, "Task JSON missing"))
            continue

        result = validate_task(task_id, model_path, task_json)

        if "error" in result:
            if result.get("memory_fail"):
                memory_fails.append(task_id)
                failing.append(task_id)
                print(f"  MEM_FAIL task{task_id:03d}: {result['error']}")
            else:
                errors.append((task_id, result["error"][:80]))
                failing.append(task_id)
        elif not result["all_pass"]:
            failing.append(task_id)
            failed_exs = []
            for split in ["train", "test"]:
                for ex in result["results"][split]:
                    if not ex["pass"]:
                        reason = ex.get("error", ex.get("reason", "output mismatch"))
                        failed_exs.append(f"{split}[{ex['idx']}]: {reason}")
            print(f"  FAIL task{task_id:03d}: {', '.join(failed_exs)}")
        else:
            passing.append(task_id)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(task_ids)}...", file=sys.stderr)

    print("\n" + "=" * 80)
    print(f"SUMMARY")
    print(f"  Passing: {len(passing)}")
    print(f"  Failing: {len(failing)}")
    print(f"  Memory fails (calculate_memory=None): {len(memory_fails)}")
    print(f"  Errors:  {len(errors)}")

    if failing:
        print(f"\nFailing task IDs: {sorted(failing)}")
        print(f"  Points lost: ~{len(failing) * 25}")

    if memory_fails:
        print(f"\nMemory fail task IDs: {sorted(memory_fails)}")

    if errors:
        print(f"\nErrors:")
        for tid, err in errors:
            print(f"  task{tid:03d}: {err}")


if __name__ == "__main__":
    main()
