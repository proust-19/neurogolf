#!/usr/bin/env python3
"""
Visualize a single task: shows taskXXX.json data + ONNX model output.
Usage: python3 visualize_task.py <task_number> [--task-dir DIR] [--model-dir DIR]
"""

import sys, os, json, numpy as np, argparse, onnxruntime as ort
from pathlib import Path


def load_task_json(task_num, task_dir):
    path = os.path.join(task_dir, f"task{task_num:03d}.json")
    with open(path) as f:
        return json.load(f)


def load_onnx(model_path):
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    return sess, inp, out


def run_onnx(sess, inp, examples):
    outputs = []
    for ex in examples:
        arr = np.array(ex, dtype=np.float32).reshape(1, 10, 30, 30)
        result = sess.run(None, {inp.name: arr})[0]
        outputs.append(result.reshape(30, 30).astype(int).tolist())
    return outputs


def print_grid(grid, label="", color=True):
    """Print a grid with ANSI colors for ARC colors."""
    colors = {
        0: "\033[40m  ",
        1: "\033[44m  ",
        2: "\033[41m  ",
        3: "\033[42m  ",
        4: "\033[43m  ",
        5: "\033[45m  ",
        6: "\033[46m  ",
        7: "\033[47m  ",
        8: "\033[100m  ",
        9: "\033[101m  ",
    }
    reset = "\033[0m"
    if label:
        print(f"  {label}")
    for row in grid:
        if color:
            print("    " + "".join(colors.get(v, "??") for v in row) + reset)
        else:
            print("    " + " ".join(str(v) for v in row))


def truncate_grid(grid, max_rows=20, max_cols=20):
    """Truncate grid to max size, return truncated flag."""
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0
    truncated = False
    if h > max_rows:
        grid = grid[:max_rows]
        truncated = True
    if w > max_cols:
        grid = [row[:max_cols] for row in grid]
        truncated = True
    return grid, truncated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task_num", type=int)
    parser.add_argument("--task-dir", default="neurogolf-2026")
    parser.add_argument("--model-dir", default="fixed")
    parser.add_argument("--max-grid", type=int, default=20)
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()

    task_num = args.task_num
    task_path = os.path.join(args.task_dir, f"task{task_num:03d}.json")
    model_path = os.path.join(args.model_dir, f"task{task_num:03d}.onnx")

    if not os.path.exists(task_path):
        print(f"ERROR: {task_path} not found")
        sys.exit(1)

    td = load_task_json(task_num, args.task_dir)
    train = td.get("train", [])
    test = td.get("test", [])
    color = not args.no_color

    print(f"\n{'=' * 60}")
    print(
        f"  TASK {task_num:03d}  |  Train examples: {len(train)}  |  Test examples: {len(test)}"
    )
    print(f"{'=' * 60}")

    # Show train examples
    if train:
        print(f"\n--- TRAIN ({len(train)} examples) ---")
        for i, ex in enumerate(train[:3]):  # show first 3
            inp = ex["input"]
            out = ex["output"]
            h_in, w_in = len(inp), len(inp[0]) if inp else 0
            h_out, w_out = len(out), len(out[0]) if out else 0
            print(
                f"\n  Example {i + 1}:  input {h_in}x{w_in}  →  output {h_out}x{w_out}"
            )
            inp_trunc, trunc_in = truncate_grid(inp, args.max_grid, args.max_grid)
            out_trunc, trunc_out = truncate_grid(out, args.max_grid, args.max_grid)
            print_grid(inp_trunc, "INPUT:", color)
            if trunc_in:
                print(f"    (truncated to {args.max_grid}x{args.max_grid})")
            print_grid(out_trunc, "OUTPUT:", color)
            if trunc_out:
                print(f"    (truncated to {args.max_grid}x{args.max_grid})")

    # Show ONNX model output if available
    if os.path.exists(model_path):
        print(f"\n--- ONNX MODEL OUTPUT ({model_path}) ---")
        try:
            sess, inp_info, out_info = load_onnx(model_path)
            print(f"  Input: {inp_info.name} {inp_info.shape}")
            print(f"  Output: {out_info.name} {out_info.shape}")

            # Run on train inputs
            for i, ex in enumerate(train[:3]):
                inp_arr = np.array(ex["input"], dtype=np.float32)
                expected = ex["output"]
                result = run_onnx(sess, inp_info, [ex["input"]])[0]
                exp_arr = np.array(expected)
                match = np.array_equal(result, exp_arr)
                total = exp_arr.size
                correct = int(np.sum(result == exp_arr))
                print(
                    f"\n  Train {i + 1}: {'✓ MATCH' if match else f'✗ {correct}/{total} correct ({100 * correct / total:.1f}%)'}"
                )
                print_grid(
                    truncate_grid(result, args.max_grid, args.max_grid)[0],
                    "ONNX OUTPUT:",
                    color,
                )

            # Run on test inputs
            if test:
                print(f"\n  --- Test predictions ---")
                for i, ex in enumerate(test[:5]):
                    result = run_onnx(sess, inp_info, [ex["input"]])[0]
                    if "output" in ex:
                        exp_arr = np.array(ex["output"])
                        match = np.array_equal(result, exp_arr)
                        correct = int(np.sum(result == exp_arr))
                        total = exp_arr.size
                        print(
                            f"\n  Test {i + 1}: {'✓ MATCH' if match else f'✗ {correct}/{total} correct ({100 * correct / total:.1f}%)'}"
                        )
                    else:
                        print(f"\n  Test {i + 1}: (no expected output)")
                    print_grid(
                        truncate_grid(result, args.max_grid, args.max_grid)[0],
                        "ONNX OUTPUT:",
                        color,
                    )

        except Exception as e:
            print(f"  ERROR running ONNX: {e}")
    else:
        print(f"\n  (No ONNX model found at {model_path})")

    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    main()
