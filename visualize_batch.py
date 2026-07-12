#!/usr/bin/env python3
"""
Batch visualize tasks: shows task data + ONNX match info for identification.
Usage: python3 visualize_batch.py 0 10    # tasks 0-9
       python3 visualize_batch.py 137 137 # just task 137
"""

import sys, os, json, numpy as np, argparse, onnxruntime as ort
from pathlib import Path

COLORS = {
    0: ".",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
}


def grid_str(grid):
    return "\n".join("".join(COLORS.get(v, "?") for v in row) for row in grid)


def load_onnx(path):
    try:
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        return sess, inp
    except:
        return None, None


def run_model(sess, inp, examples):
    results = []
    for ex in examples:
        try:
            arr = np.array(ex, dtype=np.float32).reshape(1, 10, 30, 30)
            out = sess.run(None, {inp.name: arr})[0]
            results.append(out)
        except Exception as e:
            results.append(f"ERROR: {e}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("start", type=int)
    parser.add_argument("end", type=int, nargs="?", default=None)
    parser.add_argument("--task-dir", default="neurogolf-2026")
    parser.add_argument("--model-dir", default="fixed")
    parser.add_argument("--max-train", type=int, default=2)
    args = parser.parse_args()

    end = args.end if args.end is not None else args.start

    for tid in range(args.start, end + 1):
        task_path = os.path.join(args.task_dir, f"task{tid:03d}.json")
        model_path = os.path.join(args.model_dir, f"task{tid:03d}.onnx")

        if not os.path.exists(task_path):
            print(f"task{tid:03d}: NOT FOUND")
            continue

        with open(task_path) as f:
            td = json.load(f)

        train = td.get("train", [])
        test = td.get("test", [])

        print(f"\n{'=' * 50}")
        print(f"TASK {tid:03d}  |  train={len(train)}  test={len(test)}")
        print(f"{'=' * 50}")

        # Show train examples
        for i, ex in enumerate(train[: args.max_train]):
            inp = ex["input"]
            out = ex["output"]
            h_in, w_in = len(inp), len(inp[0]) if inp else 0
            h_out, w_out = len(out), len(out[0]) if out else 0
            print(f"\nExample {i + 1}: {h_in}x{w_in} -> {h_out}x{w_out}")
            print("INPUT:                OUTPUT:")

            # Side by side
            max_h = max(h_in, h_out)
            inp_padded = inp + [[0] * w_in] * (max_h - h_in)
            out_padded = out + [[0] * w_out] * (max_h - h_out)

            for r in range(min(max_h, 25)):
                in_row = "".join(COLORS.get(v, "?") for v in inp_padded[r][:25])
                out_row = "".join(COLORS.get(v, "?") for v in out_padded[r][:25])
                print(f"  {in_row}    {out_row}")
            if max_h > 25:
                print(f"  ... ({max_h} rows total)")

        # Show ONNX model info
        if os.path.exists(model_path):
            sess, inp_info = load_onnx(model_path)
            if sess:
                # Get model size
                sz = os.path.getsize(model_path)
                sz_str = (
                    f"{sz / 1024:.1f}KB"
                    if sz < 1024 * 1024
                    else f"{sz / (1024 * 1024):.1f}MB"
                )

                # Run on train inputs
                matches = 0
                total = 0
                for ex in train:
                    try:
                        arr = np.array(ex["input"], dtype=np.float32).reshape(
                            1, 10, 30, 30
                        )
                        result = sess.run(None, {inp_info.name: arr})[0]
                        expected = np.array(ex["output"])
                        if result.shape == expected.shape:
                            if np.array_equal(result, expected):
                                matches += 1
                            total += 1
                    except:
                        pass

                match_str = (
                    f"{matches}/{total} train match" if total > 0 else "no train match"
                )
                print(f"\nONNX: {sz_str} | {match_str} | input={inp_info.shape}")
            else:
                print(f"\nONNX: ERROR loading model")
        else:
            print(f"\nONNX: no model")


if __name__ == "__main__":
    main()
