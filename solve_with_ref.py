"""Use solver_ref to solve task137 and task175."""

import json
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort
import time
import sys

# Import solver_ref
sys.path.insert(0, ".")
import solver_ref as sr


def validate_model(model, td):
    try:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess = ort.InferenceSession(model.SerializeToString(), opts)
    except Exception as e:
        print(f"  Session error: {e}")
        return False
    examples = td["train"] + td.get("test", [])
    for i, ex in enumerate(examples):
        inp = sr.to_onehot(ex["input"])
        exp = sr.to_onehot(ex["output"])
        try:
            out = sess.run(["output"], {"input": inp})[0]
            out_bin = (out > 0.0).astype(np.float32)
        except Exception as e:
            print(f"  Run error on example {i}: {e}")
            return False
        if not np.array_equal(out_bin, exp):
            print(f"  FAIL on example {i}")
            return False
    return True


for task_id in [137, 175]:
    print(f"\nSolving task{task_id:03d}...")
    with open(f"neurogolf-2026/task{task_id:03d}.json") as f:
        td = json.load(f)

    # Try all solvers from solver_ref
    solvers = [
        ("identity", sr.s_identity),
        ("color_map", sr.s_color_map),
        ("transpose", sr.s_transpose),
        ("flip", sr.s_flip),
        ("rotate", sr.s_rotate),
        ("tile", sr.s_tile),
        ("upscale", sr.s_upscale),
        ("concat", sr.s_concat),
        ("constant", sr.s_constant),
        ("crop", sr.s_crop),
    ]

    for name, solver in solvers:
        try:
            result = solver(td)
            if result is not None:
                print(f"  {name}: Found solution")
                valid = validate_model(result, td)
                if valid:
                    print(f"  {name}: VALID!")
                    onnx.save(result, f"fixed/task{task_id:03d}_solved.onnx")
                    break
                else:
                    print(f"  {name}: Invalid")
        except Exception as e:
            pass

    # Try Conv solvers
    for name, solver_fn in [
        ("conv_fixed", sr.solve_conv_fixed),
        ("conv_variable", sr.solve_conv_variable),
    ]:
        try:
            result = solver_fn(
                td, f"/tmp/task{task_id:03d}_test.onnx", time_budget=30.0
            )
            if result is not None:
                print(f"  {name}: Found solution")
                valid = validate_model(result, td)
                if valid:
                    print(f"  {name}: VALID!")
                    onnx.save(result, f"fixed/task{task_id:03d}_solved.onnx")
                    break
                else:
                    print(f"  {name}: Invalid")
        except Exception as e:
            print(f"  {name}: Error: {e}")

    # Check if we solved it
    import os

    if os.path.exists(f"fixed/task{task_id:03d}_solved.onnx"):
        print(f"  Task{task_id:03d} SOLVED!")
    else:
        print(f"  Task{task_id:03d} NOT solved")
