"""Try solving task137 and task175 with the Conv solver."""

import json
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort
import time

BATCH, CH, GH, GW = 1, 10, 30, 30
GRID_SHAPE = [BATCH, CH, GH, GW]
DT = TensorProto.FLOAT
IR = 10
OPSET = [helper.make_opsetid("", 10)]


def to_onehot(grid):
    arr = np.zeros((1, CH, GH, GW), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            arr[0, v, r, c] = 1.0
    return arr


def mk(nodes, inits=None):
    x = helper.make_tensor_value_info("input", DT, GRID_SHAPE)
    y = helper.make_tensor_value_info("output", DT, GRID_SHAPE)
    g = helper.make_graph(nodes, "g", [x], [y], initializer=inits or [])
    return helper.make_model(g, ir_version=IR, opset_imports=OPSET)


def get_exs(td):
    return [
        (np.array(ex["input"], dtype=np.int64), np.array(ex["output"], dtype=np.int64))
        for ex in td["train"] + td.get("test", [])
    ]


def _lstsq_conv(exs_raw, ks, use_bias):
    pad = ks // 2
    feat = 10 * ks * ks + (1 if use_bias else 0)
    if feat > 20000:
        return None
    patches, targets = [], []
    for inp_g, out_g in exs_raw:
        ih, iw = inp_g.shape
        oh_enc = np.zeros((10, ih, iw), dtype=np.float64)
        for c in range(10):
            oh_enc[c] = inp_g == c
        oh_pad = np.pad(oh_enc, ((0, 0), (pad, pad), (pad, pad)))
        oh, ow = out_g.shape
        for r in range(oh):
            for c in range(ow):
                p = oh_pad[:, r : r + ks, c : c + ks].flatten()
                if use_bias:
                    p = np.append(p, 1.0)
                patches.append(p)
                targets.append(int(out_g[r, c]))
    n_patches = len(patches)
    if feat > 5000 and n_patches > 2000:
        return None
    P = np.array(patches, dtype=np.float64)
    T = np.array(targets, dtype=np.int64)
    T_oh = np.zeros((len(T), 10), dtype=np.float64)
    for i, t in enumerate(T):
        T_oh[i, t] = 1.0
    WT = np.linalg.lstsq(P, T_oh, rcond=None)[0]
    if not np.array_equal(np.argmax(P @ WT, axis=1), T):
        return None
    if use_bias:
        Wconv = WT[:-1].T.reshape(10, 10, ks, ks).astype(np.float32)
        B = WT[-1].astype(np.float32)
    else:
        Wconv = WT.T.reshape(10, 10, ks, ks).astype(np.float32)
        B = None
    return Wconv, B


def solve_conv_fixed(td, time_budget=30.0):
    exs = get_exs(td)
    for inp, out in exs:
        if inp.shape != out.shape:
            return None
    shapes = set(inp.shape for inp, _ in exs)
    if len(shapes) != 1:
        return None
    t_start = time.time()
    for use_bias in [False, True]:
        for ks in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]:
            if time.time() - t_start > time_budget:
                return None
            result = _lstsq_conv(exs, ks, use_bias)
            if result is None:
                continue
            Wconv, B = result
            W_oh = numpy_helper.from_array(Wconv, "W")
            inits = [W_oh]
            if B is not None:
                inits.append(numpy_helper.from_array(B, "B"))
                model = mk(
                    [
                        helper.make_node(
                            "Conv",
                            ["input", "W", "B"],
                            ["output"],
                            kernel_shape=[ks, ks],
                            pads=[pad, pad, pad, pad],
                        )
                    ],
                    inits,
                )
            else:
                pad = ks // 2
                model = mk(
                    [
                        helper.make_node(
                            "Conv",
                            ["input", "W"],
                            ["output"],
                            kernel_shape=[ks, ks],
                            pads=[pad, pad, pad, pad],
                        )
                    ],
                    inits,
                )
            return model
    return None


def validate(model, td):
    try:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess = ort.InferenceSession(model.SerializeToString(), opts)
    except:
        return False
    examples = td["train"] + td.get("test", [])
    for ex in examples:
        inp = to_onehot(ex["input"])
        exp = to_onehot(ex["output"])
        try:
            out = sess.run(["output"], {"input": inp})[0]
            out = (out > 0.0).astype(np.float32)
        except:
            return False
        if not np.array_equal(out, exp):
            return False
    return True


for task_id in [137, 175]:
    print(f"\nTrying task{task_id:03d}...")
    with open(f"neurogolf-2026/task{task_id:03d}.json") as f:
        td = json.load(f)

    exs = get_exs(td)
    shapes = set(inp.shape for inp, _ in exs)
    print(f"  Input shapes: {shapes}")

    model = solve_conv_fixed(td, time_budget=60.0)
    if model is not None:
        print(f"  Found Conv solution!")
        valid = validate(model, td)
        print(f"  Validation: {valid}")
        if valid:
            onnx.save(model, f"fixed/task{task_id:03d}_conv.onnx")
            print(f"  Saved to fixed/task{task_id:03d}_conv.onnx")
    else:
        print(f"  No Conv solution found")
