"""Analyze task137 and task175 patterns."""

import json
import numpy as np

for task_id in [137, 175]:
    print(f"\n{'=' * 60}")
    print(f"Task {task_id}")
    print(f"{'=' * 60}")

    with open(f"neurogolf-2026/task{task_id:03d}.json") as f:
        examples = json.load(f)

    for i, ex in enumerate(examples["train"][:2]):
        inp = np.array(ex["input"])
        out = np.array(ex["output"])

        print(f"\nTrain[{i}]: input {inp.shape}, output {out.shape}")

        # Find non-zero positions in input
        non_zero = np.argwhere(inp != 0)
        if len(non_zero) > 0:
            print(f"  Input non-zero positions (row, col, value):")
            for r, c in non_zero[:10]:
                print(f"    ({r}, {c}) = {inp[r, c]}")

        # Find non-zero positions in output
        non_zero_out = np.argwhere(out != 0)
        if len(non_zero_out) > 0:
            print(f"  Output non-zero positions (row, col, value):")
            for r, c in non_zero_out[:15]:
                print(f"    ({r}, {c}) = {out[r, c]}")

        # Check if it's a color mapping
        if inp.shape == out.shape:
            unique_in = set(inp.flat)
            unique_out = set(out.flat)
            print(f"  Unique input values: {sorted(unique_in)}")
            print(f"  Unique output values: {sorted(unique_out)}")

            # Check color mapping
            cm = {}
            for iv, ov in zip(inp.flat, out.flat):
                iv, ov = int(iv), int(ov)
                if iv in cm and cm[iv] != ov:
                    print(f"  NOT a color map: {iv} maps to both {cm[iv]} and {ov}")
                    break
                cm[iv] = ov
            else:
                print(f"  Color mapping: {cm}")

        # Check if output shape depends on input shape
        if i == 0:
            if len(examples["train"]) > 1:
                inp2 = np.array(examples["train"][1]["input"])
                out2 = np.array(examples["train"][1]["output"])
                print(f"\n  Train[1] shapes: input {inp2.shape}, output {out2.shape}")
                print(f"  Same input shape? {inp.shape == inp2.shape}")
                print(f"  Same output shape? {out.shape == out2.shape}")

        # Show full grids for small examples
        if inp.size <= 400:
            print(f"\n  Input grid:")
            for r in range(inp.shape[0]):
                print(f"    {inp[r].tolist()}")
            print(f"\n  Output grid:")
            for r in range(out.shape[0]):
                print(f"    {out[r].tolist()}")
