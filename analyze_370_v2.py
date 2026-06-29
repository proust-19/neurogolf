"""Task 370: Verify the 'reflection + diagonal propagation' hypothesis."""

import json, sys, numpy as np

sys.path.insert(
    0, "/home/purshotamkumar/projects/neurogolf/neurogolf-2026/neurogolf_utils"
)
import neurogolf_utils as nu

nu._NEUROGOLF_DIR = "/home/purshotamkumar/projects/neurogolf/neurogolf-2026/"

with open("/home/purshotamkumar/projects/neurogolf/neurogolf-2026/task370.json") as f:
    examples = json.load(f)


def to_grid(ex):
    return np.array(ex)


def compute_reflection_propagation(inp_grid, seed, hole_cells):
    """
    Algorithm hypothesis:
    1. For each hole cell, compute vector from seed: (hr - sr, hc - sc)
    2. Ray goes in OPPOSITE direction: step k adds k * (-v_r, -v_c) to seed
    3. All ray positions get seed color
    """
    h, w = inp_grid.shape
    sr, sc = seed
    seed_val = inp_grid[sr, sc]
    output = inp_grid.copy()

    for hr, hc in hole_cells:
        dr = hr - sr  # vector from seed to hole
        dc = hc - sc
        # Go in opposite direction
        k = 1
        while True:
            rr = sr - k * dr
            rc = sc - k * dc
            if 0 <= rr < h and 0 <= rc < w:
                output[rr, rc] = seed_val
                k += 1
            else:
                break

    return output


def test_on_example(ex, name):
    inp = np.array(ex["input"])
    out_expected = np.array(ex["output"])
    h, w = inp.shape

    # Find hole cells (value 0)
    hole_mask = inp == 0
    hole_cells = list(zip(*np.where(hole_mask)))

    # Find seed: single non-bg, non-zero cell
    colors = inp[inp != 0].flatten()
    from collections import Counter

    bg = Counter(colors.tolist()).most_common(1)[0][0] if len(colors) > 0 else -1

    non_bg_mask = (inp != 0) & (inp != bg)
    seed_positions = list(zip(*np.where(non_bg_mask)))

    if len(seed_positions) != 1:
        print(
            f"  {name}: ERROR - found {len(seed_positions)} seed positions (expected 1)"
        )
        return False

    seed_pos = seed_positions[0]
    seed_val = inp[seed_pos]

    # Compute predicted output
    pred = compute_reflection_propagation(inp, seed_pos, hole_cells)

    match = np.all(pred == out_expected)

    if not match:
        diff = pred != out_expected
        false_pos = np.where(
            (pred == seed_val) & (out_expected != seed_val) & (out_expected != bg)
        )
        false_neg = np.where((pred != seed_val) & (out_expected == seed_val))

        fp_count = len(false_pos[0])
        fn_count = len(false_neg[0])

        if fp_count <= 5 and fn_count <= 5:
            print(f"  {name}: ALMOST MATCH ({fp_count} fp, {fn_count} fn)")
            return False
        else:
            print(f"  {name}: FAIL ({fp_count} fp, {fn_count} fn)")
            return False

    print(f"  {name}: PASS")
    return True


print("=== Testing reflection-propagation hypothesis ===\n")

all_pass = True
for i, ex in enumerate(examples["train"]):
    if not test_on_example(ex, f"train_{i}"):
        all_pass = False

for i, ex in enumerate(examples.get("test", [])):
    if not test_on_example(ex, f"test_{i}"):
        all_pass = False

# Test first 50 arc-gen
for i, ex in enumerate(examples.get("arc-gen", [])[:50]):
    if not test_on_example(ex, f"arc-gen_{i}"):
        all_pass = False

print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
