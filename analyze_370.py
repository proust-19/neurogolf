"""Analyze task 370: understand the exact transformation rule."""

import json
import sys
import numpy as np

sys.path.insert(
    0, "/home/purshotamkumar/projects/neurogolf/neurogolf-2026/neurogolf_utils"
)
import neurogolf_utils as nu

# Load task from local path
with open("/home/purshotamkumar/projects/neurogolf/neurogolf-2026/task370.json") as f:
    examples = json.load(f)

# Override for full dataset path if needed
nu._NEUROGOLF_DIR = "/home/purshotamkumar/projects/neurogolf/neurogolf-2026/"


def one_hot(grid):
    arr = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            if r < 30 and c < 30:
                arr[0, int(val), r, c] = 1.0
    return arr


def from_one_hot(arr):
    ch, h, w = arr.shape[1], arr.shape[2], arr.shape[3]
    grid = []
    for r in range(h):
        row = []
        for c in range(w):
            colors = [ch for ch in range(10) if arr[0, ch, r, c] > 0.5]
            row.append(colors[0] if len(colors) == 1 else -1)
        if any(v != -1 for v in row):
            while row and row[-1] == -1:
                row.pop()
            grid.append(row)
    while grid and not grid[-1]:
        grid.pop()
    return grid


def print_grid(grid, label=""):
    print(f"\n{label} ({len(grid)}x{len(grid[0]) if grid else 0}):")
    for r, row in enumerate(grid):
        print(f"  {r:2d}: " + "".join(str(c) if c != -1 else "." for c in row))


def analyze_example(ex, idx):
    inp = ex["input"]
    out = ex["output"]
    h, w = len(inp), len(inp[0])

    inp_hot = one_hot(inp)
    out_hot = one_hot(out)

    # Find hole cells (value 0)
    hole_cells = [(r, c) for r in range(h) for c in range(w) if inp[r][c] == 0]

    # Find bg color (most common non-zero color)
    colors = [inp[r][c] for r in range(h) for c in range(w) if inp[r][c] != 0]
    from collections import Counter

    bg = Counter(colors).most_common(1)[0][0]

    # Find seed: single non-bg, non-zero cell
    seed_cells = [
        (r, c) for r in range(h) for c in range(w) if inp[r][c] != 0 and inp[r][c] != bg
    ]
    seed_val = inp[seed_cells[0][0]][seed_cells[0][1]] if seed_cells else None

    # Find new cells (in output but not in input)
    new_cells = []
    for r in range(h):
        for c in range(w):
            if out[r][c] != inp[r][c] and out[r][c] != bg:
                new_cells.append((r, c, out[r][c]))

    print(f"\n{'=' * 60}")
    print(f"Example {idx}: {h}x{w}, bg={bg}, seed={seed_val} at {seed_cells}")
    print(f"  Hole cells ({len(hole_cells)}): {sorted(hole_cells)}")
    print(f"  New seed-colored cells ({len(new_cells)}):")
    for r, c, v in sorted(new_cells):
        print(f"    ({r:2d},{c:2d}) = {v}")

    # Test: reflection of hole cells through seed
    if seed_cells:
        sr, sc = seed_cells[0]
        reflected = set()
        for hr, hc in hole_cells:
            rr = 2 * sr - hr
            rc = 2 * sc - hc
            if 0 <= rr < h and 0 <= rc < w:
                reflected.add((rr, rc))

        new_set = set((r, c) for r, c, v in new_cells)
        match = reflected & new_set
        extra = new_set - reflected
        missing = reflected - new_set

        print(f"  Reflection match: {len(match)}/{len(new_set)}")
        print(f"  Matching reflected: {sorted(match)}")
        if extra:
            print(f"  Extra (not reflected): {sorted(extra)}")
        if missing:
            print(
                f"  Missing (should be reflected but not in output): {sorted(missing)}"
            )

    # Print grids for visual inspection
    print(f"\n  Input grid:")
    for r, row in enumerate(inp):
        line = f"    {r:2d}: " + "".join(str(c) for c in row)
        print(line)

    print(f"\n  Output grid:")
    for r, row in enumerate(out):
        line = f"    {r:2d}: " + "".join(str(c) for c in row)
        print(line)

    # Difference map
    print(f"\n  Diff (+ = new seed, * = new other, . = unchanged):")
    for r in range(h):
        line = f"    {r:2d}: "
        for c in range(w):
            if inp[r][c] != out[r][c] and out[r][c] != bg:
                line += "+"
            elif inp[r][c] != out[r][c]:
                line += "*"
            else:
                line += "."
        print(line)


for i, ex in enumerate(examples["train"]):
    analyze_example(ex, i)

for i, ex in enumerate(examples.get("test", [])):
    analyze_example(ex, f"test_{i}")

for i, ex in enumerate(examples.get("arc-gen", [])):
    analyze_example(ex, f"arc-gen_{i}")
