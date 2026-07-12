#!/usr/bin/env python3
"""
Export task visualization as PNG images for easier identification.
Usage: python3 export_task_images.py 137 137    # single task
       python3 export_task_images.py 0 10       # batch
"""

import sys, os, json, numpy as np, argparse
from PIL import Image, ImageDraw, ImageFont

# ARC color palette
COLORS = [
    (0, 0, 0),  # 0: black
    (0, 113, 188),  # 1: blue
    (255, 0, 0),  # 2: red
    (0, 200, 0),  # 3: green
    (255, 200, 0),  # 4: yellow
    (180, 0, 180),  # 5: purple
    (0, 200, 200),  # 6: teal
    (255, 128, 0),  # 7: orange
    (128, 128, 128),  # 8: gray
    (200, 200, 200),  # 9: light gray
]


def grid_to_image(grid, cell_size=20):
    h, w = len(grid), len(grid[0])
    img = Image.new("RGB", (w * cell_size, h * cell_size))
    draw = ImageDraw.Draw(img)
    for r in range(h):
        for c in range(w):
            color = COLORS[grid[r][c]]
            x0, y0 = c * cell_size, r * cell_size
            draw.rectangle(
                [x0, y0, x0 + cell_size - 1, y0 + cell_size - 1],
                fill=color,
                outline=(64, 64, 64),
            )
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("start", type=int)
    parser.add_argument("end", type=int, nargs="?", default=None)
    parser.add_argument("--task-dir", default="neurogolf-2026")
    parser.add_argument("--out-dir", default="task_images")
    parser.add_argument("--cell-size", type=int, default=15)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    end = args.end if args.end is not None else args.start

    for tid in range(args.start, end + 1):
        task_path = os.path.join(args.task_dir, f"task{tid:03d}.json")
        if not os.path.exists(task_path):
            continue

        with open(task_path) as f:
            td = json.load(f)

        train = td.get("train", [])
        test = td.get("test", [])

        # Create side-by-side image for each example
        all_imgs = []
        for i, ex in enumerate(train[:3]):
            inp_img = grid_to_image(ex["input"], args.cell_size)
            out_img = grid_to_image(ex["output"], args.cell_size)

            # Side by side with arrow
            w = inp_img.width + 30 + out_img.width
            h = max(inp_img.height, out_img.height) + 30
            combined = Image.new("RGB", (w, h), (255, 255, 255))
            combined.paste(inp_img, (0, 30))
            combined.paste(out_img, (inp_img.width + 30, 30))

            # Add labels
            draw = ImageDraw.Draw(combined)
            draw.text((5, 5), f"Train {i + 1} INPUT", fill=(0, 0, 0))
            draw.text((inp_img.width + 10, 5), "OUTPUT", fill=(0, 0, 0))

            all_imgs.append(combined)

        # Stack vertically
        if all_imgs:
            total_h = sum(img.height + 10 for img in all_imgs)
            max_w = max(img.width for img in all_imgs)
            final = Image.new("RGB", (max_w, total_h), (255, 255, 255))

            y = 0
            for img in all_imgs:
                final.paste(img, (0, y))
                y += img.height + 10

            out_path = os.path.join(args.out_dir, f"task{tid:03d}.png")
            final.save(out_path)
            print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
