#!/usr/bin/env python3
"""Generate a 1xN grid image of pallet stacks for selected strategies.

Usage:
  python python/scripts/generate_pallet_grid.py --input <path/to/results.json> --out out.png

The script will look for per-strategy GIFs in the same output folder under `gifs/`.
If a GIF for a strategy is missing, a textual placeholder will be generated from the metrics
found in `results.json`.

Parallel extraction/resizing uses a worker pool sized to ~70%% of available CPUs by default.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

try:
    from PIL import Image, ImageDraw, ImageFont, ImageSequence
except Exception as e:
    print("Missing dependency: Pillow is required. Install with: pip install pillow")
    raise


DEFAULT_STRATEGIES = [
    "walle_scoring",
    "surface_contact",
    "best_fit_decreasing",
    "hybrid_adaptive",
    "extreme_points",
]


def load_results(path: str) -> dict:
    import json

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_last_frame(gif_path: str) -> Image.Image:
    im = Image.open(gif_path)
    last_frame = None
    for frame in ImageSequence.Iterator(im):
        last_frame = frame.convert("RGBA")
    if last_frame is None:
        raise RuntimeError(f"Unable to read frames from {gif_path}")
    return last_frame


def make_placeholder(name: str, metrics: Dict[str, float], size=(600, 600)) -> Image.Image:
    img = Image.new("RGBA", size, (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
        font_small = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
        font_small = font

    title = name.replace("_", " ").title()
    try:
        bbox = draw.textbbox((0, 0), title, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    except Exception:
        tw, th = font.getsize(title)
    draw.text(((size[0] - tw) / 2, 20), title, fill="black", font=font)

    # Metrics lines
    lines = []
    if metrics:
        if "avg_closed_fill" in metrics:
            lines.append(f"Avg closed fill: {metrics['avg_closed_fill']*100:.1f}%")
        if "placement_rate" in metrics:
            lines.append(f"Placement rate: {metrics['placement_rate']*100:.1f}%")
        if "total_placed" in metrics:
            lines.append(f"Placed boxes: {metrics['total_placed']}")

    if not lines:
        lines = ["No visual available", "(showing summary placeholder)"]

    y = 80
    for line in lines:
        try:
            bbox = draw.textbbox((0, 0), line, font=font_small)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except Exception:
            tw, th = font_small.getsize(line)
        draw.text(((size[0] - tw) / 2, y), line, fill="black", font=font_small)
        y += th + 6

    # draw a light border
    draw.rectangle([2, 2, size[0] - 3, size[1] - 3], outline=(180, 180, 180))
    return img


def process_strategy(strategy: str, gifs_dir: str, metrics: Dict[str, Dict]) -> Image.Image:
    gif_file = os.path.join(gifs_dir, f"{strategy}.gif")
    if os.path.exists(gif_file):
        try:
            img = extract_last_frame(gif_file)
            return img
        except Exception:
            pass

    # fallback to placeholder using metrics
    m = metrics.get(strategy, {})
    return make_placeholder(strategy, m)


def resize_keep_aspect(img: Image.Image, target_height: int) -> Image.Image:
    w, h = img.size
    if h == target_height:
        return img
    new_w = int(w * (target_height / h))
    return img.resize((new_w, target_height), Image.LANCZOS)


def build_grid(images: List[Image.Image], out_path: str, bg=(255, 255, 255)) -> None:
    # normalize heights (choose min height to avoid upscaling)
    heights = [im.size[1] for im in images]
    target_h = min(heights) if heights else 600
    resized = [resize_keep_aspect(im, target_h) for im in images]

    total_w = sum(im.size[0] for im in resized)
    out = Image.new("RGBA", (total_w, target_h), bg + (255,))
    x = 0
    for im in resized:
        out.paste(im, (x, 0), im)
        x += im.size[0]

    out.save(out_path)


def gather_metrics(data: dict) -> Dict[str, Dict]:
    # Build a simple lookup per strategy using the first matching entry
    out: Dict[str, Dict] = {}
    for r in data.get("phase1_baseline", []):
        s = r.get("strategy")
        if s and s not in out:
            out[s] = {
                "avg_closed_fill": r.get("avg_closed_fill"),
                "placement_rate": r.get("placement_rate"),
                "total_placed": r.get("total_placed"),
            }
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to results.json")
    parser.add_argument("--out", default="pallet_grid.png", help="Output image file")
    parser.add_argument("--strategies", nargs="*", default=DEFAULT_STRATEGIES,
                        help="Strategy keys to include (default: top-5 list)")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of parallel workers (0 -> auto 70%% of CPUs)")
    args = parser.parse_args(argv)

    data = load_results(args.input)
    out_dir = os.path.dirname(os.path.abspath(args.input))
    gifs_dir = os.path.join(out_dir, "gifs")

    metrics = gather_metrics(data)

    cpu_count = os.cpu_count() or 1
    if args.workers and args.workers > 0:
        workers = args.workers
    else:
        workers = max(1, int(math.ceil(cpu_count * 0.7)))

    strategies = args.strategies
    print(f"Rendering strategies: {strategies}")
    print(f"Using workers: {workers} (approx 70%% of {cpu_count})")

    images = [None] * len(strategies)
    with ThreadPoolExecutor(max_workers=min(workers, len(strategies))) as ex:
        futures = {ex.submit(process_strategy, s, gifs_dir, metrics): i for i, s in enumerate(strategies)}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                images[i] = fut.result()
            except Exception as e:
                print(f"Failed to process {strategies[i]}: {e}")
                images[i] = make_placeholder(strategies[i], metrics.get(strategies[i], {}))

    # Ensure all images present
    images = [img if img is not None else make_placeholder(s, metrics.get(s, {}))
              for img, s in zip(images, strategies)]

    build_grid(images, args.out)
    print(f"Saved grid: {args.out}")


if __name__ == "__main__":
    main()
