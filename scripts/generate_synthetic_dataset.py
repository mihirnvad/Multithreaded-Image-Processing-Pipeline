#!/usr/bin/env python3
"""Generate a reproducible synthetic image dataset for the feature pipeline.

Each image mixes the structures that corner/feature detectors respond to:
gradients, sensor-like noise, filled and outlined shapes, polygons,
checkerboards, line fields and text. A configurable fraction of images is
deliberately degraded (blur, under-exposure, low contrast) and the ground
truth is written to ``manifest.csv``, so the visualization module's
image-quality panel can be checked against known defects.

Example:
    python3 scripts/generate_synthetic_dataset.py --count 200 --output data/test_set
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

FONTS = [
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
]
WORDS = ["ORB", "FAST", "BRIEF", "pthread", "mutex", "queue", "OpenCV", "feature", "pipeline", "keypoint"]
DEGRADATIONS = ["blur", "dark", "low_contrast"]


def random_color(rng: np.random.Generator) -> tuple[int, int, int]:
    return tuple(int(c) for c in rng.integers(0, 256, size=3))


def render_image(index: int, width: int, height: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + index)

    # Smooth two-colour gradient background.
    xs = np.linspace(0.0, 1.0, width, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, height, dtype=np.float32)
    angle = rng.uniform(0, 2 * np.pi)
    t = np.cos(angle) * xs[None, :] + np.sin(angle) * ys[:, None]
    t = (t - t.min()) / max(float(t.max() - t.min()), 1e-6)
    c0 = rng.integers(0, 256, size=3).astype(np.float32)
    c1 = rng.integers(0, 256, size=3).astype(np.float32)
    img = (c0[None, None, :] * (1 - t[..., None]) + c1[None, None, :] * t[..., None]).astype(np.uint8)

    scale = min(width, height)

    # Checkerboard patches: dense, strong corners.
    for _ in range(rng.integers(1, 4)):
        cell = int(rng.integers(max(8, scale // 60), max(9, scale // 20)))
        cols, rows = int(rng.integers(3, 9)), int(rng.integers(3, 9))
        x0 = int(rng.integers(0, max(1, width - cols * cell)))
        y0 = int(rng.integers(0, max(1, height - rows * cell)))
        a, b = random_color(rng), random_color(rng)
        for r in range(rows):
            for c in range(cols):
                color = a if (r + c) % 2 == 0 else b
                cv2.rectangle(img, (x0 + c * cell, y0 + r * cell), (x0 + (c + 1) * cell, y0 + (r + 1) * cell),
                              color, cv2.FILLED)

    # Filled and outlined rectangles and circles.
    for _ in range(rng.integers(15, 40)):
        p1 = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        thickness = int(rng.choice([cv2.FILLED, 2, 4, 6]))
        if rng.random() < 0.5:
            p2 = (int(p1[0] + rng.integers(-scale // 4, scale // 4)), int(p1[1] + rng.integers(-scale // 4, scale // 4)))
            cv2.rectangle(img, p1, p2, random_color(rng), thickness, cv2.LINE_AA)
        else:
            cv2.circle(img, p1, int(rng.integers(scale // 80, scale // 8)), random_color(rng), thickness, cv2.LINE_AA)

    # Random polygons.
    for _ in range(rng.integers(5, 15)):
        n = int(rng.integers(3, 8))
        center = rng.integers([0, 0], [width, height])
        pts = (center + rng.integers(-scale // 8, scale // 8, size=(n, 2))).astype(np.int32)
        if rng.random() < 0.5:
            cv2.fillPoly(img, [pts], random_color(rng), cv2.LINE_AA)
        else:
            cv2.polylines(img, [pts], True, random_color(rng), int(rng.integers(1, 5)), cv2.LINE_AA)

    # Line field.
    for _ in range(rng.integers(20, 60)):
        p1 = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        p2 = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        cv2.line(img, p1, p2, random_color(rng), int(rng.integers(1, 4)), cv2.LINE_AA)

    # Text: many small high-contrast corners.
    for _ in range(rng.integers(5, 15)):
        text = " ".join(rng.choice(WORDS, size=int(rng.integers(1, 4))))
        origin = (int(rng.integers(0, int(width * 0.8))), int(rng.integers(scale // 20, height)))
        font_scale = float(rng.uniform(0.6, 3.0)) * scale / 1080
        cv2.putText(img, text, origin, int(rng.choice(FONTS)), font_scale, random_color(rng),
                    int(rng.integers(1, 5)), cv2.LINE_AA)

    # Sensor-like Gaussian noise.
    sigma = float(rng.uniform(3, 12))
    noise = rng.normal(0.0, sigma, size=img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def degrade(img: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    if kind == "blur":
        k = int(rng.integers(6, 12)) * 2 + 1
        return cv2.GaussianBlur(img, (k, k), 0)
    if kind == "dark":
        return cv2.convertScaleAbs(img, alpha=float(rng.uniform(0.15, 0.3)), beta=0)
    if kind == "low_contrast":
        alpha = float(rng.uniform(0.15, 0.3))
        return cv2.convertScaleAbs(img, alpha=alpha, beta=127 * (1 - alpha))
    return img


def make_one(args: tuple[int, int, int, int, float, int, str]) -> tuple[str, str]:
    index, width, height, seed, degrade_fraction, quality, out_dir = args
    img = render_image(index, width, height, seed)
    rng = np.random.default_rng((seed + index) * 7919)
    label = "clean"
    if rng.random() < degrade_fraction:
        label = str(rng.choice(DEGRADATIONS))
        img = degrade(img, label, rng)
    name = f"synthetic_{index:04d}.jpg"
    if not cv2.imwrite(os.path.join(out_dir, name), img, [cv2.IMWRITE_JPEG_QUALITY, quality]):
        raise RuntimeError(f"failed to write {name}")
    return name, label


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=200, help="number of images (default: 200)")
    parser.add_argument("--output", type=Path, default=Path("data/test_set"), help="output directory")
    parser.add_argument("--width", type=int, default=1920, help="image width (default: 1920)")
    parser.add_argument("--height", type=int, default=1080, help="image height (default: 1080)")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    parser.add_argument("--degrade-fraction", type=float, default=0.15,
                        help="fraction of images to blur/darken/flatten (default: 0.15)")
    parser.add_argument("--quality", type=int, default=92, help="JPEG quality (default: 92)")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="parallel generator processes")
    args = parser.parse_args()

    if args.count <= 0 or args.width < 64 or args.height < 64:
        parser.error("--count must be positive and images at least 64x64")

    args.output.mkdir(parents=True, exist_ok=True)
    jobs = [(i, args.width, args.height, args.seed, args.degrade_fraction, args.quality, str(args.output))
            for i in range(args.count)]
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        results = list(pool.map(make_one, jobs, chunksize=4))

    with open(args.output / "manifest.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "degradation"])
        writer.writerows(results)

    degraded = sum(1 for _, label in results if label != "clean")
    total_mb = sum((args.output / name).stat().st_size for name, _ in results) / 1e6
    print(f"wrote {len(results)} images ({args.width}x{args.height}, {total_mb:.1f} MB, "
          f"{degraded} degraded) to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
