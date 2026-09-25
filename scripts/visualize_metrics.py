#!/usr/bin/env python3
"""Visualize pipeline performance, feature maps and image quality.

Reads the outputs of ``benchmark.py`` (and optionally the annotated images and
feature maps written with ``--output --feature-maps``) and renders:

  Performance
    speedup_vs_amdahl.png     measured speedup vs. ideal linear and a fitted Amdahl curve
    throughput.png            images/second per configuration
    latency_percentiles.png   per-image processing time and end-to-end latency (p50/p95/p99)
    stage_breakdown.png       where each image's time goes: read, queue wait, process, write

  Image inspection
    image_quality.png         keypoints vs. sharpness, exposure, flagged low-quality images
    feature_maps.png          original | ORB keypoints | keypoint-density heat map

Example:
    python3 scripts/visualize_metrics.py --results results --images results/images \\
        --data data/test_set --out assets
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: CI, Docker, SSH
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Validated categorical palette (colorblind-safe in this fixed order), with
# recessive ink for text, axes and grid.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
NEUTRAL = "#8a8984"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
TARGET_SPEEDUP = 3.5

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titleweight": "semibold",
    "axes.titlesize": 13,
    "axes.titlelocation": "left",
    "axes.titlepad": 12,
    "axes.labelsize": 10.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.frameon": False,
    "legend.fontsize": 9.5,
    "lines.linewidth": 2,
    "font.family": "DejaVu Sans",
    "figure.dpi": 110,
})
DPI = 200


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def column(rows: list[dict[str, str]], name: str) -> np.ndarray:
    return np.array([float(r[name]) for r in rows])


def load_summary(results: Path) -> list[dict]:
    path = results / "benchmark_summary.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found - run scripts/benchmark.py first")
    rows = read_csv(path)
    for r in rows:
        for k, v in r.items():
            if k not in ("mode", "metrics_file"):
                r[k] = float(v)
        r["threads"] = int(r["threads"])
        r["label"] = "seq" if r["mode"] == "sequential" else str(r["threads"])
    return rows


def load_system_info(results: Path) -> dict:
    path = results / "system_info.json"
    return json.loads(path.read_text()) if path.exists() else {}


def fit_amdahl(threads: np.ndarray, speedup: np.ndarray) -> float:
    """Least-squares parallel fraction p in S(n) = 1 / ((1 - p) + p / n)."""
    candidates = np.linspace(0.0, 1.0, 10001)
    errors = [np.sum((speedup - 1.0 / ((1 - p) + p / threads)) ** 2) for p in candidates]
    return float(candidates[int(np.argmin(errors))])


def subtitle(ax, text: str) -> None:
    ax.text(0, 1.01, text, transform=ax.transAxes, fontsize=9.5, color=INK_2, va="bottom")


# ---------------------------------------------------------------------------
# Performance charts
# ---------------------------------------------------------------------------

def plot_speedup(summary: list[dict], info: dict, out: Path) -> Path:
    par = [r for r in summary if r["mode"] == "parallel"]
    threads = np.array([r["threads"] for r in par], dtype=float)
    speedup = np.array([r["speedup_vs_sequential"] for r in par])
    # Amdahl is fitted on the physical-core region, where the model applies;
    # past that point hyper-threads and memory bandwidth dominate.
    cores = info.get("physical_cores") or info.get("logical_cpus")
    fit_mask = threads <= (cores or threads.max())
    p = fit_amdahl(threads[fit_mask], speedup[fit_mask]) if fit_mask.sum() >= 2 else fit_amdahl(threads, speedup)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    n = np.logspace(0, math.log2(max(threads.max(), 16)), 200, base=2)
    ax.plot(n, n, color=NEUTRAL, linewidth=1.2, linestyle=(0, (4, 3)), label="Ideal linear speedup")
    ax.plot(n, 1 / ((1 - p) + p / n), color=SERIES[1], linewidth=2,
            label=f"Amdahl fit: p = {p:.3f}  (limit {1 / (1 - p):.1f}x)" if p < 1 else "Amdahl fit: p = 1")
    ax.plot(threads, speedup, color=SERIES[0], linewidth=2, marker="o", markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2, label="Measured", zorder=5)
    ax.axhline(TARGET_SPEEDUP, color=INK_2, linewidth=1, linestyle=":")
    ax.text(threads.min(), TARGET_SPEEDUP, f" {TARGET_SPEEDUP}x", va="bottom", ha="left", fontsize=9, color=INK_2)

    for c, name in ((info.get("physical_cores"), "physical cores"), (info.get("logical_cpus"), "logical CPUs")):
        if c and threads.min() <= c <= max(threads.max(), 16):
            ax.axvline(c, color=GRID, linewidth=1.5, zorder=0)
            ax.text(c, max(threads.max(), 16) * 0.98, f" {c} {name}",
                    rotation=90, va="top", ha="right", fontsize=8.5, color=INK_2)

    best = par[int(np.argmax(speedup))]
    ax.annotate(f"{best['speedup_vs_sequential']:.2f}x at {best['threads']} threads",
                (best["threads"], best["speedup_vs_sequential"]), textcoords="offset points", xytext=(10, -18),
                fontsize=10, color=INK, fontweight="semibold")

    ax.set_xscale("log", base=2)
    ax.set_xticks(threads)
    ax.set_xticklabels([str(int(t)) for t in threads])
    ax.set_ylim(0, max(threads.max(), 16) + 0.5)
    ax.set_xlabel("Worker threads")
    ax.set_ylabel("Speedup over sequential baseline (x)")
    ax.set_title("Throughput speedup vs. Amdahl's law")
    subtitle(ax, info.get("cpu_model", ""))
    ax.legend(loc="upper left")
    fig.tight_layout()
    path = out / "speedup_vs_amdahl.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def plot_throughput(summary: list[dict], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    labels = ["Sequential" if r["mode"] == "sequential" else f"{r['threads']} thr" for r in summary]
    values = [r["throughput_ips"] for r in summary]
    colors = [NEUTRAL if r["mode"] == "sequential" else SERIES[0] for r in summary]
    bars = ax.bar(labels, values, color=colors, width=0.62, edgecolor=SURFACE, linewidth=2)
    for bar, r in zip(bars, summary):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{r['throughput_ips']:.1f}\n{r['speedup_vs_sequential']:.2f}x",
                ha="center", va="bottom", fontsize=9, color=INK_2, linespacing=1.3)
    ax.set_ylim(0, max(values) * 1.22)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Images per second")
    ax.set_title("Pipeline throughput")
    subtitle(ax, "Median of repeated runs; label shows images/s and speedup over sequential")
    fig.tight_layout()
    path = out / "throughput.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def plot_latency(summary: list[dict], results: Path, out: Path) -> Path | None:
    configs = [(r["label"], results / r["metrics_file"]) for r in summary if (results / r["metrics_file"]).exists()]
    if not configs:
        return None
    process = [column(read_csv(p), "process_time_ms") for _, p in configs]
    total = [column(read_csv(p), "total_latency_ms") for _, p in configs]
    labels = ["seq" if lbl == "seq" else f"{lbl}" for lbl, _ in configs]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    panels = ((axes[0], process, "Processing time per image (ORB + annotate)", "ms"),
              (axes[1], total, "End-to-end latency per image (read to written)", "ms"))
    for ax, data, title, unit in panels:
        ax.boxplot(data, widths=0.55, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor="#cde2fb", edgecolor=SERIES[0], linewidth=1.2),
                   medianprops=dict(color=SERIES[0], linewidth=2),
                   whiskerprops=dict(color=INK_2, linewidth=1), capprops=dict(color=INK_2, linewidth=1))
        x = np.arange(1, len(data) + 1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        for q, marker, color, name in ((95, "^", SERIES[1], "p95"), (99, "D", INK, "p99")):
            ax.scatter(x, [np.percentile(d, q) for d in data], marker=marker, s=46, color=color,
                       edgecolor=SURFACE, linewidth=1.5, zorder=5, label=name)
        ax.set_title(title, fontsize=11.5)
        ax.set_xlabel("Worker threads")
        ax.set_ylabel(unit)
        ax.set_ylim(bottom=0)
        ax.grid(axis="x", visible=False)
    axes[0].plot([], [], color=SERIES[0], linewidth=2, label="p50 (box: p25-p75)")
    handles, lbls = axes[0].get_legend_handles_labels()
    order = [lbls.index(n) for n in ("p50 (box: p25-p75)", "p95", "p99")]
    fig.legend([handles[i] for i in order], [lbls[i] for i in order], loc="upper right", ncol=3,
               bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("Latency percentiles by thread count", x=0.01, ha="left", fontsize=13, fontweight="semibold",
                 color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = out / "latency_percentiles.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def plot_stage_breakdown(summary: list[dict], results: Path, out: Path) -> Path | None:
    stages = [("read_time_ms", "Read + decode"), ("queue_wait_ms", "Input queue wait"),
              ("process_time_ms", "ORB + annotate"), ("output_wait_ms", "Output queue wait"),
              ("write_time_ms", "Write")]
    colors = [SERIES[1], "#b7d3f6", SERIES[0], "#e4e3df", SERIES[2]]
    configs = [(r["label"], results / r["metrics_file"]) for r in summary if (results / r["metrics_file"]).exists()]
    if not configs:
        return None
    means = np.array([[column(read_csv(p), key).mean() for key, _ in stages] for _, p in configs])

    fig, ax = plt.subplots(figsize=(10, 4.8))
    y = np.arange(len(configs))[::-1]
    left = np.zeros(len(configs))
    for i, (_, name) in enumerate(stages):
        ax.barh(y, means[:, i], left=left, color=colors[i], height=0.62, edgecolor=SURFACE, linewidth=2, label=name)
        left += means[:, i]
    ax.set_yticks(y)
    ax.set_yticklabels(["Sequential" if lbl == "seq" else f"{lbl} threads" for lbl, _ in configs])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean time per image (ms)")
    ax.set_title("Where each image's time goes")
    subtitle(ax, "Queue waits grow with backpressure; the serial read/write stages bound total throughput")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=5)
    fig.tight_layout()
    path = out / "stage_breakdown.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Image inspection
# ---------------------------------------------------------------------------

def load_manifest(data: Path | None) -> dict[str, str]:
    if data is None or not (data / "manifest.csv").exists():
        return {}
    return {r["filename"]: r["degradation"] for r in read_csv(data / "manifest.csv")}


def plot_quality(metrics_path: Path, manifest: dict[str, str], out: Path) -> tuple[Path, list[dict]]:
    rows = read_csv(metrics_path)
    kps = column(rows, "num_keypoints")
    sharp = column(rows, "sharpness")
    bright = column(rows, "brightness")
    contrast = column(rows, "contrast")
    names = [r["filename"] for r in rows]
    labels = [manifest.get(Path(n).name, "unlabelled") for n in names]

    # Flag images whose sharpness is far below the dataset median (blur) or
    # whose exposure/contrast is extreme.
    sharp_threshold = float(np.median(sharp)) * 0.2
    flagged = []
    for i, name in enumerate(names):
        reasons = []
        if sharp[i] < sharp_threshold:
            reasons.append("blurry")
        if bright[i] < 50:
            reasons.append("under-exposed")
        if contrast[i] < 20:
            reasons.append("low contrast")
        if reasons:
            flagged.append({"filename": name, "reasons": ", ".join(reasons), "keypoints": int(kps[i]),
                            "sharpness": sharp[i], "truth": labels[i]})

    groups = [("clean", "Clean", NEUTRAL), ("blur", "Blurred", SERIES[0]), ("dark", "Under-exposed", SERIES[1]),
              ("low_contrast", "Low contrast", SERIES[2])]
    if not manifest:
        groups = [("unlabelled", "Images", SERIES[0])]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9))
    ax = axes[0]
    for key, name, color in groups:
        idx = [i for i, lbl in enumerate(labels) if lbl == key]
        if idx:
            ax.scatter(sharp[idx], kps[idx], s=38, color=color, edgecolor=SURFACE, linewidth=1.2,
                       label=f"{name} ({len(idx)})", alpha=0.95)
    ax.axvline(sharp_threshold, color=INK_2, linewidth=1, linestyle=":")
    ax.text(sharp_threshold, ax.get_ylim()[1], " blur threshold", fontsize=8.5, color=INK_2, va="top")
    ax.set_xscale("log")
    ax.set_xlabel("Sharpness (variance of Laplacian, log)")
    ax.set_ylabel("ORB keypoints detected")
    ax.set_title("Keypoints vs. sharpness", fontsize=11.5)
    ax.legend(loc="lower right")

    ax = axes[1]
    for key, name, color in groups:
        idx = [i for i, lbl in enumerate(labels) if lbl == key]
        if idx:
            ax.scatter(bright[idx], contrast[idx], s=38, color=color, edgecolor=SURFACE, linewidth=1.2, label=name)
    ax.set_xlabel("Mean brightness (0-255)")
    ax.set_ylabel("Contrast (intensity std. dev.)")
    ax.set_title("Exposure", fontsize=11.5)

    ax = axes[2]
    ax.hist(kps, bins=30, color=SERIES[0], edgecolor=SURFACE, linewidth=1.5)
    ax.axvline(np.median(kps), color=INK, linewidth=1.2, linestyle="--")
    ax.text(np.median(kps), ax.get_ylim()[1], f" median {np.median(kps):.0f}", fontsize=9, color=INK, va="top")
    ax.set_xlabel("ORB keypoints per image")
    ax.set_ylabel("Images")
    ax.set_title("Keypoint distribution", fontsize=11.5)
    ax.grid(axis="x", visible=False)

    fig.suptitle(f"Image quality inspection - {len(rows)} images, {len(flagged)} flagged",
                 x=0.01, ha="left", fontsize=13, fontweight="semibold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = out / "image_quality.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path, flagged


def plot_feature_maps(metrics_path: Path, images: Path, data: Path | None, manifest: dict[str, str],
                      out: Path, samples: int) -> Path | None:
    try:
        import cv2
    except ImportError:
        print("  (skipping feature_maps.png: opencv-python not installed)")
        return None
    rows = read_csv(metrics_path)
    rows.sort(key=lambda r: int(r["num_keypoints"]))
    # Show the feature-richest images, the median one, and the weakest ones:
    # the weakest are usually the degraded images the quality panel flags.
    picks_idx = sorted({len(rows) - 1, len(rows) - 2, len(rows) // 2, 1, 0})[-samples:] if len(rows) >= 5 \
        else list(range(len(rows)))
    picks = [rows[i] for i in reversed(picks_idx)]

    def stem(name: str) -> str:
        return str(Path(name).with_suffix("")).replace("\\", "/").replace("/", "_")

    def load_rgb(path: Path):
        img = cv2.imread(str(path))
        return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(len(picks), 3, figsize=(13.5, 2.75 * len(picks)))
    axes = np.atleast_2d(axes)
    for row_axes, r in zip(axes, picks):
        panels = [
            (data / r["filename"] if data else None, "Original"),
            (images / f"{stem(r['filename'])}_keypoints.jpg", f"ORB keypoints: {r['num_keypoints']}"),
            (images / f"{stem(r['filename'])}_featuremap.jpg", "Keypoint density"),
        ]
        for ax, (path, title) in zip(row_axes, panels):
            ax.axis("off")
            img = load_rgb(path) if path is not None and path.exists() else None
            if img is None:
                ax.text(0.5, 0.5, "not available", ha="center", va="center", color=INK_2, transform=ax.transAxes)
            else:
                ax.imshow(img)
            ax.set_title(title, fontsize=10, loc="left", color=INK, pad=4)
        truth = manifest.get(Path(r["filename"]).name)
        row_axes[0].text(0, -0.04, r["filename"] + (f"  ({truth})" if truth and truth != "clean" else ""),
                         transform=row_axes[0].transAxes, fontsize=8.5, color=INK_2, va="top")
    fig.suptitle("Feature maps - richest, median and weakest images", x=0.01, ha="left", fontsize=13,
                 fontweight="semibold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = out / "feature_maps.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("results"), help="benchmark.py output directory")
    parser.add_argument("--images", type=Path, default=None,
                        help="directory of *_keypoints.jpg / *_featuremap.jpg written with --output --feature-maps")
    parser.add_argument("--data", type=Path, default=None, help="original dataset (for originals + manifest.csv)")
    parser.add_argument("--metrics", type=Path, default=None,
                        help="metrics CSV for the quality panels (default: results/metrics_seq.csv)")
    parser.add_argument("--out", type=Path, default=Path("assets"), help="where to write the PNG charts")
    parser.add_argument("--samples", type=int, default=5, help="rows in the feature-map gallery")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    summary_path = args.results / "benchmark_summary.csv"
    if summary_path.exists():
        summary = load_summary(args.results)
        info = load_system_info(args.results)
        written.append(plot_speedup(summary, info, args.out))
        written.append(plot_throughput(summary, args.out))
        written += [p for p in (plot_latency(summary, args.results, args.out),
                                plot_stage_breakdown(summary, args.results, args.out)) if p]
    else:
        print(f"  (no {summary_path}; skipping performance charts)")

    metrics = args.metrics or args.results / "metrics_seq.csv"
    manifest = load_manifest(args.data)
    if metrics.exists():
        quality_path, flagged = plot_quality(metrics, manifest, args.out)
        written.append(quality_path)
        if flagged:
            print(f"\n{len(flagged)} image(s) flagged for quality review:")
            for f in flagged[:15]:
                truth = f" [ground truth: {f['truth']}]" if f["truth"] not in ("unlabelled",) else ""
                print(f"  {f['filename']:<28} {f['reasons']:<28} keypoints={f['keypoints']:<5}{truth}")
            if len(flagged) > 15:
                print(f"  ... and {len(flagged) - 15} more")
            if manifest:
                degraded = {n for n, lbl in manifest.items() if lbl != "clean"}
                hits = sum(1 for f in flagged if Path(f["filename"]).name in degraded)
                print(f"  flagged {hits}/{len(degraded)} deliberately degraded images, "
                      f"{len(flagged) - hits} false positives")
        if args.images and args.images.is_dir():
            fm = plot_feature_maps(metrics, args.images, args.data, manifest, args.out, args.samples)
            if fm:
                written.append(fm)
    else:
        print(f"  (no {metrics}; skipping image-quality charts)")

    print("\nwrote:")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
