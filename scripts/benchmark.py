#!/usr/bin/env python3
"""Benchmark the pipeline across worker-thread counts.

Runs the compiled binary once in ``--sequential`` mode (single thread, no
queues: the true baseline) and then in parallel mode with 1, 2, 4, 8 and 16
worker threads. Each configuration is repeated and the median wall-clock time
is used. Speedup is reported against the sequential baseline and against the
1-worker pipeline.

Outputs (in --results):
    benchmark_summary.csv   one row per configuration
    metrics_seq.csv         per-image metrics of the sequential run
    metrics_t<N>.csv        per-image metrics of the N-worker run
    system_info.json        CPU / host details, for reproducibility

Example:
    python3 scripts/benchmark.py --binary ./build/image_pipeline --data ./data/test_set
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path


def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def physical_cores() -> int | None:
    try:
        with open("/proc/cpuinfo") as f:
            pairs = set()
            phys = core = None
            for line in f:
                if line.startswith("physical id"):
                    phys = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    core = line.split(":", 1)[1].strip()
                elif not line.strip() and core is not None:
                    pairs.add((phys, core))
                    phys = core = None
            return len(pairs) or None
    except OSError:
        return None


def run_once(binary: Path, data: Path, metrics: Path, threads: int | None, queue_size: int,
             output: Path | None, feature_maps: bool) -> tuple[float, dict]:
    cmd = [str(binary), "--input", str(data), "--metrics", str(metrics), "--queue-size", str(queue_size), "--json"]
    if threads is None:
        cmd.append("--sequential")
    else:
        cmd += ["--threads", str(threads)]
    if output is not None:
        if output.exists():
            shutil.rmtree(output)
        cmd += ["--output", str(output)]
        if feature_maps:
            cmd.append("--feature-maps")

    start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall_s = time.perf_counter() - start
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"benchmark run failed (exit {proc.returncode}): {' '.join(cmd)}")
    summary_line = next((ln for ln in reversed(proc.stdout.splitlines()) if ln.startswith("{")), None)
    if summary_line is None:
        raise SystemExit(f"no JSON summary in output of: {' '.join(cmd)}")
    return wall_s, json.loads(summary_line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, default=Path("build/image_pipeline"))
    parser.add_argument("--data", type=Path, default=Path("data/test_set"))
    parser.add_argument("--results", type=Path, default=Path("results"), help="directory for CSV outputs")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--repeats", type=int, default=3, help="runs per configuration; the median is used")
    parser.add_argument("--queue-size", type=int, default=32)
    parser.add_argument("--save-images", action="store_true",
                        help="also write annotated images + feature maps (adds serial JPEG encoding to the writer)")
    args = parser.parse_args()

    if not args.binary.exists():
        parser.error(f"binary not found: {args.binary} (build it with cmake first)")
    if not args.data.is_dir():
        parser.error(f"dataset not found: {args.data} (run generate_synthetic_dataset.py first)")
    args.results.mkdir(parents=True, exist_ok=True)
    images_out = args.results / "images" if args.save_images else None

    info = {
        "cpu_model": cpu_model(),
        "logical_cpus": os.cpu_count(),
        "physical_cores": physical_cores(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "repeats": args.repeats,
        "queue_size": args.queue_size,
        "save_images": args.save_images,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (args.results / "system_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"CPU: {info['cpu_model']} ({info['physical_cores']} cores / {info['logical_cpus']} threads)")

    # Warm the OS page cache so the first configuration is not penalised by
    # cold disk reads that later ones would not pay.
    print("warm-up run ...")
    run_once(args.binary, args.data, args.results / "warmup.csv", max(args.threads), args.queue_size, None, False)
    (args.results / "warmup.csv").unlink(missing_ok=True)

    configs: list[tuple[str, int | None]] = [("sequential", None)] + [("parallel", t) for t in args.threads]
    rows = []
    for mode, threads in configs:
        label = "seq" if threads is None else f"t{threads}"
        metrics = args.results / f"metrics_{label}.csv"
        walls, summaries = [], []
        for _ in range(args.repeats):
            wall_s, summary = run_once(args.binary, args.data, metrics, threads, args.queue_size, images_out,
                                       args.save_images)
            walls.append(wall_s)
            summaries.append(summary)
        median_wall = statistics.median(walls)
        # Report latency figures from the repeat closest to the median time.
        rep = summaries[min(range(len(walls)), key=lambda i: abs(walls[i] - median_wall))]
        images = rep["images_processed"]
        rows.append({
            "mode": mode,
            "threads": 1 if threads is None else threads,
            "images": images,
            "wall_s_median": median_wall,
            "wall_s_min": min(walls),
            "wall_s_stdev": statistics.stdev(walls) if len(walls) > 1 else 0.0,
            "throughput_ips": images / median_wall,
            "pipeline_throughput_ips": rep["throughput_ips"],
            "mean_process_ms": rep["mean_process_ms"],
            "p50_latency_ms": rep["p50_latency_ms"],
            "p95_latency_ms": rep["p95_latency_ms"],
            "p99_latency_ms": rep["p99_latency_ms"],
            "metrics_file": metrics.name,
        })
        print(f"  {mode:>10} threads={rows[-1]['threads']:>2}  median {median_wall:7.2f}s  "
              f"{rows[-1]['throughput_ips']:7.2f} img/s")

    baseline = rows[0]["throughput_ips"]
    one_worker = next(r["throughput_ips"] for r in rows if r["mode"] == "parallel" and r["threads"] == 1) \
        if any(r["mode"] == "parallel" and r["threads"] == 1 for r in rows) else baseline
    for r in rows:
        r["speedup_vs_sequential"] = r["throughput_ips"] / baseline
        r["speedup_vs_1_worker"] = r["throughput_ips"] / one_worker
        r["parallel_efficiency"] = r["speedup_vs_sequential"] / r["threads"]

    summary_path = args.results / "benchmark_summary.csv"
    fields = ["mode", "threads", "images", "wall_s_median", "wall_s_min", "wall_s_stdev", "throughput_ips",
              "pipeline_throughput_ips", "speedup_vs_sequential", "speedup_vs_1_worker", "parallel_efficiency",
              "mean_process_ms", "p50_latency_ms", "p95_latency_ms", "p99_latency_ms", "metrics_file"]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    print(f"\n{'mode':>10} {'thr':>4} {'wall(s)':>9} {'img/s':>8} {'speedup':>8} {'eff':>6} {'p99(ms)':>9}")
    for r in rows:
        print(f"{r['mode']:>10} {r['threads']:>4} {r['wall_s_median']:>9.2f} {r['throughput_ips']:>8.2f} "
              f"{r['speedup_vs_sequential']:>7.2f}x {r['parallel_efficiency']:>6.0%} {r['p99_latency_ms']:>9.1f}")
    best = max(rows, key=lambda r: r["speedup_vs_sequential"])
    print(f"\npeak speedup: {best['speedup_vs_sequential']:.2f}x over sequential at {best['threads']} worker threads")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
