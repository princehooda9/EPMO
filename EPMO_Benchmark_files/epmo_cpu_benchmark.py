"""
Output columns:
  env_id, cpu_logical, cpu_physical, ram_gb,
  gpu_available, gpu_vram_gb,
  num_workers, batch_size,
  run1, run2, run3, load_time_median
=============================================================
"""

import os
import time
import argparse
import platform
import json
import psutil
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# CONFIG — adjust ONLY these values if needed
# ─────────────────────────────────────────────
WORKER_OPTIONS  = [0, 1, 2, 4]       # num_workers to test
BATCH_OPTIONS   = [32, 64, 128, 256]  # batch sizes to test
WARMUP_BATCHES  = 5                   # throw-away batches before timing
MEASURE_BATCHES = 30                  # batches to time
RUNS_PER_CONFIG = 3                   # repeat each config N times → take median
DATA_DIR        = "./data"            # CIFAR-10 download dir
# ─────────────────────────────────────────────


def get_hardware_profile() -> dict:
    """Collect complete hardware fingerprint."""
    profile = {}
    profile["cpu_logical"]  = psutil.cpu_count(logical=True)
    profile["cpu_physical"] = psutil.cpu_count(logical=False) or 1
    profile["ram_gb"]       = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    profile["os"]           = platform.system()
    profile["python"]       = platform.python_version()
    profile["torch"]        = torch.__version__

    if torch.cuda.is_available():
        profile["gpu_available"] = 1
        profile["gpu_name"]      = torch.cuda.get_device_name(0)
        profile["gpu_vram_gb"]   = round(
            torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2
        )
        profile["cuda_version"]  = torch.version.cuda
    else:
        profile["gpu_available"] = 0
        profile["gpu_name"]      = "none"
        profile["gpu_vram_gb"]   = 0.0
        profile["cuda_version"]  = "none"

    return profile


def load_dataset():
    """Download CIFAR-10 once and return dataset object."""
    print("  Loading CIFAR-10 dataset (downloads once) ...")
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform
    )
    print(f"  Dataset ready: {len(dataset)} samples")
    return dataset


def time_one_config(dataset, num_workers: int, batch_size: int) -> float:
    """
    Accurately time a single (workers, batch_size) config.
    
    Protocol:
      1. Build loader
      2. Run WARMUP_BATCHES batches (discard — let OS cache settle)
      3. CUDA sync if GPU present
      4. Time exactly MEASURE_BATCHES batches
      5. CUDA sync
      6. Return wall-clock seconds (4 decimal places)
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),   # keep workers alive between runs
        prefetch_factor=2 if num_workers > 0 else None,
    )

    # ── warmup ──
    for i, _ in enumerate(loader):
        if i + 1 >= WARMUP_BATCHES:
            break

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # ── measure ──
    start = time.perf_counter()
    for i, _ in enumerate(loader):
        if i + 1 >= MEASURE_BATCHES:
            break
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # Explicitly delete loader to release worker processes before next config
    del loader

    return round(elapsed, 4)


def run_benchmarks(dataset, hw: dict, env_id: str) -> pd.DataFrame:
    """
    Run all (workers × batch_size) combinations RUNS_PER_CONFIG times each.
    Returns a DataFrame with one row per (config × run), plus medians.
    """
    records = []
    total   = len(WORKER_OPTIONS) * len(BATCH_OPTIONS)
    done    = 0

    print(f"\n  Grid: {len(WORKER_OPTIONS)} worker options × "
          f"{len(BATCH_OPTIONS)} batch options = {total} configs")
    print(f"  Each config repeated {RUNS_PER_CONFIG}× → {total * RUNS_PER_CONFIG} total timed runs")
    print(f"  (This takes roughly {total * RUNS_PER_CONFIG * 15 // 60}–"
          f"{total * RUNS_PER_CONFIG * 25 // 60} minutes)\n")

    for workers in WORKER_OPTIONS:
        for batch in BATCH_OPTIONS:
            done += 1
            run_times = []

            for run in range(RUNS_PER_CONFIG):
                t = time_one_config(dataset, workers, batch)
                run_times.append(t)
                print(f"  [{done:02d}/{total}] workers={workers}, batch={batch:3d} "
                      f"run{run+1}/{RUNS_PER_CONFIG}: {t:.4f}s")

            median_t = round(float(np.median(run_times)), 4)

            records.append({
                # ── identifiers ──
                "env_id":          env_id,
                # ── hardware features ──
                "cpu_logical":     hw["cpu_logical"],
                "cpu_physical":    hw["cpu_physical"],
                "ram_gb":          hw["ram_gb"],
                "gpu_available":   hw["gpu_available"],
                "gpu_vram_gb":     hw["gpu_vram_gb"],
                # ── config parameters ──
                "num_workers":     workers,
                "batch_size":      batch,
                # ── raw runs ──
                "run1":            run_times[0],
                "run2":            run_times[1],
                "run3":            run_times[2],
                # ── target variable ──
                "load_time_median": median_t,
            })
            print(f"         → median: {median_t:.4f}s\n")

    df = pd.DataFrame(records)
    return df


def summarize_results(df: pd.DataFrame, hw: dict):
    """Print a human-readable summary of benchmark results."""
    print("\n" + "=" * 55)
    print("BENCHMARK SUMMARY")
    print("=" * 55)
    print(f"  CPU logical cores : {hw['cpu_logical']}")
    print(f"  CPU physical cores: {hw['cpu_physical']}")
    print(f"  RAM               : {hw['ram_gb']} GB")
    print(f"  GPU               : {hw['gpu_name']}")
    print(f"  GPU VRAM          : {hw['gpu_vram_gb']} GB")
    print()

    pivot = df.pivot_table(
        index="num_workers",
        columns="batch_size",
        values="load_time_median"
    )
    print("  Median load time (seconds) — workers × batch_size:")
    print(pivot.to_string())

    best_row = df.loc[df["load_time_median"].idxmin()]
    worst_row = df.loc[df["load_time_median"].idxmax()]

    print(f"\n  ✓ BEST  config : workers={int(best_row['num_workers'])}, "
          f"batch={int(best_row['batch_size'])}, "
          f"time={best_row['load_time_median']:.4f}s")
    print(f"  ✗ WORST config : workers={int(worst_row['num_workers'])}, "
          f"batch={int(worst_row['batch_size'])}, "
          f"time={worst_row['load_time_median']:.4f}s")
    print(f"\n  Range: {worst_row['load_time_median'] / best_row['load_time_median']:.1f}× "
          f"difference between best and worst config")

    # Default config (what PyTorch users typically use)
    default = df[(df["num_workers"] == 0) & (df["batch_size"] == 32)]
    if not default.empty:
        default_time = default["load_time_median"].values[0]
        savings_pct = (default_time - best_row["load_time_median"]) / default_time * 100
        print(f"\n  Default (workers=0, batch=32) time : {default_time:.4f}s")
        if savings_pct > 0:
            print(f"  Potential saving vs default        : {savings_pct:.1f}%")
        else:
            print(f"  Default IS optimal on this hardware (savings=0%)")
            print(f"  → This is an important finding: on this machine,")
            print(f"    workers=0 wins. EPMO confirms it automatically.")

    print("=" * 55)


def main():
    parser = argparse.ArgumentParser(
        description="EPMO Benchmark — collect hardware-aware DataLoader data"
    )
    parser.add_argument(
        "--env", type=str, required=True,
        help="Environment label (e.g. laptop, colab1, colab2)"
    )
    parser.add_argument(
        "--workers", type=str, default=None,
        help="Override worker options, comma-separated (e.g. '0,1,2,4')"
    )
    parser.add_argument(
        "--batches", type=str, default=None,
        help="Override batch sizes, comma-separated (e.g. '32,64,128')"
    )
    args = parser.parse_args()

    # Allow overriding grid from command line
    global WORKER_OPTIONS, BATCH_OPTIONS
    if args.workers:
        WORKER_OPTIONS = [int(x) for x in args.workers.split(",")]
    if args.batches:
        BATCH_OPTIONS = [int(x) for x in args.batches.split(",")]

    env_id = args.env.strip().lower().replace(" ", "_")
    out_file = f"epmo_data_{env_id}.csv"
    hw_file  = f"epmo_hw_{env_id}.json"

    print("=" * 55)
    print(f"EPMO Benchmark v2.0  |  Environment: {env_id}")
    print("=" * 55)

    # ── 1. Hardware profile ──
    print("\n[1/4] Profiling hardware ...")
    hw = get_hardware_profile()
    for k, v in hw.items():
        print(f"  {k}: {v}")

    # Save hardware profile
    with open(hw_file, "w") as f:
        json.dump(hw, f, indent=2)
    print(f"  → Saved: {hw_file}")

    # ── 2. Load dataset ──
    print("\n[2/4] Preparing dataset ...")
    dataset = load_dataset()

    # ── 3. Run benchmarks ──
    print("\n[3/4] Running benchmarks ...")
    df = run_benchmarks(dataset, hw, env_id)

    # ── 4. Save results ──
    print("\n[4/4] Saving results ...")
    df.to_csv(out_file, index=False)
    print(f"  → Saved: {out_file}  ({len(df)} rows)")

    # ── Summary ──
    summarize_results(df, hw)

    print(f"\n  Next step: Run this script on another machine/Colab")
    print(f"  Then combine CSVs and run: python epmo_train.py")
    print()


if __name__ == "__main__":
    main()
