"""
EPMO GPU Benchmark — epmo_gpu_benchmark.py
===========================================
Hardware-aware DataLoader benchmarking script.
Works correctly on BOTH CPU-only and GPU-enabled environments.

Key fixes vs. the CPU version (epmo_cpu_benchmark.py):
  1. GPU WARMUP: adds a forward-pass warm-up on GPU before timing begins,
     so CUDA JIT compilation doesn't inflate the first measurement.
  2. WORKER CAP: automatically caps num_workers to cpu_logical on GPU
     environments where PyTorch's own warning fires (avoids slowdown/freeze).
  3. prefetch_factor guard: only set when num_workers > 0 (already in cpu
     version — kept here too).
  4. persistent_workers guard: same as above.
  5. Adds gpu_vram_gb and gpu_name to every CSV row for richer features.
  6. Adds a GPU-specific timing note in the summary.

Output columns (identical schema to cpu version):
  env_id, cpu_logical, cpu_physical, ram_gb,
  gpu_available, gpu_vram_gb,
  num_workers, batch_size,
  run1, run2, run3, load_time_median

Usage
-----
  # Colab / GPU machine
  !python epmo_gpu_benchmark.py --env colab_t4

  # Override grid
  !python epmo_gpu_benchmark.py --env colab_a100 --workers 0,2,4 --batches 32,64,128,256

GPU Environment Check Commands (run these FIRST in Colab/terminal):
--------------------------------------------------------------------
  # 1. Check if CUDA is visible to Python / PyTorch
  python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

  # 2. Check nvidia-smi (shows GPU model, VRAM, driver version)
  !nvidia-smi

  # 3. Check CUDA version
  !nvcc --version

  # 4. In Google Colab — enable GPU runtime:
  #    Runtime → Change runtime type → Hardware accelerator → GPU → Save
  #    Then verify with:
  !nvidia-smi

  # 5. In Kaggle — enable GPU runtime:
  #    Right panel → Accelerator → GPU → Save version

=============================================================
"""

import os
import gc
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

# Detect GPU once at module level
HAS_GPU = torch.cuda.is_available()


# ══════════════════════════════════════════════
# GPU CHECK UTILITY  (run standalone to verify)
# ══════════════════════════════════════════════
def print_gpu_status():
    """
    Print a clear GPU availability report.
    Run this first to confirm your environment has GPU access before benchmarking.
    """
    print("\n" + "=" * 55)
    print("GPU ENVIRONMENT CHECK")
    print("=" * 55)
    print(f"  PyTorch version   : {torch.__version__}")
    print(f"  CUDA available    : {HAS_GPU}")

    if HAS_GPU:
        print(f"  GPU name          : {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        vram_gb = round(props.total_memory / (1024 ** 3), 2)
        print(f"  GPU VRAM          : {vram_gb} GB")
        print(f"  CUDA version      : {torch.version.cuda}")
        print(f"  GPU count         : {torch.cuda.device_count()}")
        print(f"\n  ✓ GPU runtime is ACTIVE. Benchmark will use pin_memory=True.")
    else:
        print(f"\n  ✗ No GPU detected. Running on CPU only.")
        print(f"  To enable GPU in Google Colab:")
        print(f"    Runtime → Change runtime type → GPU → Save")
        print(f"  To enable GPU in Kaggle:")
        print(f"    Right panel → Accelerator → GPU P100 → Save version")

    print("=" * 55 + "\n")


# ══════════════════════════════════════════════
# HARDWARE PROFILE
# ══════════════════════════════════════════════
def get_hardware_profile() -> dict:
    """Collect complete hardware fingerprint."""
    profile = {}
    profile["cpu_logical"]  = psutil.cpu_count(logical=True)
    profile["cpu_physical"] = psutil.cpu_count(logical=False) or 1
    profile["ram_gb"]       = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    profile["os"]           = platform.system()
    profile["python"]       = platform.python_version()
    profile["torch"]        = torch.__version__

    if HAS_GPU:
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


# ══════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════
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


# ══════════════════════════════════════════════
# GPU WARM-UP  (critical for accurate GPU timing)
# ══════════════════════════════════════════════
def warmup_gpu():
    """
    Run a tiny dummy forward pass to trigger CUDA JIT compilation.
    Without this, the very first timed run is artificially slow
    because PyTorch compiles CUDA kernels on first use.
    """
    if not HAS_GPU:
        return
    print("  [GPU] Running CUDA warm-up pass (eliminates JIT compilation cost) ...")
    dummy = torch.randn(32, 3, 32, 32, device="cuda")
    model = torch.nn.Conv2d(3, 16, 3).cuda()
    with torch.no_grad():
        _ = model(dummy)
    torch.cuda.synchronize()
    del dummy, model
    torch.cuda.empty_cache()
    print("  [GPU] Warm-up done.\n")


# ══════════════════════════════════════════════
# WORKER CAP  (GPU-specific safety)
# ══════════════════════════════════════════════
def safe_worker_count(requested: int, cpu_logical: int) -> int:
    """
    On GPU Colab instances (typically 2 logical CPUs), PyTorch warns and may
    freeze when num_workers > cpu_logical.  This function caps the request
    silently so every config still runs — and records what was actually used.
    """
    if HAS_GPU and requested > cpu_logical:
        capped = cpu_logical
        print(f"  [GPU] Capping num_workers {requested} → {capped} "
              f"(max safe for {cpu_logical} logical CPUs on this GPU instance)")
        return capped
    return requested


# ══════════════════════════════════════════════
# SINGLE CONFIG TIMING
# ══════════════════════════════════════════════
def time_one_config(dataset, num_workers: int, batch_size: int) -> float:
    """
    Accurately time a single (workers, batch_size) config.

    Protocol:
      1. Build loader with GPU-safe settings
      2. Run WARMUP_BATCHES batches (discard — let OS/CUDA cache settle)
      3. CUDA sync if GPU present
      4. Time exactly MEASURE_BATCHES batches
      5. CUDA sync
      6. Return wall-clock seconds (4 decimal places)
    """
    # Only set prefetch_factor and persistent_workers when workers > 0
    # (setting them with workers=0 raises a ValueError in PyTorch ≥ 1.8)
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=HAS_GPU,          # GPU: pin for faster host→device transfer
        drop_last=False,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"]    = 2

    loader = DataLoader(dataset, **loader_kwargs)

    # ── warmup ──
    for i, _ in enumerate(loader):
        if i + 1 >= WARMUP_BATCHES:
            break
    if HAS_GPU:
        torch.cuda.synchronize()

    # ── measure ──
    start = time.perf_counter()
    for i, _ in enumerate(loader):
        if i + 1 >= MEASURE_BATCHES:
            break
    if HAS_GPU:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # Release worker processes before the next config
    del loader
    gc.collect()

    return round(elapsed, 4)


# ══════════════════════════════════════════════
# BENCHMARK LOOP
# ══════════════════════════════════════════════
def run_benchmarks(dataset, hw: dict, env_id: str) -> pd.DataFrame:
    """
    Run all (workers × batch_size) combinations RUNS_PER_CONFIG times each.
    Returns a DataFrame with one row per config (median across runs).
    """
    cpu_logical = hw["cpu_logical"]
    records = []
    total   = len(WORKER_OPTIONS) * len(BATCH_OPTIONS)
    done    = 0

    print(f"\n  Grid: {len(WORKER_OPTIONS)} worker options × "
          f"{len(BATCH_OPTIONS)} batch options = {total} configs")
    print(f"  Each config repeated {RUNS_PER_CONFIG}× → {total * RUNS_PER_CONFIG} total timed runs")
    print(f"  GPU environment: {'YES — pin_memory=True' if HAS_GPU else 'NO — CPU only'}\n")

    for workers_requested in WORKER_OPTIONS:
        workers = safe_worker_count(workers_requested, cpu_logical)

        for batch in BATCH_OPTIONS:
            done += 1
            run_times = []

            for run in range(RUNS_PER_CONFIG):
                t = time_one_config(dataset, workers, batch)
                run_times.append(t)
                print(f"  [{done:02d}/{total}] workers={workers} (req={workers_requested}), "
                      f"batch={batch:3d}  run{run+1}/{RUNS_PER_CONFIG}: {t:.4f}s")

            median_t = round(float(np.median(run_times)), 4)

            records.append({
                # ── identifiers ──
                "env_id":           env_id,
                # ── hardware features ──
                "cpu_logical":      hw["cpu_logical"],
                "cpu_physical":     hw["cpu_physical"],
                "ram_gb":           hw["ram_gb"],
                "gpu_available":    hw["gpu_available"],
                "gpu_vram_gb":      hw["gpu_vram_gb"],
                # ── config parameters ──
                "num_workers":      workers,          # actual used (may be capped)
                "num_workers_req":  workers_requested, # what was requested
                "batch_size":       batch,
                # ── raw runs ──
                "run1":             run_times[0],
                "run2":             run_times[1],
                "run3":             run_times[2],
                # ── target variable ──
                "load_time_median": median_t,
            })
            print(f"         → median: {median_t:.4f}s\n")

    df = pd.DataFrame(records)
    return df


# ══════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════
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

    best_row  = df.loc[df["load_time_median"].idxmin()]
    worst_row = df.loc[df["load_time_median"].idxmax()]

    print(f"\n  ✓ BEST  config : workers={int(best_row['num_workers'])}, "
          f"batch={int(best_row['batch_size'])}, "
          f"time={best_row['load_time_median']:.4f}s")
    print(f"  ✗ WORST config : workers={int(worst_row['num_workers'])}, "
          f"batch={int(worst_row['batch_size'])}, "
          f"time={worst_row['load_time_median']:.4f}s")
    print(f"\n  Range: {worst_row['load_time_median'] / best_row['load_time_median']:.1f}× "
          f"difference between best and worst config")

    default = df[(df["num_workers"] == 0) & (df["batch_size"] == 32)]
    if not default.empty:
        default_time = default["load_time_median"].values[0]
        savings_pct  = (default_time - best_row["load_time_median"]) / default_time * 100
        print(f"\n  Default (workers=0, batch=32) time : {default_time:.4f}s")
        if savings_pct > 0:
            print(f"  Potential saving vs default        : {savings_pct:.1f}%")
        else:
            print(f"  Default IS optimal on this hardware (savings=0%)")
            print(f"  → EPMO finding: workers=0 wins here (typical on GPU environments")
            print(f"    where GPU compute dominates and workers add overhead).")

    if HAS_GPU:
        print(f"\n  [GPU NOTE] On GPU environments, workers=0 often wins because")
        print(f"  the GPU is the bottleneck, not the DataLoader. Low worker counts")
        print(f"  also reduce CPU-GPU synchronization overhead.")

    print("=" * 55)


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="EPMO GPU Benchmark — collect hardware-aware DataLoader data"
    )
    parser.add_argument(
        "--env", type=str, required=True,
        help="Environment label (e.g. colab_t4, kaggle_p100)"
    )
    parser.add_argument(
        "--workers", type=str, default=None,
        help="Override worker options, comma-separated (e.g. '0,1,2,4')"
    )
    parser.add_argument(
        "--batches", type=str, default=None,
        help="Override batch sizes, comma-separated (e.g. '32,64,128')"
    )
    parser.add_argument(
        "--check-gpu", action="store_true",
        help="Only print GPU status and exit (good for verifying environment)"
    )
    args = parser.parse_args()

    # ── GPU-only check mode ──
    if args.check_gpu:
        print_gpu_status()
        return

    # Allow overriding grid from command line
    global WORKER_OPTIONS, BATCH_OPTIONS
    if args.workers:
        WORKER_OPTIONS = [int(x) for x in args.workers.split(",")]
    if args.batches:
        BATCH_OPTIONS  = [int(x) for x in args.batches.split(",")]

    env_id   = args.env.strip().lower().replace(" ", "_")
    out_file = f"epmo_data_{env_id}.csv"
    hw_file  = f"epmo_hw_{env_id}.json"

    print("=" * 55)
    print(f"EPMO Benchmark v2.1-GPU  |  Environment: {env_id}")
    print("=" * 55)

    # ── 0. GPU status ──
    print_gpu_status()

    # ── 1. Hardware profile ──
    print("[1/5] Profiling hardware ...")
    hw = get_hardware_profile()
    for k, v in hw.items():
        print(f"  {k}: {v}")

    with open(hw_file, "w") as f:
        json.dump(hw, f, indent=2)
    print(f"  → Saved: {hw_file}")

    # ── 2. Load dataset ──
    print("\n[2/5] Preparing dataset ...")
    dataset = load_dataset()

    # ── 3. GPU warm-up (no-op on CPU) ──
    print("\n[3/5] GPU warm-up ...")
    warmup_gpu()

    # ── 4. Run benchmarks ──
    print("[4/5] Running benchmarks ...")
    df = run_benchmarks(dataset, hw, env_id)

    # ── 5. Save results ──
    print("\n[5/5] Saving results ...")
    df.to_csv(out_file, index=False)
    print(f"  → Saved: {out_file}  ({len(df)} rows)")

    # ── Summary ──
    summarize_results(df, hw)

    print(f"\n  Next step: Run this script on another machine/Colab")
    print(f"  Then combine CSVs with epmo_merge_data.py and run epmo_train.py")
    print()


if __name__ == "__main__":
    main()
