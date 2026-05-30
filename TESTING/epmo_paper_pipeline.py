"""
=============================================================
EPMO Paper Pipeline  —  epmo_paper_pipeline.py
=============================================================
Purpose:
  Single-entry-point script for the paper's held-out test evaluation.

What it does:
  1. Auto-detects whether the current machine has a GPU or not
  2. Runs the appropriate benchmark script via subprocess
       CPU → epmo_cpu_benchmark.py
       GPU → epmo_gpu_benchmark.py
     This generates a real-measured CSV (one row per config)
  3. Loads that CSV automatically
  4. Loads the correct trained model  (model_cpu.pkl or model_gpu.pkl)
  5. Engineers the 11 features used in training
  6. Predicts load_time_median for all 16 configs → picks argmin
  7. Extracts Oracle (true best) and Default (workers=0, batch=32)
  8. Reports:
       - EPMO config  (num_workers, batch_size, measured time)
       - Oracle config (num_workers, batch_size, measured time)
       - Default config (num_workers, batch_size, measured time)
       - Saving % vs Default
       - Oracle Saving % vs Default
       - Regret % vs Oracle
       - PctOracle (% of oracle saving captured)
  9. Saves results to  epmo_test_results.json  for use by the
     case-study notebook (epmo_casestudies.ipynb)

Usage:
  python epmo_paper_pipeline.py --env ENV_LABEL
  python epmo_paper_pipeline.py --env ENV_LABEL 

Requirements (must be in the same folder):
  epmo_cpu_benchmark.py
  epmo_gpu_benchmark.py
  model_cpu.pkl
  model_gpu.pkl
=============================================================
"""

import os
import sys
import json
import pickle
import argparse
import subprocess
import numpy as np
import pandas as pd

# ── constants (must match epmo_train.py exactly) ──────────
WORKER_OPTIONS = [0, 1, 2, 4]
BATCH_OPTIONS  = [32, 64, 128, 256]
TARGET         = "load_time_median"
DEFAULT_W      = 0
DEFAULT_B      = 32

FEATS = [
    "cpu_logical", "ram_gb", "gpu_available", "gpu_vram_gb",
    "num_workers", "batch_size",
    "workers_x_cpu", "batch_x_ram", "cpu_per_worker",
    "workers_x_gpu", "batch_x_gpu",
]


# ══════════════════════════════════════════════════════════
# STEP 1 — GPU DETECTION
# ══════════════════════════════════════════════════════════

def detect_gpu() -> bool:
    """Return True if CUDA GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        print("  WARNING: PyTorch not found. Assuming CPU-only environment.")
        return False


# ══════════════════════════════════════════════════════════
# STEP 2 — RUN BENCHMARK VIA SUBPROCESS
# ══════════════════════════════════════════════════════════

def run_benchmark(has_gpu: bool, env_id: str) -> str:
    """
    Call the correct benchmark script as a subprocess.
    Returns the path to the generated CSV file.
    """
    script = "epmo_gpu_benchmark.py" if has_gpu else "epmo_cpu_benchmark.py"
    csv_path = f"epmo_data_{env_id}.csv"

    if not os.path.exists(script):
        sys.exit(f"ERROR: {script} not found in current directory.")

    print(f"\n[2/4] Running benchmark: {script} --env {env_id}")
    print(f"      This will take several minutes. Please wait.\n")
    print("=" * 55)

    result = subprocess.run(
        [sys.executable, script, "--env", env_id],
        check=True   # raises CalledProcessError if benchmark fails
    )

    print("=" * 55)

    if not os.path.exists(csv_path):
        sys.exit(f"ERROR: Benchmark completed but {csv_path} was not created.")

    print(f"\n  Benchmark CSV ready: {csv_path}")
    return csv_path


# ══════════════════════════════════════════════════════════
# STEP 3 — LOAD CSV + FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════

def load_and_engineer(csv_path: str) -> pd.DataFrame:
    """Load benchmark CSV and add the 11 training features."""
    df = pd.read_csv(csv_path)

    # Guard: add gpu_vram_gb if missing (CPU benchmark may omit it)
    if "gpu_vram_gb" not in df.columns:
        df["gpu_vram_gb"] = 0.0
    if "gpu_available" not in df.columns:
        df["gpu_available"] = 0

    # Engineer interaction features — identical to epmo_train.py
    df["workers_x_cpu"]  = df["num_workers"] * df["cpu_logical"]
    df["batch_x_ram"]    = df["batch_size"]  / df["ram_gb"]
    df["cpu_per_worker"] = df["cpu_logical"] / (df["num_workers"] + 1)
    df["workers_x_gpu"]  = df["num_workers"] * df["gpu_available"]
    df["batch_x_gpu"]    = df["batch_size"]  * df["gpu_available"]

    return df


# ══════════════════════════════════════════════════════════
# STEP 4 — LOAD MODEL + PREDICT
# ══════════════════════════════════════════════════════════

def load_model(has_gpu: bool):
    """Load the correct pkl model."""
    path = "model_gpu.pkl" if has_gpu else "model_cpu.pkl"
    if not os.path.exists(path):
        sys.exit(f"ERROR: {path} not found. Run epmo_train.py first.")
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"\n  Model loaded: {path}")
    return model


def predict_and_evaluate(df: pd.DataFrame, model) -> dict:
    """
    Run model over all 16 configs, find EPMO best, Oracle, Default.
    Returns a results dict.
    """
    X = df[FEATS].values
    preds = model.predict(X)

    # EPMO: argmin of predicted times
    epmo_idx  = int(np.argmin(preds))
    epmo_row  = df.iloc[epmo_idx]

    # Oracle: argmin of actual measured times
    oracle_idx = int(df[TARGET].values.argmin())
    oracle_row = df.iloc[oracle_idx]

    # Default: workers=0, batch=32
    default_mask = (df["num_workers"] == DEFAULT_W) & (df["batch_size"] == DEFAULT_B)
    if default_mask.sum() == 0:
        # fallback: worst measured time
        default_t = float(df[TARGET].max())
        default_w, default_b = DEFAULT_W, DEFAULT_B
    else:
        default_row = df[default_mask].iloc[0]
        default_t   = float(default_row[TARGET])
        default_w   = int(default_row["num_workers"])
        default_b   = int(default_row["batch_size"])

    epmo_t   = float(epmo_row[TARGET])
    oracle_t = float(oracle_row[TARGET])

    # Metrics
    saving_pct   = (default_t - epmo_t)   / default_t * 100 if default_t > 0 else 0.0
    oracle_saving = (default_t - oracle_t) / default_t * 100 if default_t > 0 else 0.0
    regret_pct   = max(0.0, (epmo_t - oracle_t) / oracle_t * 100) if oracle_t > 0 else 0.0

    if abs(default_t - oracle_t) > 1e-6:
        pct_oracle = min((default_t - epmo_t) / (default_t - oracle_t) * 100, 100.0)
    else:
        pct_oracle = 100.0

    config_hit = int(epmo_idx == oracle_idx)

    results = {
        "epmo": {
            "num_workers":       int(epmo_row["num_workers"]),
            "batch_size":        int(epmo_row["batch_size"]),
            "load_time_median":  round(epmo_t, 4),
            "predicted_time":    round(float(preds[epmo_idx]), 4),
        },
        "oracle": {
            "num_workers":       int(oracle_row["num_workers"]),
            "batch_size":        int(oracle_row["batch_size"]),
            "load_time_median":  round(oracle_t, 4),
        },
        "default": {
            "num_workers":       default_w,
            "batch_size":        default_b,
            "load_time_median":  round(default_t, 4),
        },
        "metrics": {
            "saving_pct":        round(saving_pct, 2),
            "oracle_saving_pct": round(oracle_saving, 2),
            "regret_pct":        round(regret_pct, 2),
            "pct_oracle":        round(pct_oracle, 2),
            "config_hit":        config_hit,
        },
        "environment": {
            "gpu_available":     int(df["gpu_available"].iloc[0]),
            "cpu_logical":       int(df["cpu_logical"].iloc[0]),
            "ram_gb":            float(df["ram_gb"].iloc[0]),
            "gpu_vram_gb":       float(df["gpu_vram_gb"].iloc[0]),
        },
        "all_configs": df[["num_workers", "batch_size", TARGET]].to_dict(orient="records"),
    }
    return results


# ══════════════════════════════════════════════════════════
# PRINT REPORT
# ══════════════════════════════════════════════════════════

def print_report(results: dict, env_id: str):
    m   = results["metrics"]
    env = results["environment"]
    gpu_str = (f"GPU  ({env['gpu_vram_gb']}GB VRAM)"
               if env["gpu_available"] else "CPU only")

    print("\n" + "=" * 60)
    print(f"  EPMO TEST RESULTS  —  {env_id.upper()}")
    print("=" * 60)
    print(f"  Environment : {gpu_str}")
    print(f"  CPU logical : {env['cpu_logical']} cores")
    print(f"  RAM         : {env['ram_gb']} GB")
    print()

    e = results["epmo"]
    o = results["oracle"]
    d = results["default"]

    print(f"  {'Config':<12} {'Workers':>8} {'Batch':>7} {'Time (s)':>10}")
    print(f"  {'─'*42}")
    hit = "✓" if results["metrics"]["config_hit"] else "✗"
    print(f"  {'EPMO '+hit:<12} {e['num_workers']:>8} {e['batch_size']:>7} "
          f"{e['load_time_median']:>10.4f}")
    print(f"  {'Oracle':<12} {o['num_workers']:>8} {o['batch_size']:>7} "
          f"{o['load_time_median']:>10.4f}")
    print(f"  {'Default':<12} {d['num_workers']:>8} {d['batch_size']:>7} "
          f"{d['load_time_median']:>10.4f}")
    print()
    print(f"  Saving vs Default  : {m['saving_pct']:+.2f}%")
    print(f"  Oracle Saving      : {m['oracle_saving_pct']:+.2f}%")
    print(f"  Regret vs Oracle   : {m['regret_pct']:.2f}%")
    print(f"  % of Oracle Saving : {m['pct_oracle']:.1f}%")
    print(f"  Exact Config Match : {'YES ✓' if m['config_hit'] else 'NO ✗'}")

    print()
    if m["regret_pct"] < 2:
        verdict = "EXCELLENT  — EPMO ≈ Oracle"
    elif m["regret_pct"] < 5:
        verdict = "SUCCESS    — Within 5% of Oracle"
    elif m["regret_pct"] < 15:
        verdict = "ACCEPTABLE — Within 15% of Oracle"
    else:
        verdict = "POOR       — Regret > 15%"
    print(f"  Verdict : {verdict}")
    print("=" * 60)


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="EPMO Paper Pipeline — benchmark + evaluate in one command"
    )
    parser.add_argument(
        "--env", type=str, required=True,
        help="Environment label, e.g. khushi_laptop, colab_t4"
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="(Optional) Skip benchmark and use an existing CSV directly"
    )
    args = parser.parse_args()

    env_id = args.env.strip().lower().replace(" ", "_")

    print("=" * 60)
    print("  EPMO Paper Pipeline  v1.0")
    print("=" * 60)

    # ── Step 1: Detect hardware ──
    print("\n[1/4] Detecting hardware ...")
    has_gpu = detect_gpu()
    domain  = "GPU" if has_gpu else "CPU"
    print(f"  GPU detected: {has_gpu}  →  Using Model_{('B' if has_gpu else 'A')} ({domain})")

    # ── Step 2: Run benchmark (or skip if CSV provided) ──
    if args.csv:
        csv_path = args.csv
        if not os.path.exists(csv_path):
            sys.exit(f"ERROR: --csv file not found: {csv_path}")
        print(f"\n[2/4] Skipping benchmark — using provided CSV: {csv_path}")
    else:
        csv_path = run_benchmark(has_gpu, env_id)

    # ── Step 3: Load CSV + engineer features ──
    print(f"\n[3/4] Loading and engineering features from {csv_path} ...")
    df = load_and_engineer(csv_path)
    print(f"  Rows loaded : {len(df)}")
    print(f"  Configs     : {len(df)} (expected 16)")

    # ── Step 4: Load model + predict + evaluate ──
    print(f"\n[4/4] Loading model and evaluating ...")
    model   = load_model(has_gpu)
    results = predict_and_evaluate(df, model)

    # ── Print report ──
    print_report(results, env_id)

    # ── Save results for notebook ──
    out_path = "epmo_test_results.json"
    results["env_id"] = env_id
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")
    print(f"  → Open epmo_casestudies.ipynb to run case studies A, B, C\n")


if __name__ == "__main__":
    main()
