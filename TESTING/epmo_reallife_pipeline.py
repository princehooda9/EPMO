"""
=============================================================
EPMO Real-Life Pipeline  —  epmo_reallife_pipeline.py
=============================================================
Purpose:
  Zero-benchmark inference script for real-world deployment.
  No dataset download. No timing runs. Instant recommendation.

What it does:
  1. Detects hardware profile from the current machine
       cpu_logical, ram_gb, gpu_available, gpu_vram_gb
  2. Selects the correct trained model
       GPU present → model_gpu.pkl  (Model_B)
       CPU only    → model_cpu.pkl  (Model_A)
  3. Engineers all 16 candidate configs × 11 features
  4. Predicts load_time_median for each config
  5. Picks argmin → EPMO recommended config
  6. Compares against hardcoded default (workers=0, batch=32)
  7. Prints a clean recommendation table
  8. Saves output to epmo_reallife_results.json for the
     case-study notebook (epmo_casestudies.ipynb)

Usage:
  python epmo_reallife_pipeline.py

Requirements (must be in the same folder):
  model_cpu.pkl
  model_gpu.pkl
=============================================================
"""

import os
import sys
import json
import pickle
import platform
import numpy as np

# ── constants (must match epmo_train.py exactly) ──────────
WORKER_OPTIONS = [0, 1, 2, 4]
BATCH_OPTIONS  = [32, 64, 128, 256]

DEFAULT_W = 0    # PyTorch out-of-box default
DEFAULT_B = 32   # PyTorch out-of-box default

FEATS = [
    "cpu_logical", "ram_gb", "gpu_available", "gpu_vram_gb",
    "num_workers", "batch_size",
    "workers_x_cpu", "batch_x_ram", "cpu_per_worker",
    "workers_x_gpu", "batch_x_gpu",
]


# ══════════════════════════════════════════════════════════
# STEP 1 — HARDWARE DETECTION
# ══════════════════════════════════════════════════════════

def detect_hardware() -> dict:
    """
    Collect hardware profile from the current machine.
    No dataset loading. No benchmarking. Pure hardware introspection.
    """
    try:
        import psutil
    except ImportError:
        sys.exit("ERROR: psutil not installed. Run: pip install psutil")

    try:
        import torch
        has_gpu      = torch.cuda.is_available()
        gpu_vram_gb  = (round(torch.cuda.get_device_properties(0).total_memory
                              / (1024 ** 3), 2) if has_gpu else 0.0)
        gpu_name     = torch.cuda.get_device_name(0) if has_gpu else "none"
    except ImportError:
        has_gpu     = False
        gpu_vram_gb = 0.0
        gpu_name    = "none"

    hw = {
        "cpu_logical":   psutil.cpu_count(logical=True),
        "cpu_physical":  psutil.cpu_count(logical=False) or 1,
        "ram_gb":        round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "gpu_available": int(has_gpu),
        "gpu_vram_gb":   gpu_vram_gb,
        "gpu_name":      gpu_name,
        "os":            platform.system(),
        "python":        platform.python_version(),
    }
    return hw


# ══════════════════════════════════════════════════════════
# STEP 2 — LOAD MODEL
# ══════════════════════════════════════════════════════════

def load_model(has_gpu: bool):
    """Load the correct pkl model based on GPU availability."""
    path = "model_gpu.pkl" if has_gpu else "model_cpu.pkl"
    if not os.path.exists(path):
        sys.exit(f"ERROR: {path} not found. Run epmo_train.py first.")
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model, path


# ══════════════════════════════════════════════════════════
# STEP 3 — BUILD CANDIDATE FEATURES
# ══════════════════════════════════════════════════════════

def build_candidates(hw: dict) -> list:
    """
    Build the full 16-row candidate grid (4 workers × 4 batch sizes)
    with all 11 engineered features — exactly as in epmo_train.py.
    """
    candidates = []
    for w in WORKER_OPTIONS:
        for b in BATCH_OPTIONS:
            row = {
                "cpu_logical":   hw["cpu_logical"],
                "ram_gb":        hw["ram_gb"],
                "gpu_available": hw["gpu_available"],
                "gpu_vram_gb":   hw["gpu_vram_gb"],
                "num_workers":   w,
                "batch_size":    b,
                "workers_x_cpu": w * hw["cpu_logical"],
                "batch_x_ram":   b / hw["ram_gb"],
                "cpu_per_worker": hw["cpu_logical"] / (w + 1),
                "workers_x_gpu": w * hw["gpu_available"],
                "batch_x_gpu":   b * hw["gpu_available"],
            }
            candidates.append(row)
    return candidates


# ══════════════════════════════════════════════════════════
# STEP 4 — PREDICT + SELECT
# ══════════════════════════════════════════════════════════

def predict_best(model, candidates: list, hw: dict) -> dict:
    """
    Run model over all 16 candidates, return EPMO recommendation
    alongside the default config for comparison.
    """
    X = [[row[f] for f in FEATS] for row in candidates]
    preds = model.predict(X)

    best_idx   = int(np.argmin(preds))
    best_cand  = candidates[best_idx]
    best_pred  = float(preds[best_idx])

    # All predictions for the full table
    all_results = []
    for i, (cand, pred) in enumerate(zip(candidates, preds)):
        all_results.append({
            "num_workers":    cand["num_workers"],
            "batch_size":     cand["batch_size"],
            "predicted_time": round(float(pred), 4),
            "rank":           0,  # filled below
        })

    # Rank by predicted time
    sorted_idxs = np.argsort(preds)
    for rank, idx in enumerate(sorted_idxs):
        all_results[idx]["rank"] = rank + 1

    results = {
        "recommended": {
            "num_workers":    best_cand["num_workers"],
            "batch_size":     best_cand["batch_size"],
            "predicted_time": round(best_pred, 4),
        },
        "default": {
            "num_workers":    DEFAULT_W,
            "batch_size":     DEFAULT_B,
            "predicted_time": round(float(preds[
                next(i for i, c in enumerate(candidates)
                     if c["num_workers"] == DEFAULT_W and c["batch_size"] == DEFAULT_B)
            ]), 4),
        },
        "all_configs": all_results,
        "environment": {
            "gpu_available":  hw["gpu_available"],
            "gpu_name":       hw["gpu_name"],
            "cpu_logical":    hw["cpu_logical"],
            "cpu_physical":   hw["cpu_physical"],
            "ram_gb":         hw["ram_gb"],
            "gpu_vram_gb":    hw["gpu_vram_gb"],
            "os":             hw["os"],
        },
    }

    # Estimated saving vs default (predicted times only — no real measurement)
    default_pred = results["default"]["predicted_time"]
    epmo_pred    = results["recommended"]["predicted_time"]
    if default_pred > 0:
        results["predicted_saving_pct"] = round(
            (default_pred - epmo_pred) / default_pred * 100, 2
        )
    else:
        results["predicted_saving_pct"] = 0.0

    return results


# ══════════════════════════════════════════════════════════
# PRINT RECOMMENDATION
# ══════════════════════════════════════════════════════════

def print_recommendation(results: dict, model_path: str):
    env  = results["environment"]
    rec  = results["recommended"]
    dft  = results["default"]
    sav  = results["predicted_saving_pct"]

    gpu_str = (f"GPU  |  {env['gpu_name']}  ({env['gpu_vram_gb']} GB VRAM)"
               if env["gpu_available"] else "CPU only")

    print("\n" + "=" * 60)
    print("  EPMO REAL-LIFE RECOMMENDATION")
    print("=" * 60)
    print(f"  OS          : {env['os']}")
    print(f"  CPU cores   : {env['cpu_logical']} logical / {env['cpu_physical']} physical")
    print(f"  RAM         : {env['ram_gb']} GB")
    print(f"  Compute     : {gpu_str}")
    print(f"  Model used  : {model_path}")
    print()
    print(f"  {'Config':<16} {'num_workers':>12} {'batch_size':>11} {'Pred Time (s)':>14}")
    print(f"  {'─'*56}")
    print(f"  {'EPMO Recommended':<16} {rec['num_workers']:>12} "
          f"{rec['batch_size']:>11} {rec['predicted_time']:>14.4f}")
    print(f"  {'Default':<16} {dft['num_workers']:>12} "
          f"{dft['batch_size']:>11} {dft['predicted_time']:>14.4f}")
    print()
    if sav > 0:
        print(f"  Predicted speedup vs default : {sav:.1f}% faster")
    elif sav == 0:
        print(f"  Default IS the optimal config on this hardware.")
    else:
        print(f"  Default slightly better by prediction ({abs(sav):.1f}%)")
        print(f"  → This is expected on some GPU machines where workers=0 wins.")

    print()
    print(f"  Top-3 Predicted Configs:")
    top3 = sorted(results["all_configs"], key=lambda x: x["rank"])[:3]
    for cfg in top3:
        print(f"    #{cfg['rank']}  workers={cfg['num_workers']:<2}  "
              f"batch={cfg['batch_size']:<4}  "
              f"pred={cfg['predicted_time']:.4f}s")

    print()
    print(f"  NOTE: Predicted times are model estimates, not measured values.")
    print(f"        For measured validation, use epmo_paper_pipeline.py")
    print("=" * 60)


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  EPMO Real-Life Pipeline  v1.0")
    print("=" * 60)

    # ── Step 1: Detect hardware ──
    print("\n[1/3] Detecting hardware ...")
    hw = detect_hardware()
    print(f"  CPU logical : {hw['cpu_logical']} cores")
    print(f"  RAM         : {hw['ram_gb']} GB")
    print(f"  GPU         : {'YES — ' + hw['gpu_name'] if hw['gpu_available'] else 'NO'}")
    if hw["gpu_available"]:
        print(f"  GPU VRAM    : {hw['gpu_vram_gb']} GB")

    # ── Step 2: Load model ──
    print(f"\n[2/3] Loading model ...")
    model, model_path = load_model(bool(hw["gpu_available"]))
    domain = "GPU (Model_B)" if hw["gpu_available"] else "CPU (Model_A)"
    print(f"  Model loaded: {model_path}  [{domain}]")

    # ── Step 3: Predict ──
    print(f"\n[3/3] Predicting optimal config ...")
    candidates = build_candidates(hw)
    results    = predict_best(model, candidates, hw)

    # ── Print recommendation ──
    print_recommendation(results, model_path)

    # ── Save for notebook ──
    out_path = "epmo_reallife_results.json"
    results["pipeline"] = "reallife"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")
    print(f"  → Open epmo_casestudies.ipynb to run case studies A, B, C\n")


if __name__ == "__main__":
    main()
