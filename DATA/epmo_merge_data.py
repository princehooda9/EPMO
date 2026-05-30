"""
=============================================================
EPMO Data Merger  v4.0  —  epmo_merge_data.py
=============================================================
Merges all Training CSVs into two domain-specific datasets:
  epmo_data_cpu.csv  →  Model_A training data (gpu_available=0)
  epmo_data_gpu.csv  →  Model_B training data (gpu_available=1)

Usage:
  python epmo_merge_data.py

Expected files in same folder:
  Training_1cpu.csv  ... Training_12cpu.csv   (CPU domain)
  Training_1gpu.csv  ... Training_8gpu.csv    (GPU domain)

Outputs:
  epmo_data_cpu.csv   ← feed to epmo_train.py
  epmo_data_gpu.csv   ← feed to epmo_train.py
  epmo_data_all.csv   ← combined reference
=============================================================
"""

import os, sys
import pandas as pd
import numpy as np
from collections import defaultdict

CPU_FILES = [f"Training_{i}cpu.csv" for i in range(1, 13)]
GPU_FILES = [f"Training_{i}gpu.csv" for i in range(1, 9)]
TARGET = "load_time_median"
DEFAULT_W, DEFAULT_B = 0, 32


def load_files(file_list, domain_label):
    dfs, missing = [], []
    print(f"\n  Loading {domain_label} files:")
    for fname in file_list:
        if not os.path.exists(fname):
            missing.append(fname); print(f"    ✗ {fname} — NOT FOUND"); continue
        df = pd.read_csv(fname)
        env = df["env_id"].iloc[0]; gpu = df["gpu_available"].iloc[0]
        cpu = df["cpu_logical"].iloc[0]; ram = df["ram_gb"].iloc[0]
        vram = df["gpu_vram_gb"].iloc[0]
        print(f"    ✓ {fname:<25} env={env:<8} cpu={cpu:>2} "
              f"ram={ram:>5.1f} vram={vram:>5.2f} gpu_available={gpu}")
        dfs.append(df)
    if missing:
        print(f"\n  WARNING: {len(missing)} files not found: {missing}")
    if not dfs:
        print(f"  ERROR: No {domain_label} files loaded."); sys.exit(1)
    return pd.concat(dfs, ignore_index=True)


def validate_domain(df, expected_gpu, name):
    wrong = df[df["gpu_available"] != expected_gpu]
    if len(wrong) > 0:
        print(f"\n  ⚠ WARNING [{name}]: {len(wrong)} rows have "
              f"gpu_available != {expected_gpu}")
        print(f"    Environments: {wrong['env_id'].unique().tolist()}")
    else:
        print(f"  ✓ All {name} rows have gpu_available={expected_gpu}")


def check_uniqueness(df, name):
    fingerprints = defaultdict(list)
    for env, grp in df.groupby("env_id"):
        key = (int(grp["cpu_logical"].iloc[0]),
               round(float(grp["ram_gb"].iloc[0]), 0),
               round(float(grp["gpu_vram_gb"].iloc[0]), 1))
        fingerprints[key].append(env)
    print(f"\n  Hardware diversity [{name}]:")
    for (cpu, ram, vram), envs in sorted(fingerprints.items()):
        flag = "⚠ SAME HW" if len(envs) > 1 else "✓"
        print(f"    {flag} cpu={cpu:>2} ram={ram:>5.0f} vram={vram:>5.1f} → {envs}")


def add_features(df):
    df = df.copy()
    df["workers_x_cpu"]  = df["num_workers"] * df["cpu_logical"]
    df["batch_x_ram"]    = df["batch_size"]  / df["ram_gb"]
    df["cpu_per_worker"] = df["cpu_logical"] / (df["num_workers"] + 1)
    df["workers_x_gpu"]  = df["num_workers"] * df["gpu_available"]
    df["batch_x_gpu"]    = df["batch_size"]  * df["gpu_available"]
    return df


def print_summary(df, name):
    print(f"\n  {'─'*65}")
    print(f"  {name}")
    print(f"  {'─'*65}")
    print(f"  {'Env':<10} {'CPU':>4} {'RAM':>6} {'VRAM':>6} "
          f"{'Best config':>14} {'Default(s)':>11} {'Saving%':>9}")
    print(f"  {'─'*65}")
    for env, grp in df.groupby("env_id"):
        best = grp.loc[grp[TARGET].idxmin()]
        def_mask = (grp["num_workers"]==DEFAULT_W) & (grp["batch_size"]==DEFAULT_B)
        default_t = float(grp[def_mask][TARGET].iloc[0]) if def_mask.sum() else None
        saving = ((default_t - float(best[TARGET])) / default_t * 100
                  if default_t else 0)
        cpu  = int(grp["cpu_logical"].iloc[0])
        ram  = float(grp["ram_gb"].iloc[0])
        vram = float(grp["gpu_vram_gb"].iloc[0])
        cfg  = f"w={int(best.num_workers)},b={int(best.batch_size)}"
        dt   = f"{default_t:.4f}" if default_t else "N/A"
        print(f"  {env:<10} {cpu:>4} {ram:>6.1f} {vram:>6.2f} "
              f"{cfg:>14} {dt:>11} {saving:>+9.1f}%")


def main():
    print("=" * 60)
    print("EPMO Data Merger  v4.0")
    print("CPU domain → Model_A  |  GPU domain → Model_B")
    print("=" * 60)

    cpu_df = load_files(CPU_FILES, "CPU (Model_A)")
    gpu_df = load_files(GPU_FILES, "GPU (Model_B)")

    print("\n[Domain Validation]")
    validate_domain(cpu_df, 0, "CPU")
    validate_domain(gpu_df, 1, "GPU")

    check_uniqueness(cpu_df, "CPU")
    check_uniqueness(gpu_df, "GPU")

    cpu_df = add_features(cpu_df)
    gpu_df = add_features(gpu_df)

    print_summary(cpu_df, "Model_A — CPU environments")
    print_summary(gpu_df, "Model_B — GPU environments")

    cpu_df.to_csv("epmo_data_cpu.csv", index=False)
    gpu_df.to_csv("epmo_data_gpu.csv", index=False)
    combined = pd.concat([cpu_df, gpu_df], ignore_index=True)
    combined.to_csv("epmo_data_all.csv", index=False)

    print(f"\n{'='*60}")
    print(f"SAVED:")
    print(f"  epmo_data_cpu.csv — {len(cpu_df)} rows, "
          f"{cpu_df['env_id'].nunique()} CPU environments")
    print(f"  epmo_data_gpu.csv — {len(gpu_df)} rows, "
          f"{gpu_df['env_id'].nunique()} GPU environments")
    print(f"  epmo_data_all.csv — {len(combined)} rows total")
    print(f"\nNEXT STEP:  python epmo_train.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
