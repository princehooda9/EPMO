"""
=============================================================
EPMO Training & Evaluation  v4.0  —  epmo_train.py
=============================================================
Two-model architecture:
  Model_A — trained on CPU environments (gpu_available=0)
  Model_B — trained on GPU environments (gpu_available=1)

Each model uses 5 algorithms + LOOCV + full metrics including Regret%.

Usage:
  python epmo_train.py

Requires:
  epmo_data_cpu.csv   (from epmo_merge_data.py)
  epmo_data_gpu.csv   (from epmo_merge_data.py)

Outputs (in epmo_outputs/):
  model_cpu.pkl, model_gpu.pkl
  fig_1_heatmaps_cpu.png, fig_1_heatmaps_gpu.png
  fig_2_model_comparison_cpu.png, fig_2_model_comparison_gpu.png
  fig_3_speedup_cpu.png, fig_3_speedup_gpu.png
  fig_4_amdahl.png
  fig_5_feature_importance_cpu.png, fig_5_feature_importance_gpu.png
  fig_6_decision_tree_cpu.png
  fig_7_pred_vs_actual_cpu.png, fig_7_pred_vs_actual_gpu.png
  table_loocv_cpu.csv, table_loocv_gpu.csv
  table_summary_cpu.csv, table_summary_gpu.csv
  paper_numbers.txt
=============================================================
"""

import os, sys, pickle, warnings, copy
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble         import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree             import DecisionTreeRegressor, export_text, plot_tree
from sklearn.linear_model     import Ridge
from sklearn.pipeline         import Pipeline
from sklearn.preprocessing    import StandardScaler
from sklearn.metrics          import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("  NOTE: XGBoost not installed. Skipping XGB model.")
    print("        Install with: pip install xgboost")

# ── style ─────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 11, "axes.titlesize": 12,
    "axes.labelsize": 11, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})
C = ["#2563EB", "#DC2626", "#16A34A", "#D97706", "#7C3AED", "#0891B2"]
OUT_DIR = "epmo_outputs"

# ── constants ─────────────────────────────────────────────
TARGET      = "load_time_median"
DEFAULT_W, DEFAULT_B = 0, 32
WORKER_OPTIONS = [0, 1, 2, 4]
BATCH_OPTIONS  = [32, 64, 128, 256]

CPU_FEATS = ["cpu_logical", "ram_gb", "gpu_available", "gpu_vram_gb",
             "num_workers", "batch_size",
             "workers_x_cpu", "batch_x_ram", "cpu_per_worker",
             "workers_x_gpu", "batch_x_gpu"]

GPU_FEATS = ["cpu_logical", "ram_gb", "gpu_available", "gpu_vram_gb",
             "num_workers", "batch_size",
             "workers_x_cpu", "batch_x_ram", "cpu_per_worker",
             "workers_x_gpu", "batch_x_gpu"]


# ══════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════

def load(path):
    if not os.path.exists(path):
        sys.exit(f"ERROR: {path} not found. Run epmo_merge_data.py first.")
    df = pd.read_csv(path)
    # ensure engineered features exist (in case merge was run without them)
    if "workers_x_cpu" not in df.columns:
        df["workers_x_cpu"]  = df["num_workers"] * df["cpu_logical"]
        df["batch_x_ram"]    = df["batch_size"]  / df["ram_gb"]
        df["cpu_per_worker"] = df["cpu_logical"] / (df["num_workers"] + 1)
        df["workers_x_gpu"]  = df["num_workers"] * df["gpu_available"]
        df["batch_x_gpu"]    = df["batch_size"]  * df["gpu_available"]
    return df


# ══════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════

def make_models():
    models = {
        "Random Forest":     RandomForestRegressor(
            n_estimators=300, max_depth=None,
            min_samples_leaf=1, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.04,
            max_depth=4, subsample=0.8, random_state=42),
        "Decision Tree":     DecisionTreeRegressor(
            max_depth=4, min_samples_leaf=1, random_state=42),
        "Ridge Regression":  Pipeline([
            ("sc", StandardScaler()), ("rg", Ridge(alpha=10.0))]),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=300, learning_rate=0.04, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0, n_jobs=-1)
    return models


# ══════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════

def default_time_for_env(env_df):
    mask = (env_df["num_workers"]==DEFAULT_W) & (env_df["batch_size"]==DEFAULT_B)
    return float(env_df[mask][TARGET].iloc[0]) if mask.sum() else float(env_df[TARGET].max())


def eval_fold(model, X_tr, y_tr, X_te, y_te, test_df):
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)

    mae  = mean_absolute_error(y_te, preds)
    rmse = np.sqrt(mean_squared_error(y_te, preds))
    r2   = r2_score(y_te, preds)

    true_best_idx = int(np.argmin(y_te.values))
    oracle_time   = float(y_te.iloc[true_best_idx])
    true_w = int(test_df["num_workers"].iloc[true_best_idx])
    true_b = int(test_df["batch_size"].iloc[true_best_idx])

    pred_best_idx = int(np.argmin(preds))
    epmo_time     = float(y_te.iloc[pred_best_idx])
    pred_w = int(test_df["num_workers"].iloc[pred_best_idx])
    pred_b = int(test_df["batch_size"].iloc[pred_best_idx])

    top3_idx   = np.argsort(preds)[:3]
    config_hit = int(pred_best_idx == true_best_idx)
    top3_hit   = int(true_best_idx in top3_idx)

    default_t = default_time_for_env(test_df)
    saving_pct = (default_t - epmo_time) / default_t * 100 if default_t > 0 else 0
    oracle_saving = (default_t - oracle_time) / default_t * 100 if default_t > 0 else 0

    # Regret% = how much worse than oracle (0% = perfect)
    if oracle_time > 0:
        regret_pct = (epmo_time - oracle_time) / oracle_time * 100
        regret_pct = max(0.0, regret_pct)   # can't be negative
    else:
        regret_pct = 0.0

    # PctOracle: what % of oracle saving did EPMO achieve
    if abs(default_t - oracle_time) > 1e-6:
        pct_oracle = min((default_t - epmo_time) / (default_t - oracle_time) * 100, 100.0)
    else:
        pct_oracle = 100.0   # oracle == default, EPMO also picks default → 100%

    return dict(
        mae=round(mae,5), rmse=round(rmse,5), r2=round(r2,4),
        config_acc=config_hit, top3_acc=top3_hit,
        default_time=round(default_t,4),
        epmo_time=round(epmo_time,4),
        oracle_time=round(oracle_time,4),
        saving_pct=round(saving_pct,1),
        oracle_saving_pct=round(oracle_saving,1),
        regret_pct=round(regret_pct,2),
        pct_oracle=round(pct_oracle,1),
        pred_w=pred_w, pred_b=pred_b,
        true_w=true_w, true_b=true_b,
        preds=preds, y_true=y_te.values,
    )


# ══════════════════════════════════════════════════════════
# LOOCV
# ══════════════════════════════════════════════════════════

def run_loocv(df, feat_cols, domain_name):
    envs    = df["env_id"].unique().tolist()
    models  = make_models()
    results = {n: {} for n in models}

    print(f"\n  {len(envs)}-fold LOOCV — {domain_name}")
    for test_env in envs:
        train_df = df[df["env_id"] != test_env].copy()
        test_df  = df[df["env_id"] == test_env].copy().reset_index(drop=True)
        X_tr = train_df[feat_cols].values
        y_tr = train_df[TARGET]
        X_te = test_df[feat_cols].values
        y_te = test_df[TARGET].reset_index(drop=True)

        for name, model in models.items():
            m   = copy.deepcopy(model)
            res = eval_fold(m, X_tr, y_tr, X_te, y_te, test_df)
            results[name][test_env] = res
            h = "✓" if res["config_acc"] else "✗"
            print(f"    {h} [{test_env:<8}] {name:<22} "
                  f"MAE={res['mae']:.4f}  "
                  f"{'HIT' if res['config_acc'] else 'MISS'}  "
                  f"Saving={res['saving_pct']:+.1f}%  "
                  f"Regret={res['regret_pct']:.2f}%")

    # retrain on full data for figures
    fitted = {}
    X_all, y_all = df[feat_cols].values, df[TARGET].values
    for name, model in models.items():
        m = copy.deepcopy(model)
        m.fit(X_all, y_all)
        fitted[name] = m

    return results, fitted, envs


# ══════════════════════════════════════════════════════════
# AGGREGATE
# ══════════════════════════════════════════════════════════

def aggregate(results, envs):
    rows = []
    for name, env_res in results.items():
        for env, m in env_res.items():
            rows.append({"model": name, "test_env": env,
                "MAE": m["mae"], "RMSE": m["rmse"], "R2": m["r2"],
                "ConfigAcc": m["config_acc"], "Top3Acc": m["top3_acc"],
                "DefaultT": m["default_time"], "EPMO_T": m["epmo_time"],
                "OracleT": m["oracle_time"],
                "Saving%": m["saving_pct"],
                "Oracle%": m["oracle_saving_pct"],
                "Regret%": m["regret_pct"],
                "PctOracle": m["pct_oracle"],
                "Pred_w": m["pred_w"], "Pred_b": m["pred_b"],
                "True_w": m["true_w"], "True_b": m["true_b"]})
    loocv_df = pd.DataFrame(rows)

    summary_rows = []
    for name, grp in loocv_df.groupby("model"):
        summary_rows.append({"Model": name,
            "MAE_mean":       round(grp["MAE"].mean(), 5),
            "MAE_std":        round(grp["MAE"].std(),  5),
            "ConfigAcc_mean": round(grp["ConfigAcc"].mean(), 3),
            "Top3Acc_mean":   round(grp["Top3Acc"].mean(), 3),
            "Saving%_mean":   round(grp["Saving%"].mean(), 1),
            "Regret%_mean":   round(grp["Regret%"].mean(), 2),
            "PctOracle_mean": round(grp["PctOracle"].mean(), 1)})
    summary_df = pd.DataFrame(summary_rows).sort_values("MAE_mean")
    return loocv_df, summary_df


# ══════════════════════════════════════════════════════════
# PRINT TABLES
# ══════════════════════════════════════════════════════════

def print_tables(loocv_df, summary_df, domain_name):
    print(f"\n{'='*72}")
    print(f"RESULTS — {domain_name}")
    print(f"{'='*72}")
    print(f"  {'Model':<22} {'Env':<10} {'MAE':>7} {'CA':>4} "
          f"{'Sav%':>7} {'Reg%':>7} {'Orac%':>7} Config")
    print(f"  {'─'*70}")
    for _, r in loocv_df.sort_values(["model","test_env"]).iterrows():
        h = "✓" if r["ConfigAcc"] else "✗"
        cfg = f"w={int(r['Pred_w'])},b={int(r['Pred_b'])}"
        print(f"  {r['model']:<22} {r['test_env']:<10} {r['MAE']:>7.4f} "
              f"{h:>4} {r['Saving%']:>+7.1f}% {r['Regret%']:>6.2f}% "
              f"{r['PctOracle']:>6.1f}%  {cfg}")

    print(f"\n  SUMMARY (mean across folds):")
    print(f"  {'Model':<22} {'MAE±std':>16} {'ConfigAcc':>11} "
          f"{'Saving%':>9} {'Regret%':>9} {'PctOracle':>11}")
    print(f"  {'─'*82}")
    for _, r in summary_df.iterrows():
        print(f"  {r['Model']:<22} "
              f"{r['MAE_mean']:.4f}±{r['MAE_std']:.4f}  "
              f"{r['ConfigAcc_mean']*100:>9.0f}%  "
              f"{r['Saving%_mean']:>8.1f}%  "
              f"{r['Regret%_mean']:>8.2f}%  "
              f"{r['PctOracle_mean']:>10.1f}%")


# ══════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════

def _save(name):
    p = os.path.join(OUT_DIR, name)
    plt.savefig(p, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")


def fig_heatmaps(df, domain_name, suffix):
    envs = df["env_id"].unique()
    n    = len(envs)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5*cols, 4.2*rows))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for idx, env in enumerate(envs):
        ax  = axes[idx]
        sub = df[df["env_id"] == env]
        pivot = sub.pivot_table(index="num_workers", columns="batch_size",
                                values=TARGET, aggfunc="median")
        vmin, vmax = pivot.values.min(), pivot.values.max()
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r",
                       vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_xlabel("Batch Size"); ax.set_ylabel("Workers")
        br, bc = divmod(int(np.argmin(pivot.values)), pivot.shape[1])
        ax.plot(bc, br, "w*", markersize=14)
        hw = sub.iloc[0]
        gpu_str = f"GPU={hw.gpu_vram_gb:.1f}GB" if hw.gpu_available else "GPU=none"
        ax.set_title(f"{env}\nCPU={int(hw.cpu_logical)} "
                     f"RAM={hw.ram_gb:.0f}GB {gpu_str}", fontsize=8)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(j, i, f"{pivot.values[i,j]:.2f}", ha="center",
                        va="center", fontsize=6,
                        color="white" if pivot.values[i,j]>(vmin+vmax)/2 else "black")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # hide unused axes
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"DataLoader Heatmaps — {domain_name} (★=optimal)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    _save(f"fig_1_heatmaps_{suffix}.png")


def fig_model_comparison(summary_df, domain_name, suffix):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    names = summary_df["Model"].tolist()
    maes  = summary_df["MAE_mean"].tolist()
    stds  = summary_df["MAE_std"].tolist()

    ax = axes[0]
    bars = ax.bar(names, maes, yerr=stds, capsize=5,
                  color=C[:len(names)], width=0.5, edgecolor="white",
                  error_kw={"elinewidth": 1.5})
    ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=8)
    ax.set_ylabel("MAE (seconds)")
    ax.set_title(f"MAE — {domain_name}\n(mean±std, LOOCV)")
    ax.set_ylim(0, max(maes)*1.5)
    ax.set_xticklabels(names, rotation=15, ha="right")

    ax = axes[1]
    cacc = [summary_df[summary_df["Model"]==n]["ConfigAcc_mean"].values[0]*100 for n in names]
    t3   = [summary_df[summary_df["Model"]==n]["Top3Acc_mean"].values[0]*100   for n in names]
    x, w = np.arange(len(names)), 0.35
    b1 = ax.bar(x-w/2, cacc, w, label="Exact Match", color=C[0])
    b2 = ax.bar(x+w/2, t3,   w, label="Top-3 Match", color=C[1])
    ax.bar_label(b1, fmt="%.0f%%", padding=3, fontsize=8)
    ax.bar_label(b2, fmt="%.0f%%", padding=3, fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 130)
    ax.set_title(f"Config Accuracy — {domain_name}\n(LOOCV)")
    ax.legend()
    plt.tight_layout()
    _save(f"fig_2_model_comparison_{suffix}.png")


def fig_speedup(loocv_df, summary_df, domain_name, suffix):
    # use best model by ConfigAcc — deployment metric
    best_name = summary_df.sort_values("ConfigAcc_mean", ascending=False).iloc[0]["Model"]
    best_df   = loocv_df[loocv_df["model"] == best_name].reset_index(drop=True)
    envs = best_df["test_env"].tolist()
    x, w = np.arange(len(envs)), 0.25

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: time bars
    ax = axes[0]
    b1 = ax.bar(x-w,   best_df["DefaultT"], w, label="Static Default", color=C[2])
    b2 = ax.bar(x,     best_df["EPMO_T"],   w, label="EPMO Predicted",  color=C[0])
    b3 = ax.bar(x+w,   best_df["OracleT"],  w, label="Oracle (best)",   color=C[1])
    for bars in [b1,b2,b3]: ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=6)
    for i, row in best_df.iterrows():
        if abs(row["Saving%"]) > 1:
            ax.annotate(f"{row['Saving%']:+.1f}%",
                        xy=(x[i], max(row["DefaultT"], row["EPMO_T"])*1.03),
                        ha="center", fontsize=8, color=C[0], fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(envs, rotation=15, fontsize=8)
    ax.set_ylabel("DataLoader Time (s, 30 batches)")
    ax.set_title(f"Speedup — {domain_name}\n({best_name}, LOOCV)")
    ax.legend(fontsize=8)

    # Right: Regret% bar chart
    ax = axes[1]
    colors = [C[1] if r < 5 else C[3] if r < 15 else C[0]
              for r in best_df["Regret%"]]
    bars = ax.bar(envs, best_df["Regret%"], color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=8)
    ax.axhline(5, color="green", linestyle="--", lw=1.2, label="5% threshold")
    ax.set_ylabel("Regret% vs Oracle")
    ax.set_title(f"Regret% — {domain_name}\n(lower is better, <5% = success)")
    ax.set_xticklabels(envs, rotation=15, fontsize=8)
    ax.legend(fontsize=8)

    plt.tight_layout()
    _save(f"fig_3_speedup_{suffix}.png")


def fig_amdahl(cpu_df):
    """CPU-only Amdahl analysis."""
    sub  = cpu_df[cpu_df["batch_size"] == 32].copy()
    base = {}
    for env in cpu_df["env_id"].unique():
        mask = ((cpu_df["env_id"]==env) &
                (cpu_df["num_workers"]==0) &
                (cpu_df["batch_size"]==32))
        if mask.sum():
            base[env] = float(cpu_df[mask][TARGET].iloc[0])

    w_range = np.linspace(0.5, 5, 200)
    fig, ax = plt.subplots(figsize=(8, 5))
    for P, ls, lbl in [(0.7,"--","P=0.7"),(0.8,"-","P=0.8"),(0.9,":","P=0.9")]:
        ax.plot(w_range, 1.0/((1-P)+P/w_range), ls=ls, color="gray", lw=1.2,
                label=f"Amdahl {lbl}")
    for i, env in enumerate(cpu_df["env_id"].unique()):
        env_sub = sub[(sub["env_id"]==env) & (sub["num_workers"]>0)]
        if env not in base or env_sub.empty: continue
        ax.scatter(env_sub["num_workers"].values,
                   base[env]/env_sub[TARGET].values,
                   s=60, zorder=5, color=C[i % len(C)],
                   label=f"{env}", edgecolors="white", linewidths=0.4)
    ax.axhline(1.0, color="black", lw=0.8, linestyle=":")
    ax.set_xlabel("Number of Workers")
    ax.set_ylabel("Speedup vs workers=0  (batch=32)")
    ax.set_title("Empirical Speedup vs Amdahl's Law\n(CPU environments, batch_size=32)")
    ax.legend(fontsize=7, ncol=3)
    ax.set_xlim(0, 5); ax.set_ylim(bottom=0)
    plt.tight_layout()
    _save("fig_4_amdahl.png")


def fig_feature_importance(fitted, feat_cols, domain_name, suffix):
    for model_name in ["Random Forest", "Gradient Boosting", "XGBoost"]:
        m = fitted.get(model_name)
        if m is None or not hasattr(m, "feature_importances_"): continue
        imp = m.feature_importances_
        idx = np.argsort(imp)
        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.barh([feat_cols[i] for i in idx], imp[idx],
                       color=C[0], edgecolor="white")
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        ax.set_xlabel("Importance Score")
        ax.set_title(f"Feature Importance — {domain_name}\n({model_name})")
        plt.tight_layout()
        _save(f"fig_5_feature_importance_{suffix}_{model_name.lower().replace(' ','_')}.png")
        break  # just do best tree-based model


def fig_decision_tree(fitted, feat_cols, suffix):
    dt = fitted.get("Decision Tree")
    if dt is None: return
    fig, ax = plt.subplots(figsize=(22, 8))
    plot_tree(dt, feature_names=feat_cols, filled=True, rounded=True,
              fontsize=6, impurity=False, precision=3, ax=ax)
    ax.set_title(f"Decision Tree — CPU DataLoader Time Predictor (depth=4)", fontsize=11)
    _save(f"fig_6_decision_tree_{suffix}.png")
    rules = export_text(dt, feature_names=feat_cols, max_depth=4)
    rpath = os.path.join(OUT_DIR, f"decision_tree_rules_{suffix}.txt")
    with open(rpath, "w") as f: f.write(rules)
    print(f"  Saved: {rpath}")


def fig_pred_vs_actual(results, df, domain_name, suffix):
    # best model by aggregate MAE
    model_cacc = {}
    for name, env_res in results.items():
        model_cacc[name] = np.mean([r["config_acc"] for r in env_res.values()])
    best_name = max(model_cacc, key=model_cacc.get)
    all_true, all_pred, all_env = [], [], []
    for env, res in results[best_name].items():
        all_true.extend(res["y_true"])
        all_pred.extend(res["preds"])
        all_env.extend([env]*len(res["y_true"]))

    fig, ax = plt.subplots(figsize=(6, 5))
    envs = list(dict.fromkeys(all_env))
    for i, env in enumerate(envs):
        mask = [e == env for e in all_env]
        ax.scatter([all_true[j] for j,m in enumerate(mask) if m],
                   [all_pred[j] for j,m in enumerate(mask) if m],
                   s=35, color=C[i % len(C)], label=env, alpha=0.8,
                   edgecolors="white", linewidths=0.3)
    lim = max(max(all_true), max(all_pred)) * 1.05
    ax.plot([0,lim],[0,lim],"k--",lw=1,label="Perfect prediction")
    ax.set_xlim(0,lim); ax.set_ylim(0,lim)
    ax.set_xlabel("Actual Time (s)"); ax.set_ylabel("Predicted Time (s)")
    ax.set_title(f"{best_name}: Predicted vs Actual\n{domain_name} — LOOCV")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    _save(f"fig_7_pred_vs_actual_{suffix}.png")


# ══════════════════════════════════════════════════════════
# PAPER NUMBERS
# ══════════════════════════════════════════════════════════

def write_paper_numbers(cpu_loocv, cpu_summary, gpu_loocv, gpu_summary,
                        cpu_fitted, gpu_fitted, cpu_feats, gpu_feats):
    lines = ["="*65, "EPMO PAPER NUMBERS  v4.0", "="*65]

    # best by ConfigAcc — consistent with pkl and fig_3
    best_cpu_model = cpu_summary.sort_values("ConfigAcc_mean", ascending=False).iloc[0]["Model"]
    best_gpu_model = gpu_summary.sort_values("ConfigAcc_mean", ascending=False).iloc[0]["Model"]

    for domain, loocv_df, summary_df, fitted, feats, forced in [
        ("Model_A (CPU)", cpu_loocv, cpu_summary, cpu_fitted, cpu_feats, best_cpu_model),
        ("Model_B (GPU)", gpu_loocv, gpu_summary, gpu_fitted, gpu_feats, best_gpu_model),
    ]:
        best = summary_df[summary_df["Model"] == forced].iloc[0]
        lines += [
            f"",
            f"{'─'*45}",
            f"{domain}",
            f"{'─'*45}",
            f"  Best algorithm:  {best['Model']}",
            f"  MAE:             {best['MAE_mean']:.4f} ± {best['MAE_std']:.4f}s",
            f"  ConfigAcc:       {best['ConfigAcc_mean']*100:.0f}%",
            f"  Top-3 Acc:       {best['Top3Acc_mean']*100:.0f}%",
            f"  Avg Saving%:     {best['Saving%_mean']:+.1f}%",
            f"  Avg Regret%:     {best['Regret%_mean']:.2f}%",
            f"  PctOracle:       {best['PctOracle_mean']:.1f}%",
            f"",
            f"  Per-fold ({best['Model']}):",
        ]
        best_rows = loocv_df[loocv_df["model"]==best["Model"]]
        for _, r in best_rows.iterrows():
            h = "✓" if r["ConfigAcc"] else "✗"
            lines.append(
                f"    {h} {r['test_env']:<10} MAE={r['MAE']:.4f}  "
                f"Saving={r['Saving%']:+.1f}%  "
                f"Regret={r['Regret%']:.2f}%  "
                f"Oracle={r['PctOracle']:.1f}%")

        # feature importance
        for name in ["Random Forest","Gradient Boosting","XGBoost"]:
            m = fitted.get(name)
            if m and hasattr(m, "feature_importances_"):
                imp_sorted = sorted(zip(feats, m.feature_importances_),
                                    key=lambda x: x[1], reverse=True)
                lines += [f"", f"  Top features ({name}):"]
                for feat, imp in imp_sorted[:5]:
                    lines.append(f"    {feat:<22} {imp:.4f}")
                break

    lines += ["", "="*65,
              "REGRET INTERPRETATION:",
              "  Regret < 2%  → Excellent (EPMO ≈ Oracle)",
              "  Regret < 5%  → Success threshold",
              "  Regret 5-15% → Acceptable",
              "  Regret > 15% → Poor prediction",
              "="*65]

    text = "\n".join(lines)
    path = os.path.join(OUT_DIR, "paper_numbers.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  Saved: {path}")
    print("\n" + text)


# ══════════════════════════════════════════════════════════
# INFERENCE FUNCTION
# ══════════════════════════════════════════════════════════

def predict_best_config(hardware_profile: dict,
                        model_cpu, model_gpu,
                        cpu_feats: list, gpu_feats: list) -> dict:
    """
    Given a hardware profile dict, predict the optimal DataLoader config.

    hardware_profile keys:
      cpu_logical, ram_gb, gpu_available, gpu_vram_gb

    Returns:
      {'num_workers': int, 'batch_size': int, 'predicted_time': float,
       'model_used': str}
    """
    if hardware_profile.get("gpu_available", 0) == 1:
        model, feats, model_name = model_gpu, gpu_feats, "Model_B (GPU)"
    else:
        model, feats, model_name = model_cpu, cpu_feats, "Model_A (CPU)"

    hw = hardware_profile
    candidates = []
    for w in WORKER_OPTIONS:
        for b in BATCH_OPTIONS:
            row = {
                "cpu_logical":   hw.get("cpu_logical", 4),
                "ram_gb":        hw.get("ram_gb", 8.0),
                "gpu_available": hw.get("gpu_available", 0),
                "gpu_vram_gb":   hw.get("gpu_vram_gb", 0.0),
                "num_workers":   w,
                "batch_size":    b,
                "workers_x_cpu": w * hw.get("cpu_logical", 4),
                "batch_x_ram":   b / hw.get("ram_gb", 8.0),
                "cpu_per_worker": hw.get("cpu_logical", 4) / (w + 1),
                "workers_x_gpu": w * hw.get("gpu_available", 0),
                "batch_x_gpu":   b * hw.get("gpu_available", 0),
            }
            feat_vals = [[row[f] for f in feats]]
            pred_time = float(model.predict(feat_vals)[0])
            candidates.append({"num_workers": w, "batch_size": b,
                                "predicted_time": pred_time})

    best = min(candidates, key=lambda x: x["predicted_time"])
    best["model_used"] = model_name
    return best


def save_inference_function(out_dir):
    """Save a standalone predict.py for deployment."""
    code = '''"""
EPMO Inference  —  predict.py
Usage:
  python predict.py
Requires: model_cpu.pkl, model_gpu.pkl in same folder
"""
import pickle, numpy as np

WORKER_OPTIONS = [0, 1, 2, 4]
BATCH_OPTIONS  = [32, 64, 128, 256]
FEATS = ["cpu_logical","ram_gb","gpu_available","gpu_vram_gb",
         "num_workers","batch_size","workers_x_cpu","batch_x_ram",
         "cpu_per_worker","workers_x_gpu","batch_x_gpu"]

with open("model_cpu.pkl","rb") as f: model_cpu = pickle.load(f)
with open("model_gpu.pkl","rb") as f: model_gpu = pickle.load(f)

def predict_best_config(hw: dict) -> dict:
    gpu = hw.get("gpu_available", 0)
    model = model_gpu if gpu else model_cpu
    name  = "Model_B (GPU)" if gpu else "Model_A (CPU)"
    best_time, best_cfg = float("inf"), {}
    for w in WORKER_OPTIONS:
        for b in BATCH_OPTIONS:
            row = [hw.get("cpu_logical",4), hw.get("ram_gb",8.0),
                   gpu, hw.get("gpu_vram_gb",0.0), w, b,
                   w*hw.get("cpu_logical",4), b/hw.get("ram_gb",8.0),
                   hw.get("cpu_logical",4)/(w+1), w*gpu, b*gpu]
            t = float(model.predict([row])[0])
            if t < best_time:
                best_time = t
                best_cfg = {"num_workers":w,"batch_size":b,
                            "predicted_time":round(t,4),"model":name}
    return best_cfg

if __name__ == "__main__":
    import psutil, torch
    hw = {
        "cpu_logical":  psutil.cpu_count(logical=True),
        "ram_gb":       round(psutil.virtual_memory().total/(1024**3),2),
        "gpu_available":int(torch.cuda.is_available()),
        "gpu_vram_gb":  round(torch.cuda.get_device_properties(0).total_memory/(1024**3),2)
                        if torch.cuda.is_available() else 0.0,
    }
    print("Hardware:", hw)
    result = predict_best_config(hw)
    print("\\nEPMO Recommended Config:")
    print(f"  num_workers  = {result['num_workers']}")
    print(f"  batch_size   = {result['batch_size']}")
    print(f"  predicted_time = {result['predicted_time']}s")
    print(f"  model_used   = {result['model']}")
'''
    path = os.path.join(out_dir, "predict.py")
    with open(path, "w") as f:
        f.write(code)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("="*65)
    print("EPMO Training & Evaluation  v4.0")
    print("Two-model architecture: Model_A (CPU) | Model_B (GPU)")
    print("="*65)

    # ── Load ──
    print("\n[1/6] Loading data ...")
    cpu_df = load("epmo_data_cpu.csv")
    gpu_df = load("epmo_data_gpu.csv")
    print(f"  CPU: {len(cpu_df)} rows, {cpu_df['env_id'].nunique()} envs")
    print(f"  GPU: {len(gpu_df)} rows, {gpu_df['env_id'].nunique()} envs")

    # ── LOOCV ──
    print("\n[2/6] Running LOOCV ...")
    cpu_results, cpu_fitted, cpu_envs = run_loocv(cpu_df, CPU_FEATS, "Model_A CPU")
    gpu_results, gpu_fitted, gpu_envs = run_loocv(gpu_df, GPU_FEATS, "Model_B GPU")

    # ── Aggregate ──
    print("\n[3/6] Aggregating results ...")
    cpu_loocv, cpu_summary = aggregate(cpu_results, cpu_envs)
    gpu_loocv, gpu_summary = aggregate(gpu_results, gpu_envs)

    cpu_loocv.to_csv(os.path.join(OUT_DIR,"table_loocv_cpu.csv"), index=False)
    cpu_summary.to_csv(os.path.join(OUT_DIR,"table_summary_cpu.csv"), index=False)
    gpu_loocv.to_csv(os.path.join(OUT_DIR,"table_loocv_gpu.csv"), index=False)
    gpu_summary.to_csv(os.path.join(OUT_DIR,"table_summary_gpu.csv"), index=False)

    print_tables(cpu_loocv, cpu_summary, "Model_A — CPU")
    print_tables(gpu_loocv, gpu_summary, "Model_B — GPU")

    # ── Save models (best by ConfigAcc — deployment metric) ──
    best_cpu_name = cpu_summary.sort_values("ConfigAcc_mean", ascending=False).iloc[0]["Model"]
    best_gpu_name = gpu_summary.sort_values("ConfigAcc_mean", ascending=False).iloc[0]["Model"]
    with open(os.path.join(OUT_DIR,"model_cpu.pkl"),"wb") as f:
        pickle.dump(cpu_fitted[best_cpu_name], f)
    with open(os.path.join(OUT_DIR,"model_gpu.pkl"),"wb") as f:
        pickle.dump(gpu_fitted[best_gpu_name], f)
    print(f"\n  Best CPU model: {best_cpu_name} → model_cpu.pkl")
    print(f"  Best GPU model: {best_gpu_name} → model_gpu.pkl")

    # ── Figures ──
    print("\n[4/6] Generating figures ...")
    fig_heatmaps(cpu_df, "CPU Environments", "cpu")
    fig_heatmaps(gpu_df, "GPU Environments", "gpu")
    fig_model_comparison(cpu_summary, "Model_A CPU", "cpu")
    fig_model_comparison(gpu_summary, "Model_B GPU", "gpu")
    fig_speedup(cpu_loocv, cpu_summary, "Model_A CPU", "cpu")
    fig_speedup(gpu_loocv, gpu_summary, "Model_B GPU", "gpu")
    fig_amdahl(cpu_df)
    fig_feature_importance(cpu_fitted, CPU_FEATS, "Model_A CPU", "cpu")
    fig_feature_importance(gpu_fitted, GPU_FEATS, "Model_B GPU", "gpu")
    fig_decision_tree(cpu_fitted, CPU_FEATS, "cpu")
    fig_pred_vs_actual(cpu_results, cpu_df, "Model_A CPU", "cpu")
    fig_pred_vs_actual(gpu_results, gpu_df, "Model_B GPU", "gpu")

    # ── Paper numbers ──
    print("\n[5/6] Writing paper numbers ...")
    write_paper_numbers(cpu_loocv, cpu_summary, gpu_loocv, gpu_summary,
                        cpu_fitted, gpu_fitted, CPU_FEATS, GPU_FEATS)

    # ── Inference function ──
    print("\n[6/6] Saving inference script ...")
    save_inference_function(OUT_DIR)

    print(f"\n{'='*65}")
    print(f"DONE. All outputs in: {OUT_DIR}/")
    print(f"  Models:  model_cpu.pkl, model_gpu.pkl")
    print(f"  Figures: fig_1 through fig_7 (cpu + gpu variants)")
    print(f"  Tables:  table_loocv_cpu/gpu.csv, table_summary_cpu/gpu.csv")
    print(f"  Paper:   paper_numbers.txt")
    print(f"  Deploy:  predict.py")
    print(f"\nNEXT: python epmo_test.py  (held-out test evaluation)")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
