#!/usr/bin/env python3
"""Generate the final seaborn figures from the pulled ARES result JSONs.

Unlike ``visualize_results.py`` (which auto-scans a directory), this script uses
a *curated* list of runs so we can:
  * relabel the CPU ``dominant_replay`` run as the v-features (391-feature)
    variant — pyCLAD's JsonOutputWriter names every replay run
    ``continual_dominant_replay.json`` regardless of ``--use-v-features``, so the
    391-feature run overwrote the file name but its ContinualAverage (0.7288...)
    matches the v-features Slurm log exactly;
  * include ``dominant_naive`` (ContinualAverage 0.599) — NOTE this run used a
    reduced config (epoch=5, batch_size=1024, ~7.6 s total) and is not directly
    comparable to the heavier runs; the two ares_results/{cpu,gpu} copies are
    byte-identical, so it appears once.

Produces, under ``plots/``:
  * heatmap_<run>.png            — per-run 6x6 learned x evaluated ROC-AUC
  * heatmaps_grid.png            — all runs in one figure
  * continual_average.png        — ContinualAverage per model/strategy (+ static)
  * transfer_metrics.png         — Forward / Backward Transfer per run
  * training_times.png           — total train time per run (log scale, by device)
  * training_time_per_concept.png— train time per concept (line)
  * summary_heatmap.png          — model x run ContinualAverage overview

Usage:
  uv run python scripts/plot_results.py
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "plots"

R1 = ROOT / "ares_results"
R2 = ROOT / "ares_results2" / "fraud_results2"


@dataclass(frozen=True)
class Run:
    label: str          # display label
    model: str          # OCGNN / CoLA / DOMINANT
    strategy: str       # Naive / Cumulative / Replay
    features: int       # input feature count (52 base, 391 with V-features)
    device: str         # gpu / cpu
    continual: Path     # continual_*.json
    static: Path | None # static_*.json (optional)


# Curated set of completed, trustworthy runs. Order controls plot order.
RUNS: list[Run] = [
    Run("OCGNN / Naive",        "OCGNN",    "Naive",      52, "gpu",
        R1 / "gpu/outputs/continual_ocgnn_naive.json",      R1 / "gpu/outputs/static_ocgnn.json"),
    Run("OCGNN / Cumulative",   "OCGNN",    "Cumulative", 52, "gpu",
        R1 / "gpu/outputs/continual_ocgnn_cumulative.json", R1 / "gpu/outputs/static_ocgnn.json"),
    Run("OCGNN / Replay",       "OCGNN",    "Replay",     52, "gpu",
        R1 / "gpu/outputs/continual_ocgnn_replay.json",     R1 / "gpu/outputs/static_ocgnn.json"),
    Run("CoLA / Naive",         "CoLA",     "Naive",      52, "gpu",
        R1 / "gpu/outputs/continual_cola_naive.json",       R1 / "gpu/outputs/static_cola.json"),
    Run("CoLA / Cumulative",    "CoLA",     "Cumulative", 52, "cpu",
        R2 / "cpu/outputs/continual_cola_cumulative.json",  R2 / "cpu/outputs/static_cola.json"),
    Run("CoLA / Replay",        "CoLA",     "Replay",     52, "cpu",
        R2 / "cpu/outputs/continual_cola_replay.json",      R2 / "cpu/outputs/static_cola.json"),
    Run("DOMINANT / Naive",     "DOMINANT", "Naive",      52, "cpu",
        R1 / "gpu/outputs/continual_dominant_naive.json",   R1 / "gpu/outputs/static_dominant.json"),
    Run("DOMINANT / Replay",    "DOMINANT", "Replay",     52, "gpu",
        R2 / "gpu/outputs/continual_dominant_replay.json",  R2 / "gpu/outputs/static_dominant.json"),
    Run("DOMINANT / Replay (V-feat)", "DOMINANT", "Replay", 391, "cpu",
        R2 / "cpu/outputs/continual_dominant_replay.json",  R2 / "cpu/outputs/static_dominant.json"),
]


def cm_key(d: dict) -> dict:
    return next(d[k] for k in d if k.startswith("concept_metric_callback"))


def load_run(run: Run) -> dict:
    d = json.loads(run.continual.read_text())
    cm = cm_key(d)
    times = d.get("time_evaluation_callback", {}).get("time_by_concept", {})
    static = json.loads(run.static.read_text()) if run.static and run.static.exists() else None
    return {"cm": cm, "times": times, "static": static}


def tidy_label(run: Run) -> str:
    return run.label


def build_frames():
    matrix_rows, summary_rows, time_rows, static_rows = [], [], [], []
    for run in RUNS:
        data = load_run(run)
        cm = data["cm"]
        order = cm.get("concepts_order", [])
        matrix = cm.get("metric_matrix", {})
        metrics = cm.get("metrics", {})

        for learned, row in matrix.items():
            for evaluated, val in row.items():
                matrix_rows.append({
                    "run": run.label, "model": run.model, "strategy": run.strategy,
                    "learned": learned, "evaluated": evaluated,
                    "roc_auc": float(val) if val is not None else math.nan,
                })

        summary_rows.append({
            "run": run.label, "model": run.model, "strategy": run.strategy,
            "device": run.device, "features": run.features,
            "ContinualAverage": metrics.get("ContinualAverage", math.nan),
            "ForwardTransfer": metrics.get("ForwardTransfer", math.nan),
            "BackwardTransfer": metrics.get("BackwardTransfer", math.nan),
            "static_average": (data["static"] or {}).get("average_roc_auc", math.nan),
        })

        total_train = 0.0
        for concept, t in data["times"].items():
            tr = float(t.get("train_time", math.nan))
            total_train += tr if not math.isnan(tr) else 0.0
            time_rows.append({
                "run": run.label, "model": run.model, "strategy": run.strategy,
                "device": run.device, "concept": concept, "train_time": tr,
            })
        summary_rows[-1]["total_train_time"] = total_train

        st = data["static"]
        if st:
            for concept, vals in st.get("per_concept", {}).items():
                static_rows.append({
                    "run": run.label, "model": run.model, "concept": concept,
                    "roc_auc": float(vals["roc_auc"]),
                })

    return (pd.DataFrame(matrix_rows), pd.DataFrame(summary_rows),
            pd.DataFrame(time_rows), pd.DataFrame(static_rows))


def sanitize(s: str) -> str:
    return (s.replace(" ", "").replace("/", "_").replace("(", "")
            .replace(")", "").replace("-", "_"))


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_heatmaps(df_matrix: pd.DataFrame):
    for run, group in df_matrix.groupby("run", sort=False):
        pivot = group.pivot_table(index="learned", columns="evaluated",
                                  values="roc_auc", aggfunc="mean")
        pivot = pivot.sort_index().reindex(sorted(pivot.columns), axis=1)
        plt.figure(figsize=(7, 5.5))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="rocket_r",
                    vmin=0.3, vmax=0.8, linewidths=0.5, linecolor="white",
                    cbar_kws={"label": "ROC-AUC"})
        plt.title(f"Continual ROC-AUC matrix — {run}")
        plt.xlabel("evaluated concept")
        plt.ylabel("learned concept")
        plt.tight_layout()
        plt.savefig(OUT / f"heatmap_{sanitize(run)}.png", dpi=150)
        plt.close()


def plot_heatmaps_grid(df_matrix: pd.DataFrame):
    runs = list(dict.fromkeys(df_matrix["run"]))
    ncol = 4
    nrow = math.ceil(len(runs) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.6 * nrow))
    axes = axes.flatten()
    for ax, run in zip(axes, runs):
        group = df_matrix[df_matrix["run"] == run]
        pivot = group.pivot_table(index="learned", columns="evaluated",
                                  values="roc_auc", aggfunc="mean")
        pivot = pivot.sort_index().reindex(sorted(pivot.columns), axis=1)
        sns.heatmap(pivot, ax=ax, annot=True, fmt=".2f", cmap="rocket_r",
                    vmin=0.3, vmax=0.8, cbar=False,
                    xticklabels=[c.replace("month_", "m") for c in pivot.columns],
                    yticklabels=[c.replace("month_", "m") for c in pivot.index])
        ax.set_title(run, fontsize=10)
        ax.set_xlabel(""); ax.set_ylabel("")
    for ax in axes[len(runs):]:
        ax.axis("off")
    fig.suptitle("Continual ROC-AUC matrices (learned × evaluated)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "heatmaps_grid.png", dpi=150)
    plt.close(fig)


def plot_continual_average(df_sum: pd.DataFrame):
    df = df_sum.melt(id_vars=["run", "model", "strategy"],
                     value_vars=["ContinualAverage", "static_average"],
                     var_name="kind", value_name="roc_auc")
    df["kind"] = df["kind"].map({"ContinualAverage": "Continual avg",
                                 "static_average": "Static baseline"})
    plt.figure(figsize=(11, 5.5))
    ax = sns.barplot(data=df, x="run", y="roc_auc", hue="kind",
                     palette=["#2a6f97", "#bdbdbd"])
    ax.axhline(0.5, ls="--", lw=1, color="crimson", label="random (0.5)")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=8, padding=2)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("ROC-AUC")
    ax.set_xlabel("")
    ax.set_title("Continual average vs. static baseline ROC-AUC per run")
    plt.xticks(rotation=25, ha="right")
    ax.legend(title="", loc="upper left")
    plt.tight_layout()
    plt.savefig(OUT / "continual_average.png", dpi=150)
    plt.close()


def plot_transfer(df_sum: pd.DataFrame):
    df = df_sum.melt(id_vars=["run"],
                     value_vars=["ForwardTransfer", "BackwardTransfer"],
                     var_name="metric", value_name="value")
    plt.figure(figsize=(11, 5.5))
    ax = sns.barplot(data=df, x="run", y="value", hue="metric",
                     palette={"ForwardTransfer": "#1b9e77",
                              "BackwardTransfer": "#d95f02"})
    ax.axhline(0.0, lw=1, color="black")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=8, padding=2)
    ax.set_ylabel("transfer (ROC-AUC delta)")
    ax.set_xlabel("")
    ax.set_title("Forward & Backward Transfer per run")
    plt.xticks(rotation=25, ha="right")
    ax.legend(title="")
    plt.tight_layout()
    plt.savefig(OUT / "transfer_metrics.png", dpi=150)
    plt.close()


def _fmt_duration(v: float) -> str:
    return f"{v:.0f} s" if v < 90 else (f"{v/60:.1f} min" if v < 3600 else f"{v/3600:.1f} h")


def plot_training_times(df_sum: pd.DataFrame):
    df = df_sum.copy()
    df["label"] = df["run"] + "  (" + df["device"] + ")"
    df = df.sort_values("total_train_time")
    colors = df["device"].map({"gpu": "#4477aa", "cpu": "#ee6677"}).tolist()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(df["label"], df["total_train_time"], color=colors)
    ax.set_xscale("log")
    ax.bar_label(bars, labels=[_fmt_duration(v) for v in df["total_train_time"]],
                 fontsize=8, padding=3)
    ax.set_xlabel("total training time over 6 concepts (s, log scale)")
    ax.set_ylabel("")
    ax.set_title("Training cost per run — DOMINANT dwarfs the lighter detectors")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c)
               for c in ("#4477aa", "#ee6677")]
    ax.legend(handles, ["gpu", "cpu"], title="device", loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT / "training_times.png", dpi=150)
    plt.close()


def plot_training_time_per_concept(df_time: pd.DataFrame):
    df = df_time.copy()
    df["concept"] = df["concept"].str.replace("month_", "m")
    plt.figure(figsize=(10, 5.5))
    ax = sns.lineplot(data=df, x="concept", y="train_time", hue="run",
                      marker="o", sort=False)
    ax.set_yscale("log")
    ax.set_ylabel("train time per concept (s, log scale)")
    ax.set_xlabel("concept (temporal month)")
    ax.set_title("Per-concept training time across the stream")
    ax.legend(title="", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(OUT / "training_time_per_concept.png", dpi=150)
    plt.close()


def plot_summary_heatmap(df_sum: pd.DataFrame):
    pivot = df_sum.pivot_table(index="model", columns="strategy",
                               values="ContinualAverage", aggfunc="max")
    # keep a sensible strategy order
    cols = [c for c in ["Naive", "Cumulative", "Replay"] if c in pivot.columns]
    pivot = pivot.reindex(columns=cols)
    plt.figure(figsize=(6.5, 4))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="rocket_r",
                vmin=0.3, vmax=0.8, linewidths=0.5, linecolor="white",
                cbar_kws={"label": "ContinualAverage ROC-AUC"})
    plt.title("ContinualAverage by model × strategy\n(DOMINANT/Replay = best available, incl. V-features)")
    plt.xlabel("strategy"); plt.ylabel("model")
    plt.tight_layout()
    plt.savefig(OUT / "summary_heatmap.png", dpi=150)
    plt.close()


def main():
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.7)
    OUT.mkdir(parents=True, exist_ok=True)

    df_matrix, df_sum, df_time, df_static = build_frames()

    plot_heatmaps(df_matrix)
    plot_heatmaps_grid(df_matrix)
    plot_continual_average(df_sum)
    plot_transfer(df_sum)
    plot_training_times(df_sum)
    plot_training_time_per_concept(df_time)
    plot_summary_heatmap(df_sum)

    print(f"Wrote {len(list(OUT.glob('*.png')))} figures to {OUT}")
    print("\nSummary table:")
    cols = ["run", "device", "features", "ContinualAverage",
            "ForwardTransfer", "BackwardTransfer", "static_average", "total_train_time"]
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(df_sum[cols].to_string(index=False))


if __name__ == "__main__":
    main()
