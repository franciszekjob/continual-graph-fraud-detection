#!/usr/bin/env python3
"""Visualize continual/static experiment results using seaborn.

Scans the `ares_results` folder for JSON outputs and produces:
- heatmaps of continual metric matrices
- per-concept comparison bars (static vs continual @learn vs @final)
- summary metric bar plots (ContinualAverage vs static average)
- forward/backward transfer bars

Usage:
  uv run python scripts/visualize_results.py --results-dir ares_results --out-dir ares_results/plots
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_results(results_dir: Path) -> Dict[str, Any]:
    results = {"continual": [], "static": []}
    for p in results_dir.rglob("*.json"):
        try:
            payload = json.loads(p.read_text())
        except Exception:
            continue
        name = p.name
        if (
            name.startswith("continual_")
            or "concept_metric_callback_ROC-AUC" in payload
        ):
            # continual-style file
            info = payload.get("concept_metric_callback_ROC-AUC") or payload.get(
                "concept_metric_callback_ROC-AUC"
            )
            if info is None:
                # sometimes wrapped under different keys; try to locate
                for k in payload:
                    if k.startswith("concept_metric_callback"):
                        info = payload[k]
                        break
            model = payload.get("model", {}).get("name") or payload.get("model")
            strategy = payload.get("strategy", {}).get("name") or "continual"
            results["continual"].append(
                {
                    "path": str(p),
                    "model": model,
                    "strategy": strategy,
                    "info": info,
                }
            )
        elif name.startswith("static_") or "per_concept" in payload:
            model = payload.get("model") or payload.get("model") or p.stem
            results["static"].append({"path": str(p), "model": model, "info": payload})
    return results


def build_dataframes(results: Dict[str, Any]):
    rows_matrix = []
    rows_cont_summary = []
    rows_static = []

    for item in results["continual"]:
        model = item["model"]
        strategy = item["strategy"]
        info = item["info"] or {}
        metrics = info.get("metrics", {})
        concepts = info.get("concepts_order", [])
        matrix = info.get("metric_matrix", {})
        for learned, row in matrix.items():
            for evaluated, val in row.items():
                rows_matrix.append(
                    {
                        "model": model,
                        "strategy": strategy,
                        "learned": learned,
                        "evaluated": evaluated,
                        "roc_auc": float(val) if val is not None else float("nan"),
                    }
                )
        for mname, mval in metrics.items():
            rows_cont_summary.append(
                {"model": model, "strategy": strategy, "metric": mname, "value": mval}
            )

    for item in results["static"]:
        model = item["model"]
        info = item["info"] or {}
        per = info.get("per_concept", {})
        avg = info.get("average_roc_auc")
        for concept, vals in per.items():
            rows_static.append(
                {
                    "model": model,
                    "concept": concept,
                    "roc_auc": float(vals.get("roc_auc")),
                }
            )
        if avg is not None:
            rows_static.append(
                {"model": model, "concept": "average", "roc_auc": float(avg)}
            )

    df_matrix = pd.DataFrame(rows_matrix)
    df_cont = pd.DataFrame(rows_cont_summary)
    df_static = pd.DataFrame(rows_static)

    if not df_matrix.empty:
        df_matrix = df_matrix.drop_duplicates(
            subset=["model", "strategy", "learned", "evaluated"], keep="last"
        )
    if not df_cont.empty:
        df_cont = df_cont.drop_duplicates(
            subset=["model", "strategy", "metric"], keep="last"
        )
    if not df_static.empty:
        df_static = df_static.drop_duplicates(subset=["model", "concept"], keep="last")

    return df_matrix, df_cont, df_static


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def plot_heatmaps(df_matrix: pd.DataFrame, out_dir: Path):
    ensure_dir(out_dir)
    if df_matrix.empty:
        return
    for (model, strategy), group in df_matrix.groupby(["model", "strategy"]):
        pivot = group.pivot_table(
            index="learned",
            columns="evaluated",
            values="roc_auc",
            aggfunc="mean",
        )
        pivot = pivot.sort_index().reindex(sorted(pivot.columns), axis=1)
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="vlag", vmin=0.0, vmax=1.0)
        plt.title(f"Continual ROC-AUC matrix — {model} / {strategy}")
        plt.xlabel("evaluated concept")
        plt.ylabel("learned concept")
        out_file = out_dir / f"heatmap_{sanitize(model)}_{sanitize(strategy)}.png"
        plt.tight_layout()
        plt.savefig(out_file)
        plt.close()


def sanitize(s: str) -> str:
    return (
        s.replace(" ", "_").replace("/", "_").replace("\n", "_").replace("-", "_")
        if isinstance(s, str)
        else str(s)
    )


def plot_per_concept_comparisons(
    df_matrix: pd.DataFrame, df_static: pd.DataFrame, out_dir: Path
):
    ensure_dir(out_dir)
    if df_matrix.empty or df_static.empty:
        return
    # Determine concepts order from data
    concepts = sorted({c for c in df_matrix["evaluated"].unique()})
    last_learned_map = {}
    for (model, strategy), group in df_matrix.groupby(["model", "strategy"]):
        learned_order = sorted(group["learned"].unique())
        if not learned_order:
            continue
        last = learned_order[-1]
        last_learned_map[(model, strategy)] = last
        static_subset = df_static[
            df_static["model"].str.contains(model.split("-")[-1], na=False)
        ]

        static_vals = {
            row["concept"]: row["roc_auc"] for _, row in static_subset.iterrows()
        }

        data = []
        for concept in concepts:
            static_v = static_vals.get(concept, float("nan"))
            at_learn = group[
                (group["learned"] == concept) & (group["evaluated"] == concept)
            ]["roc_auc"]
            at_final = group[
                (group["learned"] == last) & (group["evaluated"] == concept)
            ]["roc_auc"]
            data.append(
                {
                    "model": model,
                    "strategy": strategy,
                    "concept": concept,
                    "static": float(static_v)
                    if not math.isnan(static_v)
                    else float("nan"),
                    "continual_at_learn": float(at_learn.iloc[0])
                    if not at_learn.empty
                    else float("nan"),
                    "continual_at_final": float(at_final.iloc[0])
                    if not at_final.empty
                    else float("nan"),
                }
            )
        df_plot = pd.DataFrame(data).melt(
            id_vars=["model", "strategy", "concept"],
            value_vars=["static", "continual_at_learn", "continual_at_final"],
            var_name="source",
            value_name="roc_auc",
        )
        plt.figure(figsize=(10, 5))
        sns.barplot(data=df_plot, x="concept", y="roc_auc", hue="source")
        plt.ylim(0, 1)
        plt.title(f"Per-concept ROC-AUC — {model} / {strategy}")
        plt.tight_layout()
        out_file = out_dir / f"per_concept_{sanitize(model)}_{sanitize(strategy)}.png"
        plt.savefig(out_file)
        plt.close()


def plot_summary_metrics(df_cont: pd.DataFrame, df_static: pd.DataFrame, out_dir: Path):
    ensure_dir(out_dir)
    if df_cont.empty or df_static.empty:
        return
    # ContinualAverage per model/strategy
    cont_avg = df_cont[df_cont["metric"] == "ContinualAverage"].copy()
    cont_avg["model_short"] = cont_avg["model"].astype(str)
    # Static average
    static_avg_rows = []
    for model, group in df_static.groupby("model"):
        avg_row = group[group["concept"] == "average"]
        if not avg_row.empty:
            static_avg_rows.append(
                {"model": model, "value": float(avg_row.iloc[0]["roc_auc"])}
            )
    if not static_avg_rows:
        return
    df_static_avg = pd.DataFrame(static_avg_rows)

    # Merge and plot
    plt.figure(figsize=(8, 5))
    # plot continual average grouped by model/strategy
    sns.barplot(data=cont_avg, x="model", y="value", hue="strategy")
    plt.ylim(0, 1)
    plt.title("ContinualAverage (ROC-AUC) per model/strategy")
    plt.tight_layout()
    plt.savefig(out_dir / "continual_average_by_model_strategy.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.barplot(data=df_static_avg, x="model", y="value")
    plt.ylim(0, 1)
    plt.title("Static average ROC-AUC (trained on month_1)")
    plt.tight_layout()
    plt.savefig(out_dir / "static_average_by_model.png")
    plt.close()


def plot_transfer_metrics(df_cont: pd.DataFrame, out_dir: Path):
    ensure_dir(out_dir)
    if df_cont.empty:
        return
    df_tf = df_cont[
        df_cont["metric"].isin(["ForwardTransfer", "BackwardTransfer"])
    ].copy()
    if df_tf.empty:
        return
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_tf, x="model", y="value", hue="metric")
    plt.title("Forward / Backward Transfer per model/strategy")
    plt.tight_layout()
    plt.savefig(out_dir / "transfer_metrics_by_model.png")
    plt.close()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("ares_results"))
    parser.add_argument("--out-dir", type=Path, default=Path("ares_results/plots"))
    args = parser.parse_args(argv)

    results = load_results(args.results_dir)
    df_matrix, df_cont, df_static = build_dataframes(results)

    out = Path(args.out_dir)
    ensure_dir(out)

    plot_heatmaps(df_matrix, out)
    plot_per_concept_comparisons(df_matrix, df_static, out)
    plot_summary_metrics(df_cont, df_static, out)
    plot_transfer_metrics(df_cont, out)

    print("Plots written to:", out)


if __name__ == "__main__":
    main()
