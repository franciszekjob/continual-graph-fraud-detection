import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.loader import load_ieee_cis_windowed
from src.models.adapter import GraphAnomalyDetector
from src.models.replay import GraphAnomalyDetectorWithReplay
from src.evaluation.metrics import compute_roc_auc, print_metrics_table

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def parse_args():
    p = argparse.ArgumentParser(description="Graph-based fraud detection experiment")

    # Data
    p.add_argument("--n-windows", type=int, default=6)
    p.add_argument("--max-rows-per-window", type=int, default=50_000)
    p.add_argument("--v-cols-max", type=int, default=39, help="Use V1..V<n> as node features (max 339)")

    # Replay
    p.add_argument("--buffer-size", type=int, default=800)
    p.add_argument("--replay-ratio", type=float, default=0.15)

    # Model
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--hidden-channels", type=int, default=64)

    return p.parse_args()


def run_continual_eval(detector_cls, windows, detector_kwargs=None):
    detector_kwargs = detector_kwargs or {}
    T = len(windows)
    auc_matrix = np.full((T, T), np.nan)

    detector = detector_cls(**detector_kwargs)

    for t in range(T):
        print(f"  Training on window {t}...")
        detector.fit(windows[t])

        for j in range(T):
            y_true = windows[j]["isFraud"].values
            if len(np.unique(y_true)) < 2:
                continue
            scores = detector.score_samples(windows[j])
            auc_matrix[t, j] = compute_roc_auc(y_true, scores)
            print(f"    AUC[train={t}, eval={j}] = {auc_matrix[t, j]:.4f}")

    return auc_matrix


def plot_auc_matrices(baseline_matrix, replay_matrix, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, matrix, title in [
        (axes[0], baseline_matrix, "Baseline (no CL)"),
        (axes[1], replay_matrix, "Replay"),
    ]:
        im = ax.imshow(matrix, vmin=0.4, vmax=1.0, cmap="RdYlGn")
        ax.set_title(title)
        ax.set_xlabel("Eval window")
        ax.set_ylabel("Train window")
        plt.colorbar(im, ax=ax)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if not np.isnan(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)

    plt.tight_layout()
    path = os.path.join(save_dir, "auc_matrices.png")
    plt.savefig(path, dpi=150)
    print(f"\nSaved plot to {path}")


def main():
    args = parse_args()

    # Apply V-cols range before importing preprocessor defaults
    from src.data import preprocessor
    preprocessor.NUMERIC_FEATURE_COLS = (
        ["TransactionAmt"]
        + [f"C{i}" for i in range(1, 15)]
        + [f"D{i}" for i in range(1, 16)]
        + [f"V{i}" for i in range(1, args.v_cols_max + 1)]
    )

    print(f"Config: windows={args.n_windows}, rows/window={args.max_rows_per_window}, "
          f"V1-V{args.v_cols_max}, epochs={args.epochs}, hidden={args.hidden_channels}, "
          f"buffer={args.buffer_size}, replay_ratio={args.replay_ratio}")

    print("\nLoading IEEE-CIS dataset (chunked)...")
    windows = load_ieee_cis_windowed(DATA_DIR, n_windows=args.n_windows, max_rows_per_window=args.max_rows_per_window)

    for i, w in enumerate(windows):
        print(f"  Window {i}: {len(w)} rows, {w['isFraud'].mean():.2%} fraud")

    model_kwargs = {"epoch": args.epochs, "hid_dim": args.hidden_channels}

    print("\n--- Running BASELINE (no continual learning) ---")
    baseline_matrix = run_continual_eval(GraphAnomalyDetector, windows, detector_kwargs={"model_kwargs": model_kwargs})

    print("\n--- Running REPLAY ---")
    replay_matrix = run_continual_eval(
        GraphAnomalyDetectorWithReplay,
        windows,
        detector_kwargs={
            "buffer_size": args.buffer_size,
            "replay_ratio": args.replay_ratio,
            "model_kwargs": model_kwargs,
        },
    )

    print("\n=== RESULTS ===")
    print_metrics_table(baseline_matrix, replay_matrix)

    plot_auc_matrices(baseline_matrix, replay_matrix, RESULTS_DIR)


if __name__ == "__main__":
    main()
