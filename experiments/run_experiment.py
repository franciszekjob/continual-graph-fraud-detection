import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.loader import load_ieee_cis
from src.data.preprocessor import temporal_split
from src.models.adapter import GraphAnomalyDetector
from src.models.replay import GraphAnomalyDetectorWithReplay
from src.evaluation.metrics import compute_roc_auc, print_metrics_table

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
N_WINDOWS = 6
MAX_ROWS_PER_WINDOW = 50_000
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


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
    print("Loading IEEE-CIS dataset...")
    df = load_ieee_cis(DATA_DIR)
    print(f"Loaded {len(df)} transactions, {df['isFraud'].mean():.2%} fraud rate")

    print(f"\nSplitting into {N_WINDOWS} temporal windows...")
    windows = temporal_split(df, n_windows=N_WINDOWS)

    # Limit size for prototype speed
    windows = [w.head(MAX_ROWS_PER_WINDOW) for w in windows]
    for i, w in enumerate(windows):
        print(f"  Window {i}: {len(w)} rows, {w['isFraud'].mean():.2%} fraud")

    print("\n--- Running BASELINE (no continual learning) ---")
    baseline_matrix = run_continual_eval(GraphAnomalyDetector, windows)

    print("\n--- Running REPLAY ---")
    replay_matrix = run_continual_eval(
        GraphAnomalyDetectorWithReplay,
        windows,
        detector_kwargs={"buffer_size": 800, "replay_ratio": 0.15},
    )

    print("\n=== RESULTS ===")
    print_metrics_table(baseline_matrix, replay_matrix)

    plot_auc_matrices(baseline_matrix, replay_matrix, RESULTS_DIR)


if __name__ == "__main__":
    main()
