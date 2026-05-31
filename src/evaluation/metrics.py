import numpy as np
from sklearn.metrics import roc_auc_score


def compute_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, scores)


def compute_forgetting(auc_matrix: np.ndarray) -> float:
    """
    Average forgetting across tasks.
    For each task j, forgetting = max AUC achieved on j before final evaluation - final AUC on j.
    auc_matrix[i, j] = AUC of model after training on window i, evaluated on window j.
    """
    T = auc_matrix.shape[0]
    forgetting_per_task = []
    for j in range(T - 1):
        past_aucs = [auc_matrix[i, j] for i in range(j + 1, T) if not np.isnan(auc_matrix[i, j])]
        best_past = auc_matrix[j, j]
        if past_aucs:
            final = past_aucs[-1]
            forgetting_per_task.append(best_past - final)
    return float(np.nanmean(forgetting_per_task)) if forgetting_per_task else float("nan")


def compute_bwt(auc_matrix: np.ndarray) -> float:
    """
    Backward Transfer: how learning new tasks affects performance on previous ones.
    BWT = mean over j<T of (final_AUC_on_j - AUC_on_j_right_after_training_j)
    """
    T = auc_matrix.shape[0]
    bwt_vals = []
    for j in range(T - 1):
        final = auc_matrix[T - 1, j]
        initial = auc_matrix[j, j]
        if not np.isnan(final) and not np.isnan(initial):
            bwt_vals.append(final - initial)
    return float(np.nanmean(bwt_vals)) if bwt_vals else float("nan")


def compute_fwt(auc_matrix: np.ndarray) -> float:
    """
    Forward Transfer: how training on past tasks helps on future unseen tasks.
    FWT = mean over j>0 of (AUC on j before training on j - random baseline 0.5)
    """
    T = auc_matrix.shape[0]
    fwt_vals = []
    for j in range(1, T):
        # AUC on window j evaluated after training on window j-1 (before seeing j)
        auc_before = auc_matrix[j - 1, j]
        if not np.isnan(auc_before):
            fwt_vals.append(auc_before - 0.5)
    return float(np.nanmean(fwt_vals)) if fwt_vals else float("nan")


def print_metrics_table(baseline_matrix: np.ndarray, replay_matrix: np.ndarray) -> None:
    print(f"\n{'Metric':<20} {'Baseline':>12} {'Replay':>12}")
    print("-" * 46)

    for label, matrix in [("Baseline", baseline_matrix), ("Replay", replay_matrix)]:
        pass

    T = baseline_matrix.shape[0]
    diag_baseline = [baseline_matrix[i, i] for i in range(T) if not np.isnan(baseline_matrix[i, i])]
    diag_replay = [replay_matrix[i, i] for i in range(T) if not np.isnan(replay_matrix[i, i])]

    rows = [
        ("Avg ROC-AUC", np.nanmean(diag_baseline), np.nanmean(diag_replay)),
        ("Forgetting", compute_forgetting(baseline_matrix), compute_forgetting(replay_matrix)),
        ("BWT", compute_bwt(baseline_matrix), compute_bwt(replay_matrix)),
        ("FWT", compute_fwt(baseline_matrix), compute_fwt(replay_matrix)),
    ]
    for name, b_val, r_val in rows:
        print(f"{name:<20} {b_val:>12.4f} {r_val:>12.4f}")
