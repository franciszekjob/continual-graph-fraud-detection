"""Static offline baseline: train once on month 1, freeze, evaluate forward.

This is the contrast case for the continual strategies — no sequential
updates, so any score decay across later months quantifies concept drift.
"""

from __future__ import annotations

import logging
from typing import Any

from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from sklearn.metrics import f1_score, roc_auc_score

from fraud_detection.adapter import PyGODAdapter

logger = logging.getLogger(__name__)


def run_static_baseline(adapter: PyGODAdapter, dataset: ConceptsDataset) -> dict[str, Any]:
    train_concepts = list(dataset.train_concepts())
    test_concepts = list(dataset.test_concepts())

    first = train_concepts[0]
    logger.info("Static baseline: training %s once on %s", adapter.name(), first.name)
    adapter.fit(first.data)

    per_concept: dict[str, dict[str, float]] = {}
    for concept in test_concepts:
        y_pred, anomaly_scores = adapter.predict(concept.data)
        per_concept[concept.name] = {
            "roc_auc": float(roc_auc_score(concept.labels, anomaly_scores)),
            "f1": float(f1_score(concept.labels, y_pred, zero_division=0)),
        }
        logger.info(
            "Static eval %s: ROC-AUC=%.4f F1=%.4f",
            concept.name,
            per_concept[concept.name]["roc_auc"],
            per_concept[concept.name]["f1"],
        )

    aucs = [m["roc_auc"] for m in per_concept.values()]
    return {
        "mode": "static_offline",
        "model": adapter.name(),
        "trained_on": first.name,
        "per_concept": per_concept,
        "average_roc_auc": sum(aucs) / len(aucs),
    }
