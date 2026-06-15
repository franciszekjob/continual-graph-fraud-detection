"""End-to-end execution pipeline.

Builds the global transaction graph, the 6-concept temporal stream, and runs
the requested PyGOD detector in:
  1. continual mode — a pyCLAD strategy (Naive / Replay / Cumulative) executed
     through the concept stream by ConceptAwareScenario, with ROC-AUC tracked
     across every (learned concept, evaluated concept) pair;
  2. static offline mode — train once on month 1, evaluate frozen on months 1-6.

Example:
    uv run fraud-detection --data-dir data --model dominant --strategy replay
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib

import torch
from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
from pyclad.callbacks.evaluation.time_evaluation import TimeEvaluationCallback
from pyclad.metrics.base.roc_auc import RocAuc
from pyclad.metrics.continual.average_continual import ContinualAverage
from pyclad.metrics.continual.backward_transfer import BackwardTransfer
from pyclad.metrics.continual.forward_transfer import ForwardTransfer
from pyclad.output.json_writer import JsonOutputWriter
from pyclad.scenarios.concept_aware import ConceptAwareScenario
from pyclad.strategies.baselines.cumulative import CumulativeStrategy
from pyclad.strategies.baselines.naive import NaiveStrategy
from pyclad.strategies.replay.buffers.adaptive_balanced import (
    AdaptiveBalancedReplayBuffer,
)
from pyclad.strategies.replay.replay import ReplayEnhancedStrategy
from pyclad.strategies.replay.selection.random import RandomSelection
from pygod.detector import CoLA, DOMINANT, OCGNN, AnomalyDAE

from fraud_detection.adapter import PyGODAdapter
from fraud_detection.baseline import run_static_baseline
from fraud_detection.concepts import build_concepts_dataset
from fraud_detection.config import ConceptConfig, DataConfig, GraphConfig, ModelConfig
from fraud_detection.graph import build_transaction_graph
from fraud_detection.preprocessing import load_and_preprocess

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "dominant": DOMINANT,
    "cola": CoLA,
    "ocgnn": OCGNN,
    "anomalydae": AnomalyDAE,
}


def make_strategy(name: str, model: PyGODAdapter, replay_buffer_size: int):
    if name == "naive":
        return NaiveStrategy(model)
    if name == "cumulative":
        return CumulativeStrategy(model)
    if name == "replay":
        buffer = AdaptiveBalancedReplayBuffer(
            selection_method=RandomSelection(), max_size=replay_buffer_size
        )
        return ReplayEnhancedStrategy(model, buffer)
    raise ValueError(f"Unknown strategy '{name}'")


def run_continual(
    graph,
    dataset,
    model_cfg: ModelConfig,
    strategy_name: str,
    replay_buffer_size: int,
    output_dir: pathlib.Path,
) -> dict:
    adapter = PyGODAdapter(
        graph, MODEL_REGISTRY[model_cfg.name], model_cfg.detector_params()
    )
    strategy = make_strategy(strategy_name, adapter, replay_buffer_size)
    metric_callback = ConceptMetricCallback(
        base_metric=RocAuc(),
        metrics=[ContinualAverage(), BackwardTransfer(), ForwardTransfer()],
    )
    callbacks = [metric_callback, TimeEvaluationCallback()]

    ConceptAwareScenario(dataset, strategy=strategy, callbacks=callbacks).run()

    out_path = output_dir / f"continual_{model_cfg.name}_{strategy_name}.json"
    JsonOutputWriter(out_path).write([adapter, dataset, strategy, *callbacks])
    logger.info("Continual results written to %s", out_path)
    return metric_callback.info()


def extract_continual_summary(callback_info: dict) -> tuple[dict, dict, list[str]]:
    """Pull (metric_matrix, summarized metrics, concept order) out of the
    ConceptMetricCallback info dict (its top-level key embeds the metric name)."""
    key = next(k for k in callback_info if k.startswith("concept_metric_callback"))
    payload = callback_info[key]
    return payload["metric_matrix"], payload["metrics"], payload["concepts_order"]


def print_comparison(callback_info: dict, static_results: dict) -> None:
    matrix, summary, order = extract_continual_summary(callback_info)
    last_learned = order[-1]

    print("\n=== ROC-AUC per monthly concept: continual vs static offline ===")
    header = f"{'concept':<10}{'static (frozen)':>17}{'continual @learn':>18}{'continual @final':>18}"
    print(header)
    print("-" * len(header))
    for concept in order:
        static_auc = static_results["per_concept"][concept]["roc_auc"]
        at_learn = matrix[concept].get(concept, float("nan"))
        at_final = matrix[last_learned].get(concept, float("nan"))
        print(f"{concept:<10}{static_auc:>17.4f}{at_learn:>18.4f}{at_final:>18.4f}")

    print("\nContinual summary metrics:")
    for name, value in summary.items():
        print(f"  {name}: {value:.4f}")
    print(f"Static average ROC-AUC: {static_results['average_roc_auc']:.4f}")
    print(
        "\n(@learn = evaluated right after that concept was learned; "
        "@final = evaluated after the whole stream — the gap shows "
        "forgetting/adaptation. Static column never updates after month 1.)"
    )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continual GNN fraud detection")
    parser.add_argument("--data-dir", type=pathlib.Path, default=pathlib.Path("data"))
    parser.add_argument(
        "--output-dir", type=pathlib.Path, default=pathlib.Path("outputs")
    )
    parser.add_argument("--model", choices=sorted(MODEL_REGISTRY), default="dominant")
    parser.add_argument(
        "--strategy", choices=["naive", "replay", "cumulative"], default="naive"
    )
    parser.add_argument(
        "--mode", choices=["both", "continual", "static"], default="both"
    )
    parser.add_argument("--n-concepts", type=int, default=6)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hid-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.004)
    parser.add_argument("--contamination", type=float, default=0.035)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="PyGOD mini-batch size; >0 enables NeighborLoader sampling",
    )
    parser.add_argument(
        "--num-neigh",
        type=int,
        default=10,
        help="neighbors sampled per layer by NeighborLoader",
    )
    parser.add_argument(
        "--max-neighbors-per-key",
        type=int,
        default=5,
        help="temporal edges per node per entity key",
    )
    parser.add_argument("--replay-buffer-size", type=int, default=20_000)
    parser.add_argument("--use-v-features", action="store_true")
    parser.add_argument(
        "--train-with-fraud",
        action="store_true",
        help="keep fraud rows in train concepts (default: normal-only)",
    )
    parser.add_argument(
        "--nrows", type=int, default=None, help="row cap for quick smoke tests"
    )
    parser.add_argument(
        "--cpu", action="store_true", help="force CPU even if CUDA is available"
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = DataConfig(
        data_dir=args.data_dir, use_v_features=args.use_v_features, nrows=args.nrows
    )
    graph_cfg = GraphConfig(max_neighbors_per_key=args.max_neighbors_per_key)
    concept_cfg = ConceptConfig(
        n_concepts=args.n_concepts,
        train_ratio=args.train_ratio,
        normal_only_training=not args.train_with_fraud,
    )
    model_cfg = ModelConfig(
        name=args.model,
        hid_dim=args.hid_dim,
        num_layers=args.num_layers,
        epoch=args.epochs,
        lr=args.lr,
        contamination=args.contamination,
        batch_size=args.batch_size,
        num_neigh=args.num_neigh,
        gpu=0 if (torch.cuda.is_available() and not args.cpu) else -1,
    )
    logger.info("Device: %s", "cuda:0" if model_cfg.gpu >= 0 else "cpu")
    logger.info("Threads: %s", torch.get_num_threads())
    logger.info("Threads inter-op: %s", torch.get_num_interop_threads())

    edge_cols = {c for group in graph_cfg.edge_key_groups for c in group}
    prepared = load_and_preprocess(data_cfg, edge_cols)
    graph = build_transaction_graph(prepared, graph_cfg)
    dataset = build_concepts_dataset(
        prepared.labels, prepared.transaction_dt, concept_cfg
    )

    continual_info = None
    if args.mode in ("both", "continual"):
        continual_info = run_continual(
            graph,
            dataset,
            model_cfg,
            args.strategy,
            args.replay_buffer_size,
            args.output_dir,
        )

    static_results = None
    if args.mode in ("both", "static"):
        adapter = PyGODAdapter(
            graph, MODEL_REGISTRY[model_cfg.name], model_cfg.detector_params()
        )
        static_results = run_static_baseline(adapter, dataset)
        out_path = args.output_dir / f"static_{model_cfg.name}.json"
        out_path.write_text(json.dumps(static_results, indent=2))
        logger.info("Static baseline results written to %s", out_path)

    if continual_info and static_results:
        print_comparison(continual_info, static_results)
    elif continual_info:
        matrix, summary, order = extract_continual_summary(continual_info)
        print(
            json.dumps(
                {"metric_matrix": matrix, "summary": summary, "order": order}, indent=2
            )
        )
    elif static_results:
        print(json.dumps(static_results, indent=2))


if __name__ == "__main__":
    main()
