# Continual Graph Anomaly Detection — IEEE-CIS Fraud

A continual/lifelong anomaly-detection pipeline for the [IEEE-CIS Fraud
Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data)
dataset. It builds a single global **homogeneous transaction graph**, streams
it through **6 monthly temporal concepts** with
[pyCLAD](https://github.com/lifelonglab/pyclad), and detects anomalies with
[PyGOD](https://github.com/pygod-team/pygod) GNN detectors (DOMINANT, CoLA,
OCGNN, AnomalyDAE) via a custom PyGOD→pyCLAD adapter.

## Setup (uv)

```bash
# 1. Install uv (once): https://docs.astral.sh/uv/
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. From the project root, create the venv and install everything
uv sync

# 3. Install the compiled NeighborLoader sampling backend (REQUIRED —
#    PyG's neighbor sampling needs pyg-lib or torch-sparse). Match the
#    suffix to your installed torch version and CUDA flavor:
uv pip install pyg-lib -f https://data.pyg.org/whl/torch-2.12.0+cpu.html    # local CPU dev
uv pip install pyg-lib -f https://data.pyg.org/whl/torch-2.12.0+cu124.html  # GPU cluster
```

(`uv run python -c "import torch; print(torch.__version__)"` tells you which
wheel index page to use.)

`pyproject.toml` pins a CUDA 12.4 PyTorch index for Linux (the GPU cluster);
on macOS/Windows uv installs the default CPU build automatically. Adjust the
`pytorch-cu124` index URL if the cluster runs a different CUDA version.

## Data

```bash
mkdir -p data
kaggle competitions download -c ieee-fraud-detection -p data
unzip data/ieee-fraud-detection.zip -d data
```

Only `data/train_transaction.csv` is required (it contains `isFraud` labels
and `TransactionDT`).

## Usage

```bash
# Full run: DOMINANT, Naive continual strategy + static offline baseline
uv run fraud-detection --data-dir data --model dominant --strategy naive

# Replay strategy with CoLA, V-features enabled (cluster-sized run)
uv run fraud-detection --model cola --strategy replay --use-v-features --epochs 100

# Quick smoke test on a laptop (50k rows, few epochs, CPU)
uv run fraud-detection --nrows 50000 --epochs 5 --batch-size 1024 --cpu
```

Key flags: `--model {dominant,cola,ocgnn,anomalydae}`,
`--strategy {naive,replay,cumulative}`, `--mode {both,continual,static}`,
`--batch-size`, `--num-neigh`, `--n-concepts`, `--replay-buffer-size`.
Results land in `outputs/` as JSON; a comparison table is printed at the end:

```
=== ROC-AUC per monthly concept: continual vs static offline ===
concept      static (frozen)  continual @learn  continual @final
month_1               0.71..            0.72..            0.69..
...
```

## How it works

### Graph design (RAM-conscious)

- **Nodes** = individual transactions; node features are
  StandardScaler-normalized numericals (`TransactionAmt`, `C1–C14`, `D1–D15`,
  optionally `V1–V339`) plus frequency-encoded categoricals (`ProductCD`,
  `card1–card6`, `addr`, email domains, `M1–M9`).
- **Edges** connect transactions sharing identical composite entity keys:
  `card1..card6`, `(card1, addr1)`, `(card1, P_emaildomain)`. Within an entity
  group each transaction links to its `--max-neighbors-per-key` most recent
  predecessors rather than all pairs — edge count stays **O(N·k)** instead of
  the O(N²) blow-up a clique per hub entity would cause.
- Everything is assembled into one global `torch_geometric.data.Data` with
  `x`, `edge_index`, `y`, and per-node `transaction_id` / `transaction_dt`
  mapping attributes used for concept indexing.

### Temporal concepts

`TransactionDT` (seconds offset, ~6 months span) is bucketed into 6
equal-width intervals → `month_1 … month_6`. Each concept is split temporally
(first 70% train / last 30% test); train concepts keep only legitimate
transactions by default (one-class regime — pass `--train-with-fraud` to
disable). Concepts are wrapped into a pyCLAD `ConceptsDataset` with 6 train
and 6 test concepts in chronological order.

### The PyGOD→pyCLAD adapter

pyCLAD models receive plain arrays; PyGOD needs PyG `Data`. The bridge:
concept payloads are **global node indices** `(n, 1)`. `PyGODAdapter.fit(X)`
induces the subgraph spanned by those indices (`Data.subgraph`, order
preserving) and trains a fresh PyGOD detector on it with `batch_size > 0` and
`num_neigh` set — PyGOD then trains through PyG's **`NeighborLoader`**
(mini-batch neighborhood sampling) so GPU memory stays bounded no matter how
many indices a strategy hands over. `predict(X)` scores the induced subgraph
and returns a pyCLAD `PredictionResults(y_pred, anomaly_scores)` aligned
row-by-row with `X`.

Because index arrays flow through pyCLAD unchanged, replay buffers and the
cumulative strategy work untouched: **strategies differ in which node sets
they hand to `fit`** (Naive = current concept, Replay = current + balanced
buffer, Cumulative = all seen), and the detector is retrained from scratch on
that set — the same semantics as pyCLAD's bundled PyOD adapters.

### Evaluation scenarios

1. **Continual** — `ConceptAwareScenario` walks the 6-concept stream with the
   chosen strategy; `ConceptMetricCallback(RocAuc)` records the full
   learned×evaluated ROC-AUC matrix plus `ContinualAverage`,
   `BackwardTransfer`, `ForwardTransfer`.
2. **Static offline** — the same detector is trained once on `month_1` train
   data, frozen, and evaluated on all 6 test months (ROC-AUC + F1). Score
   decay over months quantifies concept drift; the delta to the continual
   column quantifies the value of lifelong adaptation.

## Project layout

```
src/fraud_detection/
├── config.py          # dataclass configs (data / graph / concepts / model)
├── preprocessing.py   # CSV loading, scaling, frequency encoding
├── graph.py           # global homogeneous graph assembly
├── concepts.py        # 6 monthly concepts -> pyCLAD ConceptsDataset
├── adapter.py         # PyGODAdapter (pyCLAD Model <-> PyGOD detector)
├── baseline.py        # static offline baseline
└── run_experiment.py  # CLI: continual vs static comparison
```

## Sizing notes (128 GB RAM / GPU node)

- Full dataset: ~590k nodes; base features ≈ 110 MB, with `--use-v-features`
  ≈ 900 MB float32. Edges at default settings ≈ 15–20M — the global graph fits
  in a few GB of host RAM.
- The GPU footprint is controlled entirely by `--batch-size` × `--num-neigh`
  × `--num-layers` (NeighborLoader subgraphs), not by graph size. Increase
  `--batch-size` (e.g. 16384) on an A100/H100 to shorten epochs.
- **Reconstruction detectors (DOMINANT, AnomalyDAE) build a dense N×N
  adjacency target in host RAM** (a PyGOD-internal design), so their cost is
  quadratic in the *fitted node set*: one monthly concept (~70k nodes ≈ 20 GB)
  is fine on a 128 GB node, but the `cumulative` strategy on the full stream
  is not — pair `cumulative` with `ocgnn` or `cola`, which have no dense
  adjacency. The adapter also self-loops isolated nodes, working around a
  PyGOD `to_dense_adj` shape bug on subgraphs with trailing isolated nodes.
- Preprocessing + graph build are single-process pandas/numpy and take a few
  minutes; rerun cost is dominated by detector training.
