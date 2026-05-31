import numpy as np
import pandas as pd

from src.models.adapter import GraphAnomalyDetector


class ReplayBuffer:
    def __init__(self, max_size: int = 500):
        self.max_size = max_size
        self._buffer: pd.DataFrame = pd.DataFrame()

    def update(self, df: pd.DataFrame, scores: np.ndarray) -> None:
        n_keep = min(self.max_size // 2, len(df))
        n_anomalous = n_keep // 2
        n_random = n_keep - n_anomalous

        top_indices = np.argsort(scores)[-n_anomalous:]
        random_indices = np.random.choice(len(df), size=min(n_random, len(df)), replace=False)
        combined = np.union1d(top_indices, random_indices)
        new_samples = df.iloc[combined]

        self._buffer = pd.concat([self._buffer, new_samples], ignore_index=True)
        if len(self._buffer) > self.max_size:
            self._buffer = self._buffer.sample(self.max_size, random_state=42).reset_index(drop=True)

    def sample(self, n: int) -> pd.DataFrame:
        if len(self._buffer) == 0 or n == 0:
            return pd.DataFrame()
        n = min(n, len(self._buffer))
        return self._buffer.sample(n, random_state=42).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self._buffer)


class GraphAnomalyDetectorWithReplay(GraphAnomalyDetector):
    def __init__(self, buffer_size: int = 500, replay_ratio: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.buffer = ReplayBuffer(buffer_size)
        self.replay_ratio = replay_ratio

    def fit(self, df: pd.DataFrame) -> "GraphAnomalyDetectorWithReplay":
        replay_df = self.buffer.sample(int(len(df) * self.replay_ratio))
        if len(replay_df) > 0:
            augmented_df = pd.concat([df, replay_df], ignore_index=True)
        else:
            augmented_df = df

        graph = self.graph_builder.build(augmented_df)
        self.model.fit(graph)

        scores = self.score_samples(df)
        self.buffer.update(df, scores)
        return self
