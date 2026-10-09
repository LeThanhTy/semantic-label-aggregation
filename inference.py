import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from preprocessing import normalize_text

DEFAULT_THRESHOLD = 0.30


class NewsClassifier:
    """Inference-only wrapper for the trained Word2Vec + LSTM model."""

    def __init__(self, artifact_dir: str | Path):
        self.artifact_dir = Path(artifact_dir)
        self.model_path = self.artifact_dir / "word2vec_lstm.keras"
        self.vocab_path = self.artifact_dir / "vocab.json"
        self.labels_path = self.artifact_dir / "labels.json"
        self.group_path = self.artifact_dir / "label_to_group.json"
        self.cluster_names_path = self.artifact_dir / "cluster_names.json"

        for path in [self.model_path, self.vocab_path, self.labels_path]:
            if not path.exists():
                raise FileNotFoundError(f"Missing required artifact: {path}")

        self.model = tf.keras.models.load_model(self.model_path, compile=False)
        self.labels = self._load_json(self.labels_path)
        vocab = self._load_json(self.vocab_path)

        if not isinstance(vocab, list) or len(vocab) < 2:
            raise ValueError("vocab.json must contain the saved Keras vocabulary list.")
        if not isinstance(self.labels, list) or not self.labels:
            raise ValueError("labels.json must contain a non-empty label list.")

        input_shape = self.model.input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        self.sequence_length = int(input_shape[1])

        # Reconstruct the exact TextVectorization configuration used by the notebook.
        self.vectorizer = layers.TextVectorization(
            max_tokens=len(vocab),
            standardize=None,
            split="whitespace",
            output_mode="int",
            output_sequence_length=self.sequence_length,
        )
        self.vectorizer.set_vocabulary(vocab)

        output_shape = self.model.output_shape
        if isinstance(output_shape, list):
            output_shape = output_shape[0]
        if int(output_shape[-1]) != len(self.labels):
            raise ValueError(
                f"Model outputs {output_shape[-1]} labels but labels.json contains "
                f"{len(self.labels)} labels."
            )

        self.label_to_group = {}
        if self.group_path.exists():
            raw = self._load_json(self.group_path)
            if isinstance(raw, dict):
                self.label_to_group = raw

        self.cluster_names = {}
        if self.cluster_names_path.exists():
            raw = self._load_json(self.cluster_names_path)
            if isinstance(raw, dict):
                self.cluster_names = {str(k): str(v) for k, v in raw.items()}

    @staticmethod
    def _load_json(path: Path):
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _pretty_group(raw_group: str) -> str:
        """Turn 'Nhóm 5 [khoa học / ai / chuyển đổi số]' into a readable title."""
        if not isinstance(raw_group, str):
            return str(raw_group)
        if "[" in raw_group and "]" in raw_group:
            inside = raw_group.split("[", 1)[1].rsplit("]", 1)[0].strip()
            return inside.title()
        return raw_group

    def predict(self, text: str, threshold: float = DEFAULT_THRESHOLD) -> Dict:
        clean = normalize_text(text)
        if not clean:
            return {
                "clean_text": "",
                "probabilities": np.zeros(len(self.labels), dtype=np.float32),
                "predictions": [],
                "groups": [],
                "token_count": 0,
            }

        ids = self.vectorizer(tf.constant([clean]))
        probabilities = self.model.predict(ids, verbose=0)[0].astype(float)

        predictions = [
            {
                "label": self.labels[i],
                "probability": float(probabilities[i]),
            }
            for i in range(len(self.labels))
            if probabilities[i] >= threshold
        ]
        predictions.sort(key=lambda x: x["probability"], reverse=True)

        groups = []
        seen = set()
        for item in predictions:
            raw_group = self.label_to_group.get(item["label"])
            if raw_group is None:
                continue
            group_name = self._pretty_group(raw_group)
            if group_name not in seen:
                groups.append(group_name)
                seen.add(group_name)

        token_count = len(clean.split())
        return {
            "clean_text": clean,
            "probabilities": probabilities,
            "predictions": predictions,
            "groups": groups,
            "token_count": token_count,
        }

    def top_predictions(self, probabilities: np.ndarray, k: int = 10) -> List[Dict]:
        indices = np.argsort(probabilities)[::-1][: min(k, len(self.labels))]
        return [
            {
                "rank": rank,
                "label": self.labels[i],
                "probability": float(probabilities[i]),
            }
            for rank, i in enumerate(indices, 1)
        ]
