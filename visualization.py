from pathlib import Path

import numpy as np
import pandas as pd


def load_csv(path: Path):
    if path.exists():
        return pd.read_csv(path)
    return None


def load_label_semantic_data(artifact_dir: Path):
    centroids_path = artifact_dir / "label_centroids.npy"
    assignments_path = artifact_dir / "label_cluster_assignments.csv"
    if not centroids_path.exists() or not assignments_path.exists():
        return None, None
    return np.load(centroids_path), pd.read_csv(assignments_path)
