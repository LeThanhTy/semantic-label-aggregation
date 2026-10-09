import argparse
import csv
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np
from sklearn.decomposition import PCA

from inference import DEFAULT_THRESHOLD, NewsClassifier

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
STATIC_FILES = {
    "/": (BASE_DIR / "templates" / "index.html", "text/html; charset=utf-8"),
    "/index.html": (BASE_DIR / "templates" / "index.html", "text/html; charset=utf-8"),
    "/static/style.css": (BASE_DIR / "static" / "style.css", "text/css; charset=utf-8"),
    "/static/script.js": (BASE_DIR / "static" / "script.js", "text/javascript; charset=utf-8"),
}
MAX_REQUEST_BYTES = 1_000_000


class DashboardServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], classifier: NewsClassifier):
        super().__init__(address, DashboardHandler)
        self.classifier = classifier


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlsplit(self.path).path
        if path in STATIC_FILES:
            self._serve_static(path)
            return

        handlers = {
            "/api/config": self._get_config,
            "/api/labels": self._get_labels,
            "/api/semantic-space": self._get_semantic_space,
            "/api/experiments": self._get_experiments,
        }
        handler = handlers.get(path)
        if handler is None:
            self._send_json({"error": "Not found."}, status=404)
            return
        handler()

    def do_POST(self):
        if urlsplit(self.path).path != "/api/predict":
            self._send_json({"error": "Not found."}, status=404)
            return

        content_length = self.headers.get("Content-Length")
        try:
            body_length = int(content_length) if content_length is not None else -1
        except ValueError:
            self._send_json({"error": "Invalid Content-Length header."}, status=400)
            return
        if body_length < 0 or body_length > MAX_REQUEST_BYTES:
            self._send_json({"error": "Request body must be between 0 and 1,000,000 bytes."}, status=413)
            return

        try:
            payload = json.loads(self.rfile.read(body_length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json({"error": "Request body must be valid UTF-8 JSON."}, status=400)
            return
        if not isinstance(payload, dict):
            self._send_json({"error": "Request body must be a JSON object."}, status=400)
            return

        text = payload.get("text")
        threshold = payload.get("threshold", DEFAULT_THRESHOLD)
        if not isinstance(text, str) or not text.strip():
            self._send_json({"error": "The text field must contain a non-empty article."}, status=400)
            return
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            self._send_json({"error": "Threshold must be a number between 0 and 1."}, status=400)
            return
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            self._send_json({"error": "Threshold must be a finite number between 0 and 1."}, status=400)
            return

        result = self.server.classifier.predict(text, float(threshold))
        self._send_json({
            "groups": result["groups"],
            "predictions": result["predictions"],
            "top_predictions": self.server.classifier.top_predictions(result["probabilities"], 10),
            "token_count": result["token_count"],
            "clean_text": result["clean_text"],
        })

    def _get_config(self):
        classifier = self.server.classifier
        self._send_json({
            "default_threshold": DEFAULT_THRESHOLD,
            "model": {
                "label_count": len(classifier.labels),
                "embedding_dimensions": 100,
                "lstm_units": 128,
                "sequence_length": classifier.sequence_length,
            },
            "artifacts": {
                "semantic_space": (ARTIFACT_DIR / "label_centroids.npy").exists()
                and (ARTIFACT_DIR / "label_cluster_assignments.csv").exists(),
                "model_comparison": (ARTIFACT_DIR / "model_comparison.csv").exists(),
                "raw_vs_filtered_metrics": (ARTIFACT_DIR / "raw_vs_filtered_metrics.csv").exists(),
            },
        })

    def _get_labels(self):
        classifier = self.server.classifier
        labels = []
        for label in classifier.labels:
            raw_group = classifier.label_to_group.get(label)
            labels.append({
                "label": label,
                "group": classifier._pretty_group(raw_group) if raw_group else None,
            })
        self._send_json({"labels": labels})

    def _get_semantic_space(self):
        centroids_path = ARTIFACT_DIR / "label_centroids.npy"
        assignments_path = ARTIFACT_DIR / "label_cluster_assignments.csv"
        missing = [
            path.name
            for path in (centroids_path, assignments_path)
            if not path.exists()
        ]
        if missing:
            self._send_json({"available": False, "missing_artifacts": missing})
            return

        centroids = np.load(centroids_path)
        with assignments_path.open("r", encoding="utf-8", newline="") as file:
            assignments = list(csv.DictReader(file))
        if centroids.ndim != 2 or centroids.shape[0] != len(assignments):
            self._send_json(
                {"error": "Semantic centroid rows must match the label assignments."},
                status=500,
            )
            return
        if centroids.shape[0] < 2 or centroids.shape[1] < 2:
            self._send_json({"error": "At least two valid label centroids are required."}, status=500)
            return

        points = PCA(n_components=2, random_state=42).fit_transform(centroids)
        self._send_json({
            "available": True,
            "points": [
                {
                    "label": row.get("label", f"Nhãn {index + 1}"),
                    "cluster_id": row.get("cluster_id", ""),
                    "x": float(point[0]),
                    "y": float(point[1]),
                }
                for index, (row, point) in enumerate(zip(assignments, points))
            ],
        })

    def _get_experiments(self):
        results = {}
        for key, filename in (
            ("model_comparison", "model_comparison.csv"),
            ("raw_vs_filtered_metrics", "raw_vs_filtered_metrics.csv"),
        ):
            path = ARTIFACT_DIR / filename
            if not path.exists():
                results[key] = None
                continue
            with path.open("r", encoding="utf-8-sig", newline="") as file:
                results[key] = list(csv.DictReader(file))
        self._send_json(results)

    def _serve_static(self, path: str):
        file_path, content_type = STATIC_FILES[path]
        if not file_path.is_file():
            self._send_json({"error": f"Required interface file is missing: {file_path.name}."}, status=404)
            return
        content = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, payload: dict[str, Any], status: int = 200):
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)


def main():
    parser = argparse.ArgumentParser(description="Serve the Vietnamese News Intelligence dashboard.")
    parser.add_argument("--host", default="127.0.0.1", help="Interface to bind (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on (default: 8000).")
    args = parser.parse_args()

    classifier = NewsClassifier(ARTIFACT_DIR)
    server = DashboardServer((args.host, args.port), classifier)
    print(f"Dashboard ready at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
