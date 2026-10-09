from __future__ import annotations

import csv
import json
import math
import threading
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sklearn.decomposition import PCA


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

ARTIFACT_DIR = BASE_DIR / "artifacts"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Vietnamese News Intelligence",
    description="Vietnamese Multi-Label News Classification with Word2Vec + LSTM",
    version="1.0.0",
)


# ============================================================
# STATIC FILES & TEMPLATES
# ============================================================

if STATIC_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=STATIC_DIR),
        name="static",
    )

templates = Jinja2Templates(directory=TEMPLATES_DIR)


# ============================================================
# LAZY MODEL LOADING
# ============================================================
#
# IMPORTANT:
# Do NOT load TensorFlow / NewsClassifier when Uvicorn starts.
#
# Render needs the web server to bind to $PORT quickly.
# The model will only be loaded when /api/predict,
# /api/config, /api/labels, etc. actually need it.
# ============================================================

_classifier = None
_classifier_lock = threading.Lock()


def get_classifier():
    """
    Load NewsClassifier only when it is actually needed.

    This prevents TensorFlow + .keras model loading from
    blocking FastAPI startup on Render.
    """
    global _classifier

    if _classifier is None:
        with _classifier_lock:
            if _classifier is None:
                from inference import NewsClassifier

                _classifier = NewsClassifier(ARTIFACT_DIR)

    return _classifier


# ============================================================
# BASIC ROUTES
# ============================================================


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Main dashboard page.
    """
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
        },
    )


@app.get("/health")
async def health():
    """
    Lightweight health-check endpoint.

    This endpoint intentionally does NOT load TensorFlow
    or the ML model.
    """
    return {
        "status": "ok",
        "service": "Vietnamese News Intelligence",
        "model_loaded": _classifier is not None,
    }


# ============================================================
# PREDICTION API
# ============================================================


@app.post("/api/predict")
async def predict(data: dict[str, Any]):
    """
    Predict labels for a Vietnamese news article.

    Expected JSON:

    {
        "text": "...",
        "threshold": 0.30
    }
    """

    # --------------------------------------------------------
    # Validate payload
    # --------------------------------------------------------

    if not isinstance(data, dict):
        return JSONResponse(
            status_code=400,
            content={
                "error": "Request body must be a JSON object."
            },
        )

    text = data.get("text")

    if not isinstance(text, str) or not text.strip():
        return JSONResponse(
            status_code=400,
            content={
                "error": "The text field must contain a non-empty article."
            },
        )

    threshold = data.get("threshold", 0.30)

    if isinstance(threshold, bool):
        return JSONResponse(
            status_code=400,
            content={
                "error": "Threshold must be a number between 0 and 1."
            },
        )

    if not isinstance(threshold, (int, float)):
        return JSONResponse(
            status_code=400,
            content={
                "error": "Threshold must be a number between 0 and 1."
            },
        )

    threshold = float(threshold)

    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Threshold must be a finite number between 0 and 1."
            },
        )

    # --------------------------------------------------------
    # Run model
    # --------------------------------------------------------

    try:
        classifier = get_classifier()

        result = classifier.predict(
            text,
            threshold,
        )

        top_predictions = classifier.top_predictions(
            result["probabilities"],
            10,
        )

        return {
            "groups": result["groups"],
            "predictions": result["predictions"],
            "top_predictions": top_predictions,
            "token_count": result["token_count"],
            "clean_text": result["clean_text"],
            "threshold": threshold,
        }

    except FileNotFoundError as exc:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Required model artifact is missing.",
                "detail": str(exc),
            },
        )

    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Model prediction failed.",
                "detail": str(exc),
            },
        )


# ============================================================
# MODEL CONFIGURATION API
# ============================================================


@app.get("/api/config")
async def get_config():
    """
    Return model and artifact information.
    """

    try:
        classifier = get_classifier()

        return {
            "default_threshold": 0.30,
            "model": {
                "name": "Word2Vec + LSTM",
                "label_count": len(classifier.labels),
                "embedding_dimensions": 100,
                "lstm_units": 128,
                "sequence_length": classifier.sequence_length,
                "task": "Vietnamese Multi-Label Text Classification",
            },
            "artifacts": {
                "semantic_space": (
                    (ARTIFACT_DIR / "label_centroids.npy").exists()
                    and (
                        ARTIFACT_DIR
                        / "label_cluster_assignments.csv"
                    ).exists()
                ),
                "model_comparison": (
                    ARTIFACT_DIR
                    / "model_comparison.csv"
                ).exists(),
                "raw_vs_filtered_metrics": (
                    ARTIFACT_DIR
                    / "raw_vs_filtered_metrics.csv"
                ).exists(),
            },
        }

    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Could not load model configuration.",
                "detail": str(exc),
            },
        )


# ============================================================
# LABEL API
# ============================================================


@app.get("/api/labels")
async def get_labels():
    """
    Return all fine-grained labels and their semantic groups.
    """

    try:
        classifier = get_classifier()

        labels = []

        for label in classifier.labels:
            raw_group = classifier.label_to_group.get(label)

            group = None

            if raw_group:
                group = classifier._pretty_group(raw_group)

            labels.append(
                {
                    "label": label,
                    "group": group,
                }
            )

        return {
            "count": len(labels),
            "labels": labels,
        }

    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Could not load labels.",
                "detail": str(exc),
            },
        )


# ============================================================
# SEMANTIC LABEL SPACE
# ============================================================


@app.get("/api/semantic-space")
async def get_semantic_space():
    """
    Return 2D PCA coordinates of semantic label embeddings.

    PCA is used ONLY for visualization.
    The actual semantic grouping is still determined by
    the saved cluster assignments.
    """

    centroids_path = (
        ARTIFACT_DIR / "label_centroids.npy"
    )

    assignments_path = (
        ARTIFACT_DIR
        / "label_cluster_assignments.csv"
    )

    missing = []

    if not centroids_path.exists():
        missing.append(
            centroids_path.name
        )

    if not assignments_path.exists():
        missing.append(
            assignments_path.name
        )

    if missing:
        return {
            "available": False,
            "missing_artifacts": missing,
            "points": [],
        }

    try:
        centroids = np.load(
            centroids_path
        )

        with assignments_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as file:
            assignments = list(
                csv.DictReader(file)
            )

        if centroids.ndim != 2:
            return JSONResponse(
                status_code=500,
                content={
                    "error": (
                        "Semantic centroid data "
                        "must be a 2D matrix."
                    )
                },
            )

        if centroids.shape[0] != len(assignments):
            return JSONResponse(
                status_code=500,
                content={
                    "error": (
                        "Semantic centroid rows must "
                        "match the label assignments."
                    )
                },
            )

        if (
            centroids.shape[0] < 2
            or centroids.shape[1] < 2
        ):
            return JSONResponse(
                status_code=500,
                content={
                    "error": (
                        "At least two valid label "
                        "centroids are required."
                    )
                },
            )

        # ----------------------------------------------------
        # PCA ONLY FOR VISUALIZATION
        # ----------------------------------------------------

        points = PCA(
            n_components=2,
            random_state=42,
        ).fit_transform(centroids)

        result = []

        for index, (row, point) in enumerate(
            zip(assignments, points)
        ):
            result.append(
                {
                    "label": row.get(
                        "label",
                        f"Nhãn {index + 1}",
                    ),
                    "cluster_id": row.get(
                        "cluster_id",
                        "",
                    ),
                    "x": float(point[0]),
                    "y": float(point[1]),
                }
            )

        return {
            "available": True,
            "points": result,
        }

    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "error": (
                    "Could not load semantic "
                    "label space."
                ),
                "detail": str(exc),
            },
        )


# ============================================================
# EXPERIMENT RESULTS
# ============================================================


@app.get("/api/experiments")
async def get_experiments():
    """
    Load experiment CSV files if they exist.
    """

    results = {}

    experiment_files = {
        "model_comparison": (
            "model_comparison.csv"
        ),
        "raw_vs_filtered_metrics": (
            "raw_vs_filtered_metrics.csv"
        ),
    }

    for key, filename in experiment_files.items():

        path = ARTIFACT_DIR / filename

        if not path.exists():
            results[key] = None
            continue

        try:
            with path.open(
                "r",
                encoding="utf-8-sig",
                newline="",
            ) as file:

                results[key] = list(
                    csv.DictReader(file)
                )

        except Exception as exc:

            results[key] = {
                "error": str(exc)
            }

    return results


# ============================================================
# OPTIONAL: MODEL STATUS
# ============================================================


@app.get("/api/status")
async def get_status():
    """
    Return current backend/model status.
    """

    required_artifacts = [
        "word2vec_lstm.keras",
        "vocab.json",
        "labels.json",
        "label_to_group.json",
    ]

    artifacts = {}

    for filename in required_artifacts:

        path = ARTIFACT_DIR / filename

        artifacts[filename] = {
            "exists": path.exists(),
            "size_mb": (
                round(
                    path.stat().st_size
                    / (1024 * 1024),
                    2,
                )
                if path.exists()
                else None
            ),
        }

    return {
        "status": "ok",
        "model_loaded": _classifier is not None,
        "artifacts": artifacts,
    }


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
