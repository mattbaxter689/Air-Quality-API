import mlflow.pytorch
import mlflow.pytorch
import mlflow
import torch.nn as nn
from sklearn.pipeline import Pipeline
import joblib
import os
import logging

logger = logging.getLogger(name="weather_api")


def load_mlflow_model() -> tuple[nn.Module, Pipeline]:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URL"))
    client = mlflow.tracking.MlflowClient(
        tracking_uri=os.getenv("MLFLOW_TRACKING_URL")
    )
    registered_model_name = "air_quality"
    # Get all versions for this model
    all_versions = client.search_model_versions(
        f"name='{registered_model_name}'"
    )

    # Filter by your custom tag 'status' == 'Production'
    production_versions = [v for v in all_versions]
    for v in production_versions:
        logger.info(
            f"Version: {v.version}, Run ID: {v.run_id}, Source: {v.source}"
        )
    if not production_versions:
        raise RuntimeError("No Production model versions found")

    # Choose the latest version by creation timestamp or version number
    latest_version = max(
        production_versions,
        key=lambda v: int(v.version),  # or v.creation_timestamp if you prefer
    )

    model_uri = f"models:/{registered_model_name}/{latest_version.version}"
    model = mlflow.pytorch.load_model(model_uri)

    run_id = latest_version.run_id
    preprocessor_path = "preprocessing_pipeline.joblib"
    preprocessor = client.download_artifacts(
        run_id=run_id, path=preprocessor_path, dst_path="processor"
    )
    processor = joblib.load(preprocessor)
    return model, processor
