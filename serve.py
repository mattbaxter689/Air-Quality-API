from fastapi import FastAPI
from fastapi.exceptions import HTTPException
import mlflow.pytorch
from api.base_models.models import AirQuality, PredictionResult
from pydantic import ValidationError
import mlflow.pytorch
import mlflow
from threading import Lock
import time
import torch.nn as nn
import torch
from api.logger.init_logger import create_logger
from sklearn.pipeline import Pipeline
from sklearn import set_config
import asyncio
import pandas as pd
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from dotenv import load_dotenv
import joblib
import numpy as np
import os

model_cache = {}
cache_lock = Lock()
logger = create_logger(name="weather_api")
load_dotenv()
set_config(transform_output="pandas")


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


async def refresh_model() -> None:
    while True:
        await asyncio.sleep(3600)
        try:
            model, preprocessor = load_mlflow_model()

            with cache_lock:
                model_cache["model"] = model
                model_cache["preprocessor"] = preprocessor

            logger.info("Model refreshed")
        except Exception as e:
            logger.error(f"Failed to refresh model: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    logger.info("Loading model and preprocessor on API startup")
    model, preprocessor = load_mlflow_model()
    with cache_lock:
        model_cache["model"] = model
        model_cache["preprocessor"] = preprocessor

    asyncio.create_task(refresh_model())
    yield
    logger.info("Shutting down API ...")


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def test() -> None:
    return {"Message": "Hello Air Quality API"}


@app.post(
    "/predict",
    summary="Forecast next hour's qir quality, using last 12 hours worth of time points",
    description="Predicts the next hour's air quality using the last 12 hours worth of air quality data",
)
async def predict(input: AirQuality) -> PredictionResult:

    if len(input.sequence) != 16:
        logger.error(
            f"Data did not contain exactly 16 time points. Contained: {len(input.sequence)}"
        )
        raise ValueError(
            f"Data must contain exactly 16 points, contained {len(input.sequence)}"
        )

    try:
        logger.info("Validating input with Base Model")
        AirQuality.model_validate(input)
    except ValidationError as e:
        logger.error(f"Validation error on input data: {e}")
        raise e

    with cache_lock:
        model = model_cache.get("model")
        preprocessor = model_cache.get("preprocessor")

    if not model or not preprocessor:
        raise HTTPException(status_code=503, detail="Model not available")

    try:
        # Extract features
        logger.info("Extracting data")
        df = pd.DataFrame(
            [item.model_dump() for item in input.sequence]
        ).rename(columns={"time": "_time"})
        df = df.sort_values(by="_time", ascending=True)

        # Apply preprocessing and create tensor
        logger.info("Transforming data and creating pytorch tensor")
        transformed = preprocessor.transform(df)
        future_cols = [
            col for col in transformed.columns if "cos" in col or "sin" in col
        ]
        past_cols = [
            col for col in transformed.columns if col not in future_cols
        ]

        past_tensor = torch.tensor(
            transformed.iloc[:12][past_cols].values, dtype=torch.float32
        ).unsqueeze(0)

        future_tensor = torch.tensor(
            transformed.iloc[-4:][future_cols].values, dtype=torch.float32
        ).unsqueeze(0)

        logger.info("Getting prediction")
        model.eval()
        with torch.no_grad():
            prediction = model(past_tensor, future_tensor)
        predicted_inverse = np.expm1(prediction)
        logger.info(
            f"Forecasted 4-hour air quality: {predicted_inverse.tolist()}"
        )
        return PredictionResult(prediction=predicted_inverse.tolist())

    except Exception as e:
        logger.error(f"Error generating predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    with cache_lock:
        model_loaded = "model" in model_cache
        preprocessor_loaded = "preprocessor" in model_cache

    if model_loaded and preprocessor_loaded:
        return {"status": "ok"}
    else:
        return {"status": "unavailable"}
