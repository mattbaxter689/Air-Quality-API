from fastapi import FastAPI
from fastapi.exceptions import HTTPException
from api.base_models.models import AirQuality, PredictionResult
from pydantic import ValidationError
from threading import Lock
import torch
from api.logger.init_logger import create_logger
from api.middleware.bucket_middleware import TokenBucketMiddleware
from api.utils.manager import PredictionProcessor
from api.utils.utils import load_mlflow_model
from sklearn import set_config
import asyncio
import pandas as pd
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from dotenv import load_dotenv
import numpy as np
import os

model_cache = {}
cache_lock = Lock()
logger = create_logger(name="weather_api")

load_dotenv()
set_config(transform_output="pandas")


class ModelManager:
    """Manager for model cache"""

    @staticmethod
    def load_model() -> tuple:
        """Load model and preprocessor from MLflow"""
        return load_mlflow_model()

    @staticmethod
    def cache_model(model, preprocessor) -> None:
        """Thread-safe model caching"""
        with cache_lock:
            model_cache["model"] = model
            model_cache["preprocessor"] = preprocessor

    @staticmethod
    def get_model() -> tuple:
        """Thread-safe model retrieval"""
        with cache_lock:
            return model_cache.get("model"), model_cache.get("preprocessor")

    @staticmethod
    def is_model_available() -> bool:
        """Check if model and preprocessor are loaded"""
        with cache_lock:
            return "model" in model_cache and "preprocessor" in model_cache


async def refresh_model() -> None:
    while True:
        await asyncio.sleep(3600)
        try:
            model, preprocessor = ModelManager.load_model()
            ModelManager.cache_model(model, preprocessor)
            logger.info("Model refreshed")
        except Exception as e:
            logger.error(f"Failed to refresh model: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    logger.info("Loading model and preprocessor on API startup")

    try:
        model, preprocessor = ModelManager.load_model()
        ModelManager.cache_model(model, preprocessor)
        logger.info("Model loaded on startup successfully")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")

    refresh_task = asyncio.create_task(refresh_model())

    yield

    logger.info("Beginning API shutdown ...")

    refresh_task.cancel()
    try:
        await refresh_task
    except asyncio.CancelledError:
        logger.info("Background model refresh task cancelled")
    logger.info("Shutting down API...")


app = FastAPI(lifespan=lifespan)
logger.info("Adding Redis token bucket middleware")
app.add_middleware(
    TokenBucketMiddleware,
    redis_host=os.getenv("REDIS_HOST"),
    redis_port=6379,
    bucket_size=10,
    refill_rate=1,
)


@app.get("/")
async def test() -> None:
    return {"Message": "Hello Air Quality API"}


@app.post(
    "/predict",
    summary="Forecast next hour's qir quality, using last 12 hours worth of time points",
    description="Predicts the next hour's air quality using the last 12 hours worth of air quality data",
    response_model=PredictionResult,
)
async def predict(input: AirQuality) -> PredictionResult:

    PredictionProcessor.validate_input_length(input.sequence)

    try:
        logger.info("Validating input with Base Model")
        AirQuality.model_validate(input)
    except ValidationError as e:
        logger.error(f"Validation error on input data: {e}")
        raise HTTPException(
            status_code=422, detail=f"Input validation failed: {e}"
        )

    model, preprocessor = ModelManager.load_model()

    if not model or not preprocessor:
        raise HTTPException(
            status_code=503, detail="Model/Processor not available"
        )

    try:
        logger.info("Processing input data")
        df = PredictionProcessor.prepare_dataframe(input.sequence)

        logger.info("Applying preprocessing transformations")
        transformed = preprocessor.transform(df)

        logger.info("Creating PyTorch tensors")
        past_tensor, future_tensor = PredictionProcessor.create_tensors(
            transformed
        )

        logger.info("Generating prediction")
        prediction_list = PredictionProcessor.generate_prediction(
            model, past_tensor, future_tensor
        )

        logger.info(f"Forecasted 4-hour air quality: {prediction_list}")
        return PredictionResult(prediction=prediction_list)

    except Exception as e:
        logger.error(f"Error generating predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    model_available = ModelManager.is_model_available()
    return {
        "status": "healthy" if model_available else "unhealthy",
        "model_loaded": model_available,
        "service": "air-quality-prediction",
    }
