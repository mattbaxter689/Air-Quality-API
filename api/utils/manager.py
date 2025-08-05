import logging
import torch
import pandas as pd
import numpy as np

logger = logging.getLogger("weather_api")

EXPECTED_SEQUENCE_LENGTH = 16
WINDOW_SIZE = 12
FORECAST_LENGTH = 4


class PredictionProcessor:
    """
    Handles data preprocessing and prediction logic
    """

    @staticmethod
    def validate_input_length(sequence: list) -> None:
        """Validate input sequence length"""
        if len(sequence) != EXPECTED_SEQUENCE_LENGTH:
            error_msg = f"Data must contain exactly {EXPECTED_SEQUENCE_LENGTH} points, contained {len(sequence)}"
            logger.error(
                f"Data did not contain exactly {EXPECTED_SEQUENCE_LENGTH} time points. Contained: {len(sequence)}"
            )
            raise ValueError(error_msg)

    @staticmethod
    def prepare_dataframe(sequence: list) -> pd.DataFrame:
        """Convert sequence to sorted DataFrame"""
        df = pd.DataFrame([item.model_dump() for item in sequence]).rename(
            columns={"time": "_time"}
        )
        return df.sort_values(by="_time", ascending=True)

    @staticmethod
    def create_tensors(
        transformed_df: pd.DataFrame,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create past and future tensors from transformed data"""
        future_cols = [
            col
            for col in transformed_df.columns
            if "cos" in col or "sin" in col
        ]
        past_cols = [
            col for col in transformed_df.columns if col not in future_cols
        ]

        past_tensor = torch.tensor(
            transformed_df.iloc[:WINDOW_SIZE][past_cols].values,
            dtype=torch.float32,
        ).unsqueeze(0)

        future_tensor = torch.tensor(
            transformed_df.iloc[-FORECAST_LENGTH:][future_cols].values,
            dtype=torch.float32,
        ).unsqueeze(0)

        return past_tensor, future_tensor

    @staticmethod
    def generate_prediction(
        model, past_tensor: torch.Tensor, future_tensor: torch.Tensor
    ) -> list:
        """Generate and transform prediction"""
        model.eval()
        with torch.no_grad():
            prediction = model(past_tensor, future_tensor)

        # Convert back from log scale
        predicted_inverse = np.expm1(prediction)
        return predicted_inverse.tolist()
