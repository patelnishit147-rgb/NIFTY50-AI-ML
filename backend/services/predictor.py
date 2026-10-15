
from pathlib import Path

import joblib
import pandas as pd
import yfinance as yf


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "nifty50_random_forest.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "nifty50.csv"


# Features expected by the trained Random Forest model
FEATURES = [
    "Close",
    "Return_1d",
    "Return_5d",
    "MA_5",
    "MA_10",
    "MA_20",
    "Volatility_10",
    "High_Low_Range",
]


def load_model():
    """Load the saved Random Forest model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def refresh_market_data():
    """Download the latest available NIFTY 50 daily data and save it to CSV."""
    data = yf.download(
        "^NSEI",
        period="10y",
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        raise ValueError(
            "Yahoo Finance returned no NIFTY 50 data. "
            "Check your internet connection and try again."
        )

    # Handle different yfinance column formats
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.columns = [
        str(column).strip()
        for column in data.columns
    ]

    required_columns = ["Close", "High", "Low"]
    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Downloaded data is missing columns: {missing_columns}. "
            f"Received columns: {list(data.columns)}"
        )

    # Ensure price columns contain numeric values
    for column in required_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = data.dropna(subset=required_columns)

    if data.empty:
        raise ValueError(
            "Downloaded data contains no valid price records."
        )

    # Save the refreshed dataset
    DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(
        DATA_PATH,
        index_label="Date",
    )

    print(
        f"Market data refreshed: {len(data)} rows. "
        f"Latest available date: {data.index[-1]}"
    )


def load_latest_features():
    """Calculate the features required for the latest market observation."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    required_columns = ["Close", "High", "Low"]
    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    # Convert price columns to numeric values
    for column in required_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(subset=required_columns).copy()

    if df.empty:
        raise ValueError(
            "No valid numeric market prices are available."
        )

    # Calculate model features
    df["Return_1d"] = df["Close"].pct_change()
    df["Return_5d"] = df["Close"].pct_change(periods=5)

    df["MA_5"] = df["Close"].rolling(window=5).mean()
    df["MA_10"] = df["Close"].rolling(window=10).mean()
    df["MA_20"] = df["Close"].rolling(window=20).mean()

    df["Volatility_10"] = (
        df["Return_1d"].rolling(window=10).std()
    )

    df["High_Low_Range"] = df["High"] - df["Low"]

    missing_features = [
        column
        for column in FEATURES
        if column not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing model features: {missing_features}"
        )

    # Get the most recent row with all features available
    latest = df[FEATURES].dropna().tail(1)

    if latest.empty:
        raise ValueError(
            "Not enough valid historical data to calculate the model features."
        )

    return latest


def predict_market():
    """Refresh market data and return the latest model prediction."""
    # Fetch the latest available data every time prediction is requested
    refresh_market_data()

    # Load the trained model and calculate current features
    model = load_model()
    features = load_latest_features()

    # Generate prediction
    prediction = float(model.predict(features)[0])
    latest_close = float(features["Close"].iloc[0])

    predicted_change = prediction - latest_close

    # Return values used by the FastAPI endpoint and React dashboard
    return {
        "latest_close": latest_close,
        "predicted_close": prediction,
        "predicted_change": predicted_change,
    }
