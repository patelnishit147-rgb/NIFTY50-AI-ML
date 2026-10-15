
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

from backend.services.predictor import predict_market

router = APIRouter(
    prefix="/api/market",
    tags=["Market"],
)


@router.get("/prediction")
def get_market_prediction():
    """Return the latest NIFTY 50 model prediction."""
    try:
        return predict_market()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc


@router.get("/history")
def get_market_history():
    """Return the latest 30 historical closing prices."""
    try:
        # Project root: NIFTY50-AI-ML
        project_root = Path(__file__).resolve().parents[2]
        csv_path = project_root / "data" / "nifty50.csv"

        if not csv_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Dataset not found: {csv_path}",
            )

        df = pd.read_csv(csv_path)

        # Identify the closing-price column.
        close_col = next(
            (
                col for col in df.columns
                if str(col).strip().lower() == "close"
            ),
            None,
        )

        if close_col is None:
            raise HTTPException(
                status_code=422,
                detail="The CSV does not contain a Close column.",
            )

        # Identify an optional date column.
        date_col = next(
            (
                col for col in df.columns
                if str(col).strip().lower()
                in ("date", "datetime", "timestamp")
            ),
            None,
        )

        # Keep the most recent 30 rows.
        df = df.tail(30).copy()
        df[close_col] = pd.to_numeric(
            df[close_col], errors="coerce"
        )
        df = df.dropna(subset=[close_col])

        history = []

        for index, (_, row) in enumerate(df.iterrows()):
            history.append({
                "date": (
                    str(row[date_col])
                    if date_col and pd.notna(row[date_col])
                    else str(index + 1)
                ),
                "close": float(row[close_col]),
            })

        return {"history": history}

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to load market history: {exc}",
        ) from exc
