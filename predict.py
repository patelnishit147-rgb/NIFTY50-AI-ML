
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

data_path = Path("data/nifty50.csv")
model_path = Path("models/nifty50_random_forest.joblib")

if not data_path.exists() or not model_path.exists():
    raise FileNotFoundError(
        "Check that data/nifty50.csv and the saved model exist."
    )

df = pd.read_csv(data_path, header=[0, 1], index_col=0)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df.loc[:, ~df.columns.duplicated()]
df.index = pd.to_datetime(df.index)
df = df.sort_index()

close = pd.to_numeric(df["Close"], errors="coerce")
high = pd.to_numeric(df["High"], errors="coerce")
low = pd.to_numeric(df["Low"], errors="coerce")

features = pd.DataFrame(index=df.index)
features["Close"] = close
features["Return_1d"] = close.pct_change()
features["Return_5d"] = close.pct_change(5)
features["MA_5"] = close.rolling(5).mean()
features["MA_10"] = close.rolling(10).mean()
features["MA_20"] = close.rolling(20).mean()
features["Volatility_10"] = features["Return_1d"].rolling(10).std()
features["High_Low_Range"] = (high - low) / close

features = features.replace([np.inf, -np.inf], np.nan).dropna()

latest = features.iloc[[-1]]
model = joblib.load(model_path)

prediction = model.predict(latest)[0]
last_close = latest["Close"].iloc[0]
last_date = latest.index[0]

print("\nNIFTY 50 — Five-Trading-Day Forecast")
print("------------------------------------")
print("Latest available data date:", last_date.date())
print(f"Latest closing price: {last_close:.2f}")
print(f"Predicted close in 5 trading days: {prediction:.2f}")
print(f"Predicted change: {prediction - last_close:.2f} points")
print(
    f"Predicted change (%): "
    f"{(prediction / last_close - 1) * 100:.2f}%"
)
print("\nThis is an experimental model prediction, not a guarantee.")
