
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load downloaded NIFTY 50 data
df = pd.read_csv("data/nifty50.csv", header=[0, 1], index_col=0)

# Handle Yahoo Finance multi-level CSV columns
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df.loc[:, ~df.columns.duplicated()]
df.index = pd.to_datetime(df.index)
df = df.sort_index()

# Keep numeric market data
close = pd.to_numeric(df["Close"], errors="coerce")
high = pd.to_numeric(df["High"], errors="coerce")
low = pd.to_numeric(df["Low"], errors="coerce")

data = pd.DataFrame(index=df.index)
data["Close"] = close
data["Return_1d"] = close.pct_change()
data["Return_5d"] = close.pct_change(5)
data["MA_5"] = close.rolling(5).mean()
data["MA_10"] = close.rolling(10).mean()
data["MA_20"] = close.rolling(20).mean()
data["Volatility_10"] = data["Return_1d"].rolling(10).std()
data["High_Low_Range"] = (high - low) / close

# Target: closing price five trading sessions ahead
data["Target"] = close.shift(-5)

data = data.replace([np.inf, -np.inf], np.nan).dropna()

if len(data) < 100:
    raise ValueError("Not enough valid data to train the model.")

features = [
    "Close",
    "Return_1d",
    "Return_5d",
    "MA_5",
    "MA_10",
    "MA_20",
    "Volatility_10",
    "High_Low_Range",
]

X = data[features]
y = data["Target"]

# Chronological split: earliest 80% for training
split = int(len(data) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# Train Random Forest
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluate on later, unseen dates
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("NIFTY 50 Random Forest Model")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print(f"MAE: {mae:.2f} index points")
print(f"RMSE: {rmse:.2f} index points")
print(f"Actual close (test): {y_test.iloc[-1]:.2f}")
print(f"Predicted close (test): {predictions[-1]:.2f}")

# Save model and the latest features
Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/nifty50_random_forest.joblib")
X.iloc[[-1]].to_csv("models/latest_features.csv")

print("Model saved to models/nifty50_random_forest.joblib")
