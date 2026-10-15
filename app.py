
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="NIFTY 50 AI Predictor",
    page_icon="📈",
    layout="wide",
)

DATA_PATH = Path("data/nifty50.csv")
MODEL_PATH = Path("models/nifty50_random_forest.joblib")

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


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, header=[0, 1], index_col=0)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.loc[:, ~df.columns.duplicated()]
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    for col in ["Close", "High", "Low"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=["Close", "High", "Low"])


def make_features(df):
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    features = pd.DataFrame(index=df.index)
    features["Close"] = close
    features["Return_1d"] = close.pct_change()
    features["Return_5d"] = close.pct_change(5)
    features["MA_5"] = close.rolling(5).mean()
    features["MA_10"] = close.rolling(10).mean()
    features["MA_20"] = close.rolling(20).mean()
    features["Volatility_10"] = (
        features["Return_1d"].rolling(10).std()
    )
    features["High_Low_Range"] = (high - low) / close

    return features.replace([np.inf, -np.inf], np.nan).dropna()


st.title("📈 NIFTY 50 AI Market Predictor")
st.caption("Machine Learning project | Random Forest | NSE: ^NSEI")



if not DATA_PATH.exists() or not MODEL_PATH.exists():
    st.error(
        "Dataset or trained model not found. Check that data/nifty50.csv "
        "and models/nifty50_random_forest.joblib exist."
    )
    st.stop()

try:
    df = load_data()
    features = make_features(df)
    model = joblib.load(MODEL_PATH)

    if features.empty:
        st.error("Not enough historical data to calculate features.")
        st.stop()

    latest = features.iloc[[-1]][FEATURES]
    latest_close = float(latest["Close"].iloc[0])
    forecast = float(model.predict(latest)[0])
    change = forecast - latest_close
    change_pct = change / latest_close * 100
    latest_date = latest.index[0]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Latest Closing Price",
        f"{latest_close:,.2f}",
        help=f"Latest available data: {latest_date.date()}",
    )
    col2.metric(
        "Predicted Close (5 Trading Days)",
        f"{forecast:,.2f}",
        delta=f"{change_pct:+.2f}%",
    )
    col3.metric(
        "Predicted Price Change",
        f"{change:+,.2f} points",
    )

    st.caption(f"Forecast based on data through {latest_date.date()}.")

    st.subheader("Historical NIFTY 50 Closing Prices")
    days = st.selectbox("Chart period", [30, 90, 180, 365], index=2)

    chart_data = df["Close"].tail(days)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(chart_data.index, chart_data.values, label="Closing price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Index points")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Model Evaluation")

    evaluation = features.copy()
    evaluation["Target"] = df["Close"].reindex(
        evaluation.index
    ).shift(-5)

    evaluation = evaluation.dropna(subset=["Target"])
    split = int(len(evaluation) * 0.8)

    if split > 0 and len(evaluation) - split > 0:
        test = evaluation.iloc[split:]
        test_predictions = model.predict(test[FEATURES])
        actual = test["Target"].to_numpy()

        mae = mean_absolute_error(actual, test_predictions)
        rmse = np.sqrt(mean_squared_error(actual, test_predictions))

        m1, m2 = st.columns(2)
        m1.metric("Holdout MAE", f"{mae:,.2f} points")
        m2.metric("Holdout RMSE", f"{rmse:,.2f} points")

        st.caption(
            "These metrics are calculated on the later portion of the "
            "historical dataset using the saved model. They are not a "
            "guarantee of future performance."
        )

    with st.expander("View latest historical data"):
        st.dataframe(df.tail(10), use_container_width=True)

except Exception as exc:
    st.error(f"Unable to load the dashboard: {exc}")
