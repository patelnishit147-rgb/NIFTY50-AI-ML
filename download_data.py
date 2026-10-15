
import yfinance as yf
from pathlib import Path

# Download NIFTY 50 historical data
ticker = "^NSEI"

print("Downloading NIFTY 50 data...")

data = yf.download(
    ticker,
    period="10y",
    interval="1d",
    auto_adjust=True,
    progress=False
)

if data.empty:
    print("Download failed. Please try again.")
else:
    # Create data folder
    Path("data").mkdir(exist_ok=True)

    # Save dataset
    data.to_csv("data/nifty50.csv")

    print("Download successful!")
    print("Total trading days:", len(data))
    print(data.tail())
    print("Saved to data/nifty50.csv")
