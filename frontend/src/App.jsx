
import { useEffect, useState } from "react";
import "./App.css";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  BarChart,
  Bar,
} from "recharts";

const API_BASE = "https://nifty50-ai-ml.onrender.com/api/market";

const formatPrice = (value) =>
  Number(value).toLocaleString("en-IN", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });

function App() {
  const [prediction, setPrediction] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [historyError, setHistoryError] = useState("");

  async function getPrediction() {
    setLoading(true);
    setError("");
    setHistoryError("");

    try {
      const response = await fetch(`${API_BASE}/prediction`);

      if (!response.ok) {
        throw new Error(
          `Prediction request failed (${response.status}).`
        );
      }

      const data = await response.json();

      if (
        data.latest_close == null ||
        data.predicted_close == null
      ) {
        throw new Error("The API response is missing price values.");
      }

      setPrediction(data);
    } catch (err) {
      setError(err.message || "Unable to load prediction.");
    } finally {
      setLoading(false);
    }
  }

  async function getHistory() {
    setHistoryError("");

    try {
      const response = await fetch(`${API_BASE}/history`);

      if (!response.ok) {
        throw new Error(`History request failed (${response.status}).`);
      }

      const data = await response.json();

      // Supports either an array or { history: [...] }.
      const rows = Array.isArray(data)
        ? data
        : Array.isArray(data.history)
        ? data.history
        : Array.isArray(data.data)
        ? data.data
        : null;

      if (!rows) {
        throw new Error(
          "Unexpected history format. Check the API response."
        );
      }

      const chartRows = rows
        .map((item, index) => {
          const close = Number(item.close ?? item.Close);
          const date = item.date ?? item.Date ?? `Day ${index + 1}`;

          return {
            date: String(date),
            close,
          };
        })
        .filter((item) => Number.isFinite(item.close))
        .slice(-30);

      if (chartRows.length === 0) {
        throw new Error("No historical closing prices were found.");
      }

      setHistory(chartRows);
    } catch (err) {
      setHistoryError(err.message || "Unable to load market history.");
    }
  }

  async function refreshDashboard() {
    await Promise.all([getPrediction(), getHistory()]);
  }

  useEffect(() => {
    refreshDashboard();
  }, []);

  const comparisonData = prediction
    ? [
        {
          name: "Latest Close",
          price: Number(prediction.latest_close),
        },
        {
          name: "Predicted Close",
          price: Number(prediction.predicted_close),
        },
      ]
    : [];

  const change = prediction
    ? Number(
        prediction.predicted_change ??
          Number(prediction.predicted_close) -
            Number(prediction.latest_close)
      )
    : 0;

  const changePercent =
    prediction && Number(prediction.latest_close) !== 0
      ? (change / Number(prediction.latest_close)) * 100
      : 0;

  return (
    <main className="container">
      <header className="dashboard-header">
        <div>
          <h1>MarketMind AI</h1>
          <p>NIFTY 50 AI/ML Market Prediction Dashboard</p>
        </div>

        <button onClick={refreshDashboard} disabled={loading}>
          {loading ? "Refreshing..." : "Refresh Dashboard"}
        </button>
      </header>

      {error && <p className="error">{error}</p>}

      {prediction && (
        <>
          <section className="stats-grid">
            <article className="card stat-card">
              <p>Latest Close</p>
              <h2>₹{formatPrice(prediction.latest_close)}</h2>
            </article>

            <article className="card stat-card">
              <p>Predicted Close</p>
              <h2>₹{formatPrice(prediction.predicted_close)}</h2>
            </article>

            <article className="card stat-card">
              <p>Predicted Change</p>
              <h2 className={change >= 0 ? "positive" : "negative"}>
                {change >= 0 ? "+" : ""}₹{formatPrice(change)}
              </h2>
            </article>

            <article className="card stat-card">
              <p>Predicted Change (%)</p>
              <h2 className={changePercent >= 0 ? "positive" : "negative"}>
                {changePercent >= 0 ? "+" : ""}
                {changePercent.toFixed(2)}%
              </h2>
            </article>
          </section>

          <section className="card chart-card">
            <h2>Latest vs Predicted Price</h2>
            <p>Comparison of the latest close and model prediction.</p>

            <div className="chart">
  <ResponsiveContainer width="100%" height={300}>
    <BarChart
      data={[
        {
          name: "Latest Close",
          price: Number(prediction.latest_close),
        },
        {
          name: "Predicted Close",
          price: Number(prediction.predicted_close),
        },
      ]}
      margin={{ top: 20, right: 20, left: 20, bottom: 10 }}
    >
      <CartesianGrid strokeDasharray="3 3" stroke="#344256" />
      <XAxis dataKey="name" stroke="#cbd5e1" />
      <YAxis
        stroke="#cbd5e1"
        domain={["dataMin - 200", "dataMax + 200"]}
        tickFormatter={(value) =>
          `₹${Number(value).toLocaleString("en-IN")}`
        }
      />
      <Tooltip
        formatter={(value) => [
          `₹${Number(value).toLocaleString("en-IN", {
            maximumFractionDigits: 2,
          })}`,
          "Price",
        ]}
      />
      <Legend />
      <Bar dataKey="price" name="Price (INR)" fill="#67e8f9" />
    </BarChart>
  </ResponsiveContainer>
</div>
          </section>
        </>
      )}

      <section className="card chart-card">
        <h2>NIFTY 50 Historical Closing Prices</h2>
        <p>Most recent available observations from your history API.</p>

        {historyError && <p className="error">{historyError}</p>}

        {history.length > 0 ? (
          <>
            <div className="chart">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={history}
                  margin={{ top: 10, right: 20, left: 10, bottom: 10 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#344256" />
                  <XAxis
                    dataKey="date"
                    stroke="#cbd5e1"
                    tick={{ fontSize: 11 }}
                    minTickGap={25}
                  />
                  <YAxis
                    stroke="#cbd5e1"
                    domain={["auto", "auto"]}
                    tickFormatter={(value) =>
                      Number(value).toLocaleString("en-IN")
                    }
                  />
                  <Tooltip
                    formatter={(value) => `₹${formatPrice(value)}`}
                    labelFormatter={(label) => `Date: ${label}`}
                    contentStyle={{
                      backgroundColor: "#101a2c",
                      border: "1px solid #344256",
                      borderRadius: "8px",
                    }}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="close"
                    name="Closing Price"
                    stroke="#38bdf8"
                    strokeWidth={3}
                    dot={false}
                    activeDot={{ r: 5 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <p className="chart-note">
              Showing {history.length} historical observations.
            </p>
          </>
        ) : (
          !historyError && <p>Loading historical market data...</p>
        )}
      </section>

      <footer className="dashboard-footer">
        Predictions are estimates from your model, not guaranteed market outcomes.
      </footer>
    </main>
  );
}

export default App;
