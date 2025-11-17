# app.py
import io
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template, jsonify, request, send_file
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# optional yfinance
try:
    import yfinance as yf
    HAVE_YFINANCE = True
except Exception:
    HAVE_YFINANCE = False

app = Flask(__name__)

DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset.csv")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


def load_from_csv():
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH, parse_dates=["Date"])
        df = df.set_index("Date").sort_index()
        df = df[["Close"]].dropna()
        return df
    return None


def synthetic_prices(start=100.0, n=400, mu=0.0006, sigma=0.02, seed=42):
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=mu, scale=sigma, size=n)
    prices = start * np.exp(np.cumsum(returns))
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n)
    return pd.DataFrame({"Close": prices}, index=dates)


def fetch_stock(ticker="AAPL", days=300):
    df_csv = load_from_csv()
    if df_csv is not None:
        return df_csv.tail(min(days, len(df_csv)))
    if HAVE_YFINANCE:
        try:
            df = yf.download(ticker, period=f"{days}d", progress=False)
            if not df.empty and "Close" in df.columns:
                return df[["Close"]].dropna()
        except Exception:
            pass
    return synthetic_prices(n=days)


def make_features(df):
    df = df.copy()
    df["ret"] = df["Close"].pct_change()
    for lag in (1, 2, 3):
        df[f"lag_{lag}"] = df["ret"].shift(lag)
    for w in (5, 10, 20):
        df[f"ma_{w}"] = df["Close"].rolling(w).mean()
    df = df.dropna()
    return df


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/summary")
def api_summary():
    ticker = request.args.get("ticker", "AAPL")
    days = int(request.args.get("days", 300))
    df = fetch_stock(ticker, days=days)
    if df is None or df.empty:
        return jsonify({"error": "no data"}), 400
    desc = df["Close"].describe().to_dict()
    return jsonify({
        "ticker": ticker,
        "start": df.index.min().strftime("%Y-%m-%d"),
        "end": df.index.max().strftime("%Y-%m-%d"),
        "describe": desc,
        "n_rows": len(df)
    })


@app.route("/api/plot")
def api_plot():
    ticker = request.args.get("ticker", "AAPL")
    days = int(request.args.get("days", 300))
    df = fetch_stock(ticker, days=days)
    if df is None or df.empty:
        return jsonify({"error": "no data"}), 400

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df.index, df["Close"], label=f"{ticker} Close")
    ax.set_title(f"{ticker} Close ({df.index.min().strftime('%Y-%m-%d')} → {df.index.max().strftime('%Y-%m-%d')})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True)
    ax.legend()

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.route("/api/train_predict", methods=["POST"])
def api_train_predict():
    payload = request.get_json(force=True)
    ticker = payload.get("ticker", "AAPL")
    days = int(payload.get("days", 300))
    model_type = payload.get("model", "rf")
    horizon = int(payload.get("horizon", 1))

    df = fetch_stock(ticker, days=days + 60)
    if df is None or df.empty:
        return jsonify({"error": "no data"}), 400

    data = make_features(df)
    data[f"future_{horizon}"] = df["Close"].shift(-horizon).loc[data.index]
    data = data.dropna()
    feature_cols = [c for c in data.columns if c.startswith("lag_") or c.startswith("ma_")]
    if len(feature_cols) == 0 or len(data) < 12:
        return jsonify({"error": "not enough data for features"}), 400

    X = data[feature_cols].values
    y = data[f"future_{horizon}"].values

    # time-aware split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    if model_type == "linear":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=0)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = float(np.mean(np.abs(preds - y_test)))

    next_pred = float(model.predict(data[feature_cols].iloc[-1].values.reshape(1, -1))[0])

    last_rows = data[["Close"] + feature_cols + [f"future_{horizon}"]].tail(10)
    last_rows.index = last_rows.index.strftime("%Y-%m-%d")
    last_rows_json = last_rows.reset_index().to_dict(orient="records")

    # optionally compute feature importances for RF
    feature_importances = None
    if hasattr(model, "feature_importances_"):
        feature_importances = dict(zip(feature_cols, list(map(float, model.feature_importances_))))

    return jsonify({
        "ticker": ticker,
        "model": model_type,
        "horizon": horizon,
        "mae_test": round(mae, 4),
        "next_pred_close": round(next_pred, 4),
        "last_rows": last_rows_json,
        "feature_columns": feature_cols,
        "feature_importances": feature_importances
    })


if __name__ == "__main__":
    print("Starting Flask app...")
    if not os.path.exists(STATIC_DIR):
        os.makedirs(STATIC_DIR, exist_ok=True)
    app.run(debug=True)

