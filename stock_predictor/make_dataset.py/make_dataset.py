# make_dataset.py  -- creates dataset.csv (synthetic historical Close prices)

import numpy as np
import pandas as pd

# ---------------- CONFIG ----------------
n_days = 400            # how many rows to generate (>= 200 recommended)
start_price = 100.0     # starting stock price
mu = 0.0006             # daily expected return
sigma = 0.02            # daily volatility
seed = 42               # random seed for reproducibility
# ----------------------------------------


def generate_prices():
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=mu, scale=sigma, size=n_days)
    prices = start_price * np.exp(np.cumsum(returns))
    return prices


def make_dataset():
    prices = generate_prices()

    # business-day-like dates ending today
    end = pd.Timestamp.today().normalize()
    dates = pd.bdate_range(end=end, periods=n_days)

    df = pd.DataFrame({
        "Date": dates,
        "Close": np.round(prices, 2)
    })

    # optional engineered columns
    df["Daily_Return_%"] = df["Close"].pct_change() * 100
    df["5d_MA"] = df["Close"].rolling(window=5).mean()

    df.to_csv("dataset.csv", index=False)
    print(
        f"Saved dataset.csv with {len(df)} rows "
        f"from {df.Date.min().date()} to {df.Date.max().date()}"
    )


if __name__ == "__main__":
    make_dataset()

