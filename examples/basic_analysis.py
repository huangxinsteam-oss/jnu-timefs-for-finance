"""
Basic financial analysis example using jnu-timefs-for-finance.

This script demonstrates the full workflow:
  1. Create a TimeFS store.
  2. Load synthetic OHLCV data via FinanceLoader.
  3. Enrich with technical indicators via FinanceAnalyser.
  4. Print a summary report.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone

import pandas as pd

from timefs.core import TimeFS
from finance.loader import FinanceLoader
from finance.analysis import FinanceAnalyser


def make_sample_data(ticker: str, start: str = "2023-01-02", periods: int = 60) -> pd.DataFrame:
    """Generate a simple random-walk OHLCV dataset."""
    import numpy as np

    rng = np.random.default_rng(42)
    dates = pd.date_range(start, periods=periods, freq="B", tz="UTC")
    close = 100.0 + (rng.standard_normal(periods)).cumsum()
    spread = rng.uniform(0.5, 2.0, size=periods)
    df = pd.DataFrame(
        {
            "date": dates,
            "open": close - spread / 2,
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": rng.integers(500_000, 2_000_000, size=periods).astype(float),
        }
    )
    return df


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = TimeFS(tmpdir)
        loader = FinanceLoader(store)
        analyser = FinanceAnalyser(store)

        for ticker in ("AAPL", "TSLA", "GOOG"):
            df = make_sample_data(ticker)
            n = loader.load_dataframe(df, namespace=ticker, date_column="date")
            print(f"[{ticker}] Loaded {n} timedots into TimeFS store.")

        print("\n--- Summary Statistics: AAPL ---")
        summary = analyser.summary("AAPL")
        print(summary.to_string())

        print("\n--- Enriched OHLCV (last 5 rows): AAPL ---")
        enriched = analyser.enrich("AAPL", sma_windows=[10, 20])
        print(enriched.tail(5).to_string())

        print("\n--- Pairwise Correlation (daily returns) ---")
        corr = analyser.correlation(["AAPL", "TSLA", "GOOG"])
        print(corr.to_string())

        print("\n--- Value at Risk (95%) ---")
        for ticker in ("AAPL", "TSLA", "GOOG"):
            var, cvar = analyser.value_at_risk(ticker)
            print(f"  {ticker}: VaR={var:.4%}  CVaR={cvar:.4%}")


if __name__ == "__main__":
    main()
