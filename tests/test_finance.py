"""Tests for the finance loader, indicators, and analysis modules."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pandas as pd
import pytest

from timefs.core import TimeFS
from finance.loader import FinanceLoader
from finance.indicators import Indicators
from finance.analysis import FinanceAnalyser


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_store(tmp_path):
    return TimeFS(tmp_path / "store")


@pytest.fixture
def sample_ohlcv_df():
    """Simple 10-day OHLCV DataFrame for AAPL."""
    dates = pd.date_range("2024-01-02", periods=10, freq="D", tz="UTC")
    data = {
        "date": dates,
        "open":   [185.0, 186.0, 184.5, 187.0, 188.0, 187.5, 189.0, 190.0, 191.0, 192.0],
        "high":   [188.0, 188.5, 187.0, 189.0, 190.0, 190.5, 191.0, 192.0, 193.0, 194.0],
        "low":    [184.0, 185.0, 183.5, 186.0, 187.0, 186.5, 188.0, 189.0, 190.0, 191.0],
        "close":  [186.5, 187.0, 185.0, 188.0, 189.0, 188.5, 190.0, 191.0, 192.0, 193.0],
        "volume": [1_000_000] * 10,
    }
    return pd.DataFrame(data)


@pytest.fixture
def populated_store(tmp_store, sample_ohlcv_df):
    loader = FinanceLoader(tmp_store)
    loader.load_dataframe(sample_ohlcv_df, namespace="AAPL", date_column="date")
    return tmp_store


# ---------------------------------------------------------------------------
# FinanceLoader
# ---------------------------------------------------------------------------


class TestFinanceLoader:
    def test_load_dataframe_returns_count(self, tmp_store, sample_ohlcv_df):
        loader = FinanceLoader(tmp_store)
        n = loader.load_dataframe(sample_ohlcv_df, namespace="AAPL", date_column="date")
        assert n == 10

    def test_load_dataframe_persists_data(self, tmp_store, sample_ohlcv_df):
        loader = FinanceLoader(tmp_store)
        loader.load_dataframe(sample_ohlcv_df, namespace="AAPL", date_column="date")
        assert tmp_store.count("AAPL") == 10

    def test_get_returns_dataframe(self, tmp_store, sample_ohlcv_df):
        loader = FinanceLoader(tmp_store)
        loader.load_dataframe(sample_ohlcv_df, namespace="AAPL", date_column="date")
        df = loader.get("AAPL")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        assert "close" in df.columns

    def test_get_with_date_range(self, tmp_store, sample_ohlcv_df):
        loader = FinanceLoader(tmp_store)
        loader.load_dataframe(sample_ohlcv_df, namespace="AAPL", date_column="date")
        start = datetime(2024, 1, 5, tzinfo=timezone.utc)
        end = datetime(2024, 1, 8, tzinfo=timezone.utc)
        df = loader.get("AAPL", start=start, end=end)
        assert len(df) == 4

    def test_load_csv(self, tmp_store, sample_ohlcv_df, tmp_path):
        csv_path = tmp_path / "aapl.csv"
        sample_ohlcv_df.to_csv(csv_path, index=False)
        loader = FinanceLoader(tmp_store)
        n = loader.load_csv(csv_path, namespace="AAPL_CSV", date_column="date")
        assert n == 10

    def test_load_dataframe_missing_date_column_raises(self, tmp_store, sample_ohlcv_df):
        loader = FinanceLoader(tmp_store)
        with pytest.raises(ValueError, match="'timestamp'"):
            loader.load_dataframe(sample_ohlcv_df, namespace="AAPL", date_column="timestamp")

    def test_case_insensitive_columns(self, tmp_store):
        df = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"),
            "Close": [100.0, 101.0, 102.0],
        })
        loader = FinanceLoader(tmp_store)
        n = loader.load_dataframe(df, namespace="TEST", date_column="Date")
        assert n == 3


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------


class TestIndicators:
    @pytest.fixture
    def close(self):
        return pd.Series(
            [100, 102, 101, 105, 107, 106, 110, 112, 111, 115,
             114, 118, 120, 119, 122, 124, 123, 127, 129, 130],
            dtype=float,
        )

    def test_sma_length(self, close):
        sma = Indicators.sma(close, window=5)
        assert len(sma) == len(close)

    def test_sma_name(self, close):
        assert Indicators.sma(close, window=20).name == "SMA_20"

    def test_ema_name(self, close):
        assert Indicators.ema(close, window=12).name == "EMA_12"

    def test_macd_columns(self, close):
        macd = Indicators.macd(close)
        assert set(macd.columns) == {"macd", "signal", "histogram"}

    def test_rsi_range(self, close):
        rsi = Indicators.rsi(close, window=5)
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_rsi_name(self, close):
        assert Indicators.rsi(close).name == "RSI"

    def test_bollinger_bands_columns(self, close):
        bb = Indicators.bollinger_bands(close)
        assert set(bb.columns) == {"bb_middle", "bb_upper", "bb_lower"}

    def test_bollinger_upper_above_lower(self, close):
        bb = Indicators.bollinger_bands(close).dropna()
        assert (bb["bb_upper"] >= bb["bb_lower"]).all()

    def test_atr_positive(self):
        high = pd.Series([110.0, 112, 111, 115, 114], dtype=float)
        low = pd.Series([105.0, 108, 107, 110, 109], dtype=float)
        close = pd.Series([108.0, 110, 109, 113, 112], dtype=float)
        atr = Indicators.atr(high, low, close, window=3).dropna()
        assert (atr > 0).all()

    def test_obv_monotone_up(self):
        close = pd.Series([100.0, 101, 102, 103, 104])
        volume = pd.Series([1000.0] * 5)
        obv = Indicators.obv(close, volume)
        assert obv.iloc[-1] > obv.iloc[0]

    def test_daily_returns_first_nan(self, close):
        ret = Indicators.daily_returns(close)
        assert math.isnan(ret.iloc[0])

    def test_cumulative_returns_nonnegative_for_rising(self, close):
        cum = Indicators.cumulative_returns(close).dropna()
        assert cum.iloc[-1] > 0


# ---------------------------------------------------------------------------
# FinanceAnalyser
# ---------------------------------------------------------------------------


class TestFinanceAnalyser:
    def test_get_ohlcv(self, populated_store):
        analyser = FinanceAnalyser(populated_store)
        df = analyser.get_ohlcv("AAPL")
        assert len(df) == 10

    def test_summary_keys(self, populated_store):
        analyser = FinanceAnalyser(populated_store)
        stats = analyser.summary("AAPL")
        assert not stats.empty
        assert "ann_volatility" in stats.index
        assert "total_return" in stats.index

    def test_enrich_adds_indicators(self, populated_store):
        analyser = FinanceAnalyser(populated_store)
        df = analyser.enrich("AAPL", sma_windows=[3], ema_windows=[3])
        assert "SMA_3" in df.columns
        assert "EMA_3" in df.columns
        assert "RSI" in df.columns
        assert "macd" in df.columns
        assert "bb_middle" in df.columns
        assert "ATR" in df.columns
        assert "OBV" in df.columns
        assert "daily_return" in df.columns

    def test_correlation_returns_square_matrix(self, tmp_store, sample_ohlcv_df):
        loader = FinanceLoader(tmp_store)
        loader.load_dataframe(sample_ohlcv_df, namespace="A")
        loader.load_dataframe(
            sample_ohlcv_df.assign(close=sample_ohlcv_df["close"] * 1.1),
            namespace="B",
        )
        analyser = FinanceAnalyser(tmp_store)
        corr = analyser.correlation(["A", "B"])
        assert corr.shape == (2, 2)

    def test_value_at_risk_returns_tuple(self, populated_store):
        analyser = FinanceAnalyser(populated_store)
        var, cvar = analyser.value_at_risk("AAPL")
        assert isinstance(var, float)
        assert isinstance(cvar, float)
        assert var <= 0 or math.isnan(var)  # VaR should be negative or nan (too little data)

    def test_summary_missing_namespace_returns_empty(self, tmp_store):
        analyser = FinanceAnalyser(tmp_store)
        stats = analyser.summary("MISSING")
        assert stats.empty

    def test_var_missing_namespace_returns_nan(self, tmp_store):
        analyser = FinanceAnalyser(tmp_store)
        var, cvar = analyser.value_at_risk("MISSING")
        assert math.isnan(var) and math.isnan(cvar)
