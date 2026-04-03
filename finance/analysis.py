"""
finance.analysis
~~~~~~~~~~~~~~~~

High-level analysis and reporting utilities that combine TimeFS retrieval
with technical indicator calculations.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from timefs.core import TimeFS
from .loader import FinanceLoader
from .indicators import Indicators


class FinanceAnalyser:
    """
    End-to-end financial analysis over a TimeFS-backed store.

    Parameters
    ----------
    store : TimeFS
        Initialised TimeFS store containing OHLCV data.
    """

    def __init__(self, store: TimeFS) -> None:
        self.store = store
        self.loader = FinanceLoader(store)

    # ------------------------------------------------------------------
    # Data access helpers
    # ------------------------------------------------------------------

    def get_ohlcv(
        self,
        namespace: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Return OHLCV data for *namespace* as a DataFrame."""
        return self.loader.get(namespace, start=start, end=end)

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def summary(self, namespace: str) -> pd.DataFrame:
        """Return descriptive statistics for the close price series.

        Parameters
        ----------
        namespace : str
            Ticker or group name.

        Returns
        -------
        pandas.DataFrame
            ``describe()`` output plus annualised volatility and total return.
        """
        df = self.get_ohlcv(namespace)
        if df.empty or "close" not in df.columns:
            return pd.DataFrame()

        close = df["close"]
        stats = close.describe().to_frame()

        daily_ret = Indicators.daily_returns(close).dropna()
        ann_vol = daily_ret.std() * (252 ** 0.5)
        total_ret = (close.iloc[-1] / close.iloc[0]) - 1

        extra = pd.DataFrame(
            {"close": {"ann_volatility": ann_vol, "total_return": total_ret}}
        )
        return pd.concat([stats, extra])

    # ------------------------------------------------------------------
    # Indicator enrichment
    # ------------------------------------------------------------------

    def enrich(
        self,
        namespace: str,
        sma_windows: Optional[List[int]] = None,
        ema_windows: Optional[List[int]] = None,
        include_rsi: bool = True,
        include_macd: bool = True,
        include_bbands: bool = True,
        include_atr: bool = True,
        include_obv: bool = True,
        include_returns: bool = True,
    ) -> pd.DataFrame:
        """Return the OHLCV DataFrame enriched with technical indicators.

        Parameters
        ----------
        namespace : str
            Ticker or group name.
        sma_windows : list[int], optional
            Windows for Simple Moving Average (default: [20, 50]).
        ema_windows : list[int], optional
            Windows for Exponential Moving Average (default: [12, 26]).
        include_rsi : bool
            Include RSI(14) column.
        include_macd : bool
            Include MACD, signal, and histogram columns.
        include_bbands : bool
            Include Bollinger Band columns.
        include_atr : bool
            Include ATR(14) column (requires high/low/close).
        include_obv : bool
            Include OBV column (requires close/volume).
        include_returns : bool
            Include daily and cumulative return columns.

        Returns
        -------
        pandas.DataFrame
        """
        df = self.get_ohlcv(namespace)
        if df.empty or "close" not in df.columns:
            return df

        close = df["close"]

        if sma_windows is None:
            sma_windows = [20, 50]
        if ema_windows is None:
            ema_windows = [12, 26]

        for w in sma_windows:
            df[f"SMA_{w}"] = Indicators.sma(close, w)
        for w in ema_windows:
            df[f"EMA_{w}"] = Indicators.ema(close, w)

        if include_rsi:
            df["RSI"] = Indicators.rsi(close)

        if include_macd:
            macd_df = Indicators.macd(close)
            df = pd.concat([df, macd_df], axis=1)

        if include_bbands:
            bb_df = Indicators.bollinger_bands(close)
            df = pd.concat([df, bb_df], axis=1)

        if include_atr and {"high", "low", "close"}.issubset(df.columns):
            df["ATR"] = Indicators.atr(df["high"], df["low"], df["close"])

        if include_obv and {"close", "volume"}.issubset(df.columns):
            df["OBV"] = Indicators.obv(df["close"], df["volume"])

        if include_returns:
            df["daily_return"] = Indicators.daily_returns(close)
            df["cumulative_return"] = Indicators.cumulative_returns(close)

        return df

    # ------------------------------------------------------------------
    # Correlation analysis
    # ------------------------------------------------------------------

    def correlation(
        self,
        namespaces: List[str],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Compute pairwise Pearson correlation of daily returns.

        Parameters
        ----------
        namespaces : list[str]
            Tickers to compare.
        start, end : datetime, optional
            Time range filter applied to all tickers.

        Returns
        -------
        pandas.DataFrame
            Correlation matrix (NaN where insufficient data).
        """
        returns: Dict[str, pd.Series] = {}
        for ns in namespaces:
            df = self.get_ohlcv(ns, start=start, end=end)
            if not df.empty and "close" in df.columns:
                returns[ns] = Indicators.daily_returns(df["close"])
        if not returns:
            return pd.DataFrame()
        return pd.DataFrame(returns).corr()

    # ------------------------------------------------------------------
    # Risk metrics
    # ------------------------------------------------------------------

    def value_at_risk(
        self,
        namespace: str,
        confidence: float = 0.95,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> Tuple[float, float]:
        """Historical Value at Risk (VaR) and Conditional VaR (CVaR).

        Parameters
        ----------
        namespace : str
            Ticker or group name.
        confidence : float
            Confidence level (default 0.95 → 95% VaR).
        start, end : datetime, optional
            Time range filter.

        Returns
        -------
        tuple[float, float]
            ``(VaR, CVaR)`` as fractions (e.g. ``-0.02`` means −2%).
        """
        df = self.get_ohlcv(namespace, start=start, end=end)
        if df.empty or "close" not in df.columns:
            return (float("nan"), float("nan"))

        ret = Indicators.daily_returns(df["close"]).dropna()
        var = ret.quantile(1 - confidence)
        cvar = ret[ret <= var].mean()
        return (float(var), float(cvar))
