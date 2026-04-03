"""
finance.indicators
~~~~~~~~~~~~~~~~~~

Common technical indicators for financial time series analysis.

All functions accept a :class:`pandas.DataFrame` (or :class:`pandas.Series`)
and return a :class:`pandas.Series` unless stated otherwise, so they
compose naturally with a TimeFS-backed DataFrame.
"""

from __future__ import annotations

import pandas as pd


class Indicators:
    """
    Namespace for technical indicator calculations.

    All methods are *static* so the class can be used without instantiation::

        from finance.indicators import Indicators
        sma = Indicators.sma(df["close"], window=20)
    """

    # ------------------------------------------------------------------
    # Trend indicators
    # ------------------------------------------------------------------

    @staticmethod
    def sma(series: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average.

        Parameters
        ----------
        series : pandas.Series
            Price series (typically ``close``).
        window : int
            Look-back window in bars.

        Returns
        -------
        pandas.Series
            Rolling mean with the same index as *series*.
        """
        return series.rolling(window=window).mean().rename(f"SMA_{window}")

    @staticmethod
    def ema(series: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average.

        Parameters
        ----------
        series : pandas.Series
            Price series.
        window : int
            Span for the EMA calculation (``adjust=False``).

        Returns
        -------
        pandas.Series
        """
        return series.ewm(span=window, adjust=False).mean().rename(f"EMA_{window}")

    @staticmethod
    def macd(
        series: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.DataFrame:
        """Moving Average Convergence Divergence.

        Parameters
        ----------
        series : pandas.Series
            Close price series.
        fast : int
            Fast EMA period (default 12).
        slow : int
            Slow EMA period (default 26).
        signal : int
            Signal line EMA period (default 9).

        Returns
        -------
        pandas.DataFrame
            Columns: ``macd``, ``signal``, ``histogram``.
        """
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return pd.DataFrame(
            {"macd": macd_line, "signal": signal_line, "histogram": histogram}
        )

    # ------------------------------------------------------------------
    # Momentum indicators
    # ------------------------------------------------------------------

    @staticmethod
    def rsi(series: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index.

        Parameters
        ----------
        series : pandas.Series
            Close price series.
        window : int
            Look-back window (default 14).

        Returns
        -------
        pandas.Series
            RSI values in the range [0, 100].
        """
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
        avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
        rs = avg_gain / avg_loss.replace(0, float("nan"))
        rsi = 100 - (100 / (1 + rs))
        return rsi.rename("RSI")

    # ------------------------------------------------------------------
    # Volatility indicators
    # ------------------------------------------------------------------

    @staticmethod
    def bollinger_bands(
        series: pd.Series,
        window: int = 20,
        num_std: float = 2.0,
    ) -> pd.DataFrame:
        """Bollinger Bands.

        Parameters
        ----------
        series : pandas.Series
            Close price series.
        window : int
            Rolling window for the middle band (default 20).
        num_std : float
            Number of standard deviations for upper/lower bands (default 2).

        Returns
        -------
        pandas.DataFrame
            Columns: ``bb_middle``, ``bb_upper``, ``bb_lower``.
        """
        middle = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        upper = middle + num_std * std
        lower = middle - num_std * std
        return pd.DataFrame(
            {"bb_middle": middle, "bb_upper": upper, "bb_lower": lower}
        )

    @staticmethod
    def atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14,
    ) -> pd.Series:
        """Average True Range.

        Parameters
        ----------
        high, low, close : pandas.Series
            OHLCV price columns.
        window : int
            Look-back window (default 14).

        Returns
        -------
        pandas.Series
        """
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.ewm(com=window - 1, min_periods=window).mean().rename("ATR")

    # ------------------------------------------------------------------
    # Volume indicators
    # ------------------------------------------------------------------

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume.

        Parameters
        ----------
        close : pandas.Series
            Close price series.
        volume : pandas.Series
            Volume series.

        Returns
        -------
        pandas.Series
        """
        direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        direction.iloc[0] = 0
        return (direction * volume).cumsum().rename("OBV")

    # ------------------------------------------------------------------
    # Returns
    # ------------------------------------------------------------------

    @staticmethod
    def daily_returns(close: pd.Series) -> pd.Series:
        """Percentage daily returns.

        Returns
        -------
        pandas.Series
        """
        return close.pct_change().rename("daily_return")

    @staticmethod
    def cumulative_returns(close: pd.Series) -> pd.Series:
        """Cumulative returns relative to the first bar.

        Returns
        -------
        pandas.Series
        """
        return ((1 + close.pct_change()).cumprod() - 1).rename("cumulative_return")
