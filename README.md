# jnu-timefs-for-finance

A financial data analysis project built on **TimeFS** — a filesystem-backed
time series datastore inspired by
[abhishekkr/timefs](https://github.com/abhishekkr/timefs).

## Overview

`jnu-timefs-for-finance` organises OHLCV (Open / High / Low / Close / Volume)
financial data in a hierarchical directory tree on disk so individual time
points ("timedots") can be written and retrieved without loading entire
datasets into memory.  On top of this store it provides a suite of technical
indicators and analysis utilities.

```
<store_root>/
└── <namespace>/          ← ticker symbol, e.g. AAPL
    └── <YYYY>/
        └── <MM>/
            └── <DD>/
                └── <HH>/
                    └── <MM>/
                        └── <SS>.json   ← timedot
```

## Project Structure

```
jnu-timefs-for-finance/
├── timefs/               # Core TimeFS implementation
│   ├── core.py           # TimeFS store, TimeDot model
│   └── query.py          # TimeQuery → pandas DataFrame bridge
├── finance/              # Financial layer
│   ├── loader.py         # FinanceLoader: ingest CSV / DataFrame → TimeFS
│   ├── indicators.py     # Technical indicators (SMA, EMA, MACD, RSI, …)
│   └── analysis.py       # FinanceAnalyser: summary, enrich, correlation, VaR
├── tests/                # pytest test suite
│   ├── test_timefs.py
│   └── test_finance.py
├── examples/
│   └── basic_analysis.py # End-to-end usage demo
├── requirements.txt
└── setup.py
```

## Installation

```bash
pip install -e ".[dev]"
```

Dependencies: `pandas`, `numpy` (runtime); `pytest` (dev).

## Quick Start

```python
import tempfile
import pandas as pd
from timefs.core import TimeFS
from finance.loader import FinanceLoader
from finance.analysis import FinanceAnalyser

# 1. Create a TimeFS store
store = TimeFS("/path/to/store")

# 2. Load OHLCV data from a DataFrame or CSV
loader = FinanceLoader(store)
df = pd.read_csv("aapl.csv")
loader.load_dataframe(df, namespace="AAPL", date_column="date")

# 3. Analyse
analyser = FinanceAnalyser(store)

# Descriptive statistics + annualised volatility + total return
print(analyser.summary("AAPL"))

# OHLCV enriched with SMA, EMA, MACD, RSI, Bollinger Bands, ATR, OBV
enriched = analyser.enrich("AAPL", sma_windows=[20, 50])
print(enriched.tail())

# Pairwise daily-return correlation
print(analyser.correlation(["AAPL", "TSLA", "GOOG"]))

# Historical VaR and CVaR at 95 % confidence
var, cvar = analyser.value_at_risk("AAPL", confidence=0.95)
print(f"VaR={var:.2%}  CVaR={cvar:.2%}")
```

## Technical Indicators

| Indicator | Function | Category |
|-----------|----------|----------|
| Simple Moving Average | `Indicators.sma(close, window)` | Trend |
| Exponential Moving Average | `Indicators.ema(close, window)` | Trend |
| MACD | `Indicators.macd(close, fast, slow, signal)` | Trend |
| Relative Strength Index | `Indicators.rsi(close, window=14)` | Momentum |
| Bollinger Bands | `Indicators.bollinger_bands(close, window, num_std)` | Volatility |
| Average True Range | `Indicators.atr(high, low, close, window=14)` | Volatility |
| On-Balance Volume | `Indicators.obv(close, volume)` | Volume |
| Daily Returns | `Indicators.daily_returns(close)` | Returns |
| Cumulative Returns | `Indicators.cumulative_returns(close)` | Returns |

## Running Tests

```bash
python -m pytest tests/ -v
```

## Running the Example

```bash
python examples/basic_analysis.py
```
