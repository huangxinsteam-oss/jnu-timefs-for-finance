"""
finance.loader
~~~~~~~~~~~~~~

Utilities for ingesting financial OHLCV data into a TimeFS store.

Supported sources
-----------------
* CSV files with columns: ``date``, ``open``, ``high``, ``low``, ``close``,
  ``volume``  (column names are case-insensitive).
* ``pandas.DataFrame`` objects with the same schema.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from timefs.core import TimeDot, TimeFS
from timefs.query import TimeQuery


class FinanceLoader:
    """Load financial OHLCV data into a :class:`~timefs.core.TimeFS` store.

    Parameters
    ----------
    store : TimeFS
        The underlying TimeFS instance to write data into.
    """

    OHLCV_COLUMNS = {"open", "high", "low", "close", "volume"}

    def __init__(self, store: TimeFS) -> None:
        self.store = store
        self.query = TimeQuery(store)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def load_dataframe(
        self,
        df: pd.DataFrame,
        namespace: str,
        date_column: str = "date",
    ) -> int:
        """Persist every row of *df* as a :class:`~timefs.core.TimeDot`.

        Parameters
        ----------
        df : pandas.DataFrame
            Must contain at minimum a date/timestamp column and at least one
            value column.
        namespace : str
            Ticker symbol or any logical grouping name (e.g. ``"AAPL"``).
        date_column : str
            Name of the column that holds the date/timestamp.

        Returns
        -------
        int
            Number of timedots written.
        """
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        if date_column.lower() not in df.columns:
            raise ValueError(
                f"Column '{date_column}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

        date_col = date_column.lower()
        timestamps = pd.to_datetime(df[date_col], utc=True)
        value_cols = [c for c in df.columns if c != date_col]

        dots: List[TimeDot] = []
        for ts, (_, row) in zip(timestamps, df.iterrows()):
            value: Dict[str, float] = {}
            for col in value_cols:
                raw = row[col]
                try:
                    value[col] = float(raw)
                except (TypeError, ValueError):
                    value[col] = raw
            dots.append(TimeDot(timestamp=ts.to_pydatetime(), namespace=namespace, value=value))

        self.store.write_many(dots)
        return len(dots)

    def load_csv(
        self,
        path: Union[str, Path],
        namespace: str,
        date_column: str = "date",
        **read_csv_kwargs,
    ) -> int:
        """Read a CSV file and persist it into the TimeFS store.

        Parameters
        ----------
        path : str | Path
            Path to the CSV file.
        namespace : str
            Ticker symbol or logical group name.
        date_column : str
            Name of the date/timestamp column in the CSV.
        **read_csv_kwargs
            Extra keyword arguments forwarded to :func:`pandas.read_csv`.

        Returns
        -------
        int
            Number of timedots written.
        """
        df = pd.read_csv(path, **read_csv_kwargs)
        return self.load_dataframe(df, namespace, date_column=date_column)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get(
        self,
        namespace: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Retrieve stored data for *namespace* as a DataFrame.

        Parameters
        ----------
        namespace : str
            Ticker or group name to retrieve.
        start : datetime, optional
            Inclusive lower bound.
        end : datetime, optional
            Inclusive upper bound.

        Returns
        -------
        pandas.DataFrame
            OHLCV (or whatever fields were stored) indexed by UTC timestamp.
        """
        return self.query.to_dataframe(namespace, start=start, end=end)
