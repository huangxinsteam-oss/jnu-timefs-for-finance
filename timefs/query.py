"""
timefs.query
~~~~~~~~~~~~

High-level query helpers that turn raw :class:`~timefs.core.TimeDot`
streams into pandas DataFrames.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from .core import TimeFS


class TimeQuery:
    """Convenience wrapper for querying a :class:`TimeFS` store."""

    def __init__(self, store: TimeFS) -> None:
        self.store = store

    def to_dataframe(
        self,
        namespace: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Return a :class:`pandas.DataFrame` for *namespace*.

        The index is a ``DatetimeIndex`` (UTC).  Each column corresponds to
        a key in the ``value`` dict of the stored :class:`~timefs.core.TimeDot`.

        Parameters
        ----------
        namespace : str
            The data namespace to query (e.g. ``"AAPL"``).
        start : datetime, optional
            Inclusive start of the time range.
        end : datetime, optional
            Inclusive end of the time range.

        Returns
        -------
        pandas.DataFrame
            Sorted by timestamp ascending; empty frame when no data exist.
        """
        records = []
        for dot in self.store.iter_namespace(namespace, start=start, end=end):
            row = {"timestamp": dot.timestamp}
            row.update(dot.value)
            records.append(row)

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
        return df
