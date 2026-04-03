"""
timefs.core
~~~~~~~~~~~

Core TimeFS implementation.  Time series values are stored as JSON files
in a hierarchical directory tree::

    <root>/<namespace>/<YYYY>/<MM>/<DD>/<HH>/<MM>/<SS>.json

This mirrors the design of abhishekkr/timefs (Go), adapted for Python and
extended to support multi-field records (e.g. OHLCV financial data).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class TimeDot:
    """A single timestamped data point stored in the filesystem."""

    timestamp: datetime
    namespace: str
    value: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "namespace": self.namespace,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TimeDot":
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            namespace=data["namespace"],
            value=data["value"],
        )


class TimeFS:
    """
    Filesystem-backed time series store.

    Parameters
    ----------
    root : str | Path
        Root directory where all time series data will be stored.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dot_path(self, namespace: str, ts: datetime) -> Path:
        """Return the filesystem path for a single timedot."""
        t = ts.astimezone(timezone.utc)
        return (
            self.root
            / namespace
            / f"{t.year:04d}"
            / f"{t.month:02d}"
            / f"{t.day:02d}"
            / f"{t.hour:02d}"
            / f"{t.minute:02d}"
            / f"{t.second:02d}.json"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(self, dot: TimeDot) -> Path:
        """Persist a :class:`TimeDot` to the filesystem.

        Returns the path of the written file.
        """
        path = self._dot_path(dot.namespace, dot.timestamp)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dot.to_dict(), indent=2), encoding="utf-8")
        return path

    def read(self, namespace: str, ts: datetime) -> Optional[TimeDot]:
        """Read a single :class:`TimeDot` by namespace and exact timestamp.

        Returns *None* when the timedot does not exist.
        """
        path = self._dot_path(namespace, ts)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return TimeDot.from_dict(data)

    def write_many(self, dots: Iterable[TimeDot]) -> List[Path]:
        """Persist multiple :class:`TimeDot` objects at once."""
        return [self.write(dot) for dot in dots]

    def namespaces(self) -> List[str]:
        """Return the list of namespaces (top-level directories) stored."""
        if not self.root.exists():
            return []
        return sorted(
            entry.name
            for entry in self.root.iterdir()
            if entry.is_dir()
        )

    def iter_namespace(
        self,
        namespace: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> Iterable[TimeDot]:
        """Yield all :class:`TimeDot` objects for *namespace*.

        Parameters
        ----------
        namespace : str
            The data namespace (e.g. a ticker symbol such as ``"AAPL"``).
        start : datetime, optional
            Inclusive lower bound; UTC-naive datetimes are treated as UTC.
        end : datetime, optional
            Inclusive upper bound; UTC-naive datetimes are treated as UTC.
        """
        ns_root = self.root / namespace
        if not ns_root.exists():
            return

        def _to_utc(dt: Optional[datetime]) -> Optional[datetime]:
            if dt is None:
                return None
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)

        utc_start = _to_utc(start)
        utc_end = _to_utc(end)

        for json_file in sorted(ns_root.rglob("*.json")):
            data = json.loads(json_file.read_text(encoding="utf-8"))
            dot = TimeDot.from_dict(data)
            ts = _to_utc(dot.timestamp)
            if utc_start is not None and ts < utc_start:
                continue
            if utc_end is not None and ts > utc_end:
                continue
            yield dot

    def delete(self, namespace: str, ts: datetime) -> bool:
        """Delete the timedot at *namespace* / *ts*.

        Returns *True* if the file was removed, *False* if it did not exist.
        """
        path = self._dot_path(namespace, ts)
        if path.exists():
            path.unlink()
            return True
        return False

    def count(self, namespace: str) -> int:
        """Return the number of timedots stored for *namespace*."""
        ns_root = self.root / namespace
        if not ns_root.exists():
            return 0
        return sum(1 for _ in ns_root.rglob("*.json"))
