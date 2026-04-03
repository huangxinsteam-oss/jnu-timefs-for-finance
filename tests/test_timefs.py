"""Tests for the timefs core and query modules."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from timefs.core import TimeDot, TimeFS
from timefs.query import TimeQuery


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_store(tmp_path):
    """Return a fresh TimeFS instance backed by a temporary directory."""
    return TimeFS(tmp_path / "store")


@pytest.fixture
def sample_dot():
    ts = datetime(2024, 1, 15, 9, 30, 0, tzinfo=timezone.utc)
    return TimeDot(
        timestamp=ts,
        namespace="AAPL",
        value={"open": 185.0, "high": 188.0, "low": 184.5, "close": 187.5, "volume": 1_000_000},
    )


# ---------------------------------------------------------------------------
# TimeDot
# ---------------------------------------------------------------------------


class TestTimeDot:
    def test_to_dict_roundtrip(self, sample_dot):
        d = sample_dot.to_dict()
        restored = TimeDot.from_dict(d)
        assert restored.namespace == sample_dot.namespace
        assert restored.timestamp == sample_dot.timestamp
        assert restored.value == sample_dot.value

    def test_from_dict_preserves_value_types(self, sample_dot):
        d = sample_dot.to_dict()
        restored = TimeDot.from_dict(d)
        assert isinstance(restored.value["close"], float)


# ---------------------------------------------------------------------------
# TimeFS – write / read
# ---------------------------------------------------------------------------


class TestTimeFSWriteRead:
    def test_write_creates_file(self, tmp_store, sample_dot):
        path = tmp_store.write(sample_dot)
        assert path.exists()

    def test_write_file_contains_valid_json(self, tmp_store, sample_dot):
        path = tmp_store.write(sample_dot)
        data = json.loads(path.read_text())
        assert data["namespace"] == "AAPL"
        assert data["value"]["close"] == 187.5

    def test_read_existing(self, tmp_store, sample_dot):
        tmp_store.write(sample_dot)
        dot = tmp_store.read(sample_dot.namespace, sample_dot.timestamp)
        assert dot is not None
        assert dot.value["close"] == 187.5

    def test_read_nonexistent_returns_none(self, tmp_store):
        ts = datetime(2023, 1, 1, tzinfo=timezone.utc)
        assert tmp_store.read("XYZ", ts) is None

    def test_write_many(self, tmp_store):
        dots = [
            TimeDot(
                timestamp=datetime(2024, 1, i, tzinfo=timezone.utc),
                namespace="TSLA",
                value={"close": float(200 + i)},
            )
            for i in range(1, 6)
        ]
        paths = tmp_store.write_many(dots)
        assert len(paths) == 5
        assert all(p.exists() for p in paths)


# ---------------------------------------------------------------------------
# TimeFS – namespaces / count / delete
# ---------------------------------------------------------------------------


class TestTimeFSMeta:
    def test_namespaces_empty(self, tmp_store):
        assert tmp_store.namespaces() == []

    def test_namespaces_populated(self, tmp_store, sample_dot):
        tmp_store.write(sample_dot)
        assert "AAPL" in tmp_store.namespaces()

    def test_count_zero(self, tmp_store):
        assert tmp_store.count("MISSING") == 0

    def test_count_nonzero(self, tmp_store, sample_dot):
        tmp_store.write(sample_dot)
        assert tmp_store.count("AAPL") == 1

    def test_delete_existing(self, tmp_store, sample_dot):
        tmp_store.write(sample_dot)
        removed = tmp_store.delete(sample_dot.namespace, sample_dot.timestamp)
        assert removed is True
        assert tmp_store.read(sample_dot.namespace, sample_dot.timestamp) is None

    def test_delete_nonexistent(self, tmp_store):
        ts = datetime(2020, 1, 1, tzinfo=timezone.utc)
        assert tmp_store.delete("GHOST", ts) is False


# ---------------------------------------------------------------------------
# TimeFS – iter_namespace with date filtering
# ---------------------------------------------------------------------------


class TestTimeFSIterNamespace:
    def _make_dots(self, namespace="GOOG"):
        return [
            TimeDot(
                timestamp=datetime(2024, 3, d, tzinfo=timezone.utc),
                namespace=namespace,
                value={"close": float(d * 10)},
            )
            for d in range(1, 6)
        ]

    def test_iter_all(self, tmp_store):
        dots = self._make_dots()
        tmp_store.write_many(dots)
        result = list(tmp_store.iter_namespace("GOOG"))
        assert len(result) == 5

    def test_iter_start_filter(self, tmp_store):
        dots = self._make_dots()
        tmp_store.write_many(dots)
        start = datetime(2024, 3, 3, tzinfo=timezone.utc)
        result = list(tmp_store.iter_namespace("GOOG", start=start))
        assert all(d.timestamp >= start for d in result)
        assert len(result) == 3

    def test_iter_end_filter(self, tmp_store):
        dots = self._make_dots()
        tmp_store.write_many(dots)
        end = datetime(2024, 3, 2, tzinfo=timezone.utc)
        result = list(tmp_store.iter_namespace("GOOG", end=end))
        assert all(d.timestamp <= end for d in result)
        assert len(result) == 2

    def test_iter_range_filter(self, tmp_store):
        dots = self._make_dots()
        tmp_store.write_many(dots)
        start = datetime(2024, 3, 2, tzinfo=timezone.utc)
        end = datetime(2024, 3, 4, tzinfo=timezone.utc)
        result = list(tmp_store.iter_namespace("GOOG", start=start, end=end))
        assert len(result) == 3

    def test_iter_missing_namespace_yields_nothing(self, tmp_store):
        result = list(tmp_store.iter_namespace("MISSING"))
        assert result == []


# ---------------------------------------------------------------------------
# TimeQuery
# ---------------------------------------------------------------------------


class TestTimeQuery:
    def _populate(self, store, namespace="MSFT"):
        dots = [
            TimeDot(
                timestamp=datetime(2024, 5, d, tzinfo=timezone.utc),
                namespace=namespace,
                value={"open": 400.0 + d, "close": 401.0 + d, "volume": d * 1000},
            )
            for d in range(1, 4)
        ]
        store.write_many(dots)

    def test_to_dataframe_shape(self, tmp_store):
        self._populate(tmp_store)
        q = TimeQuery(tmp_store)
        df = q.to_dataframe("MSFT")
        assert len(df) == 3
        assert set(["open", "close", "volume"]).issubset(df.columns)

    def test_to_dataframe_sorted(self, tmp_store):
        self._populate(tmp_store)
        q = TimeQuery(tmp_store)
        df = q.to_dataframe("MSFT")
        assert df.index.is_monotonic_increasing

    def test_to_dataframe_empty_namespace(self, tmp_store):
        q = TimeQuery(tmp_store)
        df = q.to_dataframe("EMPTY")
        assert df.empty
