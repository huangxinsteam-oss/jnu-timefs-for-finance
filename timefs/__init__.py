"""
TimeFS: A filesystem-based time series datastore.

Organises time series data on disk using a hierarchical directory
structure keyed by date/time so that individual time points ("timedots")
can be written and retrieved without loading entire datasets into memory.
"""

from .core import TimeFS, TimeDot
from .query import TimeQuery

__all__ = ["TimeFS", "TimeDot", "TimeQuery"]
