"""Partitioned Parquet store (partition by ticker, date) plus a thin DuckDB
connection helper used by feature assembly.
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

MESSAGES_DIR = "messages"
FEATURES_DIR = "features"
QUEUE_POSITIONS_DIR = "queue_positions"


def write_partition(df: pd.DataFrame, base_dir: str, subdir: str) -> str:
    """Write one (ticker, date) session as its own Parquet file under
    ``<base_dir>/<subdir>/ticker=<ticker>/date=<date>/part.parquet``
    (Hive-style partitioning, queryable directly by DuckDB's
    ``read_parquet(..., hive_partitioning=true)``).
    """
    ticker = df["ticker"].iloc[0]
    date = df["date"].iloc[0]
    out_dir = Path(base_dir) / subdir / f"ticker={ticker}" / f"date={date}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "part.parquet"
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, out_path)
    return str(out_path)


def glob_pattern(base_dir: str, subdir: str) -> str:
    return str(Path(base_dir) / subdir / "**" / "*.parquet")


def partition_path(base_dir: str, subdir: str, ticker: str, date: str) -> Path:
    return Path(base_dir) / subdir / f"ticker={ticker}" / f"date={date}" / "part.parquet"


def read_partition(base_dir: str, subdir: str, ticker: str, date: str) -> pd.DataFrame:
    return pq.read_table(partition_path(base_dir, subdir, ticker, date)).to_pandas()


def connect(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:", read_only=False)
    con.execute("PRAGMA threads=4")
    return con


def messages_relation(con: duckdb.DuckDBPyConnection, base_dir: str) -> str:
    """Return a DuckDB SQL expression selecting every message-level row
    across every ticker/date partition, with ticker/date recovered from
    the Hive-style path (also present as real columns in the file, kept
    consistent by construction)."""
    pattern = glob_pattern(base_dir, MESSAGES_DIR)
    return f"read_parquet('{pattern}', hive_partitioning=true)"


def queue_positions_relation(con: duckdb.DuckDBPyConnection, base_dir: str) -> str:
    pattern = glob_pattern(base_dir, QUEUE_POSITIONS_DIR)
    return f"read_parquet('{pattern}', hive_partitioning=true)"
