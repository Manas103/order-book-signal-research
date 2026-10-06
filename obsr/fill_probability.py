"""Fill probability by queue rank, computed in one DuckDB SQL statement
built entirely from window functions (no join operator): queue-position
rows (one per ADD, from ``queue_position.py``) and EXECUTE-only message rows
are unioned into a single relation per ``(ticker, date, order_id)``, ordered
by time, and a single forward-looking window function reads off the first
EXECUTE timestamp at or after each ADD without a self-join.
"""

from __future__ import annotations

import duckdb
import pandas as pd

HORIZON_SECONDS = 1.0
# Buckets 1-5 are exact ranks; 6 means "6 or more". Collapsing at rank 5
# itself (rather than one bucket higher) was tried first and diluted "rank
# 5" with a heavy tail of much deeper, almost-never-filled orders (see
# README Findings); exact ranks 1-5 plus a separate 6+ tail is what is
# reported below.
RANK_BUCKETS = (1, 2, 3, 4, 5, 6)


def build_fill_probability_sql(queue_positions_relation: str, messages_relation: str,
                                horizon_seconds: float = HORIZON_SECONDS) -> str:
    return f"""
    WITH combined AS (
        SELECT ticker, date, order_id, add_ts AS ts, 'ADD' AS msg_type,
               rank, qty_ahead, is_touch, add_ts
        FROM {queue_positions_relation}
        UNION ALL
        SELECT ticker, date, order_id, ts, msg_type,
               NULL AS rank, NULL AS qty_ahead, NULL AS is_touch, NULL AS add_ts
        FROM {messages_relation}
        WHERE msg_type = 'EXECUTE'
    ),
    with_next_exec AS (
        SELECT *,
            MIN(CASE WHEN msg_type = 'EXECUTE' THEN ts END) OVER (
                PARTITION BY ticker, date, order_id ORDER BY ts
                ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING
            ) AS next_exec_ts
        FROM combined
    )
    SELECT ticker, date, order_id, add_ts, rank, qty_ahead, is_touch, next_exec_ts,
           (next_exec_ts IS NOT NULL AND next_exec_ts <= add_ts + {horizon_seconds}) AS filled_within_horizon
    FROM with_next_exec
    WHERE msg_type = 'ADD'
    """


def rank_bucket(rank: int) -> int:
    return min(rank, RANK_BUCKETS[-1])


def compute_fill_probability_by_rank(con: duckdb.DuckDBPyConnection, queue_positions_relation: str,
                                      messages_relation: str, touch_only: bool = False) -> pd.DataFrame:
    """``touch_only=True`` restricts the rank summary to ADD events that
    joined the best bid/ask at the instant of insertion ("at the touch"),
    which is the standard microstructure framing for "queue rank": comparing
    rank 1 against rank 5 across the *whole* book conflates queue position
    with how far from the market the price level sits (deep-book orders
    rarely fill quickly for a completely different reason). The per-order
    table returned always includes every ADD event regardless, so callers
    needing the unrestricted view still have it."""
    sql = build_fill_probability_sql(queue_positions_relation, messages_relation)
    per_order = con.execute(sql).fetchdf()
    per_order["rank_bucket"] = per_order["rank"].apply(rank_bucket)
    scope = per_order[per_order["is_touch"]] if touch_only else per_order
    summary = (
        scope.groupby("rank_bucket")
        .agg(n_orders=("order_id", "count"), n_filled=("filled_within_horizon", "sum"))
        .reset_index()
    )
    summary["fill_probability"] = summary["n_filled"] / summary["n_orders"]
    return summary, per_order
