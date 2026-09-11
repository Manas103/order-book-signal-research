"""Microstructure feature engineering, built entirely with DuckDB SQL window
functions against the partitioned Parquet message store (never
``pandas.shift()``/``.rolling()``): multi-level order flow imbalance and
queue imbalance (5 levels), spread, microprice, depth, per-type message-rate
features, realized short-horizon volatility, signed trade flow, lagged
returns and a running net-flow total, plus the three forward-horizon labels.

Every window is ``PARTITION BY ticker, date ORDER BY event_seq``: no lag,
rolling-window or forward label ever reaches across a session boundary, by
construction, not by post-hoc filtering (see tests/test_leakage.py).
"""

from __future__ import annotations

import duckdb
import pandas as pd

from .labels import HORIZONS, label_col

DEPTH = 5
COUNT_WINDOWS = (20, 50, 100)
VOL_WINDOWS = (20, 50, 100)
FLOW_WINDOWS = (10, 50, 100)
GAP_WINDOWS = (20, 100)
LAG_RETURNS = (1, 2, 3)
MSG_TYPES = ("ADD", "CANCEL", "EXECUTE", "DELETE")

FEATURE_COLUMNS = (
    ["spread", "microprice", "bid_depth_total", "ask_depth_total", "depth_imbalance"]
    + [f"ofi_l{k}" for k in range(1, DEPTH + 1)] + ["ofi_agg5"]
    + [f"qi_l{k}" for k in range(1, DEPTH + 1)] + ["qi_agg5"]
    + ["bid_sz_1", "ask_sz_1"]
    + [f"msg_count_{t.lower()}_w{w}" for t in MSG_TYPES for w in COUNT_WINDOWS]
    + [f"realized_vol_w{w}" for w in VOL_WINDOWS]
    + [f"trade_flow_imb_w{w}" for w in FLOW_WINDOWS]
    + [f"gap_mean_w{w}" for w in GAP_WINDOWS]
    + [f"lag_ret_{j}" for j in LAG_RETURNS]
    + ["cum_net_flow"]
)

LABEL_COLUMNS = [label_col(h) for h in HORIZONS]
ID_COLUMNS = ["ticker", "date", "day_index", "event_seq", "ts", "msg_type", "side", "mid"]


def _ofi_expr(k: int) -> str:
    return f"(bid_sz_{k} - LAG(bid_sz_{k}) OVER w) - (ask_sz_{k} - LAG(ask_sz_{k}) OVER w) AS ofi_l{k}"


def _qi_expr(k: int) -> str:
    return f"(bid_sz_{k} - ask_sz_{k}) / NULLIF(bid_sz_{k} + ask_sz_{k}, 0) AS qi_l{k}"


def build_feature_sql(messages_relation: str) -> str:
    ofi_cols = ",\n    ".join(_ofi_expr(k) for k in range(1, DEPTH + 1))
    qi_cols = ",\n    ".join(_qi_expr(k) for k in range(1, DEPTH + 1))
    lead_labels = ",\n    ".join(
        f"LEAD(mid, {h}) OVER w - mid AS {label_col(h)}" for h in HORIZONS
    )
    lag_rets = ",\n    ".join(
        f"mid - LAG(mid, {j}) OVER w AS lag_ret_{j}" for j in LAG_RETURNS
    )
    count_cols = ",\n    ".join(
        f"COUNT(*) FILTER (WHERE msg_type = '{t}') OVER w{w} AS msg_count_{t.lower()}_w{w}"
        for t in MSG_TYPES for w in COUNT_WINDOWS
    )
    vol_cols = ",\n    ".join(f"STDDEV_SAMP(mid_diff) OVER w{w} AS realized_vol_w{w}" for w in VOL_WINDOWS)
    flow_cols = ",\n    ".join(
        f"SUM(signed_exec_qty) OVER w{w} AS trade_flow_imb_w{w}" for w in FLOW_WINDOWS
    )
    gap_cols = ",\n    ".join(f"AVG(time_gap) OVER w{w} AS gap_mean_w{w}" for w in GAP_WINDOWS)
    window_defs = ",\n    ".join(
        f"w{w} AS (PARTITION BY ticker, date ORDER BY event_seq ROWS BETWEEN {w} PRECEDING AND CURRENT ROW)"
        for w in sorted(set(COUNT_WINDOWS) | set(VOL_WINDOWS) | set(FLOW_WINDOWS) | set(GAP_WINDOWS))
    )

    return f"""
WITH base AS (
  SELECT *,
    (bid_px_1 + ask_px_1) / 2.0 AS mid,
    (ask_px_1 - bid_px_1) AS spread,
    CASE WHEN (bid_sz_1 + ask_sz_1) = 0 THEN NULL
         ELSE (bid_px_1 * ask_sz_1 + ask_px_1 * bid_sz_1) / (bid_sz_1 + ask_sz_1) END AS microprice,
    (bid_sz_1 + bid_sz_2 + bid_sz_3 + bid_sz_4 + bid_sz_5) AS bid_depth_total,
    (ask_sz_1 + ask_sz_2 + ask_sz_3 + ask_sz_4 + ask_sz_5) AS ask_depth_total,
    CASE WHEN msg_type = 'EXECUTE' AND side = 'B' THEN qty
         WHEN msg_type = 'EXECUTE' AND side = 'S' THEN -qty
         ELSE 0 END AS signed_exec_qty
  FROM {messages_relation}
),
step1 AS (
  SELECT *,
    (bid_depth_total - ask_depth_total) AS depth_imbalance_num,
    (bid_depth_total + ask_depth_total) AS depth_imbalance_den,
    {ofi_cols},
    {qi_cols},
    mid - LAG(mid, 1) OVER w AS mid_diff,
    {lag_rets},
    ts - LAG(ts, 1) OVER w AS time_gap,
    SUM(size_delta) OVER (PARTITION BY ticker, date ORDER BY event_seq
                           ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cum_net_flow,
    {lead_labels}
  FROM base
  WINDOW w AS (PARTITION BY ticker, date ORDER BY event_seq)
)
SELECT
  ticker, date, day_index, event_seq, ts, msg_type, side, mid,
  spread, microprice, bid_depth_total, ask_depth_total,
  depth_imbalance_num / NULLIF(depth_imbalance_den, 0) AS depth_imbalance,
  ofi_l1, ofi_l2, ofi_l3, ofi_l4, ofi_l5,
  (ofi_l1 + ofi_l2 + ofi_l3 + ofi_l4 + ofi_l5) AS ofi_agg5,
  qi_l1, qi_l2, qi_l3, qi_l4, qi_l5,
  (qi_l1 + qi_l2 + qi_l3 + qi_l4 + qi_l5) / 5.0 AS qi_agg5,
  bid_sz_1, ask_sz_1,
  {count_cols},
  {vol_cols},
  {flow_cols},
  {gap_cols},
  lag_ret_1, lag_ret_2, lag_ret_3,
  cum_net_flow,
  {', '.join(label_col(h) for h in HORIZONS)}
FROM step1
WINDOW
    {window_defs}
"""


def build_features(con: duckdb.DuckDBPyConnection, messages_relation: str) -> pd.DataFrame:
    return con.execute(build_feature_sql(messages_relation)).fetchdf()
