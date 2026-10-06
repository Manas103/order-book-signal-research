"""10-minute realized-volatility binning, built the same way as
``obsr.features``: DuckDB SQL window functions, partitioned by
``(ticker, date)``, so a bin or a forward label never reaches across a
session boundary.

A bin is 600 seconds of simulated wall-clock time (``ts``), not a fixed
event count: ``obsr.simulate`` paces messages with
``Exponential(0.05)``-second gaps (mean 20 events/second), so a 10-minute
bin holds roughly 12,000 events, not a round number of them. Each bin's
realized volatility is ``sqrt(sum(mid_diff^2))`` over every tick-to-tick
mid-price change whose event falls in that bin; the forward label is the
*next* bin's own realized volatility (``LEAD(bin_rv) OVER (PARTITION BY
ticker, date ORDER BY bin_id)``), undefined for the last bin of every
session, mirroring ``obsr.labels``' forward mid-price-change convention
exactly. The "horizon-matched previous-window baseline" is simply a bin's
own, already-realized volatility used as the forecast for the next bin.
"""

from __future__ import annotations

import duckdb
import pandas as pd

BIN_SECONDS = 600  # 10 minutes


def build_bin_rv_sql(messages_relation: str) -> str:
    """One row per (ticker, date, bin_id): realized volatility and message
    count within that bin, computed from tick-to-tick *microprice* changes,
    not raw best-bid/ask mid changes. The best bid/ask is heavily quantized
    and sticky in this generator (the touch changes only a few times per
    80,000-event session; see README Findings), which makes mid-based
    realized volatility degenerate at a 10-minute scale: almost every bin
    would measure exactly 0. Microprice (size-weighted between the best bid
    and ask) moves on every size change at the touch, not only when the
    best price level itself changes, and is already one of the 43
    microstructure features (``obsr.features``), so this is the same
    quantity the forecasting model already sees, used here as the honest
    realized-volatility base.

    The null guard is stricter than ``obsr.features``' own microprice
    column (``bid_sz_1 + ask_sz_1 = 0``): during the first couple of
    seeding events of every session, exactly one side has size 0 while its
    *price* also defaults to 0, so the sum-based guard does not fire and
    the formula divides a near-zero numerator by the live side's size,
    producing one spuriously well-defined value (observed: a microprice of
    exactly 0.0 immediately followed by a jump of several thousand ticks on
    the very next event). That single artifact dominates a bin's realized
    volatility if left in; see README Findings for how this was found.
    """
    return f"""
WITH base AS (
  SELECT ticker, date, day_index, event_seq, ts,
    CASE WHEN bid_sz_1 = 0 OR ask_sz_1 = 0 THEN NULL
         ELSE (bid_px_1 * ask_sz_1 + ask_px_1 * bid_sz_1) / CAST(bid_sz_1 + ask_sz_1 AS DOUBLE) END AS microprice,
    CAST(ts / {BIN_SECONDS} AS BIGINT) AS bin_id
  FROM {messages_relation}
),
diffed AS (
  SELECT *, microprice - LAG(microprice) OVER w AS micro_diff
  FROM base
  WINDOW w AS (PARTITION BY ticker, date ORDER BY event_seq)
)
SELECT
  ticker, date, day_index, bin_id,
  COUNT(*) AS msg_count,
  SQRT(SUM(POWER(COALESCE(micro_diff, 0), 2))) AS bin_rv
FROM diffed
GROUP BY ticker, date, day_index, bin_id
"""


def build_bin_rv(con: duckdb.DuckDBPyConnection, messages_relation: str) -> pd.DataFrame:
    return con.execute(build_bin_rv_sql(messages_relation)).fetchdf()


def attach_forward_rv(con: duckdb.DuckDBPyConnection, bin_rv: pd.DataFrame) -> pd.DataFrame:
    """Adds ``fwd_rv`` (next bin's realized volatility, the forecast target)
    and keeps ``bin_rv`` itself as ``prev_rv``, the horizon-matched
    previous-window baseline forecast. NULL ``fwd_rv`` on the last bin of
    every session, same convention as ``obsr.labels.compute_forward_labels``.
    """
    con.register("bin_rv_tbl", bin_rv)
    out = con.execute(
        """
        SELECT *, bin_rv AS prev_rv,
          LEAD(bin_rv) OVER (PARTITION BY ticker, date ORDER BY bin_id) AS fwd_rv
        FROM bin_rv_tbl
        """
    ).fetchdf()
    con.unregister("bin_rv_tbl")
    return out


def last_snapshot_per_bin(con: duckdb.DuckDBPyConnection, features: pd.DataFrame) -> pd.DataFrame:
    """The microstructure-feature row from the *last* event of each
    (ticker, date, bin_id): the state the forecast is made from."""
    feat = features.copy()
    feat["bin_id"] = (feat["ts"] // BIN_SECONDS).astype("int64")
    con.register("feat_tbl", feat)
    out = con.execute(
        """
        SELECT * EXCLUDE (rn) FROM (
          SELECT *, ROW_NUMBER() OVER (
            PARTITION BY ticker, date, bin_id ORDER BY event_seq DESC
          ) AS rn
          FROM feat_tbl
        )
        WHERE rn = 1
        """
    ).fetchdf()
    con.unregister("feat_tbl")
    return out


def rmspe(actual: pd.Series, predicted: pd.Series) -> float:
    """Root mean squared percentage error: sqrt(mean(((actual-pred)/actual)^2)).
    Undefined for an exactly-zero actual; callers filter those out, same as
    ``run_research.py``'s ``dropna`` guard before scoring.
    """
    pct_err = (actual.to_numpy() - predicted.to_numpy()) / actual.to_numpy()
    return float((pct_err ** 2).mean() ** 0.5)
