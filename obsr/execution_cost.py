"""Compares two execution rules' expected effective spread on the same
replayed order flow: always cross the spread immediately, versus a
rank-aware quote-and-wait rule that posts passively at the best bid and
blends the two possible outcomes (filled within the horizon, or not) by
the *measured* fill probability for the queue depth actually observed at
that instant.

This is an expected-value accounting exercise over real decision points
pulled from the simulated order flow, not a simulated P&L path: at every
decision point the rank-aware rule's cost is
``p_fill * (bid - mid) + (1 - p_fill) * (ask_later - mid)``, where
``p_fill`` comes from the empirical fill-probability-by-queue-depth curve
built in ``fill_probability.py`` (binned by quantity ahead, not by the
coarser integer rank bucket, because quantity ahead -- unlike order-count
rank -- is directly observable from an ordinary L2 snapshot at any
arbitrary decision point) and ``ask_later`` is the real ask price observed
one horizon later in that same session, found with
``pandas.merge_asof(..., direction="forward")``. "Always crossing" pays
``ask - mid`` at every decision point, unconditionally.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_PROB_BINS = 50
DECISION_POINT_STRIDE = 500  # every 500th event_seq per session


def build_fill_probability_curve(per_order: pd.DataFrame, n_bins: int = N_PROB_BINS) -> pd.DataFrame:
    """Bins ADD events by quantity ahead into ``n_bins`` quantile buckets and
    returns one row per bucket: its mean quantity ahead and the empirical
    fraction filled within the horizon, sorted by quantity ahead for a
    nearest-value lookup."""
    df = per_order.copy()
    df["qty_bin"] = pd.qcut(df["qty_ahead"], n_bins, duplicates="drop")
    curve = (
        df.groupby("qty_bin", observed=True)
        .agg(qty_ahead=("qty_ahead", "mean"), fill_probability=("filled_within_horizon", "mean"),
             n=("order_id", "count"))
        .reset_index(drop=True)
        .sort_values("qty_ahead")
        .reset_index(drop=True)
    )
    return curve


def select_decision_points(messages: pd.DataFrame, stride: int = DECISION_POINT_STRIDE) -> pd.DataFrame:
    dp = messages[messages["event_seq"] % stride == 0].copy()
    dp = dp[(dp["bid_px_1"] > 0) & (dp["ask_px_1"] > 0)]
    dp["mid"] = (dp["bid_px_1"] + dp["ask_px_1"]) / 2.0
    return dp.sort_values("bid_sz_1").reset_index(drop=True)


def attach_fill_probability(decision_points: pd.DataFrame, curve: pd.DataFrame) -> pd.DataFrame:
    dp = decision_points.copy()
    dp["bid_sz_1"] = dp["bid_sz_1"].astype(float)
    dp = dp.sort_values("bid_sz_1")
    curve = curve.copy()
    curve["qty_ahead"] = curve["qty_ahead"].astype(float)
    merged = pd.merge_asof(dp, curve[["qty_ahead", "fill_probability"]].sort_values("qty_ahead"),
                            left_on="bid_sz_1", right_on="qty_ahead", direction="nearest")
    return merged


def attach_future_ask(decision_points: pd.DataFrame, messages: pd.DataFrame,
                       horizon_seconds: float) -> pd.DataFrame:
    """For each decision point, finds the ask price at the first message at
    or after ``ts + horizon_seconds`` in the same (ticker, date) session via
    a forward as-of join."""
    dp = decision_points.copy()
    dp["target_ts"] = dp["ts"] + horizon_seconds
    dp = dp.sort_values("target_ts")
    lookups = messages[["ticker", "date", "ts", "ask_px_1"]].sort_values("ts")
    merged = pd.merge_asof(dp, lookups, left_on="target_ts", right_on="ts", by=["ticker", "date"],
                            direction="forward", suffixes=("", "_later"))
    return merged


def compute_execution_costs(decision_points: pd.DataFrame) -> dict:
    dp = decision_points.dropna(subset=["ask_px_1_later"])
    always_crossing_cost = dp["ask_px_1"] - dp["mid"]
    rank_aware_cost = (
        dp["fill_probability"] * (dp["bid_px_1"] - dp["mid"])
        + (1.0 - dp["fill_probability"]) * (dp["ask_px_1_later"] - dp["mid"])
    )
    mean_always = float(always_crossing_cost.mean())
    mean_rank_aware = float(rank_aware_cost.mean())
    pct_reduction = (mean_always - mean_rank_aware) / mean_always * 100.0
    return {
        "n_decision_points": int(len(dp)),
        "mean_always_crossing_effective_spread_ticks": mean_always,
        "mean_rank_aware_effective_spread_ticks": mean_rank_aware,
        "pct_less_effective_spread": pct_reduction,
    }
