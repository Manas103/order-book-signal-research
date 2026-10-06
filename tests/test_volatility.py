"""Hand-computed bin realized volatility, the forward-label leakage
convention (undefined on each session's last bin, never reaching into the
next session), and the RMSPE helper, cross-checked against an independent
plain-pandas computation of the same quantities.
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest

from obsr.simulate import simulate_ticker_day
from obsr.volatility import BIN_SECONDS, attach_forward_rv, build_bin_rv, rmspe


def _small_sample():
    df1 = simulate_ticker_day("SYNA", "2025-01-06", 0, seed=1, n_msgs=4000)
    df2 = simulate_ticker_day("SYNA", "2025-01-07", 1, seed=2, n_msgs=4000)
    return pd.concat([df1, df2], ignore_index=True)


def _independent_bin_rv(raw: pd.DataFrame) -> pd.DataFrame:
    """A completely separate pandas re-implementation of build_bin_rv's SQL:
    no DuckDB, no shared code, just groupby/shift/sqrt(sum(x**2))."""
    out = raw.copy()
    both_sides_sized = (out["bid_sz_1"] != 0) & (out["ask_sz_1"] != 0)
    microprice = (out["bid_px_1"] * out["ask_sz_1"] + out["ask_px_1"] * out["bid_sz_1"]) / (
        out["bid_sz_1"] + out["ask_sz_1"]
    ).astype(float)
    out["microprice"] = microprice.where(both_sides_sized)
    out["bin_id"] = (out["ts"] // BIN_SECONDS).astype("int64")
    out["micro_diff"] = out.groupby(["ticker", "date"], sort=False)["microprice"].diff()
    out["micro_diff"] = out["micro_diff"].fillna(0.0)
    grouped = out.groupby(["ticker", "date", "bin_id"], sort=False)
    agg = grouped.agg(msg_count=("micro_diff", "size"),
                       bin_rv=("micro_diff", lambda s: float(np.sqrt((s ** 2).sum()))))
    return agg.reset_index()


def test_sql_bin_rv_matches_independent_pandas_reimplementation():
    raw = _small_sample()
    con = duckdb.connect(":memory:")
    con.register("raw_msgs", raw)
    sql_out = build_bin_rv(con, "raw_msgs")
    plain = _independent_bin_rv(raw)

    sql_out = sql_out.sort_values(["ticker", "date", "bin_id"]).reset_index(drop=True)
    plain = plain.sort_values(["ticker", "date", "bin_id"]).reset_index(drop=True)

    pd.testing.assert_series_equal(sql_out["msg_count"], plain["msg_count"], check_names=False)
    np.testing.assert_allclose(sql_out["bin_rv"].to_numpy(), plain["bin_rv"].to_numpy(), atol=1e-9)


def test_forward_rv_is_next_bins_own_rv_and_null_on_last_bin_of_session():
    raw = _small_sample()
    con = duckdb.connect(":memory:")
    con.register("raw_msgs", raw)
    bin_rv = build_bin_rv(con, "raw_msgs")
    labeled = attach_forward_rv(con, bin_rv)

    for (ticker, date), g in labeled.groupby(["ticker", "date"], sort=False):
        g = g.sort_values("bin_id").reset_index(drop=True)
        # every row except the last: fwd_rv equals the next row's own bin_rv
        for i in range(len(g) - 1):
            assert g.loc[i, "fwd_rv"] == pytest.approx(g.loc[i + 1, "bin_rv"])
        # the last bin of the session has no next bin: label must be null,
        # not accidentally pulled from the other session in this sample
        assert pd.isna(g.loc[len(g) - 1, "fwd_rv"])


def test_prev_rv_baseline_equals_the_bins_own_realized_vol():
    raw = _small_sample()
    con = duckdb.connect(":memory:")
    con.register("raw_msgs", raw)
    bin_rv = build_bin_rv(con, "raw_msgs")
    labeled = attach_forward_rv(con, bin_rv)
    pd.testing.assert_series_equal(labeled["prev_rv"], labeled["bin_rv"], check_names=False)


def test_rmspe_zero_for_a_perfect_forecast():
    actual = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert rmspe(actual, actual) == pytest.approx(0.0)


def test_rmspe_hand_computed_case():
    actual = pd.Series([10.0, 20.0])
    predicted = pd.Series([12.0, 18.0])
    # pct errors: (10-12)/10=-0.2, (20-18)/20=0.1 -> mean of squares = (0.04+0.01)/2 = 0.025
    expected = 0.025 ** 0.5
    assert rmspe(actual, predicted) == pytest.approx(expected)


def test_one_sided_empty_book_does_not_produce_a_spurious_microprice_jump():
    """The bug this test pins: a sum-based null guard (bid_sz_1 + ask_sz_1
    = 0) does not fire when exactly one side has size 0 but the empty
    side's *price* also defaults to 0 (true during the first couple of
    seeding events of every real session, see obsr/simulate.py), so the
    microprice formula divides a near-zero numerator by the live side's
    size and returns a spuriously well-defined 0.0 instead of NULL. The
    very next row then looks like a multi-thousand-tick jump. The fix
    guards on bid_sz_1 = 0 OR ask_sz_1 = 0 directly, which this constructs.
    """
    raw = pd.DataFrame({
        "ticker": ["SYNA"] * 3, "date": ["2025-01-06"] * 3, "day_index": [0, 0, 0],
        "event_seq": [0, 1, 2], "ts": [0.01, 0.02, 0.03],
        "bid_px_1": [4999, 4999, 4999], "bid_sz_1": [50, 50, 50],
        "ask_px_1": [0, 5001, 5001], "ask_sz_1": [0, 50, 50],
    })
    con = duckdb.connect(":memory:")
    con.register("raw_msgs", raw)
    out = build_bin_rv(con, "raw_msgs")
    assert len(out) == 1
    # the true microprice sequence is NULL, 5000.0, 5000.0: exactly one
    # real transition (NULL -> 5000.0, excluded by COALESCE-to-0) and one
    # zero-change step, so bin_rv must be 0.0, not several thousand.
    assert out.loc[0, "bin_rv"] == pytest.approx(0.0)
