"""Proves the day-level purged split has no overlap, and that the SQL LEAD
label generator agrees exactly with an independent plain-pandas
implementation (obsr.labels.compute_forward_labels)."""
from __future__ import annotations

import duckdb
import pandas as pd

from obsr.features import build_feature_sql
from obsr.labels import HORIZONS, compute_forward_labels, label_col
from obsr.simulate import simulate_ticker_day


def _small_sample():
    df1 = simulate_ticker_day("SYNA", "2025-01-06", 0, seed=1, n_msgs=3000)
    df2 = simulate_ticker_day("SYNA", "2025-01-07", 1, seed=2, n_msgs=3000)
    return pd.concat([df1, df2], ignore_index=True)


def test_sql_labels_match_independent_pandas_reimplementation():
    raw = _small_sample()
    con = duckdb.connect(":memory:")
    con.register("raw_msgs", raw)
    sql_out = con.execute(build_feature_sql("raw_msgs")).fetchdf()

    mid = (raw["bid_px_1"] + raw["ask_px_1"]) / 2.0
    plain = raw.assign(mid=mid)
    plain = compute_forward_labels(plain, HORIZONS)

    sql_out = sql_out.sort_values(["ticker", "date", "event_seq"]).reset_index(drop=True)
    plain = plain.sort_values(["ticker", "date", "event_seq"]).reset_index(drop=True)

    for h in HORIZONS:
        col = label_col(h)
        pd.testing.assert_series_equal(
            sql_out[col].reset_index(drop=True),
            plain[col].reset_index(drop=True),
            check_names=False, atol=1e-9,
        )


def test_label_is_null_for_last_h_rows_of_each_session_not_leaked_from_next():
    raw = _small_sample()
    con = duckdb.connect(":memory:")
    con.register("raw_msgs", raw)
    out = con.execute(build_feature_sql("raw_msgs")).fetchdf()
    for (ticker, date), g in out.groupby(["ticker", "date"], sort=False):
        g = g.sort_values("event_seq")
        for h in HORIZONS:
            col = label_col(h)
            tail = g[col].iloc[-h:]
            assert tail.isna().all(), (
                f"label {col} should be null for the last {h} rows of "
                f"{ticker}/{date}, since a real forward label there would "
                f"have to reach into the next session"
            )


def test_purged_day_split_has_no_date_overlap_and_respects_embargo():
    from scripts.run_research import purged_split, N_TRAIN_DAYS, N_EMBARGO_DAYS

    dates = pd.bdate_range("2025-01-06", periods=16).strftime("%Y-%m-%d")
    df = pd.DataFrame({"date": list(dates) * 10})
    train, test = purged_split(df)
    train_dates = set(train["date"])
    test_dates = set(test["date"])
    assert train_dates.isdisjoint(test_dates)
    assert len(train_dates) == N_TRAIN_DAYS
    sorted_dates = sorted(dates)
    embargo = set(sorted_dates[N_TRAIN_DAYS:N_TRAIN_DAYS + N_EMBARGO_DAYS])
    assert embargo.isdisjoint(train_dates)
    assert embargo.isdisjoint(test_dates)
