"""Unit checks on individual feature formulas against hand-computed values."""
from __future__ import annotations

import duckdb
import pandas as pd

from obsr.features import FEATURE_COLUMNS, build_feature_sql


def _one_row_frame(**overrides):
    depth = 5
    row = {
        "ticker": "SYNA", "date": "2025-01-06", "day_index": 0, "event_seq": 0,
        "ts": 0.0, "msg_type": "ADD", "side": "B", "order_id": 1,
        "price_ticks": 100, "qty": 10, "size_delta": 10, "total_resting_after": 10,
    }
    for k in range(1, depth + 1):
        row[f"bid_px_{k}"] = 100 - k
        row[f"bid_sz_{k}"] = 100 * k
        row[f"ask_px_{k}"] = 100 + k
        row[f"ask_sz_{k}"] = 50 * k
    row.update(overrides)
    return pd.DataFrame([row])


def test_spread_microprice_and_qi_hand_computed():
    df = _one_row_frame()
    con = duckdb.connect(":memory:")
    con.register("raw_msgs", df)
    out = con.execute(build_feature_sql("raw_msgs")).fetchdf()
    r = out.iloc[0]

    assert r["spread"] == (100 + 1) - (100 - 1)  # ask_px_1 - bid_px_1 == 2
    bid1, ask1 = 100 * 1, 50 * 1
    expected_micro = ((100 - 1) * ask1 + (100 + 1) * bid1) / (bid1 + ask1)
    assert abs(r["microprice"] - expected_micro) < 1e-9
    assert abs(r["qi_l1"] - (bid1 - ask1) / (bid1 + ask1)) < 1e-9
    assert r["bid_depth_total"] == sum(100 * k for k in range(1, 6))
    assert r["ask_depth_total"] == sum(50 * k for k in range(1, 6))


def test_all_declared_feature_columns_are_present_in_output():
    df = _one_row_frame()
    con = duckdb.connect(":memory:")
    con.register("raw_msgs", df)
    out = con.execute(build_feature_sql("raw_msgs")).fetchdf()
    missing = [c for c in FEATURE_COLUMNS if c not in out.columns]
    assert not missing, f"declared feature columns missing from SQL output: {missing}"
    assert len(FEATURE_COLUMNS) >= 40
