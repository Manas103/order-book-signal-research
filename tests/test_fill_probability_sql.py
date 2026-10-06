"""Hand-computed check of the fill-probability SQL window pass
(obsr.fill_probability.build_fill_probability_sql): a handful of ADD and
EXECUTE rows with known, by-hand-verified fill outcomes."""
from __future__ import annotations

import duckdb
import pandas as pd

from obsr.fill_probability import compute_fill_probability_by_rank


def test_fill_probability_hand_computed():
    queue_positions = pd.DataFrame.from_records([
        ("X", "2025-01-06", 1, 0.0, "B", 100, 10, 1, 0, True),    # filled at t=0.5 -> within 1.0s horizon
        ("X", "2025-01-06", 2, 0.0, "B", 100, 10, 2, 10, True),   # never executed -> not filled
        ("X", "2025-01-06", 3, 0.0, "B", 100, 10, 1, 0, True),    # executed at t=1.5 -> outside horizon
        ("X", "2025-01-06", 4, 2.0, "B", 100, 10, 1, 0, True),    # executed at t=2.3 -> within horizon
        ("Y", "2025-01-06", 1, 0.0, "B", 100, 10, 1, 0, True),    # different session, same order_id=1,
                                                                   # must not be confused with X's order 1
    ], columns=["ticker", "date", "order_id", "add_ts", "side", "price_ticks", "add_qty", "rank", "qty_ahead",
                "is_touch"])

    messages = pd.DataFrame.from_records([
        ("X", "2025-01-06", 1, 0.5, "EXECUTE", "S", 1, 100, 10),
        ("X", "2025-01-06", 2, 1.5, "EXECUTE", "S", 3, 100, 10),
        ("X", "2025-01-06", 3, 2.3, "EXECUTE", "S", 4, 100, 10),
        # order 2 (ticker X) never appears in an EXECUTE row at all
        # ticker Y's order 1 never executed either, despite the same order_id as ticker X's order 1
    ], columns=["ticker", "date", "event_seq", "ts", "msg_type", "side", "order_id", "price_ticks", "qty"])

    con = duckdb.connect(":memory:")
    con.register("queue_positions_df", queue_positions)
    con.register("messages_df", messages)

    summary, per_order = compute_fill_probability_by_rank(con, "queue_positions_df", "messages_df")
    per_order = per_order.set_index(["ticker", "order_id"])

    assert bool(per_order.loc[("X", 1), "filled_within_horizon"]) is True
    assert bool(per_order.loc[("X", 2), "filled_within_horizon"]) is False
    assert bool(per_order.loc[("X", 3), "filled_within_horizon"]) is False  # executed, but 1.0s late
    assert bool(per_order.loc[("X", 4), "filled_within_horizon"]) is True
    assert bool(per_order.loc[("Y", 1), "filled_within_horizon"]) is False

    # rank-1 bucket: X#1 (filled), X#3 (not), X#4 (filled), Y#1 (not) -> 2/4;
    # rank-2 bucket: X#2 (not) -> 0/1
    rank1 = summary.loc[summary["rank_bucket"] == 1]
    rank2 = summary.loc[summary["rank_bucket"] == 2]
    assert int(rank1["n_orders"].iloc[0]) == 4
    assert int(rank1["n_filled"].iloc[0]) == 2
    assert int(rank2["n_orders"].iloc[0]) == 1
    assert int(rank2["n_filled"].iloc[0]) == 0
