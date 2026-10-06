"""Fill probability by queue rank (one DuckDB SQL window pass) plus a
rank-aware quote-and-wait versus always-crossing execution cost comparison.

Usage: python scripts/run_queue_study.py [--data-dir data]
"""
from __future__ import annotations

import argparse
import json
import time

from obsr.execution_cost import (
    attach_fill_probability, attach_future_ask, build_fill_probability_curve,
    compute_execution_costs, select_decision_points,
)
from obsr.fill_probability import HORIZON_SECONDS, compute_fill_probability_by_rank
from obsr.store import connect, messages_relation, queue_positions_relation


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()

    t0 = time.time()
    con = connect()
    msg_rel = messages_relation(con, args.data_dir)
    qp_rel = queue_positions_relation(con, args.data_dir)

    print("Computing fill probability by queue rank, whole book (one DuckDB SQL window pass)...")
    rank_summary_all, per_order = compute_fill_probability_by_rank(con, qp_rel, msg_rel, touch_only=False)
    print(rank_summary_all.to_string(index=False))

    print("\nSame computation restricted to ADD events that joined the best bid/ask "
          "at the instant of insertion ('at the touch'):")
    rank_summary_touch, _ = compute_fill_probability_by_rank(con, qp_rel, msg_rel, touch_only=True)
    print(rank_summary_touch.to_string(index=False))
    rank1 = rank_summary_touch.loc[rank_summary_touch["rank_bucket"] == 1, "fill_probability"].iloc[0]
    rank5 = rank_summary_touch.loc[rank_summary_touch["rank_bucket"] == 5, "fill_probability"].iloc[0]
    print(f"\nat the touch, fill probability at rank 1: {rank1:.4f}   at exact rank 5: {rank5:.4f}   "
          f"({time.time() - t0:.1f}s elapsed)")

    print("\nBuilding fill-probability-by-queue-depth curve for the cost comparison "
          "(at-the-touch ADD events only, matching the decision points below)...")
    curve = build_fill_probability_curve(per_order[per_order["is_touch"]])

    print("Loading message snapshots for decision points...")
    cols = "ticker, date, event_seq, ts, bid_px_1, ask_px_1, bid_sz_1"
    messages = con.execute(f"SELECT {cols} FROM {msg_rel}").fetchdf()

    decision_points = select_decision_points(messages)
    decision_points = attach_fill_probability(decision_points, curve)
    decision_points = attach_future_ask(decision_points, messages, HORIZON_SECONDS)
    costs = compute_execution_costs(decision_points)
    print(f"\nn_decision_points={costs['n_decision_points']:,}")
    print(f"always-crossing mean effective spread: {costs['mean_always_crossing_effective_spread_ticks']:.4f} ticks")
    print(f"rank-aware quote-and-wait mean effective spread: {costs['mean_rank_aware_effective_spread_ticks']:.4f} ticks")
    print(f"rank-aware paid {costs['pct_less_effective_spread']:.2f}% less effective spread than always crossing")

    results = {
        "fill_probability_by_rank_whole_book": rank_summary_all.to_dict(orient="records"),
        "fill_probability_by_rank_at_touch": rank_summary_touch.to_dict(orient="records"),
        "fill_probability_rank_1_at_touch": float(rank1),
        "fill_probability_exact_rank_5_at_touch": float(rank5),
        "horizon_seconds": HORIZON_SECONDS,
        "execution_cost_comparison": costs,
    }
    with open("docs/queue_study_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone in {time.time() - t0:.1f}s. Results written to docs/queue_study_results.json")


if __name__ == "__main__":
    main()
