"""Replays every stored (ticker, date) session's message stream through a
fresh BookBuilder (obsr.queue_position.replay_queue_positions) to recover
each ADD event's queue rank and quantity ahead, writing the result to its
own partitioned Parquet store.

Usage: python scripts/compute_queue_positions.py [--data-dir data]
"""
from __future__ import annotations

import argparse
import time

from obsr.queue_position import replay_queue_positions
from obsr.simulate import TICKERS, N_DAYS
from obsr.store import read_partition, write_partition, MESSAGES_DIR, QUEUE_POSITIONS_DIR
import pandas as pd


def _business_dates(n_days: int) -> list[str]:
    dates = pd.bdate_range("2025-01-06", periods=n_days)
    return [d.strftime("%Y-%m-%d") for d in dates]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--n-days", type=int, default=N_DAYS)
    args = ap.parse_args()

    dates = _business_dates(args.n_days)
    t0 = time.time()
    total_adds = 0
    n_sessions = 0
    for ticker in TICKERS:
        for date in dates:
            df = read_partition(args.data_dir, MESSAGES_DIR, ticker, date).sort_values("event_seq")
            qp = replay_queue_positions(df)
            write_partition(qp, args.data_dir, QUEUE_POSITIONS_DIR)
            total_adds += len(qp)
            n_sessions += 1
            print(f"[{n_sessions:3d}] {ticker} {date}: {len(qp):,} ADD events replayed "
                  f"(total {total_adds:,}, {time.time() - t0:.1f}s elapsed)")

    print(f"\nDone. {n_sessions} sessions, {total_adds:,} total queue-position rows, "
          f"{time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
