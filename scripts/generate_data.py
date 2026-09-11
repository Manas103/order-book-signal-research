"""Generate the full synthetic message stream and write it to the
partitioned Parquet store (one file per ticker/date, Hive-style layout).

Usage: python scripts/generate_data.py [--data-dir data] [--n-days 16] [--msgs-per-day 80000]
"""
from __future__ import annotations

import argparse
import time

from obsr.simulate import TICKERS, N_DAYS, MSGS_PER_DAY, simulate_all
from obsr.store import write_partition, MESSAGES_DIR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--n-days", type=int, default=N_DAYS)
    ap.add_argument("--msgs-per-day", type=int, default=MSGS_PER_DAY)
    args = ap.parse_args()

    t0 = time.time()
    total_rows = 0
    n_sessions = 0
    for ticker, date, df in simulate_all(TICKERS, n_days=args.n_days, msgs_per_day=args.msgs_per_day):
        write_partition(df, args.data_dir, MESSAGES_DIR)
        total_rows += len(df)
        n_sessions += 1
        print(f"[{n_sessions:3d}] {ticker} {date}: {len(df):,} rows written "
              f"(total {total_rows:,}, {time.time() - t0:.1f}s elapsed)")

    print(f"\nDone. {n_sessions} sessions, {total_rows:,} total messages, "
          f"{time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
