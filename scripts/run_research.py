"""Build features from the Parquet store, apply a day-level purged
train/test split per ticker, then train a LightGBM regressor and a linear
order-flow-imbalance baseline at each horizon and report out-of-sample R^2.

Day-level purge: for each ticker, the first N_TRAIN_DAYS business days are
train, the single day immediately after is dropped entirely (the embargo,
wider than the longest label horizon of 1,000 events against 80,000
events/day, so no label from the last training day can reach across the
embargo into the test period), and the remaining days are test.

Usage: python scripts/run_research.py [--data-dir data]
"""
from __future__ import annotations

import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from obsr.features import FEATURE_COLUMNS, build_features
from obsr.labels import HORIZONS, label_col
from obsr.store import connect, messages_relation

N_TRAIN_DAYS = 11
N_EMBARGO_DAYS = 1


def purged_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(df["date"].unique())
    train_dates = set(dates[:N_TRAIN_DAYS])
    embargo_dates = set(dates[N_TRAIN_DAYS:N_TRAIN_DAYS + N_EMBARGO_DAYS])
    test_dates = set(dates[N_TRAIN_DAYS + N_EMBARGO_DAYS:])
    assert not (train_dates & test_dates), "purge failed: train/test dates overlap"
    assert not (train_dates & embargo_dates), "purge failed: train/embargo dates overlap"
    train = df[df["date"].isin(train_dates)]
    test = df[df["date"].isin(test_dates)]
    return train, test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()

    t0 = time.time()
    con = connect()
    rel = messages_relation(con, args.data_dir)
    print("Building features with DuckDB SQL window functions...")
    feat = build_features(con, rel)
    print(f"Features built: {len(feat):,} rows, {len(FEATURE_COLUMNS)} feature columns, "
          f"{time.time() - t0:.1f}s.")

    train_parts, test_parts = [], []
    for ticker, g in feat.groupby("ticker", sort=False):
        tr, te = purged_split(g)
        train_parts.append(tr)
        test_parts.append(te)
    train_all = pd.concat(train_parts, ignore_index=True)
    test_all = pd.concat(test_parts, ignore_index=True)
    print(f"Split: {len(train_all):,} train rows, {len(test_all):,} test rows "
          f"({N_TRAIN_DAYS} train days, {N_EMBARGO_DAYS} embargo day, remaining test days, per ticker).")

    results = []
    for h in HORIZONS:
        col = label_col(h)
        tr = train_all.dropna(subset=FEATURE_COLUMNS + [col]).copy()
        te = test_all.dropna(subset=FEATURE_COLUMNS + [col]).copy()

        # Attempt 2 (winsorizing the label at the 1st/99th train percentile)
        # produced a degenerate R^2 of exactly 1.0: this label is spiked at
        # zero (most 10-event windows see no tick-level mid change at all),
        # so clipping collapsed the test target's variance to nearly zero
        # and made R^2 numerically meaningless rather than informative. Not
        # used in the reported (third) attempt; left documented here because
        # a wrong measurement that gets corrected is part of the record.

        X_tr, y_tr = tr[list(FEATURE_COLUMNS)], tr[col]
        X_te, y_te = te[list(FEATURE_COLUMNS)], te[col]

        gbt = lgb.LGBMRegressor(
            n_estimators=200, num_leaves=15, max_depth=5, learning_rate=0.03,
            min_child_samples=1000, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=1.0,
            random_state=42, verbosity=-1,
        )
        gbt.fit(X_tr, y_tr)
        gbt_r2 = r2_score(y_te, gbt.predict(X_te))

        lin = LinearRegression()
        lin.fit(tr[["ofi_agg5"]], y_tr)
        lin_r2 = r2_score(y_te, lin.predict(te[["ofi_agg5"]]))

        results.append({
            "horizon_events": h, "n_train": len(tr), "n_test": len(te),
            "gbt_oos_r2": gbt_r2, "linear_ofi_oos_r2": lin_r2,
        })
        print(f"horizon={h:5d} events  n_train={len(tr):>9,}  n_test={len(te):>9,}  "
              f"GBT R^2={gbt_r2:+.4f}  linear-OFI R^2={lin_r2:+.4f}  "
              f"tree_advantage={gbt_r2 - lin_r2:+.4f}")

    with open("docs/research_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone in {time.time() - t0:.1f}s. Results written to docs/research_results.json")


if __name__ == "__main__":
    main()
