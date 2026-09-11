"""Forward mid-price-change labels at fixed event-time horizons.

The production path computes these with a DuckDB ``LEAD`` window function,
partitioned by (ticker, date) so a label never reaches across a day
boundary (see ``obsr.features``). ``compute_forward_labels`` below is an
independent, plain-pandas re-implementation used only as a known-answer
cross-check in tests: the same partition-respecting semantics, deliberately
written a completely different way (``groupby().shift()`` instead of SQL
``LEAD``), so a test can diff the two exactly on a small sample.
"""

from __future__ import annotations

import pandas as pd

HORIZONS = (10, 100, 1000)


def label_col(h: int) -> str:
    return f"fwd_mid_chg_{h}"


def compute_forward_labels(df: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """Add one ``fwd_mid_chg_<h>`` column per horizon: the mid price ``h``
    events ahead minus the current mid, computed independently within each
    (ticker, date) group so it is undefined (NaN) for the last ``h`` rows
    of every session rather than reaching into the next session.
    """
    out = df.copy()
    grouped_mid = out.groupby(["ticker", "date"], sort=False)["mid"]
    for h in horizons:
        out[label_col(h)] = grouped_mid.shift(-h) - out["mid"]
    return out
