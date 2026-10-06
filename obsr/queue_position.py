"""Per-order queue position, recovered by replaying a session's own stored
message stream through a fresh ``BookBuilder`` in event order.

The partitioned message store (``obsr/store.py``) keeps a full-depth
*aggregate* snapshot per event (``bid_px_k``/``bid_sz_k``), which is exactly
what an L2 feed would give a reader: enough to see that 4,000 shares rest at
101.25, never enough to say whose 4,000 shares or in what order they would
fill. Recovering "my order is 4th in the queue, 2,300 shares ahead of it"
needs the L3 replay this module does: for every ADD event, before applying
it, record how many orders already rest at that exact price (``rank``, 1 for
the first order ever to establish a brand-new price level) and how much
quantity is already resting there (``qty_ahead``), then apply the event
exactly as ``BookBuilder`` would and move on. ``BookBuilder.add`` always
appends to the back of its FIFO queue, so these two numbers are read directly
off the queue and level-size dict *before* the call, never recomputed.
"""

from __future__ import annotations

import pandas as pd

from .book import BookBuilder

QUEUE_POSITION_COLUMNS = [
    "ticker", "date", "order_id", "add_ts", "side", "price_ticks", "add_qty", "rank", "qty_ahead",
    "is_touch",
]


def replay_queue_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Replays one (ticker, date) session's message rows (already sorted by
    ``event_seq``) through a fresh ``BookBuilder``, returning one row per ADD
    event with its queue rank and quantity ahead at the instant it joined.
    """
    book = BookBuilder()
    rows: list[tuple] = []
    ticker = df["ticker"].iloc[0]
    date = df["date"].iloc[0]

    for row in df.itertuples(index=False):
        msg_type = row.msg_type
        if msg_type == "ADD":
            side, tick = row.side, int(row.price_ticks)
            queue = book.bid_queue if side == "B" else book.ask_queue
            levels = book.bid_level_size if side == "B" else book.ask_level_size
            rank = len(queue.get(tick, [])) + 1
            qty_ahead = levels.get(tick, 0)
            # "At the touch": this price is the best (or ties/improves on the
            # best) on its side at the instant of insertion, so it is at or
            # ahead of the current best bid/ask, not resting deeper in the book.
            best = book.best_bid() if side == "B" else book.best_ask()
            if best is None:
                is_touch = True
            elif side == "B":
                is_touch = tick >= best
            else:
                is_touch = tick <= best
            rows.append((ticker, date, int(row.order_id), float(row.ts), side, tick,
                         int(row.qty), rank, qty_ahead, is_touch))
            book.add(int(row.order_id), side, tick, int(row.qty))
        elif msg_type == "EXECUTE":
            book.execute(row.side, int(row.qty))
        elif msg_type == "CANCEL":
            book.cancel(int(row.order_id), int(row.qty))
        elif msg_type == "DELETE":
            book.delete(int(row.order_id))
        else:
            raise ValueError(f"unknown msg_type {msg_type!r}")

    return pd.DataFrame.from_records(rows, columns=QUEUE_POSITION_COLUMNS)
