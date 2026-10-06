"""Hand-computed queue-rank/qty-ahead checks, plus a reference-oracle diff
against an independent brute-force recomputation over a randomized sequence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from obsr.book import BookBuilder
from obsr.queue_position import replay_queue_positions


def _rows(events):
    """events: list of (msg_type, side, order_id, price_ticks, qty)."""
    return pd.DataFrame.from_records(
        [("X", "2025-01-06", i, float(i), *e) for i, e in enumerate(events)],
        columns=["ticker", "date", "event_seq", "ts", "msg_type", "side", "order_id", "price_ticks", "qty"],
    )


def test_hand_computed_rank_and_qty_ahead():
    events = [
        ("ADD", "B", 1, 100, 10),
        ("ADD", "B", 2, 100, 20),
        ("ADD", "B", 3, 100, 5),
        ("EXECUTE", "S", 0, 0, 10),   # hits order 1 (best bid, oldest first)
        ("ADD", "B", 4, 100, 7),
        ("CANCEL", "B", 2, 0, 15),    # order 2: 20 -> 5
        ("DELETE", "B", 3, 0, 0),
        ("ADD", "B", 5, 100, 1),
        ("ADD", "B", 6, 101, 50),     # independent price level
        ("ADD", "S", 7, 200, 30),     # independent side
        ("ADD", "S", 8, 200, 10),
        ("ADD", "B", 9, 99, 5),       # worse than the current best bid (101) -> not at the touch
    ]
    df = _rows(events)
    qp = replay_queue_positions(df).set_index("order_id")

    assert (qp.loc[1, "rank"], qp.loc[1, "qty_ahead"]) == (1, 0)
    assert (qp.loc[2, "rank"], qp.loc[2, "qty_ahead"]) == (2, 10)
    assert (qp.loc[3, "rank"], qp.loc[3, "qty_ahead"]) == (3, 30)
    # order 1 executed away, queue is now [order2(20), order3(5)]
    assert (qp.loc[4, "rank"], qp.loc[4, "qty_ahead"]) == (3, 25)
    # order2 cancelled down to 5, order3 deleted; queue is [order2(5), order4(7)]
    assert (qp.loc[5, "rank"], qp.loc[5, "qty_ahead"]) == (3, 12)
    # independent price level and independent side both start fresh
    assert (qp.loc[6, "rank"], qp.loc[6, "qty_ahead"]) == (1, 0)
    assert (qp.loc[7, "rank"], qp.loc[7, "qty_ahead"]) == (1, 0)
    assert (qp.loc[8, "rank"], qp.loc[8, "qty_ahead"]) == (2, 30)

    # touch flag: orders 1-5 all join the best bid (100 is the only bid price
    # until order 6 arrives); order 6 improves the bid to 101, so it is also
    # at the touch; orders 7-8 are the best (only) ask, so also at the touch.
    for oid in (1, 2, 3, 4, 5, 6, 7, 8):
        assert bool(qp.loc[oid, "is_touch"]) is True
    assert bool(qp.loc[9, "is_touch"]) is False


def _brute_force_rank_and_qty_ahead(history, side, tick):
    """Independent recomputation: scans the full list of still-active
    (side, tick, size) orders in the order they were added, with no
    reference to BookBuilder's own queue/level_size dicts."""
    active = [o for o in history if o["alive"] and o["side"] == side and o["tick"] == tick]
    rank = len(active) + 1
    qty_ahead = sum(o["size"] for o in active)
    return rank, qty_ahead


def test_randomized_rank_against_independent_brute_force():
    """20,000 randomized add/execute/cancel/delete operations, each ADD's
    rank and qty_ahead cross-checked against a brute-force scan over an
    independently maintained order history list (not BookBuilder's own
    queue/level_size dicts)."""
    rng = np.random.default_rng(7)
    book = BookBuilder()
    history: dict[int, dict] = {}
    events = []
    next_id = 1

    for _ in range(20_000):
        best_bid, best_ask = book.best_bid(), book.best_ask()
        action = rng.random()
        if action < 0.55 or not book.active_ids:
            side = "B" if rng.random() < 0.5 else "S"
            if side == "B":
                ref = best_bid if best_bid is not None else 1000
                tick = ref - int(rng.integers(1, 6))
                if best_ask is not None and tick >= best_ask:
                    tick = best_ask - 1
            else:
                ref = best_ask if best_ask is not None else 1010
                tick = ref + int(rng.integers(1, 6))
                if best_bid is not None and tick <= best_bid:
                    tick = best_bid + 1
            size = int(rng.integers(1, 200))
            events.append(("ADD", side, next_id, tick, size))
            history[next_id] = {"side": side, "tick": tick, "size": size, "alive": True}
            next_id += 1
        elif action < 0.75 and best_bid is not None and best_ask is not None:
            side = "B" if rng.random() < 0.5 else "S"
            qty = int(rng.integers(1, 50))
            events.append(("EXECUTE", side, 0, 0, qty))
        else:
            oid = int(book.active_ids[rng.integers(0, len(book.active_ids))])
            if rng.random() < 0.5:
                reduce_by = int(rng.integers(1, 50))
                events.append(("CANCEL", history[oid]["side"], oid, 0, reduce_by))
            else:
                events.append(("DELETE", history[oid]["side"], oid, 0, 0))

        msg_type, side, oid, tick, qty = events[-1]
        if msg_type == "ADD":
            book.add(oid, side, tick, qty)
        elif msg_type == "EXECUTE":
            filled_orders_before = set(book.active_ids)
            book.execute(side, qty)
            for gone_id in filled_orders_before - set(book.active_ids):
                if gone_id in history:
                    history[gone_id]["alive"] = False
        elif msg_type == "CANCEL":
            removed = book.cancel(oid, qty)
            history[oid]["size"] -= removed
            if history[oid]["size"] <= 0:
                history[oid]["alive"] = False
        elif msg_type == "DELETE":
            book.delete(oid)
            history[oid]["alive"] = False

    df = _rows(events)
    qp = replay_queue_positions(df)

    # Recompute each ADD's rank/qty_ahead by brute force against the order
    # history *as it stood immediately before that ADD was applied*, by
    # re-running the same event list against a second, independent ledger.
    ledger: dict[int, dict] = {}
    checked = 0
    for msg_type, side, oid, tick, qty in events:
        if msg_type == "ADD":
            before = [o for o in ledger.values() if o["alive"] and o["side"] == side and o["tick"] == tick]
            expected_rank = len(before) + 1
            expected_qty_ahead = sum(o["size"] for o in before)
            row = qp[qp["order_id"] == oid].iloc[0]
            assert row["rank"] == expected_rank
            assert row["qty_ahead"] == expected_qty_ahead
            ledger[oid] = {"side": side, "tick": tick, "size": qty, "alive": True}
            checked += 1
        elif msg_type == "EXECUTE":
            opp_side = "S" if side == "B" else "B"
            # Price-time priority across every price level on the opposite
            # side: best price first (ascending tick for asks, descending
            # for bids), FIFO within a tick preserved by Python's stable
            # sort over the insertion-ordered ledger.
            opp_orders = [o for o in ledger.items() if o[1]["alive"] and o[1]["side"] == opp_side]
            opp_orders.sort(key=lambda kv: kv[1]["tick"], reverse=(opp_side == "B"))
            remaining = qty
            for oid2, order in opp_orders:
                if remaining <= 0:
                    break
                take = min(order["size"], remaining)
                order["size"] -= take
                remaining -= take
                if order["size"] <= 0:
                    order["alive"] = False
        elif msg_type == "CANCEL":
            removed = min(ledger[oid]["size"], qty)
            ledger[oid]["size"] -= removed
            if ledger[oid]["size"] <= 0:
                ledger[oid]["alive"] = False
        elif msg_type == "DELETE":
            ledger[oid]["alive"] = False

    assert checked > 5_000, "sanity: the randomized sequence produced too few ADDs to be a meaningful check"
