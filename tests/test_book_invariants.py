"""Invariant and conservation checks on BookBuilder, independent of the
reference-oracle diff in test_reference_oracle.py."""
from __future__ import annotations

import numpy as np
import pytest

from obsr.book import BookBuilder, CrossedBookError


def test_add_then_delete_conserves_total_resting_size():
    b = BookBuilder()
    b.add(1, "B", 100, 50)
    b.add(2, "S", 101, 30)
    assert b.total_resting_size == 80
    b.delete(1)
    assert b.total_resting_size == 30
    b.delete(2)
    assert b.total_resting_size == 0


def test_crossing_add_is_rejected():
    b = BookBuilder()
    b.add(1, "B", 100, 10)
    b.add(2, "S", 101, 10)
    with pytest.raises(CrossedBookError):
        b.add(3, "B", 101, 10)  # would cross the resting ask at 101
    with pytest.raises(CrossedBookError):
        b.add(4, "S", 100, 10)  # would cross the resting bid at 100


def test_price_time_priority_on_execute():
    b = BookBuilder()
    b.add(1, "S", 101, 10)  # first in queue at best ask
    b.add(2, "S", 101, 10)  # second in queue, same level
    filled = b.execute("B", 15)
    assert filled == 15
    # order 1 (10) fully consumed, order 2 partially (5 of 10) consumed,
    # in that order, because price-time priority takes the oldest order
    # at the best level first.
    assert 1 not in b.orders
    assert b.orders[2].size == 5


def test_snapshot_never_negative_and_conserves_level_totals():
    b = BookBuilder()
    rng = np.random.default_rng(7)
    next_id = 1
    for _ in range(500):
        side = "B" if rng.random() < 0.5 else "S"
        best_bid, best_ask = b.best_bid(), b.best_ask()
        if side == "B":
            ref = best_bid if best_bid is not None else 100
            tick = ref - int(rng.integers(1, 5))
            if best_ask is not None and tick >= best_ask:
                tick = best_ask - 1
        else:
            ref = best_ask if best_ask is not None else 110
            tick = ref + int(rng.integers(1, 5))
            if best_bid is not None and tick <= best_bid:
                tick = best_bid + 1
        b.add(next_id, side, tick, int(rng.integers(1, 100)))
        next_id += 1

    bp, bs, ap, asz = b.snapshot(5)
    assert all(s >= 0 for s in bs)
    assert all(s >= 0 for s in asz)
    assert sum(b.bid_level_size.values()) + sum(b.ask_level_size.values()) == b.total_resting_size


def test_execute_stops_when_book_side_exhausted():
    b = BookBuilder()
    b.add(1, "S", 101, 10)
    filled = b.execute("B", 1000)
    assert filled == 10  # can only take what is resting, not the requested 1000
    assert b.best_ask() is None
