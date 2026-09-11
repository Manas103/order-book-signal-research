"""Diffs the fast BookBuilder against the deliberately slow, independent
SlowBookBuilder oracle over a randomized sequence of operations."""
from __future__ import annotations

import numpy as np

from obsr.book import BookBuilder
from obsr.reference_book import SlowBookBuilder


def _run_both(n_ops: int, seed: int, depth: int = 5):
    fast = BookBuilder()
    slow = SlowBookBuilder()
    rng = np.random.default_rng(seed)
    next_id = 1
    live_ids: list[int] = []

    for _ in range(n_ops):
        best_bid_f, best_ask_f = fast.best_bid(), fast.best_ask()
        action = rng.random()
        if action < 0.5 or not live_ids:
            side = "B" if rng.random() < 0.5 else "S"
            if side == "B":
                ref = best_bid_f if best_bid_f is not None else 1000
                tick = ref - int(rng.integers(1, 6))
                if best_ask_f is not None and tick >= best_ask_f:
                    tick = best_ask_f - 1
            else:
                ref = best_ask_f if best_ask_f is not None else 1010
                tick = ref + int(rng.integers(1, 6))
                if best_bid_f is not None and tick <= best_bid_f:
                    tick = best_bid_f + 1
            size = int(rng.integers(1, 200))
            fast.add(next_id, side, tick, size)
            slow.add(next_id, side, tick, size)
            live_ids.append(next_id)
            next_id += 1
        elif action < 0.65 and best_bid_f is not None and best_ask_f is not None:
            side = "B" if rng.random() < 0.5 else "S"
            qty = int(rng.integers(1, 50))
            f_filled = fast.execute(side, qty)
            s_filled = slow.execute(side, qty)
            assert f_filled == s_filled, "execute() fill quantity diverged"
            live_ids = [i for i in live_ids if i in fast.orders]
        elif action < 0.85:
            oid = live_ids[rng.integers(0, len(live_ids))]
            reduce_by = int(rng.integers(1, 100))
            f_r = fast.cancel(oid, reduce_by)
            s_r = slow.cancel(oid, reduce_by)
            assert f_r == s_r, "cancel() reduction diverged"
            if oid not in fast.orders:
                live_ids.remove(oid)
        else:
            oid = live_ids[rng.integers(0, len(live_ids))]
            f_r = fast.delete(oid)
            s_r = slow.delete(oid)
            assert f_r == s_r, "delete() removed size diverged"
            live_ids.remove(oid)

        assert fast.snapshot(depth) == slow.snapshot(depth), "book snapshot diverged"

    return fast, slow


def test_fast_book_matches_slow_oracle_exactly():
    comparisons = 0
    for seed in range(10):
        fast, slow = _run_both(n_ops=2000, seed=seed)
        assert fast.snapshot(5) == slow.snapshot(5)
        comparisons += 2000
    assert comparisons == 20000
