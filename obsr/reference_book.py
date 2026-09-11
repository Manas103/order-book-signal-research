"""A deliberately slow, obviously-correct order book, independent of
``book.BookBuilder``'s data structures. No sorted-list bookkeeping, no
incremental level-size cache: every query re-derives the answer from a
flat dict of currently-resting orders by brute-force scan and sort. This
is the reference oracle diffed exactly against the fast builder in
tests/test_reference_oracle.py.
"""

from __future__ import annotations


class SlowBookBuilder:
    def __init__(self) -> None:
        # order_id -> {side, tick, size, seq}; seq is insertion order, used
        # only to break ties within a price level (price-time priority).
        self.orders: dict[int, dict] = {}
        self._seq = 0

    def add(self, order_id: int, side: str, tick: int, size: int) -> None:
        self.orders[order_id] = {"side": side, "tick": tick, "size": size, "seq": self._seq}
        self._seq += 1

    def delete(self, order_id: int) -> int:
        o = self.orders.pop(order_id)
        return o["size"]

    def cancel(self, order_id: int, reduce_by: int) -> int:
        o = self.orders[order_id]
        reduce_by = min(reduce_by, o["size"])
        if reduce_by >= o["size"]:
            return self.delete(order_id)
        o["size"] -= reduce_by
        return reduce_by

    def execute(self, aggressor_side: str, qty: int) -> int:
        opp_side = "S" if aggressor_side == "B" else "B"
        filled = 0
        while qty > 0:
            # Recompute price-time priority from scratch every step: the
            # slow, obviously-correct way. Best price for the resting side
            # being hit: lowest tick if it is the ask side, highest if bid.
            candidates = [
                (oid, o) for oid, o in self.orders.items() if o["side"] == opp_side
            ]
            if not candidates:
                break
            if opp_side == "S":
                candidates.sort(key=lambda item: (item[1]["tick"], item[1]["seq"]))
            else:
                candidates.sort(key=lambda item: (-item[1]["tick"], item[1]["seq"]))
            oid, o = candidates[0]
            take = min(o["size"], qty)
            o["size"] -= take
            qty -= take
            filled += take
            if o["size"] <= 0:
                del self.orders[oid]
        return filled

    def snapshot(self, depth: int = 5):
        bid_levels: dict[int, int] = {}
        ask_levels: dict[int, int] = {}
        for o in self.orders.values():
            target = bid_levels if o["side"] == "B" else ask_levels
            target[o["tick"]] = target.get(o["tick"], 0) + o["size"]
        bid_ticks_sorted = sorted(bid_levels.keys(), reverse=True)[:depth]
        ask_ticks_sorted = sorted(ask_levels.keys())[:depth]
        bid_prices = list(bid_ticks_sorted)
        bid_sizes = [bid_levels[t] for t in bid_prices]
        while len(bid_prices) < depth:
            bid_prices.append(0)
            bid_sizes.append(0)
        ask_prices = list(ask_ticks_sorted)
        ask_sizes = [ask_levels[t] for t in ask_prices]
        while len(ask_prices) < depth:
            ask_prices.append(0)
            ask_sizes.append(0)
        return bid_prices, bid_sizes, ask_prices, ask_sizes
