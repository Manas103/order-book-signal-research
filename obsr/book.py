"""Fast, price-time-priority full-depth limit order book.

This is the "production" book builder: sorted price levels via ``bisect``,
O(1) top-of-book and O(1) amortized level-size lookups, FIFO order queues
per price level enforcing price-time priority. It is used both while the
synthetic message stream is generated (to pick legal cancel/delete/execute
targets) and to produce the snapshot rows written to the Parquet store.

``reference_book.SlowBookBuilder`` is an independently written, deliberately
naive re-implementation used only to prove this one correct on a sample
(see tests/test_reference_oracle.py).
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field


@dataclass
class Order:
    __slots__ = ("side", "tick", "size")
    side: str
    tick: int
    size: int


class CrossedBookError(RuntimeError):
    pass


class BookBuilder:
    """Full-depth limit order book for one (ticker, date) session.

    Sides are stored as sorted lists of active integer price ticks
    (``bid_ticks`` ascending, best bid at the end; ``ask_ticks`` ascending,
    best ask at the front), a FIFO list of resting order ids per tick, and
    an aggregate size per tick for O(1) level-size reads. ``orders`` maps
    order id -> Order for O(1) cancel/delete/execute lookups.
    """

    def __init__(self) -> None:
        self.bid_ticks: list[int] = []
        self.ask_ticks: list[int] = []
        self.bid_queue: dict[int, list[int]] = {}
        self.ask_queue: dict[int, list[int]] = {}
        self.bid_level_size: dict[int, int] = {}
        self.ask_level_size: dict[int, int] = {}
        self.orders: dict[int, Order] = {}
        # array-backed set of active order ids, O(1) random pick + removal
        self.active_ids: list[int] = []
        self._id_to_index: dict[int, int] = {}
        self.total_resting_size = 0

    # -- introspection -----------------------------------------------
    def best_bid(self) -> int | None:
        return self.bid_ticks[-1] if self.bid_ticks else None

    def best_ask(self) -> int | None:
        return self.ask_ticks[0] if self.ask_ticks else None

    def mid_ticks(self) -> float | None:
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2.0

    def snapshot(self, depth: int = 5):
        """Return (bid_prices, bid_sizes, ask_prices, ask_sizes), each a
        list of length ``depth``, best level first, zero-padded when the
        book is shallower than ``depth``."""
        bid_prices = list(reversed(self.bid_ticks[-depth:]))
        bid_sizes = [self.bid_level_size[t] for t in bid_prices]
        while len(bid_prices) < depth:
            bid_prices.append(0)
            bid_sizes.append(0)
        ask_prices = list(self.ask_ticks[:depth])
        ask_sizes = [self.ask_level_size[t] for t in ask_prices]
        while len(ask_prices) < depth:
            ask_prices.append(0)
            ask_sizes.append(0)
        return bid_prices, bid_sizes, ask_prices, ask_sizes

    # -- id bookkeeping ------------------------------------------------
    def _register_id(self, order_id: int) -> None:
        self._id_to_index[order_id] = len(self.active_ids)
        self.active_ids.append(order_id)

    def _unregister_id(self, order_id: int) -> None:
        idx = self._id_to_index.pop(order_id)
        last = self.active_ids.pop()
        if idx < len(self.active_ids):
            self.active_ids[idx] = last
            self._id_to_index[last] = idx

    def random_active_id(self, rng) -> int | None:
        if not self.active_ids:
            return None
        return self.active_ids[rng.integers(0, len(self.active_ids))]

    # -- mutations -------------------------------------------------------
    def add(self, order_id: int, side: str, tick: int, size: int) -> None:
        if size <= 0:
            raise ValueError("add size must be positive")
        if side == "B":
            if self.ask_ticks and tick >= self.ask_ticks[0]:
                raise CrossedBookError(f"bid add at {tick} would cross ask {self.ask_ticks[0]}")
            ticks, queue, levels = self.bid_ticks, self.bid_queue, self.bid_level_size
        else:
            if self.bid_ticks and tick <= self.bid_ticks[-1]:
                raise CrossedBookError(f"ask add at {tick} would cross bid {self.bid_ticks[-1]}")
            ticks, queue, levels = self.ask_ticks, self.ask_queue, self.ask_level_size
        if tick not in queue:
            queue[tick] = []
            levels[tick] = 0
            bisect.insort(ticks, tick)
        queue[tick].append(order_id)
        levels[tick] += size
        self.orders[order_id] = Order(side, tick, size)
        self._register_id(order_id)
        self.total_resting_size += size

    def _drop_level_if_empty(self, side: str, tick: int) -> None:
        ticks = self.bid_ticks if side == "B" else self.ask_ticks
        queue = self.bid_queue if side == "B" else self.ask_queue
        levels = self.bid_level_size if side == "B" else self.ask_level_size
        if not queue[tick]:
            del queue[tick]
            del levels[tick]
            i = bisect.bisect_left(ticks, tick)
            del ticks[i]

    def delete(self, order_id: int) -> int:
        """Fully remove a resting order. Returns the size that was removed."""
        order = self.orders.pop(order_id)
        queue = self.bid_queue if order.side == "B" else self.ask_queue
        levels = self.bid_level_size if order.side == "B" else self.ask_level_size
        queue[order.tick].remove(order_id)
        levels[order.tick] -= order.size
        self._drop_level_if_empty(order.side, order.tick)
        self._unregister_id(order_id)
        self.total_resting_size -= order.size
        return order.size

    def cancel(self, order_id: int, reduce_by: int) -> int:
        """Reduce a resting order's size. If it would go to zero or below,
        the order is fully deleted instead. Returns the size actually
        removed from the book."""
        order = self.orders[order_id]
        reduce_by = min(reduce_by, order.size)
        if reduce_by >= order.size:
            return self.delete(order_id)
        order.size -= reduce_by
        levels = self.bid_level_size if order.side == "B" else self.ask_level_size
        levels[order.tick] -= reduce_by
        self.total_resting_size -= reduce_by
        return reduce_by

    def execute(self, aggressor_side: str, qty: int) -> int:
        """Aggressor of ``aggressor_side`` ('B' or 'S') consumes resting
        liquidity on the opposite side, best price and then oldest order
        first (price-time priority). Returns the quantity actually filled
        (less than ``qty`` only if the book runs out of opposite liquidity)."""
        opp_ticks = self.ask_ticks if aggressor_side == "B" else self.bid_ticks
        opp_queue = self.ask_queue if aggressor_side == "B" else self.bid_queue
        opp_levels = self.ask_level_size if aggressor_side == "B" else self.bid_level_size
        opp_side = "S" if aggressor_side == "B" else "B"
        filled = 0
        while qty > 0 and opp_ticks:
            tick = opp_ticks[0] if aggressor_side == "B" else opp_ticks[-1]
            queue = opp_queue[tick]
            order_id = queue[0]
            order = self.orders[order_id]
            take = min(order.size, qty)
            order.size -= take
            opp_levels[tick] -= take
            self.total_resting_size -= take
            qty -= take
            filled += take
            if order.size <= 0:
                queue.pop(0)
                del self.orders[order_id]
                self._unregister_id(order_id)
                self._drop_level_if_empty(opp_side, tick)
        return filled
