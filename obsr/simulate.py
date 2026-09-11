"""Synthetic ITCH-style order flow generator.

This is not real Nasdaq TotalView-ITCH or LOBSTER data (see the README's
"what this is NOT" section). It is a from-scratch add/cancel/execute/delete
message stream, calibrated to three published microstructure facts rather
than to any specific real dataset:

1. Order arrivals cluster (a simple AR(1) latent "flow momentum" state
   substitutes for a genuine Hawkes self-exciting intensity: both produce
   bursty, autocorrelated order flow, and the AR(1) form is what is
   actually fit in ``obsr.model``'s linear-OFI baseline's true generative
   analogue, so it keeps the label's origin honest).
2. Order sizes are heavy-tailed (Pareto), not Gaussian.
3. Order flow imbalance has real, short-horizon-biased, decaying predictive
   power over the mid price (Cont, Kukanov & Stoikov 2014): the same
   latent momentum state that biases the aggressor side of EXECUTE
   messages also, more weakly, biases the side of resting ADD messages,
   so order flow imbalance measured from the book is correlated with the
   momentum driving near-term price drift, and that correlation decays
   geometrically (autocorrelation rho per event) as the horizon grows.

The latent momentum value itself is never written to the feature store or
used by any model; only the resulting messages and book snapshots are (see
``obsr.features``). Everything after generation treats this purely as
market data.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .book import BookBuilder

TICKERS = ["SYNA", "SYNB", "SYNC", "SYND", "SYNE"]
START_PRICE_TICKS = {"SYNA": 5_000, "SYNB": 10_000, "SYNC": 20_000, "SYND": 7_500, "SYNE": 15_000}
N_DAYS = 16
MSGS_PER_DAY = 80_000
DEPTH = 5

# Latent momentum process (never exposed as a feature).
RHO = 0.97
SIGMA = 1.0
K_EXEC = 1.6
K_ADD = 0.5

P_ADD, P_EXEC, P_CANCEL, P_DELETE = 0.42, 0.28, 0.20, 0.10
assert abs(P_ADD + P_EXEC + P_CANCEL + P_DELETE - 1.0) < 1e-9

N_SEED_LEVELS = 15


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _business_dates(n_days: int) -> list[str]:
    dates = pd.bdate_range("2025-01-06", periods=n_days)
    return [d.strftime("%Y-%m-%d") for d in dates]


def simulate_ticker_day(ticker: str, date: str, day_index: int, seed: int,
                         n_msgs: int = MSGS_PER_DAY) -> pd.DataFrame:
    """Generate one (ticker, date) session: seed an initial book, then
    ``n_msgs`` stochastic ITCH-style events, returning one row per event
    with the message fields and the resulting full-depth snapshot.
    """
    rng = np.random.default_rng(seed)
    book = BookBuilder()
    start_ticks = START_PRICE_TICKS[ticker]
    n = n_msgs

    # Pre-draw all randomness vectorized (fast); the sequential loop below
    # only does bookkeeping and array indexing, no per-event RNG dispatch.
    noise = rng.standard_normal(n)
    gaps = rng.exponential(0.05, n)
    u_type = rng.random(n)
    u_side = rng.random(n)
    u_offset = rng.geometric(0.30, n) - 1
    u_improve = rng.random(n)
    improve_amt = rng.integers(1, 4, n)
    size_pareto = rng.pareto(1.6, n)
    exec_pareto = rng.pareto(1.6, n)
    cancel_frac = rng.random(n)
    pick_frac = rng.random(n)

    def draw_size(i: int) -> int:
        return int(min(10.0 + size_pareto[i] * 20.0, 20_000))

    def draw_exec_qty(i: int) -> int:
        return int(min(5.0 + exec_pareto[i] * 30.0, 20_000))

    rows: list[tuple] = []
    next_order_id = 1
    ts = 0.0
    m = 0.0

    # -- seed initial depth so >=N_SEED_LEVELS/2... levels exist on both
    # sides before any stochastic event is applied.
    for lvl in range(N_SEED_LEVELS):
        for side, tick in (("B", start_ticks - 1 - lvl), ("S", start_ticks + 1 + lvl)):
            size = int(50 + 30 * lvl)
            oid = next_order_id
            next_order_id += 1
            book.add(oid, side, tick, size)
            ts += 0.01
            bp, bs, ap, asz = book.snapshot(DEPTH)
            rows.append(
                (ticker, date, day_index, len(rows), ts, "ADD", side, oid, tick, size, size,
                 book.total_resting_size, *bp, *bs, *ap, *asz)
            )

    n_seeded = len(rows)
    for i in range(n_seeded, n):
        m = RHO * m + SIGMA * noise[i]
        ts += gaps[i]
        u = u_type[i]
        best_bid, best_ask = book.best_bid(), book.best_ask()

        if u < P_ADD or best_bid is None or best_ask is None:
            p_buy = _sigmoid(K_ADD * m)
            side = "B" if u_side[i] < p_buy else "S"
            own_best = book.best_bid() if side == "B" else book.best_ask()
            opp_best = book.best_ask() if side == "B" else book.best_bid()
            offset = int(u_offset[i])
            if own_best is None:
                ref = opp_best if opp_best is not None else start_ticks
                tick = ref - offset - 1 if side == "B" else ref + offset + 1
            elif u_improve[i] < 0.15:
                tick = own_best + int(improve_amt[i]) if side == "B" else own_best - int(improve_amt[i])
            else:
                tick = own_best - offset if side == "B" else own_best + offset
            if opp_best is not None:
                if side == "B" and tick >= opp_best:
                    tick = opp_best - 1
                if side == "S" and tick <= opp_best:
                    tick = opp_best + 1
            size = draw_size(i)
            oid = next_order_id
            next_order_id += 1
            book.add(oid, side, tick, size)
            delta = size
            msg_type, m_side, m_oid, m_price, m_qty = "ADD", side, oid, tick, size

        elif u < P_ADD + P_EXEC:
            p_buy_aggr = _sigmoid(K_EXEC * m)
            side = "B" if u_side[i] < p_buy_aggr else "S"
            opp_tick = book.best_ask() if side == "B" else book.best_bid()
            queue = (book.ask_queue if side == "B" else book.bid_queue)[opp_tick]
            oid = queue[0]
            order = book.orders[oid]
            wanted = draw_exec_qty(i)
            fill = min(order.size, wanted)
            filled = book.execute(side, fill)
            delta = -filled
            msg_type, m_side, m_oid, m_price, m_qty = "EXECUTE", side, oid, opp_tick, filled

        elif u < P_ADD + P_EXEC + P_CANCEL:
            idx = min(int(pick_frac[i] * len(book.active_ids)), len(book.active_ids) - 1)
            oid = book.active_ids[idx]
            order = book.orders[oid]
            reduce_by = max(1, int(order.size * (0.15 + 0.6 * cancel_frac[i])))
            removed = book.cancel(oid, reduce_by)
            delta = -removed
            msg_type, m_side, m_oid, m_price, m_qty = "CANCEL", order.side, oid, order.tick, removed

        else:
            idx = min(int(pick_frac[i] * len(book.active_ids)), len(book.active_ids) - 1)
            oid = book.active_ids[idx]
            order = book.orders[oid]
            side_, tick_ = order.side, order.tick
            removed = book.delete(oid)
            delta = -removed
            msg_type, m_side, m_oid, m_price, m_qty = "DELETE", side_, oid, tick_, removed

        bp, bs, ap, asz = book.snapshot(DEPTH)
        rows.append(
            (ticker, date, day_index, len(rows), ts, msg_type, m_side, m_oid, m_price, m_qty,
             delta, book.total_resting_size, *bp, *bs, *ap, *asz)
        )

    cols = [
        "ticker", "date", "day_index", "event_seq", "ts", "msg_type", "side", "order_id",
        "price_ticks", "qty", "size_delta", "total_resting_after",
        *[f"bid_px_{k+1}" for k in range(DEPTH)],
        *[f"bid_sz_{k+1}" for k in range(DEPTH)],
        *[f"ask_px_{k+1}" for k in range(DEPTH)],
        *[f"ask_sz_{k+1}" for k in range(DEPTH)],
    ]
    return pd.DataFrame.from_records(rows, columns=cols)


def simulate_all(tickers: list[str] | None = None, n_days: int = N_DAYS,
                  msgs_per_day: int = MSGS_PER_DAY):
    """Yield (ticker, date, DataFrame) one session at a time so the caller
    can flush each session straight to its own Parquet partition without
    holding the whole 6.4M-row store in memory at once."""
    tickers = tickers or TICKERS
    dates = _business_dates(n_days)
    for t_idx, ticker in enumerate(tickers):
        for d_idx, date in enumerate(dates):
            seed = 1_000 * t_idx + d_idx
            df = simulate_ticker_day(ticker, date, d_idx, seed, n_msgs=msgs_per_day)
            yield ticker, date, df
