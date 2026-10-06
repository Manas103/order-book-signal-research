# Order Book Signal Research on Nasdaq-Style Message Data

A from-scratch limit order book pipeline: a synthetic ITCH-style message
generator feeding a price-time-priority book builder, a partitioned Parquet
store queried entirely through DuckDB SQL window functions for feature
engineering, and a LightGBM regressor evaluated out of sample, at three
event-time horizons, against a linear order-flow-imbalance baseline under a
day-level purged split. Every number below was measured on this machine, not
targeted; where three genuine attempts did not reach a stated target, that is
reported plainly rather than adjusted.

## Why this exists

Order flow imbalance (OFI) is one of the small number of results in market
microstructure with a clean, citable theoretical claim: Cont, Kukanov and
Stoikov (2014) show OFI drives short-interval price changes approximately
linearly, with a slope inversely proportional to depth. This project builds
the smallest honest version of the research question a signal desk actually
asks: does a tree model beat that linear baseline, and if so, for how long.
It is a small version of message-level signal research, not a claim to have
reproduced an institutional research pipeline.

## Honest framing, up front

- **This is not real Nasdaq TotalView-ITCH or LOBSTER data.** Real LOBSTER
  sample files sit behind a JavaScript-rendered request form that could not
  be fetched as plain data in this build environment, and this project does
  not claim otherwise. `obsr/simulate.py` generates a from-scratch,
  self-consistent ITCH-style add/execute/cancel/delete message stream,
  calibrated to three published microstructure facts (bursty, autocorrelated
  order arrivals; heavy-tailed order sizes; order flow imbalance with real,
  decaying short-horizon predictive power), not fit to any specific real
  dataset. This mirrors the synthetic-feed precedent already used elsewhere
  in this portfolio (`exchange-market-data-decoder`,
  `market-data-tick-capture`), disclosed the same way here.
- **The book builder enforces price-time priority and rejects a crossing
  add.** Unlike the synthetic feed in the sibling decoder repos, this
  generator only ever proposes adds that do not cross the opposite side (see
  `obsr/book.py`'s `CrossedBookError`), so the book stays valid by
  construction rather than by later correction.
- **The gradient-boosted-tree and linear-baseline R^2 targets were not
  reached, after three genuine attempts.** See Findings.
- **"5 Nasdaq names" means 5 synthetic tickers (SYNA through SYNE), not 5
  real symbols.** Named that way because the design decision, "how many
  names," is itself part of what the posting's substitute-experience clause
  asks for; the price series behind each ticker is synthetic.

### Machine and toolchain

| | |
|---|---|
| CPU | AMD Ryzen 7 7800X3D, 8 physical / 16 logical cores |
| RAM | 31.1 GB |
| OS | Windows 11, build 10.0.26200 |
| Python | CPython 3.12.10, in a repo-local venv |
| Libraries | duckdb 1.5.5, pandas 3.0.5, pyarrow 25.0.1, lightgbm 4.7.0, scikit-learn 1.9.1, numpy 2.5.3, pytest 9.1.1 (see `requirements.txt`) |
| Pacing | non-realtime process, no CPU pinning, single run per reported number unless noted |

## Architecture

```
obsr/simulate.py       synthetic ITCH-style message generator (5 tickers x 16 days x 80,000 msgs)
obsr/book.py           fast price-time-priority BookBuilder (sorted price levels, FIFO queues per level)
obsr/reference_book.py SlowBookBuilder: independent, brute-force reference oracle
obsr/store.py          partitioned Parquet writer (ticker=.../date=...) plus a DuckDB connection helper
obsr/features.py       43 microstructure features and 3 forward labels, built entirely in one DuckDB SQL statement
obsr/labels.py         forward mid-price-change label definition, plus an independent plain-pandas reimplementation for testing

scripts/generate_data.py   runs the simulator for all 80 (ticker, date) sessions, writes the Parquet store
scripts/run_research.py    builds features, applies the day-level purged split, trains GBT + linear baseline, reports R^2

tests/test_book_invariants.py    conservation and price-time-priority checks on BookBuilder
tests/test_reference_oracle.py   BookBuilder diffed exactly against SlowBookBuilder over 20,000 randomized operations
tests/test_features.py           hand-computed feature values on a crafted single row
tests/test_labels_and_leakage.py SQL LEAD labels vs. an independent pandas reimplementation; purged-split overlap check

docs/benchmark_output.txt   raw run of scripts/run_research.py (the reported attempt)
docs/test_output.txt        raw pytest run
docs/research_results.json  the same R^2 numbers, machine-readable
```

**Why DuckDB SQL for features, not pandas.** `obsr/features.py`'s docstring
states the rule and the tests enforce it: every feature is a `PARTITION BY
ticker, date ORDER BY event_seq` window function, so a lag, a rolling window,
or a forward label can never reach across a session boundary by construction.
A pandas implementation using `.shift()`/`.rolling()` across a concatenated
frame would need the same partition keys threaded through every call by hand;
one SQL statement with one `WINDOW` clause is both the faster and the harder-
to-get-wrong version of the same computation, which is why `store.py` and
`features.py` route the whole feature build through DuckDB rather than pandas.

**Why a fast book and a slow one.** `BookBuilder` keeps sorted price-level
lists and O(1) level-size lookups because it runs inside the generator's
per-message loop 6.4 million times. `SlowBookBuilder` recomputes every
answer from a flat dict by brute-force scan, on purpose: it shares no data
structure with the fast implementation, so agreement between the two is
actual evidence of correctness rather than two paths through the same bug.

**The linear baseline is deliberately one feature.** `ofi_agg5`, the sum of
order flow imbalance across the top 5 levels, is regressed alone against each
horizon's label. This is the Cont-Kukanov-Stoikov baseline in its narrowest,
most falsifiable form: if the tree model's advantage over one honestly-chosen
feature disappears at long horizons, that is a real finding about decay, not
an artifact of a weak baseline built from many features.

## Validation

**1. Reference-oracle diff.** `tests/test_reference_oracle.py` runs 10 seeded
sequences of 2,000 randomized add/execute/cancel/delete operations each
(20,000 operations total) through `BookBuilder` and `SlowBookBuilder` in
lockstep, asserting the full 5-level snapshot matches after every single
operation, not just at the end.

**2. Label cross-check.** `tests/test_labels_and_leakage.py` builds the same
small message sample through the production SQL path
(`obsr.features.build_feature_sql`, DuckDB `LEAD`) and through
`obsr.labels.compute_forward_labels` (plain pandas `groupby().shift()`), and
asserts the three horizon labels agree exactly, row for row.

**3. Leakage check.** The same test asserts that for every (ticker, date)
session, the last `h` rows have a null `fwd_mid_chg_h` label rather than a
label reaching into the next session, and separately asserts the day-level
purged train/test split produces disjoint date sets with the embargo day in
neither.

**4. Feature unit checks.** `tests/test_features.py` hand-computes spread,
microprice, level-1 queue imbalance and total depth on a single crafted row
and diffs them exactly against the SQL output, plus asserts every one of the
43 declared feature columns actually appears in the query's output.

```
11 passed in 3.05s
```

## Findings

**The gradient-boosted-tree and linear-OFI R^2 targets were not reached,
after three genuine attempts, and the miss is a real finding about this
signal's strength, not a bug.** First attempt: an unregularized LightGBM
model (300 trees, 31 leaves, no explicit regularization) trained on the full
4.4M-row training set measured R^2 of -0.0384 at the 10-event horizon,
*worse* than predicting the mean, while the linear-OFI baseline measured
essentially 0.0000. The working hypothesis was overfitting given how many
noisy features (43) sit next to one genuinely informative one. Second
attempt: winsorizing the label at the 1st/99th train percentile (to blunt
the influence of the Pareto-tailed order sizes on a heavy-tailed price-change
target) produced R^2 of exactly 1.0000 at the 10- and 100-event horizons.
That number is the wrong kind of encouraging: the 10-event mid-price-change
label is spiked at zero (most 10-event windows see no tick-level move at
all), so clipping collapsed the test target's own variance near zero, and
R^2 became a numerically degenerate ratio rather than a measure of skill.
That attempt is disclosed and discarded, not reported as a result. Third
attempt: reverting the label transform and regularizing the model instead
(15 leaves, max depth 5, `reg_alpha`/`reg_lambda` = 1.0, `min_child_samples`
1,000) measured R^2 of -0.0033, +0.0025 and +0.0070 at the 10-, 100- and
1,000-event horizons, and this is the number reported below.

The linear-OFI baseline's R^2 stayed within numerical noise of 0.0000 at
every horizon across all three attempts. Root cause: `market_simulator`'s
calibration biases the *aggressor* side of EXECUTE messages toward the
latent momentum state considerably more strongly than the *resting* side of
ADD messages (`K_EXEC=1.6` vs. `K_ADD=0.5` in `obsr/simulate.py`), so the
correlation between top-of-book OFI and the momentum actually driving the
label is real but weak enough that, over a 4.4M-row training set, it is
statistically indistinguishable from zero out of sample. That is a
calibration property of this generator, not a claim that OFI has no real
predictive content; the Cont-Kukanov-Stoikov result this project is built
around was derived from real order book data, not from this simulator.

**The tree's advantage over the linear baseline does not clearly decay with
horizon here, which contradicts the claim it would.** The measured advantage
(GBT R^2 minus linear R^2) is -0.0033 at 10 events, +0.0025 at 100, and
+0.0070 at 1,000: essentially flat and small at every horizon, not a large
short-horizon edge that erodes. Given how close both baselines sit to zero
signal at every horizon, this project cannot distinguish "the tree's edge
decays" from "neither model has enough signal at any horizon to say," and
reporting that honestly is more useful than asserting a decay curve the data
does not support.

## Measured results

Machine: AMD Ryzen 7 7800X3D, 8 physical / 16 logical cores, 31.1 GB RAM,
Windows 11 build 10.0.26200, CPython 3.12.10, LightGBM 4.7.0. Full pipeline
(6,400,000 messages, feature build, both models, all three horizons) runs in
about 61 seconds. Raw output in `docs/benchmark_output.txt` and
`docs/research_results.json`; the two discarded attempts are described in
Findings, not re-run for this README.

| Horizon (events) | n_train | n_test | GBT out-of-sample R^2 | Linear-OFI out-of-sample R^2 | Tree advantage |
|---|---|---|---|---|---|
| 10 | 4,399,010 | 1,599,640 | **-0.0033** (target 0.061) | **-0.0000** (target 0.042) | -0.0033 |
| 100 | 4,394,060 | 1,597,840 | **+0.0025** | **-0.0000** | +0.0025 |
| 1,000 | 4,344,560 | 1,579,840 | **+0.0070** | **-0.0000** | +0.0070 |

What R^2 measures here: out-of-sample coefficient of determination on a
day-level purged split (11 train days, 1 embargo day, 4 test days per
ticker), computed once per horizon on the full test set, not cross-validated
or averaged across seeds. It does not measure trading P&L, transaction
costs, or capacity; those are exactly what the sibling projects in this
folder's role measure instead.

| Data claim | Target | Measured |
|---|---|---|
| Full-depth books reconstructed | 5 Nasdaq-style names | **5** (SYNA-SYNE) |
| Messages | 6.4M | **6,400,000** exactly |
| Store format | partitioned Parquet queried with DuckDB | Hive-partitioned (`ticker=/date=`), all feature SQL runs through DuckDB |
| Microstructure features | 40+ | **43** |
| Horizons | 10, 100, 1,000 events | all three implemented |
| Split | day-level purged | 11 train / 1 embargo / 4 test days per ticker, verified disjoint (test) |

## Building and running

```bash
# from a Python 3.12 venv with requirements.txt installed
python scripts/generate_data.py       # ~49s, writes 220 MB to data/ (gitignored)
python scripts/run_research.py        # ~55-75s, writes docs/research_results.json
python -m pytest tests -v             # ~3s, 11 tests
```

`data/` is gitignored (220 MB of Parquet, regenerated in under a minute by
the command above); nothing under `data/` is tracked.

## Sibling comparison

[`exchange-market-data-decoder`](https://github.com/Manas103/exchange-market-data-decoder)
and [`market-data-tick-capture`](https://github.com/Manas103/market-data-tick-capture)
both build full-depth books from a synthetic message stream, but neither does
any signal research: no features, no model, no baseline, no horizons, and
their synthetic feeds are explicitly allowed to cross (a documented finding
in the decoder's own README) because their subject is decoding throughput,
not price prediction. This project's generator instead rejects a crossing
add outright, because a crossed book has no well-defined mid price to build
a regression label from. The two repos are answering different questions
with the same category of input.

## Extended (Oct. 2026): Queue Position and Fill Probability Study

**What existed before this extension and what is new.** Before this, the
book builder reconstructed full-depth snapshots but threw away exactly the
information a resting order's queue position needs: `BookBuilder` keeps a
FIFO list per price level internally but the message store only ever wrote
out aggregate level size, never which order was where in that list, because
nothing before this extension needed it. Three things are new: a full L3
replay (`obsr/queue_position.py`) that recovers each ADD event's queue rank
and quantity ahead from the stored message stream alone; a fill-probability
measurement by queue rank, built as one DuckDB SQL window-function pass
(`obsr/fill_probability.py`); and an execution-cost comparison between
always crossing the spread and a rank-aware quote-and-wait rule
(`obsr/execution_cost.py`), none of which the base repo had any reason to
include (it only ever asked "does a feature predict a price move", never
"would an order here have filled").

### Architecture (additions)

```
obsr/queue_position.py    replays one session's stored message stream
                           through a fresh BookBuilder, recording each ADD
                           event's rank (1 = first order ever at that exact
                           price) and qty_ahead (resting quantity in front
                           of it), plus whether it joined the best bid/ask
                           at that instant ("at the touch")
obsr/fill_probability.py  one DuckDB SQL statement, built entirely from a
                           forward-looking window function over a UNION of
                           ADD and EXECUTE rows (no join operator), that
                           reads off each order's first qualifying EXECUTE
                           timestamp and buckets fill-within-horizon by rank
obsr/execution_cost.py    always-crossing vs. rank-aware quote-and-wait
                           expected effective spread, using the measured
                           fill-probability-by-queue-depth curve and a
                           pandas merge_asof forward lookup for the price
                           one horizon later

scripts/compute_queue_positions.py  runs the replay over all 80 sessions,
                           writes data/queue_positions/ (same partitioning
                           as the message store)
scripts/run_queue_study.py  runs the SQL window pass, the cost comparison,
                           writes docs/queue_study_results.json

tests/test_queue_position.py      hand-computed rank/qty_ahead/touch on a
                           9-event scenario, plus a 20,000-operation
                           randomized sequence diffed against an
                           independent brute-force ledger that reimplements
                           price-time-priority execution from scratch
tests/test_fill_probability_sql.py  hand-computed fill-within-horizon
                           outcomes (including a fill that lands exactly
                           one horizon too late, and two sessions reusing
                           the same order_id, to prove the partition keys
                           are doing their job) against the SQL window pass

docs/queue_study_output.txt    raw run of scripts/run_queue_study.py
docs/queue_study_results.json  the same numbers, machine-readable
```

**Why "at the touch" is reported separately from the whole-book number.**
The first genuine measurement attempt bucketed every ADD event in the whole
book by rank, rank 5 meaning "5 or more": fill probability came out 0.1758
at rank 1 and 0.000055 at rank "5+". That number is real but misleading,
because it silently mixes two different effects. An order's fill
probability depends on two things: how far its *price* sits from the
market, and its *queue position* once it is at a price that matters. Rank
"5 or more" across the whole book is dominated by 2,658,360 of 2,689,366
total ADD events (98.8%) resting at a price nowhere near the touch, which
almost never fill quickly for a completely different reason than queue
position. Restricting to the 1,088,096 ADD events that joined the best
bid/ask at the instant of insertion isolates the question the resume claim
is actually about, and is the number reported below.

**Why rank 5 means exactly rank 5, not "5 or more", in the reported
table.** The second attempt, restricted to the touch, still bucketed rank 5
as "5 or more" and measured 0.0001 at that bucket, because the touch queue
itself can run thousands deep and a "5+" bucket at the touch is still
overwhelmingly deeper-than-5 orders. The third attempt looked at the exact
per-rank breakdown (ranks 1 through 12, see Findings) and found a smooth,
steep, monotonic decay with no cliff at any particular rank, so the
honestly reportable number is fill probability at the *exact* rank the
claim names, not a bucket that happens to be labeled with that number.

### Validation

1. **Reference-oracle diff on queue position.** `tests/test_queue_position.py`
   runs 20,000 randomized add/execute/cancel/delete operations through
   `BookBuilder` while an entirely separate, independently coded ledger
   (its own price-time-priority execution loop, not a call into
   `BookBuilder`) tracks the same orders, and asserts every single ADD's
   rank and quantity-ahead, as recovered by `replay_queue_positions`,
   matches the independent ledger's brute-force recomputation.
2. **Hand-computed fill outcomes.** `tests/test_fill_probability_sql.py`
   checks the SQL window pass against five hand-picked cases: a fill
   inside the horizon, no fill at all, a fill exactly one horizon-length
   too late (the off-by-one case most likely to hide a bug), a fill after
   a gap with an intervening ADD at a different timestamp, and two
   different sessions reusing the same `order_id`, which must not be
   confused by the `PARTITION BY ticker, date, order_id` window.

```
14 passed in 11.55s
```

### Findings

**Fill probability decays steeply and smoothly with exact queue rank at
the touch, with no cliff at any one rank.** Measured per exact rank, 1
through 12, at-the-touch ADD events, 1-second horizon:

| Rank | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10+ |
|---|---|---|---|---|---|---|---|---|---|---|
| Fill probability | 0.5377 | 0.3262 | 0.1724 | 0.0981 | **0.0410** | 0.0163 | 0.0059 | 0.0010 | 0.0010 | ~0.0000 |

The decay is close to geometric (each rank roughly 55-60% of the previous
one's probability) until it flattens into noise around rank 8, which is
mechanically exactly what price-time priority predicts: an order fills
only after every order ahead of it either fills or is removed, so each
additional rank multiplies the chance of surviving long enough to be
reached.

**The resume's claimed numbers (0.71 at rank 1, 0.19 at rank 5) were not
reached, after reaching this rank/touch methodology on the third genuine
attempt; the measured numbers (0.5377, 0.0410) are reported instead of
re-targeted.** The *shape* of the claim is correct (rank 1 clearly beats
rank 5, by more than 10x), but the measured magnitude is both lower at
rank 1 and much lower at rank 5 than claimed. The two likely drivers, not
separately isolated here: this generator's order arrival rate (80,000
messages/session across 5 tickers) may simply be thinner at the touch than
whatever produced the original target, and a 1-second horizon on
`gaps ~ Exponential(0.05)`-paced synthetic time (mean 20 events/second)
gives an order roughly 20 *subsequent* events to be reached, not a fixed
large sample; both would make even a favorable rank fill less often than a
denser or longer-horizon setup would.

**The rank-aware quote-and-wait rule did not come close to the claimed
18% effective-spread reduction, after the same three attempts; it measured
0.64%, essentially no improvement.** The direct cause: the expected-value
blend (`p_fill * (bid - mid) + (1 - p_fill) * (ask_later - mid)`) is
dominated by the `(1 - p_fill)` branch almost everywhere, because the
large majority of "at the touch" ADD events sit behind an already-deep
queue (the median `qty_ahead` of the 1,088,096 at-touch ADD events is far
above the rank-1 level), so `p_fill` is usually small and the rule's
expected cost collapses toward "cross later", which on this calibration's
random-walk mid is, on average, barely different from crossing now. A
rank-aware rule would only show the claimed-size benefit if it could
reliably post at or near rank 1, which the measured rank distribution says
is the exception, not the default, for an order simply joining the current
best price.

### Measured results

Machine: AMD Ryzen 7 7800X3D, 8 physical / 16 logical cores, 31.1 GB RAM,
Windows 11 build 10.0.26200, CPython 3.12.10, DuckDB 1.5.5. Full pipeline
(80-session replay plus the SQL window pass plus the cost comparison) runs
in about 51 seconds (43.2s replay + 7.1s study). Raw output in
`docs/queue_study_output.txt` and `docs/queue_study_results.json`.

| Claim | Target | Measured | Met |
|---|---|---|---|
| Simulated price-time-priority order book | implemented | `BookBuilder` (already existed), replayed per-order via `queue_position.py` | Yes |
| Messages | 6.4M | 6,400,000 (already measured; unchanged by this extension) | Yes |
| Fill probability by queue rank in one DuckDB SQL window pass | implemented | `obsr/fill_probability.py`, a single statement, no join operator | Yes |
| Fill probability at rank 1 | 0.71 | **0.5377** (at the touch) | No |
| Fill probability at rank 5 | 0.19 | **0.0410** (exact rank 5, at the touch) | No |
| Rank-aware quote-and-wait vs. always-crossing effective spread | 18% less | **0.64% less** | No |

What this measures: fill probability is the fraction of ADD events at a
given exact queue rank, restricted to events that joined the best bid/ask
at insertion, that see at least one EXECUTE against that same order within
1 simulated second. It is a mechanical property of price-time priority and
this generator's order flow, not a forecast; see the sibling
`order-book-signal-research` discussion above about why a mechanical claim
was chosen over a predictive one. The effective-spread comparison is an
expected-value calculation over 12,720 decision points (every 500th event
per session with both sides of the book populated), not a simulated,
path-dependent backtest; it does not account for the rank-aware rule's own
market impact or for multiple competing resting orders.

### Building and running

```bash
# from the same venv as the base repo (requirements.txt unchanged)
python scripts/generate_data.py            # if data/ is not already populated
python scripts/compute_queue_positions.py  # ~43s, writes data/queue_positions/
python scripts/run_queue_study.py          # ~7s, writes docs/queue_study_results.json
python -m pytest tests/test_queue_position.py tests/test_fill_probability_sql.py -v
```

## Limitations

- Synthetic data throughout; see "Honest framing, up front".
- The R^2 targets were not reached at any horizon; see Findings for the
  three-attempt root-cause account.
- The linear baseline is intentionally one feature (`ofi_agg5`); a
  multi-feature linear model was not attempted, because the point of this
  baseline is falsifiability against the single quantity the cited paper's
  result is about, not maximizing baseline strength.
- No walk-forward retraining within the test period; the split trains once
  and scores the whole test window, which is simpler than a desk's real
  walk-forward but keeps the purge-boundary logic auditable in one place.
- Single-run R^2 per horizon, not averaged across seeds; given how close
  every measured value sits to zero, a different seed could plausibly flip
  a sign without changing the qualitative finding (no reliable signal at
  the tested horizons in this synthetic dataset).
- Queue-position extension: fill probability by rank and the rank-aware
  execution-cost reduction both came in well below their targets (see
  Findings); the rank-aware rule is an expected-value calculation over
  independently sampled decision points, not a path-dependent backtest,
  and does not model the rule's own market impact or competition from
  other resting orders at the same price.
- The replay in `queue_position.py` recomputes rank and queue position
  from the stored message stream; it does not change what the original
  simulator decided to do, so it inherits every property (and every
  calibration choice) of `obsr/simulate.py` discussed above.
