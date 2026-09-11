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
