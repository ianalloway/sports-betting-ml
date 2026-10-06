# Historical NBA holdout benchmark

This is an offline outcome-prediction benchmark on actual archived game results. The dashboard and `python -m model.train` remain synthetic demonstrations. The historical benchmark saves a separate model and does not promote it to serving.

## Reproduce

Install `requirements.txt` with Python 3.12+ and run from the repository root:

```bash
python -m model.benchmark
```

The first run downloads about 18 MB to `data/raw/nbaallelo.csv` using public HTTPS without credentials. Subsequent runs work offline. The source bytes must match the pinned SHA-256; changed or corrupt downloads/caches fail rather than silently switching data or falling back to synthetic rows. A failed download does not produce benchmark results. To use an existing copy:

```bash
python -m model.benchmark --archive /path/to/nbaallelo.csv --output-dir benchmark-results
```

The output directory contains `report.json` (source, configuration, counts, environment, metrics, reliability bins), `predictions.csv` (game IDs, dates, outcomes, both probabilities), and `model.json` (trained only on the training period). Use a fresh output directory for each run; an unsuccessful rerun does not remove evidence from a prior run. CI runs the real benchmark and uploads these files as `historical-benchmark`. Normal pytest tests use small, explicitly invented fixtures without network access.

## Source and attribution

- **Dataset:** [FiveThirtyEight Historical NBA Elo](https://github.com/fivethirtyeight/data/blob/6d880e939ad3d11d94c137c911681b3cf718fd74/nba-elo/README.md), game information attributed upstream to [Basketball-Reference](https://www.basketball-reference.com/).
- **Fixed revision:** `6d880e939ad3d11d94c137c911681b3cf718fd74`.
- **File:** [`nba-elo/nbaallelo.csv`](https://raw.githubusercontent.com/fivethirtyeight/data/6d880e939ad3d11d94c137c911681b3cf718fd74/nba-elo/nbaallelo.csv).
- **SHA-256:** `d46ed3540ee8d9eca31b3e94cc8c777e0be5156173d814ebf65b8195e8d616bc`.
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), under the [source repository's default data license](https://github.com/fivethirtyeight/data/blob/6d880e939ad3d11d94c137c911681b3cf718fd74/README.md). The repository MIT license does not relicense the upstream dataset. Raw data is downloaded locally, not vendored here.
- **Transformations:** select NBA regular-season rows with `game_location=H` and ending-year seasons 2010–2015; retain game ID, season, date, teams, and final scores; derive rolling pregame features. Each source game has two team-perspective rows: only its home row is retained. Playoffs, ABA games, and neutral-site games are excluded. Duplicate home game IDs and invalid/missing selected results fail validation.

Only explicitly allowlisted source columns are read. Elo ratings, `forecast`, postgame ratings, and equivalent season wins never enter the features. Final scores supply labels and later dates' rolling history only.

## Evaluation protocol

1. Select the six ending-year seasons 2010–2015 (2009–10 through 2014–15): 7,133 eligible home-site regular-season games.
2. Reset each team's history at the season boundary. Require five prior games for **both** teams; use at most their ten most recent games. Warmup removes 479 games across all six seasons, leaving 6,654 feature rows. Shared `create_game_features` supplies the same feature names/definitions as the demo.
3. Construct all games on a calendar date before recording any result from that date. Input order and same-day outcome changes cannot leak into that day's inputs. No same-day tipoff ordering is assumed.
4. Fit the repository's existing XGBoost configuration once on dates before **2014-07-01**: 5,505 games, 2009-11-06 through 2014-04-16. Seed 42, one CPU worker, 100 trees, depth 4; exact parameters and package versions are in the report. No tuning, feature selection, early stopping, or probability recalibration uses holdout labels.
5. Score dates **2014-07-01 inclusive to 2015-07-01 exclusive**: 1,149 games, 2014-11-07 through 2015-04-15, after warmup. Model weights stay frozen. Earlier holdout outcomes become inputs for later dates, as in daily walk-forward prediction. This does not represent predicting the entire season on opening day.
6. Compare every holdout game to a constant probability equal to the **training** home-win rate, 0.592916. At a 0.5 decision threshold this baseline always predicts the home team. It is never refit on the holdout.

The holdout year and protocol were chosen before running the first historical evaluation, using the archive's final complete season. Future experiments on this now-inspected holdout must be described as exploratory; reserve a fresh period before claiming independent confirmation.

## Measured result

[Saved local run](benchmarks/nba-2015.json), executed 2026-10-06 with Python 3.13.15 and the pinned repository dependencies on macOS arm64:

| Metric | XGBoost | Training home-win-rate baseline |
|---|---:|---:|
| Holdout games | 1,149 | 1,149 |
| Accuracy (higher is better) | 0.6397 | 0.5727 |
| Log loss (lower is better) | 0.6333 | 0.6834 |
| Brier score (lower is better) | 0.2216 | 0.2451 |
| 10-bin ECE (lower is better) | 0.0391 | 0.0202 |

The observed holdout home-win rate is 0.572672. ECE is the count-weighted absolute difference between mean predicted probability and observed win rate in ten fixed equal-width bins. Bins are left-inclusive/right-exclusive except the final bin includes 1. Empty bins have null means and zero weight. Reliability-bin counts and means are included in the report; this is diagnostic evaluation, not a fitted calibrator.

XGBoost improves accuracy and proper scoring rules against this simple baseline, but has higher ECE. A constant predictor can have low aggregate calibration error while offering little discrimination. ECE depends on binning and sample size; it is not evidence of reliable tail probabilities or safe Kelly staking. Some bins are small. Exact floating-point results can differ across library versions/platforms; each run records its environment.

## Limits

- One retrospective, old season; no claim of current NBA predictive quality, statistical significance, or robust performance across eras. Results are descriptive, without dependence-aware uncertainty intervals.
- No historical sportsbook prices, closing lines, fees, liquidity, or executable wager timestamps. There is no ROI, CLV, Sharpe, or profitability measurement.
- The archive is a retrospectively maintained snapshot. Earlier-date filtering protects against outcome leakage in this pipeline, but cannot establish when each upstream correction became available historically.
- Only regular-season home-site games after per-team warmup are scored. Neutral sites, playoffs, early-season games, injuries, rest, roster changes, and travel are not modeled. The shortened 2011–12 season is in training.
- Rolling history resets each season; no previous-season carryover. A constant home-rate baseline is a minimal comparison, not a market or strong predictive baseline.
- Dashboard team inputs are demo statistics. This benchmark's real-data results must not be represented as validation of the deployed dashboard, synthetic CLV tab, or betting recommendations.
