# Sports Betting ML

NBA game prediction and value-bet detection using XGBoost + Kelly Criterion. Streamlit dashboard for interactive predictions.

## What This Repo Does

- Predict NBA game winners with an XGBoost classifier trained on synthetic sample games
- Identify +EV bets by comparing model probabilities to market-implied odds
- Size bets using Kelly Criterion
- Serve via Streamlit dashboard (live demo on Hugging Face)

Demo/synthetic evaluation figures live in the README. Treat them as a workflow demo, not production returns.

## Architecture

```text
app.py                # Streamlit dashboard (the UI)
model/
  train.py            # Training script (synthetic sample data)
  predict.py          # Prediction + confidence
  benchmark.py        # Real archived NBA holdout, separate from serving
  artifacts/
    model.json        # Saved XGBoost model (native format; not committed)
data/
  features.py         # Feature engineering (shared train/serve)
  historical.py       # Pinned FiveThirtyEight archive ingestion
utils/
  odds.py             # The Odds API integration + parsing
  kelly.py            # Kelly Criterion calculator
tests/                # pytest suite
pytest.ini            # pythonpath = . for imports
requirements.txt      # Python deps
Dockerfile            # Container build
docker-compose.yml    # Multi-container (app + deps)
env.example           # Template for .env (ODDS_API_KEY, optional MODEL_PATH)
demo.gif              # App demo recording
docs/                 # Documentation (architecture.svg, etc.)
```

## Key Conventions

- Python 3.12+, pip-based deps (see requirements.txt)
- **NBA-focused domain** — not multi-sport. Don't add NFL/MLB/NHL without a data source.
- Uses The Odds API for live odds (optional — app works with demo data without a key)
- Docker available for deployment
- Related repos: [nba-ratings](https://github.com/ianalloway/nba-ratings) (Elo/kelly primitives), [nba-clv-dashboard](https://github.com/ianalloway/nba-clv-dashboard) (evaluation UI)

## Commands

```bash
# Local dev
pip install -r requirements.txt
cp env.example .env   # Add ODDS_API_KEY if you want live odds
streamlit run app.py      # Opens at http://localhost:8501

# Tests / lint (same as CI)
pip install pytest ruff
python -m pytest tests/ -v
ruff check . --select E,F,W --ignore E501

# Docker
docker build -t sports-betting-ml .
docker run -p 7860:7860 --env-file .env sports-betting-ml
```

## How It Works

1. Data: synthetic NBA game rows (team stats, home/away, recent form) from `model.train.create_sample_data`
2. Model: XGBoost classifier trained on win-probability features
3. Prediction: outputs win probability per team
4. Value detection: converts betting odds to implied probability, compares to model probability
5. Bet sizing: Kelly Criterion computes optimal bet size from edge

## Performance (demo/synthetic data)

| Metric | Typical value |
|--------|---------------|
| Walk-forward CV accuracy | ~0.58 |
| Chronological holdout accuracy | ~0.61 |
| Holdout log loss | ~0.67 |
| Holdout Brier score | ~0.24 |

These figures come from `python -m model.train` on the synthetic generator — workflow demo only, not production returns. ROI/Sharpe are **not** computed by the training script.

## Historical Benchmark

`python -m model.benchmark` downloads and checksum-verifies a fixed FiveThirtyEight archive, trains on ending-year seasons 2010–2014, and holds out 2015. The daily walk-forward protocol excludes same-date results, resets rolling history each season, and freezes the trained model and home-rate baseline. See `docs/historical-benchmark.md` and `docs/benchmarks/nba-2015.json` for measured results and limits. It writes to `benchmark-results/`, never the default serving artifact. Unit tests stay offline; CI also runs the real benchmark and uploads its evidence.

## Troubleshooting

- **"No games available. Showing demo data."** — Odds API unavailable/rate-limited, invalid/missing key, or no NBA games today
- **Import errors** — reinstall in a clean venv: `python -m venv venv && source venv/bin/activate && pip install -r requirements.txt`
- **Dashboard slow on first run** — model training + odds fetch can take several seconds

## Owner

Ian Alloway (@ianalloway) — Data Scientist, sports analytics/ML.
