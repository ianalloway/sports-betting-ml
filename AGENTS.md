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
  artifacts/
    model.json        # Saved XGBoost model (native format; not committed)
data/
  features.py         # Feature engineering (shared train/serve)
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

| Metric | Value |
|--------|-------|
| Accuracy | ~68% |
| ROI (backtested) | +5.2% |
| Sharpe Ratio | 1.3 |

These figures are from a demo/synthetic dataset — treat as a workflow demo, not production returns.

## Troubleshooting

- **"No games available. Showing demo data."** — Odds API unavailable/rate-limited, invalid/missing key, or no NBA games today
- **Import errors** — reinstall in a clean venv: `python -m venv venv && source venv/bin/activate && pip install -r requirements.txt`
- **Dashboard slow on first run** — model training + odds fetch can take several seconds

## Owner

Ian Alloway (@ianalloway) — Data Scientist, sports analytics/ML.
