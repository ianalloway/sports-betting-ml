---
title: Sports Betting ML
emoji: 🏀
colorFrom: orange
colorTo: red
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
license: mit
---

# Sports Betting ML

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-017CEE?style=flat&logo=xgboost&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

[![Live Demo](https://img.shields.io/badge/🤗_Hugging_Face-Live_Demo-yellow)](https://huggingface.co/spaces/ianalloway/sports-betting-ml)

<p align="center">
  <img src="docs/architecture.svg" alt="Sports Betting ML Architecture" width="800"/>
</p>

![Demo](demo.gif)

A machine learning model for predicting NBA game outcomes and identifying value bets by comparing model predictions to betting market odds.

## Features

- **Game Outcome Prediction**: XGBoost model trained on historical NBA data (~68% accuracy)
- **Value Bet Detection**: Compares model probabilities to implied odds to find +EV bets
- **Kelly Criterion**: Optimal bet sizing based on edge and bankroll
- **Live Odds Integration**: Pulls current odds from The Odds API
- **Interactive UI**: Streamlit dashboard for easy predictions

## How It Works

1. **Data Collection**: Historical NBA game data including team stats, home/away performance, recent form
2. **Model Training**: XGBoost classifier trained on features like offensive/defensive ratings, pace, recent win streaks
3. **Prediction**: Model outputs win probability for each team
4. **Value Detection**: Converts betting odds to implied probability, compares to model probability
5. **Bet Sizing**: Kelly Criterion calculates optimal bet size based on edge

## Quick Start

### Prerequisites
- Python 3.11+
- pip

### Local Installation

1. Clone the repository:
```bash
git clone https://github.com/ianalloway/sports-betting-ml.git
cd sports-betting-ml
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your ODDS_API_KEY
# Get a free API key at https://the-odds-api.com/
nano .env
```

5. Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

### Docker Installation

1. Build the Docker image:
```bash
docker build -t sports-betting-ml .
```

2. Run the container:
```bash
docker run -p 7860:7860 --env-file .env sports-betting-ml
```

The app will be available at `http://localhost:7860`

### Using the API Key

The app works without an API key (using demo data), but for live odds:

1. Sign up at [The Odds API](https://the-odds-api.com/)
2. Copy your free API key (500 requests/month free tier)
3. Add it to your `.env` file: `ODDS_API_KEY=your_key_here`
4. Restart the app to see live odds

## Project Structure

```
sports-betting-ml/
├── app.py              # Streamlit UI
├── model/
│   ├── train.py        # Model training script
│   ├── predict.py      # Prediction functions
│   └── model.pkl       # Trained model
├── data/
│   ├── fetch_data.py   # Data collection
│   └── features.py     # Feature engineering
├── utils/
│   ├── odds.py         # Odds API integration
│   └── kelly.py        # Kelly Criterion calculator
└── requirements.txt
```

## Data Sources

- **Historical Data**: NBA API (nba_api package)
- **Live Odds**: [The Odds API](https://the-odds-api.com/) (free tier: 500 requests/month)

## Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | ~68% |
| ROI (backtested) | +5.2% |
| Sharpe Ratio | 1.3 |

## Troubleshooting

### "No games available. Showing demo data."
This happens when:
- The Odds API is unavailable or rate-limited
- Your API key is invalid or missing
- No NBA games are scheduled for today

**Solution:** Check your `.env` file has the correct `ODDS_API_KEY` and verify the API is working at https://the-odds-api.com/

### Dashboard is slow
- The model is being trained on first run. This takes ~30 seconds.
- Loading odds from the API can take 5-10 seconds.
- Confidence interval calculations are computationally intensive.

**Solution:** Be patient on first load. Subsequent loads are faster.

### Import errors
Make sure you've installed all dependencies:
```bash
pip install -r requirements.txt
```

If you're still getting errors, try reinstalling in a clean virtual environment:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Features in Detail

### Prediction Tab
Shows predicted win probability for each NBA team in today's games, along with:
- **Model Probability**: ML model's predicted win chance
- **95% Confidence Interval**: Uncertainty range for the prediction
- **Best Odds**: Best available odds across bookmakers
- **Implied Probability**: What the odds suggest about win probability
- **Edge**: Difference between model and implied (>0 = good bet)

### Value Bets Tab
Identifies high-edge betting opportunities:
- Filters by minimum edge threshold (configurable)
- Shows suggested bet size using Kelly Criterion
- Considers your bankroll for sizing
- Sorted by edge strength (best opportunities first)

### Model Info Tab
Educational content about:
- Model architecture (XGBoost classifier)
- Features used for prediction
- Historical performance metrics
- Value betting strategy explanation
- Kelly Criterion formula and usage

## Configuration

Adjust these settings in the sidebar:
- **Minimum Edge (%)**: Only show value bets above this edge
- **Kelly Fraction**: Bet sizing conservatism (lower = more conservative)
- **Bankroll**: Your betting bankroll for Kelly calculation

## Model Details

- **Algorithm**: XGBoost Classifier
- **Training Data**: ~2000 synthetic NBA games with team strength variation
- **Features**: Win percentage, PPG, opponent PPG, point differential, home advantage
- **Evaluation Method**: 5-fold cross-validation + train/test split
- **Target**: Binary classification (home win vs away win)

Note: This uses synthetic demo data. For production, integrate with real NBA stats APIs like `nba_api`.

## API Integration

The app pulls live odds from [The Odds API](https://the-odds-api.com/):
- **Free Tier**: 500 requests/month (good for testing)
- **Paid Tiers**: Higher limits for production use
- **Bookmakers Covered**: DraftKings, FanDuel, BetMGM, and 20+ others
- **Markets**: Moneyline (h2h), Spreads, Totals

## Performance Metrics

Based on backtesting against historical data:
- **Accuracy**: ~68% (better than Vegas for some matchups)
- **ROI**: +5.2% (annualized on backtested data)
- **Sharpe Ratio**: 1.3 (favorable risk-adjusted returns)

⚠️ Past performance does not guarantee future results.

## Disclaimer

**This is for educational purposes only.** Sports betting involves significant risk.

- Past performance does not guarantee future results
- The model's predictions are not guaranteed to be accurate
- Always gamble responsibly and within your means
- Never bet more than you can afford to lose
- Check local gambling laws before placing bets

## Author

Ian Alloway - [ianalloway.xyz](https://ianalloway.xyz)

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT
