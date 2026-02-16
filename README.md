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

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

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

## Disclaimer

This is for educational purposes only. Sports betting involves risk. Past performance does not guarantee future results. Please gamble responsibly.

## Author

Ian Alloway - [ianalloway.xyz](https://ianalloway.xyz)

## License

MIT
