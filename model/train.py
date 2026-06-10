"""Model training script for NBA game prediction."""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    log_loss,
)
from xgboost import XGBClassifier

MODEL_DIR = Path(__file__).parent

# Make the repo root importable so this works both as a module
# (python -m model.train) and as a script (python model/train.py).
sys.path.insert(0, str(MODEL_DIR.parent))

from data.features import prepare_training_data  # noqa: E402


def create_sample_data() -> pd.DataFrame:
    """
    Create sample training data for demonstration.
    In production, this would fetch from NBA API.
    """
    np.random.seed(42)

    teams = [
        'Boston Celtics', 'Milwaukee Bucks', 'Philadelphia 76ers', 'Cleveland Cavaliers',
        'New York Knicks', 'Brooklyn Nets', 'Miami Heat', 'Atlanta Hawks',
        'Chicago Bulls', 'Toronto Raptors', 'Indiana Pacers', 'Washington Wizards',
        'Orlando Magic', 'Charlotte Hornets', 'Detroit Pistons',
        'Denver Nuggets', 'Memphis Grizzlies', 'Sacramento Kings', 'Phoenix Suns',
        'Los Angeles Clippers', 'Golden State Warriors', 'Los Angeles Lakers',
        'New Orleans Pelicans', 'Dallas Mavericks', 'Utah Jazz', 'Minnesota Timberwolves',
        'Oklahoma City Thunder', 'Portland Trail Blazers', 'San Antonio Spurs', 'Houston Rockets'
    ]

    # Team strength ratings (higher = better)
    team_strength = {team: np.random.uniform(0.3, 0.7) for team in teams}
    # Make some teams clearly better
    team_strength['Boston Celtics'] = 0.75
    team_strength['Denver Nuggets'] = 0.72
    team_strength['Milwaukee Bucks'] = 0.70
    team_strength['Phoenix Suns'] = 0.68
    team_strength['Detroit Pistons'] = 0.28
    team_strength['San Antonio Spurs'] = 0.30

    # Valid, chronologically increasing dates (~8 games per day).
    # The old formula produced impossible dates like month 20, which
    # made "sort by date" meaningless for time-series evaluation.
    season_start = pd.Timestamp('2024-10-22')

    games = []

    # Generate 2000 sample games
    for i in range(2000):
        home_team = np.random.choice(teams)
        away_team = np.random.choice([t for t in teams if t != home_team])

        # Calculate expected scores based on team strength
        home_strength = team_strength[home_team]
        away_strength = team_strength[away_team]

        # Base scores around 110 points
        home_base = 100 + home_strength * 30
        away_base = 100 + away_strength * 30

        # Add randomness
        home_score = int(home_base + np.random.normal(0, 10))
        away_score = int(away_base + np.random.normal(0, 10))

        # Home advantage adds ~3 points
        home_score += int(np.random.normal(3, 1))

        games.append({
            'date': (season_start + pd.Timedelta(days=i // 8)).strftime('%Y-%m-%d'),
            'home_team': home_team,
            'away_team': away_team,
            'home_score': max(80, home_score),
            'away_score': max(80, away_score)
        })

    return pd.DataFrame(games)


def build_model() -> XGBClassifier:
    """Construct an untrained XGBoost model with the standard params."""
    return XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )


def train_model(X: pd.DataFrame, y: pd.Series) -> XGBClassifier:
    """Train the XGBoost model."""
    model = build_model()
    model.fit(X, y)
    return model


def evaluate_model(X: pd.DataFrame, y: pd.Series) -> dict:
    """Evaluate with a chronological holdout and walk-forward CV.

    Rows are time-ordered and every game's rolling features are built
    from earlier games only. A random train/test split therefore leaks:
    training games that happen *after* a test game carry that game's
    outcome inside their rolling statistics, inflating test accuracy.
    All splits here keep training data strictly earlier than evaluation
    data (issue #28).
    """
    # Walk-forward cross-validation: each fold trains on the past and
    # validates on the next, unseen block of games.
    cv_scores = []
    for train_idx, val_idx in TimeSeriesSplit(n_splits=5).split(X):
        fold_model = build_model()
        fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        fold_pred = fold_model.predict(X.iloc[val_idx])
        cv_scores.append(accuracy_score(y.iloc[val_idx], fold_pred))
    cv_scores = np.asarray(cv_scores)

    # Chronological holdout: train on the first 80% of the season,
    # test on the most recent 20%.
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = build_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'test_accuracy': accuracy_score(y_test, y_pred),
        # Probability quality matters more than accuracy for bet sizing:
        # Kelly stakes are driven by the predicted probabilities.
        'test_log_loss': log_loss(y_test, y_prob),
        'test_brier': brier_score_loss(y_test, y_prob),
        'classification_report': classification_report(y_test, y_pred)
    }


def main():
    """Main training pipeline."""
    print("Creating sample data...")
    df = create_sample_data()

    print("Preparing features...")
    # Single source of truth for feature engineering (data/features.py),
    # shared with serving to avoid train/serve skew (issue #28).
    X, y = prepare_training_data(df)
    print(f"Dataset size: {len(X)} games")

    print("Evaluating model (walk-forward CV + chronological holdout)...")
    metrics = evaluate_model(X, y)

    print(f"\nWalk-forward CV accuracy: {metrics['cv_mean']:.3f} (+/- {metrics['cv_std']:.3f})")
    print(f"Holdout accuracy (most recent 20%): {metrics['test_accuracy']:.3f}")
    print(f"Holdout log loss: {metrics['test_log_loss']:.3f}")
    print(f"Holdout Brier score: {metrics['test_brier']:.3f}")
    print("\nClassification Report:")
    print(metrics['classification_report'])

    # Train the final model on the full dataset
    print("Training final model on full dataset...")
    model = train_model(X, y)

    # Save model
    model_path = MODEL_DIR / "model.json"
    model.save_model(str(model_path))
    print(f"Model saved to {model_path}")

    # Feature importance
    print("\nFeature Importance:")
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importance.to_string(index=False))

    return model


if __name__ == "__main__":
    main()
