"""Model training script for NBA game prediction."""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

MODEL_DIR = Path(__file__).parent


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
    
    games = []
    
    # Generate 2000 sample games
    for i in range(2000):
        home_team = np.random.choice(teams)
        away_team = np.random.choice([t for t in teams if t != home_team])
        
        # Home advantage factor
        home_advantage = 0.03
        
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
            'date': f'2025-{(i // 100) + 1:02d}-{(i % 28) + 1:02d}',
            'home_team': home_team,
            'away_team': away_team,
            'home_score': max(80, home_score),
            'away_score': max(80, away_score)
        })
    
    return pd.DataFrame(games)


def calculate_rolling_stats(df: pd.DataFrame, team: str, idx: int, n_games: int = 10) -> dict:
    """Calculate rolling statistics for a team up to a given index."""
    team_games = df.iloc[:idx][
        (df.iloc[:idx]['home_team'] == team) | (df.iloc[:idx]['away_team'] == team)
    ].tail(n_games)
    
    if len(team_games) < 3:
        return {
            'win_pct': 0.5,
            'avg_points_for': 110,
            'avg_points_against': 110,
            'point_diff': 0
        }
    
    wins = 0
    points_for = []
    points_against = []
    
    for _, game in team_games.iterrows():
        if game['home_team'] == team:
            points_for.append(game['home_score'])
            points_against.append(game['away_score'])
            if game['home_score'] > game['away_score']:
                wins += 1
        else:
            points_for.append(game['away_score'])
            points_against.append(game['home_score'])
            if game['away_score'] > game['home_score']:
                wins += 1
    
    return {
        'win_pct': wins / len(team_games),
        'avg_points_for': np.mean(points_for),
        'avg_points_against': np.mean(points_against),
        'point_diff': np.mean(points_for) - np.mean(points_against)
    }


def prepare_features(df: pd.DataFrame) -> tuple:
    """Prepare features and targets from game data."""
    features = []
    targets = []
    
    # Start from game 50 to have enough history
    for i in range(50, len(df)):
        game = df.iloc[i]
        
        home_stats = calculate_rolling_stats(df, game['home_team'], i)
        away_stats = calculate_rolling_stats(df, game['away_team'], i)
        
        feature_dict = {
            'home_win_pct': home_stats['win_pct'],
            'away_win_pct': away_stats['win_pct'],
            'home_ppg': home_stats['avg_points_for'],
            'away_ppg': away_stats['avg_points_for'],
            'home_opp_ppg': home_stats['avg_points_against'],
            'away_opp_ppg': away_stats['avg_points_against'],
            'home_point_diff': home_stats['point_diff'],
            'away_point_diff': away_stats['point_diff'],
            'win_pct_diff': home_stats['win_pct'] - away_stats['win_pct'],
            'point_diff_diff': home_stats['point_diff'] - away_stats['point_diff'],
            'home_advantage': 1
        }
        
        features.append(feature_dict)
        targets.append(1 if game['home_score'] > game['away_score'] else 0)
    
    return pd.DataFrame(features), pd.Series(targets)


def train_model(X: pd.DataFrame, y: pd.Series) -> XGBClassifier:
    """Train the XGBoost model."""
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(X, y)
    return model


def evaluate_model(model: XGBClassifier, X: pd.DataFrame, y: pd.Series) -> dict:
    """Evaluate model performance."""
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    
    # Train/test split evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return {
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'test_accuracy': accuracy_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred)
    }


def main():
    """Main training pipeline."""
    print("Creating sample data...")
    df = create_sample_data()
    
    print("Preparing features...")
    X, y = prepare_features(df)
    print(f"Dataset size: {len(X)} games")
    
    print("Training model...")
    model = train_model(X, y)
    
    print("Evaluating model...")
    metrics = evaluate_model(model, X, y)
    
    print(f"\nCross-validation accuracy: {metrics['cv_mean']:.3f} (+/- {metrics['cv_std']:.3f})")
    print(f"Test accuracy: {metrics['test_accuracy']:.3f}")
    print("\nClassification Report:")
    print(metrics['classification_report'])
    
    # Retrain on full data
    print("\nRetraining on full dataset...")
    model = train_model(X, y)
    
    # Save model
    model_path = MODEL_DIR / "model.pkl"
    joblib.dump(model, model_path)
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
