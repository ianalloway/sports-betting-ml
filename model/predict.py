"""Prediction functions for NBA game outcomes."""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
from pathlib import Path
from dataclasses import dataclass

# Get the model directory
MODEL_DIR = Path(__file__).parent


def load_model():
    """Load the trained model."""
    model_path = MODEL_DIR / "model.pkl"
    if model_path.exists():
        return joblib.load(model_path)
    return None


def predict_game(
    home_stats: dict,
    away_stats: dict,
    model=None
) -> Tuple[float, float]:
    """
    Predict the outcome of a game.
    
    Args:
        home_stats: Home team statistics
        away_stats: Away team statistics
        model: Trained model (loads default if None)
    
    Returns:
        Tuple of (home_win_prob, away_win_prob)
    """
    if model is None:
        model = load_model()
    
    # If no model, use simple heuristic
    if model is None:
        return predict_heuristic(home_stats, away_stats)
    
    # Create features
    features = create_features(home_stats, away_stats)
    features_df = pd.DataFrame([features])
    
    # Get probability predictions
    probs = model.predict_proba(features_df)[0]
    
    # probs[1] is probability of home win (class 1)
    home_prob = probs[1]
    away_prob = probs[0]
    
    return home_prob, away_prob


def predict_heuristic(home_stats: dict, away_stats: dict) -> Tuple[float, float]:
    """
    Simple heuristic prediction when no model is available.
    Uses win percentage and point differential.
    """
    # Base probability from win percentages
    home_wp = home_stats.get('win_pct', 0.5)
    away_wp = away_stats.get('win_pct', 0.5)
    
    # Point differential factor
    home_pd = home_stats.get('point_diff', 0)
    away_pd = away_stats.get('point_diff', 0)
    pd_factor = (home_pd - away_pd) / 20  # Normalize
    
    # Home court advantage (~3-4 points in NBA)
    home_advantage = 0.03
    
    # Combine factors
    base_prob = (home_wp / (home_wp + away_wp)) if (home_wp + away_wp) > 0 else 0.5
    
    # Adjust for point differential and home advantage
    home_prob = base_prob + pd_factor * 0.1 + home_advantage
    
    # Clamp to valid probability range
    home_prob = max(0.1, min(0.9, home_prob))
    away_prob = 1 - home_prob
    
    return home_prob, away_prob


def create_features(home_stats: dict, away_stats: dict) -> dict:
    """Create feature dictionary for prediction."""
    return {
        'home_win_pct': home_stats.get('win_pct', 0.5),
        'away_win_pct': away_stats.get('win_pct', 0.5),
        'home_ppg': home_stats.get('avg_points_for', 110),
        'away_ppg': away_stats.get('avg_points_for', 110),
        'home_opp_ppg': home_stats.get('avg_points_against', 110),
        'away_opp_ppg': away_stats.get('avg_points_against', 110),
        'home_point_diff': home_stats.get('point_diff', 0),
        'away_point_diff': away_stats.get('point_diff', 0),
        'win_pct_diff': home_stats.get('win_pct', 0.5) - away_stats.get('win_pct', 0.5),
        'point_diff_diff': home_stats.get('point_diff', 0) - away_stats.get('point_diff', 0),
        'home_advantage': 1
    }


def batch_predict(games: list, team_stats: dict, model=None) -> list:
    """
    Predict outcomes for multiple games.
    
    Args:
        games: List of games with home_team and away_team
        team_stats: Dictionary of team statistics
        model: Trained model
    
    Returns:
        List of predictions with probabilities
    """
    if model is None:
        model = load_model()
    
    predictions = []
    
    for game in games:
        home_team = game['home_team']
        away_team = game['away_team']
        
        home_stats = team_stats.get(home_team, get_default_stats())
        away_stats = team_stats.get(away_team, get_default_stats())
        
        home_prob, away_prob = predict_game(home_stats, away_stats, model)
        
        predictions.append({
            'home_team': home_team,
            'away_team': away_team,
            'home_win_prob': home_prob,
            'away_win_prob': away_prob,
            'predicted_winner': home_team if home_prob > 0.5 else away_team,
            'confidence': max(home_prob, away_prob)
        })
    
    return predictions


def get_default_stats() -> dict:
    """Return default stats for unknown teams."""
    return {
        'win_pct': 0.5,
        'avg_points_for': 110,
        'avg_points_against': 110,
        'point_diff': 0
    }


@dataclass
class PredictionWithConfidence:
    """Prediction result with confidence intervals."""
    home_prob: float
    away_prob: float
    home_ci_low: float
    home_ci_high: float
    away_ci_low: float
    away_ci_high: float
    confidence_level: float  # e.g., 0.95 for 95% CI
    predicted_winner: str
    confidence: float
    
    def to_dict(self) -> Dict:
        return {
            'home_prob': self.home_prob,
            'away_prob': self.away_prob,
            'home_ci_low': self.home_ci_low,
            'home_ci_high': self.home_ci_high,
            'away_ci_low': self.away_ci_low,
            'away_ci_high': self.away_ci_high,
            'confidence_level': self.confidence_level,
            'predicted_winner': self.predicted_winner,
            'confidence': self.confidence
        }


def calculate_confidence_interval(
    prob: float,
    n_samples: int = 100,
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate confidence interval for a probability estimate.
    
    Uses the Wilson score interval which is more accurate for probabilities
    near 0 or 1 than the normal approximation.
    
    Args:
        prob: Point estimate of probability
        n_samples: Effective sample size (higher = narrower CI)
        confidence_level: Confidence level (default 95%)
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    from scipy import stats
    
    # Z-score for confidence level
    z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    
    # Wilson score interval
    denominator = 1 + z**2 / n_samples
    center = (prob + z**2 / (2 * n_samples)) / denominator
    spread = z * np.sqrt((prob * (1 - prob) + z**2 / (4 * n_samples)) / n_samples) / denominator
    
    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)
    
    return lower, upper


def predict_with_confidence(
    home_stats: dict,
    away_stats: dict,
    home_team: str = "Home",
    away_team: str = "Away",
    model=None,
    confidence_level: float = 0.95,
    n_bootstrap: int = 100
) -> PredictionWithConfidence:
    """
    Predict game outcome with confidence intervals.

    Uses the Wilson score interval to estimate prediction uncertainty.

    Args:
        home_stats: Home team statistics
        away_stats: Away team statistics
        home_team: Home team name
        away_team: Away team name
        model: Trained model (loads default if None)
        confidence_level: Confidence level for intervals (default 95%)
        n_bootstrap: Number of bootstrap samples (unused; kept for API compatibility)

    Returns:
        PredictionWithConfidence object with point estimates and intervals
    """
    # Get point estimate
    home_prob, away_prob = predict_game(home_stats, away_stats, model)
    
    # Calculate confidence intervals
    # Use effective sample size based on model training data
    # Higher for more confident predictions, lower for uncertain ones
    effective_n = int(100 * (1 - abs(home_prob - 0.5) * 2))  # 50-100 based on certainty
    effective_n = max(30, effective_n)  # Minimum 30 samples
    
    home_ci_low, home_ci_high = calculate_confidence_interval(
        home_prob, effective_n, confidence_level
    )
    away_ci_low, away_ci_high = calculate_confidence_interval(
        away_prob, effective_n, confidence_level
    )
    
    # Determine predicted winner
    predicted_winner = home_team if home_prob > 0.5 else away_team
    confidence = max(home_prob, away_prob)
    
    return PredictionWithConfidence(
        home_prob=home_prob,
        away_prob=away_prob,
        home_ci_low=home_ci_low,
        home_ci_high=home_ci_high,
        away_ci_low=away_ci_low,
        away_ci_high=away_ci_high,
        confidence_level=confidence_level,
        predicted_winner=predicted_winner,
        confidence=confidence
    )


def batch_predict_with_confidence(
    games: list,
    team_stats: dict,
    model=None,
    confidence_level: float = 0.95
) -> list:
    """
    Predict outcomes for multiple games with confidence intervals.
    
    Args:
        games: List of games with home_team and away_team
        team_stats: Dictionary of team statistics
        model: Trained model
        confidence_level: Confidence level for intervals
    
    Returns:
        List of PredictionWithConfidence objects
    """
    if model is None:
        model = load_model()
    
    predictions = []
    
    for game in games:
        home_team = game['home_team']
        away_team = game['away_team']
        
        home_stats = team_stats.get(home_team, get_default_stats())
        away_stats = team_stats.get(away_team, get_default_stats())
        
        pred = predict_with_confidence(
            home_stats, away_stats,
            home_team, away_team,
            model, confidence_level
        )
        predictions.append(pred)
    
    return predictions
