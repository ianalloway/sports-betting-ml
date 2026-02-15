"""Prediction functions for NBA game outcomes."""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from pathlib import Path

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
