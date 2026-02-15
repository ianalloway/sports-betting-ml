"""Feature engineering for NBA game prediction."""

import pandas as pd
import numpy as np
from typing import Tuple


def calculate_team_stats(games_df: pd.DataFrame, team: str, n_games: int = 10) -> dict:
    """
    Calculate rolling statistics for a team based on recent games.
    
    Args:
        games_df: DataFrame with game results
        team: Team name
        n_games: Number of recent games to consider
    
    Returns:
        Dictionary of team statistics
    """
    # Filter games where team played
    team_games = games_df[
        (games_df['home_team'] == team) | (games_df['away_team'] == team)
    ].tail(n_games)
    
    if len(team_games) == 0:
        return get_default_stats()
    
    # Calculate wins
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
        'point_diff': np.mean(points_for) - np.mean(points_against),
        'games_played': len(team_games)
    }


def get_default_stats() -> dict:
    """Return default stats for teams with no data."""
    return {
        'win_pct': 0.5,
        'avg_points_for': 110,
        'avg_points_against': 110,
        'point_diff': 0,
        'games_played': 0
    }


def create_game_features(
    home_team: str,
    away_team: str,
    home_stats: dict,
    away_stats: dict
) -> dict:
    """
    Create features for a single game prediction.
    
    Args:
        home_team: Home team name
        away_team: Away team name
        home_stats: Home team statistics
        away_stats: Away team statistics
    
    Returns:
        Dictionary of features for the game
    """
    return {
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
        'home_advantage': 1  # Home court advantage indicator
    }


def prepare_training_data(games_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare training data from historical games.
    
    Args:
        games_df: DataFrame with historical game results
    
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    features_list = []
    targets = []
    
    # Sort by date
    games_df = games_df.sort_values('date').reset_index(drop=True)
    
    # Need at least 20 games before we can make predictions
    for i in range(20, len(games_df)):
        game = games_df.iloc[i]
        historical = games_df.iloc[:i]
        
        home_stats = calculate_team_stats(historical, game['home_team'])
        away_stats = calculate_team_stats(historical, game['away_team'])
        
        features = create_game_features(
            game['home_team'],
            game['away_team'],
            home_stats,
            away_stats
        )
        
        features_list.append(features)
        targets.append(1 if game['home_score'] > game['away_score'] else 0)
    
    return pd.DataFrame(features_list), pd.Series(targets)


# NBA team name mappings for consistency
NBA_TEAMS = {
    'ATL': 'Atlanta Hawks',
    'BOS': 'Boston Celtics',
    'BKN': 'Brooklyn Nets',
    'CHA': 'Charlotte Hornets',
    'CHI': 'Chicago Bulls',
    'CLE': 'Cleveland Cavaliers',
    'DAL': 'Dallas Mavericks',
    'DEN': 'Denver Nuggets',
    'DET': 'Detroit Pistons',
    'GSW': 'Golden State Warriors',
    'HOU': 'Houston Rockets',
    'IND': 'Indiana Pacers',
    'LAC': 'Los Angeles Clippers',
    'LAL': 'Los Angeles Lakers',
    'MEM': 'Memphis Grizzlies',
    'MIA': 'Miami Heat',
    'MIL': 'Milwaukee Bucks',
    'MIN': 'Minnesota Timberwolves',
    'NOP': 'New Orleans Pelicans',
    'NYK': 'New York Knicks',
    'OKC': 'Oklahoma City Thunder',
    'ORL': 'Orlando Magic',
    'PHI': 'Philadelphia 76ers',
    'PHX': 'Phoenix Suns',
    'POR': 'Portland Trail Blazers',
    'SAC': 'Sacramento Kings',
    'SAS': 'San Antonio Spurs',
    'TOR': 'Toronto Raptors',
    'UTA': 'Utah Jazz',
    'WAS': 'Washington Wizards'
}

# Reverse mapping
TEAM_ABBREVS = {v: k for k, v in NBA_TEAMS.items()}
