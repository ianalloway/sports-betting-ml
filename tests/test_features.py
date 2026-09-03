"""Regression tests for issue #28: leakage and train/serve skew."""

import pandas as pd

from data.features import get_default_stats, prepare_training_data
from model.predict import create_features
from model.train import create_sample_data


def _small_schedule(n_games: int = 60) -> pd.DataFrame:
    """Deterministic mini-season between four teams."""
    teams = ['A', 'B', 'C', 'D']
    games = []
    for i in range(n_games):
        home = teams[i % 4]
        away = teams[(i + 1) % 4]
        games.append({
            'date': (pd.Timestamp('2024-10-01') + pd.Timedelta(days=i)).strftime('%Y-%m-%d'),
            'home_team': home,
            'away_team': away,
            'home_score': 100 + (i * 7) % 20,
            'away_score': 100 + (i * 11) % 20,
        })
    return pd.DataFrame(games)


def test_features_are_point_in_time():
    """Changing a future game's result must not change earlier features.

    This is the core no-leakage property: the feature row for game i may
    only depend on games 0..i-1. If it depended on later games, a random
    train/test split would let test outcomes leak into training features.
    """
    df = _small_schedule()
    X_before, y_before = prepare_training_data(df)

    # Flip the result of the final game only.
    df_mut = df.copy()
    df_mut.loc[df_mut.index[-1], 'home_score'] = 200

    X_after, y_after = prepare_training_data(df_mut)

    # All feature rows except the last describe games before the mutated
    # one, so they must be identical.
    pd.testing.assert_frame_equal(X_before.iloc[:-1], X_after.iloc[:-1])
    # Only the final target may differ.
    assert y_before.iloc[:-1].equals(y_after.iloc[:-1])


def test_train_and_serve_share_feature_definition():
    """Serving features must match training feature names and order."""
    df = _small_schedule()
    X_train, _ = prepare_training_data(df)

    serving = create_features(get_default_stats(), get_default_stats())

    assert list(serving.keys()) == list(X_train.columns)


def test_sample_data_dates_are_valid_and_ordered():
    """Sample dates must parse and be non-decreasing for time splits."""
    df = create_sample_data()
    dates = pd.to_datetime(df['date'], format='%Y-%m-%d')  # raises if invalid
    assert dates.is_monotonic_increasing


def test_create_features_fills_missing_stats():
    """Partial stat dicts (e.g. early season) must not raise."""
    features = create_features({'win_pct': 0.7}, {})
    assert features['home_win_pct'] == 0.7
    assert features['away_win_pct'] == 0.5
    assert features['home_advantage'] == 1
