"""Tests for feature engineering utilities."""

import pytest
import pandas as pd
import numpy as np
from data.features import (
    calculate_team_stats,
    get_default_stats,
    create_game_features,
    prepare_training_data,
    NBA_TEAMS,
    TEAM_ABBREVS,
)


def _make_games_df():
    """Create a small synthetic games DataFrame for testing."""
    np.random.seed(0)
    teams = ["Boston Celtics", "Miami Heat", "Denver Nuggets", "Phoenix Suns"]
    records = []
    for i in range(40):
        home = teams[i % len(teams)]
        away = teams[(i + 1) % len(teams)]
        records.append({
            "date": f"2025-01-{i + 1:02d}",
            "home_team": home,
            "away_team": away,
            "home_score": 105 + np.random.randint(-10, 20),
            "away_score": 103 + np.random.randint(-10, 20),
        })
    return pd.DataFrame(records)


class TestGetDefaultStats:
    def test_returns_dict(self):
        assert isinstance(get_default_stats(), dict)

    def test_win_pct_is_half(self):
        assert get_default_stats()["win_pct"] == pytest.approx(0.5)

    def test_games_played_is_zero(self):
        assert get_default_stats()["games_played"] == 0


class TestCalculateTeamStats:
    def test_returns_default_for_unknown_team(self):
        df = _make_games_df()
        stats = calculate_team_stats(df, "Unknown Team")
        assert stats == get_default_stats()

    def test_win_pct_in_valid_range(self):
        df = _make_games_df()
        stats = calculate_team_stats(df, "Boston Celtics")
        assert 0.0 <= stats["win_pct"] <= 1.0

    def test_games_played_bounded_by_n_games(self):
        df = _make_games_df()
        n = 5
        stats = calculate_team_stats(df, "Boston Celtics", n_games=n)
        assert stats["games_played"] <= n

    def test_point_diff_matches_points_for_minus_against(self):
        df = _make_games_df()
        stats = calculate_team_stats(df, "Boston Celtics")
        expected = stats["avg_points_for"] - stats["avg_points_against"]
        assert stats["point_diff"] == pytest.approx(expected)


class TestCreateGameFeatures:
    def _home(self):
        return {
            "win_pct": 0.65,
            "avg_points_for": 115,
            "avg_points_against": 108,
            "point_diff": 7,
        }

    def _away(self):
        return {
            "win_pct": 0.45,
            "avg_points_for": 109,
            "avg_points_against": 112,
            "point_diff": -3,
        }

    def test_returns_eleven_features(self):
        features = create_game_features("Home", "Away", self._home(), self._away())
        assert len(features) == 11

    def test_home_advantage_flag_is_one(self):
        features = create_game_features("Home", "Away", self._home(), self._away())
        assert features["home_advantage"] == 1

    def test_diff_features_computed_correctly(self):
        features = create_game_features("Home", "Away", self._home(), self._away())
        assert features["win_pct_diff"] == pytest.approx(0.65 - 0.45)
        assert features["point_diff_diff"] == pytest.approx(7 - (-3))


class TestPrepareTrainingData:
    def test_returns_tuple_of_dataframe_and_series(self):
        df = _make_games_df()
        X, y = prepare_training_data(df)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_feature_and_target_lengths_match(self):
        df = _make_games_df()
        X, y = prepare_training_data(df)
        assert len(X) == len(y)

    def test_target_is_binary(self):
        df = _make_games_df()
        _, y = prepare_training_data(df)
        assert set(y.unique()).issubset({0, 1})


class TestNbaTeamsMappings:
    def test_all_thirty_teams_present(self):
        assert len(NBA_TEAMS) == 30

    def test_reverse_mapping_round_trips(self):
        for abbrev, name in NBA_TEAMS.items():
            assert TEAM_ABBREVS[name] == abbrev

    def test_boston_celtics_abbreviation(self):
        assert NBA_TEAMS["BOS"] == "Boston Celtics"
