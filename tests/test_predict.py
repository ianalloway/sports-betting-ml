"""Tests for prediction functions."""

import pytest
import numpy as np
from model.predict import (
    predict_heuristic,
    create_features,
    get_default_stats,
    calculate_confidence_interval,
    predict_with_confidence,
    PredictionWithConfidence,
)


HOME_STATS = {
    "win_pct": 0.65,
    "avg_points_for": 115,
    "avg_points_against": 108,
    "point_diff": 7,
}

AWAY_STATS = {
    "win_pct": 0.45,
    "avg_points_for": 109,
    "avg_points_against": 112,
    "point_diff": -3,
}


class TestPredictHeuristic:
    def test_stronger_team_higher_probability(self):
        home_prob, away_prob = predict_heuristic(HOME_STATS, AWAY_STATS)
        assert home_prob > away_prob

    def test_probabilities_sum_to_one(self):
        home_prob, away_prob = predict_heuristic(HOME_STATS, AWAY_STATS)
        assert home_prob + away_prob == pytest.approx(1.0)

    def test_probabilities_in_valid_range(self):
        home_prob, away_prob = predict_heuristic(HOME_STATS, AWAY_STATS)
        assert 0.0 <= home_prob <= 1.0
        assert 0.0 <= away_prob <= 1.0

    def test_even_teams_near_fifty_fifty(self):
        equal_stats = get_default_stats()
        home_prob, away_prob = predict_heuristic(equal_stats, equal_stats)
        # Home advantage pushes home slightly above 0.5
        assert home_prob > 0.5
        assert home_prob < 0.6


class TestCreateFeatures:
    def test_returns_dict_with_expected_keys(self):
        features = create_features(HOME_STATS, AWAY_STATS)
        expected_keys = {
            "home_win_pct", "away_win_pct",
            "home_ppg", "away_ppg",
            "home_opp_ppg", "away_opp_ppg",
            "home_point_diff", "away_point_diff",
            "win_pct_diff", "point_diff_diff",
            "home_advantage",
        }
        assert set(features.keys()) == expected_keys

    def test_win_pct_diff_computed_correctly(self):
        features = create_features(HOME_STATS, AWAY_STATS)
        expected = HOME_STATS["win_pct"] - AWAY_STATS["win_pct"]
        assert features["win_pct_diff"] == pytest.approx(expected)

    def test_home_advantage_is_one(self):
        features = create_features(HOME_STATS, AWAY_STATS)
        assert features["home_advantage"] == 1

    def test_uses_defaults_for_missing_keys(self):
        features = create_features({}, {})
        assert features["home_win_pct"] == pytest.approx(0.5)
        assert features["home_ppg"] == pytest.approx(110)


class TestGetDefaultStats:
    def test_returns_dict(self):
        stats = get_default_stats()
        assert isinstance(stats, dict)

    def test_win_pct_is_half(self):
        assert get_default_stats()["win_pct"] == pytest.approx(0.5)

    def test_point_diff_is_zero(self):
        assert get_default_stats()["point_diff"] == pytest.approx(0.0)


class TestCalculateConfidenceInterval:
    def test_returns_tuple_of_two(self):
        result = calculate_confidence_interval(0.6)
        assert len(result) == 2

    def test_lower_less_than_upper(self):
        low, high = calculate_confidence_interval(0.6)
        assert low < high

    def test_bounds_contain_point_estimate(self):
        prob = 0.6
        low, high = calculate_confidence_interval(prob)
        assert low <= prob <= high

    def test_bounds_within_zero_one(self):
        for prob in [0.01, 0.5, 0.99]:
            low, high = calculate_confidence_interval(prob)
            assert 0.0 <= low <= 1.0
            assert 0.0 <= high <= 1.0

    def test_larger_n_produces_narrower_interval(self):
        low_n_low, low_n_high = calculate_confidence_interval(0.6, n_samples=30)
        high_n_low, high_n_high = calculate_confidence_interval(0.6, n_samples=300)
        assert (high_n_high - high_n_low) < (low_n_high - low_n_low)


class TestPredictWithConfidence:
    def test_returns_prediction_with_confidence(self):
        result = predict_with_confidence(HOME_STATS, AWAY_STATS, "Home", "Away")
        assert isinstance(result, PredictionWithConfidence)

    def test_probabilities_sum_to_one(self):
        result = predict_with_confidence(HOME_STATS, AWAY_STATS)
        assert result.home_prob + result.away_prob == pytest.approx(1.0)

    def test_ci_contains_point_estimate(self):
        result = predict_with_confidence(HOME_STATS, AWAY_STATS)
        assert result.home_ci_low <= result.home_prob <= result.home_ci_high
        assert result.away_ci_low <= result.away_prob <= result.away_ci_high

    def test_predicted_winner_matches_higher_probability(self):
        result = predict_with_confidence(HOME_STATS, AWAY_STATS, "Home", "Away")
        if result.home_prob > result.away_prob:
            assert result.predicted_winner == "Home"
        else:
            assert result.predicted_winner == "Away"

    def test_to_dict_has_all_fields(self):
        result = predict_with_confidence(HOME_STATS, AWAY_STATS)
        d = result.to_dict()
        for key in ["home_prob", "away_prob", "home_ci_low", "home_ci_high",
                    "away_ci_low", "away_ci_high", "confidence_level",
                    "predicted_winner", "confidence"]:
            assert key in d
