"""Tests for odds parsing and value-bet math."""

import math

import pytest

from utils.kelly import (
    american_to_decimal,
    american_to_implied_prob,
    calculate_edge,
    find_value_bets,
    kelly_criterion,
)
from utils.odds import get_best_h2h_odds_for_game, get_best_odds, parse_odds


def test_american_odds_conversion_positive_negative_and_zero():
    assert american_to_decimal(150) == 2.5
    assert american_to_decimal(-200) == 1.5

    with pytest.raises(ValueError, match="cannot be zero"):
        american_to_decimal(0)


def test_implied_probability_and_edge_math():
    assert math.isclose(american_to_implied_prob(150), 0.4)
    assert math.isclose(american_to_implied_prob(-200), 2 / 3)
    assert math.isclose(calculate_edge(0.55, 0.50), 5.0)


def test_kelly_criterion_clamps_no_edge_and_fractional_stake():
    assert kelly_criterion(0.40, 2.0) == 0

    # Full Kelly at +100 and 60% true win probability is 20% bankroll;
    # quarter Kelly should recommend 5%.
    assert math.isclose(kelly_criterion(0.60, 2.0, fraction=0.25), 0.05)


def test_kelly_criterion_returns_zero_when_decimal_odds_have_no_payout():
    assert kelly_criterion(0.99, 1.0) == 0
    assert kelly_criterion(0.99, 0.5) == 0


def test_find_value_bets_filters_and_sorts_by_edge():
    predictions = [
        {"team": "A", "game": "A-B", "model_prob": 0.55, "american_odds": 100},
        {"team": "B", "game": "C-D", "model_prob": 0.45, "american_odds": 100},
        {"team": "C", "game": "E-F", "model_prob": 0.55, "american_odds": 150},
    ]

    bets = find_value_bets(predictions, min_edge=3.0)

    assert [bet["team"] for bet in bets] == ["C", "A"]
    assert bets[0]["edge"] > bets[1]["edge"]
    assert bets[0]["kelly_bet"] > 0


def test_parse_odds_skips_malformed_games_and_normalizes_markets():
    raw_games = [
        {
            "id": "game-1",
            "home_team": "Home",
            "away_team": "Away",
            "commence_time": "2026-02-16T00:00:00Z",
            "bookmakers": [
                {
                    "title": "Book A",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Home", "price": -110},
                                {"name": "Away", "price": 105},
                            ],
                        }
                    ],
                }
            ],
        },
        {"id": "missing-teams"},
        "not-a-dict",
    ]

    parsed = parse_odds(raw_games)

    assert len(parsed) == 1
    assert parsed[0]["bookmakers"][0]["markets"]["h2h"]["Home"]["price"] == -110


def test_get_best_odds_selects_best_available_price_per_team():
    parsed = [
        {
            "home_team": "Home",
            "away_team": "Away",
            "commence_time": "2026-02-16T00:00:00Z",
            "bookmakers": [
                {
                    "name": "Book A",
                    "markets": {
                        "h2h": {
                            "Home": {"price": -120},
                            "Away": {"price": 110},
                        }
                    },
                },
                {
                    "name": "Book B",
                    "markets": {
                        "h2h": {
                            "Home": {"price": -105},
                            "Away": {"price": 125},
                        }
                    },
                },
            ],
        }
    ]

    best = get_best_odds(parsed)

    assert best == [
        {
            "home_team": "Home",
            "away_team": "Away",
            "commence_time": "2026-02-16T00:00:00Z",
            "home_odds": {"odds": -105, "book": "Book B"},
            "away_odds": {"odds": 125, "book": "Book B"},
        }
    ]


def test_get_best_odds_leaves_missing_prices_as_none():
    parsed = [
        {
            "home_team": "Home",
            "away_team": "Away",
            "commence_time": "2026-02-16T00:00:00Z",
            "bookmakers": [
                {
                    "name": "Book A",
                    "markets": {"h2h": {"Home": {"price": -110}}},
                }
            ],
        }
    ]

    best = get_best_odds(parsed)
    assert best[0]["home_odds"] == {"odds": -110, "book": "Book A"}
    assert best[0]["away_odds"] == {"odds": None, "book": None}


def test_get_best_h2h_odds_for_game_does_not_override_real_negative_lines():
    game = {
        "home_team": "Home",
        "away_team": "Away",
        "bookmakers": [
            {
                "name": "Book A",
                "markets": {
                    "h2h": {
                        "Home": {"price": -220},
                        "Away": {"price": 180},
                    }
                },
            }
        ],
    }

    home_odds, away_odds = get_best_h2h_odds_for_game(game)
    assert home_odds == -220
    assert away_odds == 180


def test_get_best_h2h_odds_for_game_uses_defaults_only_when_team_price_missing():
    game = {
        "home_team": "Home",
        "away_team": "Away",
        "bookmakers": [
            {
                "name": "Book A",
                "markets": {
                    "h2h": {
                        "Home": {"price": -140},
                    }
                },
            }
        ],
    }

    home_odds, away_odds = get_best_h2h_odds_for_game(game)
    assert home_odds == -140
    assert away_odds == 130


def test_get_best_h2h_odds_for_game_skips_non_numeric_and_non_dict_outcomes():
    game = {
        "home_team": "Home",
        "away_team": "Away",
        "bookmakers": [
            {"name": "Bad", "markets": {"h2h": {"Home": "not-a-dict", "Away": {"price": True}}}},
            {"name": "Good", "markets": {"h2h": {"Home": {"price": -110}, "Away": {"price": 100}}}},
        ],
    }

    home_odds, away_odds = get_best_h2h_odds_for_game(game)
    assert home_odds == -110
    assert away_odds == 100
