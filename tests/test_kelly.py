"""Tests for the Kelly Criterion and odds utilities."""

import pytest
from utils.kelly import (
    american_to_decimal,
    decimal_to_implied_prob,
    american_to_implied_prob,
    kelly_criterion,
    calculate_edge,
    find_value_bets,
)


class TestAmericanToDecimal:
    def test_positive_odds(self):
        # +200 → 3.0
        assert american_to_decimal(200) == pytest.approx(3.0)

    def test_negative_odds(self):
        # -150 → 1.6667
        assert american_to_decimal(-150) == pytest.approx(1.6667, rel=1e-3)

    def test_even_odds(self):
        # +100 → 2.0
        assert american_to_decimal(100) == pytest.approx(2.0)


class TestDecimalToImpliedProb:
    def test_even_money(self):
        assert decimal_to_implied_prob(2.0) == pytest.approx(0.5)

    def test_heavy_favourite(self):
        # 1.25 decimal → 80% implied
        assert decimal_to_implied_prob(1.25) == pytest.approx(0.8)

    def test_large_underdog(self):
        # 4.0 decimal → 25% implied
        assert decimal_to_implied_prob(4.0) == pytest.approx(0.25)


class TestAmericanToImpliedProb:
    def test_negative_american(self):
        # -200 → ~66.7%
        assert american_to_implied_prob(-200) == pytest.approx(0.6667, rel=1e-3)

    def test_positive_american(self):
        # +200 → ~33.3%
        assert american_to_implied_prob(200) == pytest.approx(0.3333, rel=1e-3)


class TestKellyCriterion:
    def test_positive_edge_returns_positive_bet(self):
        # 60% win prob on 2.0 decimal odds → positive Kelly
        bet = kelly_criterion(0.6, 2.0, fraction=1.0)
        assert bet > 0

    def test_no_edge_returns_zero(self):
        # 50% win prob on 2.0 odds → zero edge
        bet = kelly_criterion(0.5, 2.0, fraction=1.0)
        assert bet == pytest.approx(0.0)

    def test_negative_edge_returns_zero(self):
        # 40% win prob on 2.0 odds → negative Kelly, clamped to 0
        bet = kelly_criterion(0.4, 2.0, fraction=1.0)
        assert bet == pytest.approx(0.0)

    def test_fractional_kelly(self):
        full = kelly_criterion(0.6, 2.0, fraction=1.0)
        quarter = kelly_criterion(0.6, 2.0, fraction=0.25)
        assert quarter == pytest.approx(full * 0.25)

    def test_capped_at_one(self):
        # Extremely high win probability should not bet more than 100%
        bet = kelly_criterion(0.99, 100.0, fraction=1.0)
        assert bet <= 1.0


class TestCalculateEdge:
    def test_positive_edge(self):
        edge = calculate_edge(0.6, 0.5)
        assert edge == pytest.approx(10.0)

    def test_negative_edge(self):
        edge = calculate_edge(0.4, 0.5)
        assert edge == pytest.approx(-10.0)

    def test_zero_edge(self):
        edge = calculate_edge(0.5, 0.5)
        assert edge == pytest.approx(0.0)


class TestFindValueBets:
    def _make_predictions(self):
        return [
            {"team": "Team A", "game": "A vs B", "model_prob": 0.65, "american_odds": -110},
            {"team": "Team B", "game": "A vs B", "model_prob": 0.40, "american_odds": -110},
            {"team": "Team C", "game": "C vs D", "model_prob": 0.60, "american_odds": +150},
        ]

    def test_filters_by_min_edge(self):
        preds = self._make_predictions()
        value = find_value_bets(preds, min_edge=5.0)
        for bet in value:
            assert bet["edge"] >= 5.0

    def test_sorted_by_edge_descending(self):
        preds = self._make_predictions()
        value = find_value_bets(preds, min_edge=0.0)
        edges = [b["edge"] for b in value]
        assert edges == sorted(edges, reverse=True)

    def test_result_includes_kelly(self):
        preds = self._make_predictions()
        value = find_value_bets(preds, min_edge=0.0)
        for bet in value:
            assert "kelly_bet" in bet
            assert bet["kelly_bet"] >= 0

    def test_empty_when_no_value(self):
        preds = [{"team": "X", "game": "X vs Y", "model_prob": 0.40, "american_odds": -150}]
        assert find_value_bets(preds, min_edge=5.0) == []
