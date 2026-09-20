"""Tests for Closing Line Value (CLV) helpers."""

import math

import pytest

from utils.clv import (
    annotate_clv,
    attach_synthetic_closes,
    beat_close,
    clv_chart_records,
    clv_probability_points,
    filter_bets_beating_close,
    simulate_synthetic_clv_bets,
    summarize_clv,
)
from utils.kelly import american_to_implied_prob


def test_clv_probability_points_positive_when_market_moves_toward_pick():
    # Bet dog at +150 (impl 40%); close +130 (impl ~43.5%) → positive CLV.
    clv = clv_probability_points(150, 130)
    expected = (american_to_implied_prob(130) - american_to_implied_prob(150)) * 100
    assert math.isclose(clv, expected)
    assert clv > 0


def test_clv_probability_points_favorite_move_and_zero_on_equal_lines():
    # Open −150, close −180: market moved toward favorite → beat close.
    assert clv_probability_points(-150, -180) > 0
    assert math.isclose(clv_probability_points(-110, -110), 0.0)


def test_beat_close_uses_american_odds_ordering():
    assert beat_close(150, 130) is True
    assert beat_close(-110, -150) is True
    assert beat_close(-180, -150) is False
    assert beat_close(100, 100) is False


def test_summarize_clv_empty_and_mixed_slice():
    empty = summarize_clv([])
    assert empty["n_bets"] == 0
    assert empty["mean_clv_pts"] == 0.0
    assert empty["beat_close_rate"] == 0.0

    bets = [
        {"open_odds": 150, "close_odds": 130, "odds_source": "synthetic"},
        {"open_odds": -120, "close_odds": -110, "odds_source": "synthetic"},
        {"open_odds": 100, "close_odds": 100, "odds_source": "synthetic"},
    ]
    summary = summarize_clv(bets)
    assert summary["n_bets"] == 3
    assert summary["n_beat_close"] == 1
    assert math.isclose(summary["beat_close_rate"], 1 / 3)
    assert summary["source"] == "synthetic"
    # One beat (+), one lose (−), one flat → mean near first bet's CLV / 3 plus second.
    assert summary["mean_clv_pts"] == pytest.approx(
        (
            clv_probability_points(150, 130)
            + clv_probability_points(-120, -110)
            + 0.0
        )
        / 3
    )


def test_filter_bets_beating_close():
    bets = [
        {"team": "A", "open_odds": 150, "close_odds": 130},
        {"team": "B", "open_odds": -150, "close_odds": -130},
    ]
    kept = filter_bets_beating_close(bets, require_beat_close=True)
    assert [b["team"] for b in kept] == ["A"]

    all_rows = filter_bets_beating_close(bets, require_beat_close=False)
    assert len(all_rows) == 2
    assert "clv_pts" in all_rows[0]


def test_simulate_synthetic_clv_bets_is_labeled_and_reproducible():
    a = simulate_synthetic_clv_bets(n_bets=20, seed=42)
    b = simulate_synthetic_clv_bets(n_bets=20, seed=42)
    assert a == b
    assert len(a) == 20
    assert all(row["odds_source"] == "synthetic" for row in a)
    assert all("clv_pts" in row and "beat_close" in row for row in a)

    summary = summarize_clv(a)
    assert summary["n_bets"] == 20
    assert 0.0 <= summary["beat_close_rate"] <= 1.0


def test_attach_synthetic_closes_labels_source():
    value_bets = [
        {"team": "Lakers", "odds": -145, "edge": 4.0},
        {"team": "Heat", "odds": 160, "edge": 5.5},
    ]
    annotated = attach_synthetic_closes(value_bets, seed=1)
    assert len(annotated) == 2
    assert all(r["odds_source"] == "synthetic_close" for r in annotated)
    assert all(r["open_odds"] == value_bets[i]["odds"] for i, r in enumerate(annotated))
    assert all("close_odds" in r and "clv_pts" in r for r in annotated)


def test_annotate_clv_and_chart_records():
    row = annotate_clv({"team": "X", "open_odds": 120, "close_odds": 105})
    assert row["beat_close"] is True
    assert row["clv_pts"] > 0

    chart = clv_chart_records(
        [
            {"open_odds": 120, "close_odds": 105, "team": "X"},
            {"open_odds": -110, "close_odds": -110, "team": "Y"},
        ]
    )
    assert chart[0]["bet_index"] == 1
    assert chart[1]["bet_index"] == 2
    assert math.isclose(chart[1]["cumulative_mean_clv"], chart[0]["clv_pts"] / 2)
