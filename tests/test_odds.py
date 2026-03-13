"""Tests for the odds API utilities."""

import pytest
from utils.odds import parse_odds, get_best_odds, get_demo_odds


class TestGetDemoOdds:
    def test_returns_list(self):
        odds = get_demo_odds()
        assert isinstance(odds, list)
        assert len(odds) > 0

    def test_each_game_has_required_fields(self):
        for game in get_demo_odds():
            assert "home_team" in game
            assert "away_team" in game
            assert "bookmakers" in game

    def test_home_away_teams_differ(self):
        for game in get_demo_odds():
            assert game["home_team"] != game["away_team"]


class TestParseOdds:
    def test_parse_returns_same_number_of_games(self):
        raw = get_demo_odds()
        parsed = parse_odds(raw)
        assert len(parsed) == len(raw)

    def test_parsed_game_has_bookmakers(self):
        parsed = parse_odds(get_demo_odds())
        for game in parsed:
            assert "bookmakers" in game
            assert isinstance(game["bookmakers"], list)

    def test_parsed_game_preserves_teams(self):
        raw = get_demo_odds()
        parsed = parse_odds(raw)
        for original, result in zip(raw, parsed):
            assert result["home_team"] == original["home_team"]
            assert result["away_team"] == original["away_team"]

    def test_empty_input_returns_empty_list(self):
        assert parse_odds([]) == []


class TestGetBestOdds:
    def test_returns_one_entry_per_game(self):
        raw = get_demo_odds()
        parsed = parse_odds(raw)
        best = get_best_odds(parsed)
        assert len(best) == len(raw)

    def test_each_entry_has_home_and_away_odds(self):
        best = get_best_odds(parse_odds(get_demo_odds()))
        for game in best:
            assert "home_odds" in game
            assert "away_odds" in game

    def test_best_odds_for_lakers_warriors(self):
        # Demo game 1: Lakers (-150/-145) vs Warriors (+130/+125)
        # Best home (Lakers): -145 > -150
        # Best away (Warriors): +130 > +125
        best = get_best_odds(parse_odds(get_demo_odds()))
        lal_game = next(
            g for g in best if g["home_team"] == "Los Angeles Lakers"
        )
        assert lal_game["home_odds"]["odds"] == -145
        assert lal_game["away_odds"]["odds"] == 130
