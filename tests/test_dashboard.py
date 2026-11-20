"""Exercise the offline Streamlit dashboard through its public application path."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from streamlit import cache_data, cache_resource
from streamlit.testing.v1 import AppTest

from model import predict
from utils import odds


@pytest.fixture(autouse=True)
def isolated_dashboard(monkeypatch):
    # Model training at startup is covered separately by the hosted-dashboard tests.
    model = Mock()
    model.predict_proba.return_value = [[0.4, 0.6]]
    monkeypatch.setattr(predict, "load_model", lambda: model)
    cache_data.clear()
    cache_resource.clear()
    yield
    cache_data.clear()
    cache_resource.clear()


def test_demo_games_and_value_bets_share_one_prediction_and_label_the_source(monkeypatch):
    monkeypatch.setattr(odds, "API_KEY", "")
    app = Path(__file__).resolve().parents[1] / "app.py"
    with patch.object(predict, "predict_game", wraps=predict.predict_game) as predict_game:
        page = AppTest.from_file(str(app)).run(timeout=30)
    assert not page.exception
    assert predict_game.call_count == len(odds.get_demo_odds())
    assert any("sample odds" in notice.value for notice in page.tabs[0].info)
    assert any("sample odds" in notice.value for notice in page.tabs[1].info)


def test_live_game_without_a_price_has_no_bet_or_invented_moneyline(monkeypatch):
    game = {"id": "sample", "home_team": "Home", "away_team": "Away", "bookmakers": []}
    monkeypatch.setattr(odds, "get_nba_odds_with_source", lambda: ([game], "live"))
    app = Path(__file__).resolve().parents[1] / "app.py"
    page = AppTest.from_file(str(app)).run(timeout=30)
    assert not page.exception
    assert sum(item.value == "Moneyline unavailable" for item in page.tabs[0].caption) == 2
    assert all(item.label != "Best Odds" for item in page.tabs[0].metric)
    assert all(item.label != "Total Suggested Bet" for item in page.tabs[1].metric)
