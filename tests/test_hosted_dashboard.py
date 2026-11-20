from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from model.train import train_model
from utils import odds


def test_cloud_startup_without_model_or_odds_key(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "absent.json"))
    monkeypatch.setattr(odds, "API_KEY", "")
    script = Path(__file__).resolve().parents[1] / "app.py"

    with patch("model.train.train_model", wraps=train_model) as fit:
        app = AppTest.from_file(str(script), default_timeout=60).run()
        assert not app.exception
        assert "Example NBA Games" in [header.value for header in app.header]
        probabilities = [m.value for m in app.metric if m.label == "Model Probability"]
        assert len(probabilities) >= 2 * len(odds.get_demo_odds())
        assert any("synthetic" in caption.value for caption in app.caption)

        # A visitor adjusting settings reuses the fitted model.
        bankroll = next(widget for widget in app.number_input if widget.label == "Bankroll ($)")
        bankroll.set_value(2000).run()
        assert not app.exception
        fit.assert_called_once()


def test_historical_forecast_selector_matches_archived_results(monkeypatch):
    import json

    monkeypatch.setattr(odds, "API_KEY", "")
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / "docs/benchmarks/nba-2015.json").read_text())
    app = AppTest.from_file(str(root / "app.py"), default_timeout=90).run()
    for label, key in (("XGBoost", "xgboost"), ("Training home-win rate", "home_rate_baseline")):
        selector = next(widget for widget in app.selectbox if widget.label == "Historical forecast")
        if selector.value != label:
            selector.set_value(label).run()
        assert not app.exception
        values = {metric.label: metric.value for metric in app.metric}
        assert values["Historical Brier score"] == f"{report[key]['brier_score']:.3f}"
        assert values["Expected calibration error"] == f"{report[key]['ece_10_equal_width_bins']:.1%}"
        assert any("1,149 held-out games" in item.value for item in app.caption)
