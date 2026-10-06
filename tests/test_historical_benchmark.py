"""Offline contract tests. Fixtures are invented games, not benchmark evidence."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from data import historical
from model.benchmark import date_split, historical_features, probability_metrics, run_benchmark


def schedule():
    rows = []
    for year in [2014, 2015]:
        for day in range(20):
            for pair, (home, away) in enumerate([('A', 'B'), ('C', 'D')]):
                rows.append({
                    'game_id': f'{year}-{day}-{pair}', 'season': year,
                    'date': pd.Timestamp(f'{year - 1}-10-01') + pd.Timedelta(days=day),
                    'home_team': home, 'away_team': away,
                    'home_score': 90 + day % 10, 'away_score': 95 + pair,
                })
    return pd.DataFrame(rows)


def source_fixture(tmp_path):
    rows = []
    for location in ['H', 'A', 'N']:
        rows.append(dict(game_id='g1', lg_id='NBA', year_id=2015, date_game='11/1/2014',
                         is_playoffs=0, team_id='A', opp_id='B', pts=100, opp_pts=90,
                         game_location=location, elo_n=99999, forecast=0.99))
    rows.append({**rows[0], 'game_id': 'playoff', 'is_playoffs': 1})
    rows.append({**rows[0], 'game_id': 'aba', 'lg_id': 'ABA'})
    rows.append({**rows[0], 'game_id': 'old', 'year_id': 2009})
    path = tmp_path / 'fixture.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_parser_selects_only_regular_home_nba_rows_and_allowlisted_fields(tmp_path):
    games = historical.parse_games(source_fixture(tmp_path))
    assert games['game_id'].tolist() == ['g1']
    assert games.iloc[0]['date'] == pd.Timestamp('2014-11-01')
    assert list(games.columns) == ['game_id', 'season', 'date', 'home_team', 'away_team', 'home_score', 'away_score']


@pytest.mark.parametrize('failure', ['duplicate', 'tie', 'missing', 'bad_date', 'negative', 'self'])
def test_parser_rejects_invalid_games(tmp_path, failure):
    path = source_fixture(tmp_path)
    frame = pd.read_csv(path)
    if failure == 'duplicate':
        frame = pd.concat([frame, frame.iloc[[0]]])
    elif failure == 'tie':
        frame.loc[0, 'opp_pts'] = 100
    elif failure == 'missing':
        frame.loc[0, 'team_id'] = None
    elif failure == 'bad_date':
        frame.loc[0, 'date_game'] = 'not-a-date'
    elif failure == 'negative':
        frame.loc[0, 'pts'] = -1
    else:
        frame.loc[0, 'opp_id'] = 'A'
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        historical.parse_games(path)


def test_verified_cache_and_download(monkeypatch, tmp_path):
    content = b'offline test bytes'
    monkeypatch.setattr(historical, 'SOURCE_SHA256', hashlib.sha256(content).hexdigest())
    calls = []

    def get(url, timeout):
        calls.append((url, timeout))
        return SimpleNamespace(content=content, raise_for_status=lambda: None)

    monkeypatch.setattr(historical.requests, 'get', get)
    path = tmp_path / 'raw' / 'archive.csv'
    assert historical.verified_archive(path)['sha256'] == historical.SOURCE_SHA256
    assert historical.verified_archive(path)['bytes'] == len(content)
    assert len(calls) == 1
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        historical.verified_archive(path)
    assert len(calls) == 1


def test_bad_download_is_never_cached(monkeypatch, tmp_path):
    monkeypatch.setattr(historical.requests, 'get', lambda *a, **k: SimpleNamespace(
        content=b'bad', raise_for_status=lambda: None))
    path = tmp_path / 'archive.csv'
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        historical.verified_archive(path)
    assert not path.exists()


def test_current_and_future_outcomes_cannot_change_features():
    games = schedule()
    X, meta = historical_features(games)
    cutoff = pd.Timestamp('2013-10-12')
    changed = games.copy()
    changed.loc[changed['date'] >= cutoff, 'home_score'] += 100
    after, after_meta = historical_features(changed)
    pd.testing.assert_frame_equal(X.loc[meta.date <= cutoff], after.loc[after_meta.date <= cutoff])
    # Earlier holdout outcomes MAY affect later inputs: this is daily walk-forward.
    assert not X.equals(after)


def test_same_day_batch_excludes_results_even_when_a_team_appears_twice():
    games = schedule()
    extra = games.iloc[[20]].copy()
    extra['game_id'] = 'z-extra'
    games = pd.concat([games, extra], ignore_index=True)
    X, meta = historical_features(games)
    games.loc[20, 'home_score'] += 100
    after, _ = historical_features(games.sample(frac=1, random_state=42))
    same_day = meta.date == games.loc[20, 'date']
    pd.testing.assert_frame_equal(X.loc[same_day], after.loc[same_day])


def test_season_reset_and_warmup():
    games = schedule()
    X, meta = historical_features(games)
    changed = games.copy()
    changed.loc[changed.season == 2014, 'home_score'] += 100
    after, _ = historical_features(changed)
    pd.testing.assert_frame_equal(X.loc[meta.season == 2015], after.loc[meta.season == 2015])
    assert len(meta) == 60  # 5 prior games per team removed in EACH season
    assert meta.groupby('season').date.min().dt.day.tolist() == [6, 6]


def test_calendar_split_and_invalid_splits():
    _, meta = historical_features(schedule())
    train, test = date_split(meta, '2014-10-10', '2014-10-20')
    assert meta.loc[train, 'date'].max() < meta.loc[test, 'date'].min()
    assert (meta.loc[test, 'date'] >= pd.Timestamp('2014-10-10')).all()
    assert (meta.loc[test, 'date'] < pd.Timestamp('2014-10-20')).all()
    for start, end in [('2016-01-01', '2017-01-01'), ('2014-01-01', '2013-01-01')]:
        with pytest.raises(ValueError):
            date_split(meta, start, end)


def test_metrics_known_values_and_probability_endpoints():
    metrics = probability_metrics([0, 1], [0.25, 0.75])
    assert metrics['accuracy'] == 1
    assert metrics['brier_score'] == pytest.approx(0.0625)
    assert metrics['log_loss'] == pytest.approx(-np.log(0.75))
    assert metrics['ece_10_equal_width_bins'] == pytest.approx(0.25)
    endpoints = probability_metrics([0, 1], [0, 1])
    assert sum(row['count'] for row in endpoints['reliability_bins']) == 2
    assert endpoints['ece_10_equal_width_bins'] == 0
    assert probability_metrics([1, 1], [0.6, 0.6])['n_games'] == 2
    with pytest.raises(ValueError):
        probability_metrics([0], [np.nan])


def test_holdout_labels_do_not_change_fitted_model_or_baseline():
    games = schedule()
    report, predictions, model = run_benchmark(games)
    changed = games.copy()
    selected = changed.season == 2015
    changed.loc[selected, ['home_score', 'away_score']] = changed.loc[
        selected, ['away_score', 'home_score']].to_numpy()
    second_report, _, second_model = run_benchmark(changed)
    assert model.get_booster().save_raw() == second_model.get_booster().save_raw()
    assert report['baseline_training_home_win_rate'] == second_report['baseline_training_home_win_rate']
    assert predictions.home_rate_probability.nunique() == 1
    assert report['data_counts']['training_games'] == 30
    assert report['data_counts']['holdout_games'] == 30
    assert json.loads(json.dumps(report))['xgboost']['n_games'] == len(predictions)


def test_cli_saves_strict_report_and_keeps_serving_artifact_separate(monkeypatch, tmp_path):
    import sys
    from model import benchmark

    output = tmp_path / 'evidence'
    serving = tmp_path / 'serving.json'
    serving.write_text('untouched serving model')
    monkeypatch.setenv('MODEL_PATH', str(serving))
    monkeypatch.setattr(sys, 'argv', ['benchmark', '--output-dir', str(output)])
    monkeypatch.setattr(benchmark, 'verified_archive', lambda _: {'sha256': 'fixture-only'})
    monkeypatch.setattr(benchmark, 'parse_games', lambda _: schedule())
    benchmark.main()
    text = (output / 'report.json').read_text()
    assert 'NaN' not in text and 'Infinity' not in text
    report = json.loads(text)
    predictions = pd.read_csv(output / 'predictions.csv')
    assert report['source']['sha256'] == 'fixture-only'
    assert len(predictions) == report['data_counts']['holdout_games']
    assert (output / 'model.json').is_file()
    assert serving.read_text() == 'untouched serving model'
