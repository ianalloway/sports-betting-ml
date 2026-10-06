"""Offline historical benchmark, independent of the synthetic dashboard artifact.

Run: python -m model.benchmark --output-dir benchmark-results
"""

import argparse
from collections import defaultdict, deque
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from data.features import create_game_features
from data.historical import parse_games, verified_archive
from model.train import build_model


def historical_features(games: pd.DataFrame, window: int = 10, min_games: int = 5):
    """Season-reset rolling features, using only strictly earlier calendar dates.

    A day is predicted as a batch before any of its results update history.
    During the holdout, earlier holdout results update rolling inputs, but the
    fitted model and baseline stay frozen (a daily walk-forward evaluation).
    """
    if not 1 <= min_games <= window:
        raise ValueError('Require 1 <= min_games <= window')
    features, metadata = [], []
    for _, season in games.groupby('season', sort=True):
        history = defaultdict(lambda: deque(maxlen=window))
        for _, day in season.sort_values(['date', 'game_id']).groupby('date', sort=True):
            for game in day.itertuples(index=False):
                home, away = history[game.home_team], history[game.away_team]
                if len(home) < min_games or len(away) < min_games:
                    continue

                def stats(results):
                    values = np.asarray(results)
                    return {
                        'win_pct': float((values[:, 0] > values[:, 1]).mean()),
                        'avg_points_for': float(values[:, 0].mean()),
                        'avg_points_against': float(values[:, 1].mean()),
                        'point_diff': float((values[:, 0] - values[:, 1]).mean()),
                    }

                features.append(create_game_features(
                    game.home_team, game.away_team, stats(home), stats(away)
                ))
                metadata.append({
                    'game_id': game.game_id, 'date': game.date, 'season': game.season,
                    'home_team': game.home_team, 'away_team': game.away_team,
                    'home_win': int(game.home_score > game.away_score),
                })
            for game in day.itertuples(index=False):
                history[game.home_team].append((game.home_score, game.away_score))
                history[game.away_team].append((game.away_score, game.home_score))
    if not features:
        raise ValueError('No games remain after per-team warmup')
    meta = pd.DataFrame(metadata)
    order = meta.sort_values(['date', 'game_id']).index
    return pd.DataFrame(features).loc[order].reset_index(drop=True), meta.loc[order].reset_index(drop=True)


def date_split(metadata, holdout_start, holdout_end):
    start, end = pd.Timestamp(holdout_start), pd.Timestamp(holdout_end)
    if pd.isna(start) or pd.isna(end) or start >= end:
        raise ValueError('Holdout start must precede end')
    train = metadata['date'] < start
    test = (metadata['date'] >= start) & (metadata['date'] < end)
    if not train.any() or not test.any():
        raise ValueError('Both training and holdout require eligible games')
    if metadata.loc[train, 'home_win'].nunique() != 2:
        raise ValueError('Training requires both home wins and losses')
    return train, test


def probability_metrics(targets, probabilities):
    """Binary Brier/log loss and equal-width reliability bins (no recalibration)."""
    y, p = np.asarray(targets), np.asarray(probabilities, dtype=float)
    if len(y) == 0 or y.shape != p.shape or not np.isin(y, [0, 1]).all():
        raise ValueError('Expected equally sized nonempty binary targets and probabilities')
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Probabilities must be finite and in [0, 1]')
    n_bins = 10
    bin_ids = np.minimum((p * n_bins).astype(int), n_bins - 1)
    reliability, ece = [], 0.0
    for index in range(n_bins):
        selected = bin_ids == index
        count = int(selected.sum())
        mean = float(p[selected].mean()) if count else None
        observed = float(y[selected].mean()) if count else None
        if count:
            ece += count / len(y) * abs(mean - observed)
        reliability.append({
            'lower': index / n_bins, 'upper': (index + 1) / n_bins,
            'count': count, 'mean_probability': mean, 'home_win_rate': observed,
        })
    return {
        'n_games': len(y), 'accuracy': float(accuracy_score(y, p >= 0.5)),
        'log_loss': float(log_loss(y, p, labels=[0, 1])),
        'brier_score': float(brier_score_loss(y, p)),
        'mean_probability': float(p.mean()), 'home_win_rate': float(y.mean()),
        'ece_10_equal_width_bins': float(ece), 'reliability_bins': reliability,
    }


def run_benchmark(games, holdout_start='2014-07-01', holdout_end='2015-07-01'):
    X, metadata = historical_features(games)
    train, test = date_split(metadata, holdout_start, holdout_end)
    model = build_model()
    model.set_params(n_jobs=1)  # fixed CPU execution; no holdout tuning/early stopping
    model.fit(X.loc[train], metadata.loc[train, 'home_win'])
    probabilities = model.predict_proba(X.loc[test])[:, 1].astype(float)
    baseline_rate = float(metadata.loc[train, 'home_win'].mean())
    baseline = np.full(int(test.sum()), baseline_rate)
    predictions = metadata.loc[test].copy()
    predictions['xgboost_probability'] = probabilities
    predictions['home_rate_probability'] = baseline
    report = {
        'configuration': {
            'holdout_start_inclusive': holdout_start, 'holdout_end_exclusive': holdout_end,
            'rolling_window': 10, 'min_prior_games_per_team': 5, 'reset_each_season': True,
            'same_date_results_excluded': True, 'holdout_protocol': 'daily walk-forward; model frozen',
            'features': list(X.columns), 'model_parameters': model.get_params(),
        },
        'data_counts': {
            'eligible_raw_games': len(games), 'feature_rows': len(metadata),
            'warmup_excluded': len(games) - len(metadata), 'training_games': int(train.sum()),
            'holdout_games': int(test.sum()), 'outside_split': int((~(train | test)).sum()),
            'training_first_date': metadata.loc[train, 'date'].min().date().isoformat(),
            'training_last_date': metadata.loc[train, 'date'].max().date().isoformat(),
            'holdout_first_date': metadata.loc[test, 'date'].min().date().isoformat(),
            'holdout_last_date': metadata.loc[test, 'date'].max().date().isoformat(),
        },
        'baseline_training_home_win_rate': baseline_rate,
        'xgboost': probability_metrics(predictions['home_win'], probabilities),
        'home_rate_baseline': probability_metrics(predictions['home_win'], baseline),
    }
    return report, predictions, model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=Path('data/raw/nbaallelo.csv'))
    parser.add_argument('--output-dir', type=Path, default=Path('benchmark-results'))
    args = parser.parse_args()
    provenance = verified_archive(args.archive)
    games = parse_games(args.archive)  # fixed 2010–2015 ending-year seasons
    report, predictions, model = run_benchmark(games)
    report['source'] = provenance
    report['source']['selection'] = 'NBA, regular season, home rows, ending-year seasons 2010–2015'
    report['environment'] = {
        'python': platform.python_version(), 'platform': platform.platform(),
        **{name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'xgboost']},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / 'predictions.csv', index=False)
    model.save_model(args.output_dir / 'model.json')
    # XGBoost includes an unset missing-value parameter represented by NaN.
    params = report['configuration']['model_parameters']
    report['configuration']['model_parameters'] = {
        key: value for key, value in params.items()
        if value is not None and not (isinstance(value, float) and np.isnan(value))
    }
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    for name in ['xgboost', 'home_rate_baseline']:
        metrics = report[name]
        print(f"{name}: n={metrics['n_games']} accuracy={metrics['accuracy']:.4f} "
              f"log_loss={metrics['log_loss']:.4f} Brier={metrics['brier_score']:.4f} "
              f"ECE={metrics['ece_10_equal_width_bins']:.4f}")
    print(f'Report, predictions, and separate benchmark model saved to {args.output_dir}')


if __name__ == '__main__':
    main()
