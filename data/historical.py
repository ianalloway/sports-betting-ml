"""Pinned FiveThirtyEight NBA results; never uses the archive's Elo/forecast fields."""

import hashlib
from pathlib import Path

import pandas as pd
import requests

SOURCE_REVISION = '6d880e939ad3d11d94c137c911681b3cf718fd74'
SOURCE_URL = (
    f'https://raw.githubusercontent.com/fivethirtyeight/data/{SOURCE_REVISION}'
    '/nba-elo/nbaallelo.csv'
)
SOURCE_SHA256 = 'd46ed3540ee8d9eca31b3e94cc8c777e0be5156173d814ebf65b8195e8d616bc'
SOURCE_DOCUMENTATION = (
    f'https://github.com/fivethirtyeight/data/blob/{SOURCE_REVISION}/nba-elo/README.md'
)
COLUMNS = [
    'game_id', 'lg_id', 'year_id', 'date_game', 'is_playoffs',
    'team_id', 'opp_id', 'pts', 'opp_pts', 'game_location',
]


def verified_archive(path: Path) -> dict:
    """Download only when missing, and refuse changed or corrupt cached bytes."""
    path = Path(path)
    if path.exists():
        content = path.read_bytes()
    else:
        response = requests.get(SOURCE_URL, timeout=60)
        response.raise_for_status()
        content = response.content
    digest = hashlib.sha256(content).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(f'Archive SHA-256 mismatch: expected {SOURCE_SHA256}, got {digest}')
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return {
        'name': 'FiveThirtyEight Historical NBA Elo (game results only)',
        'url': SOURCE_URL,
        'revision': SOURCE_REVISION,
        'sha256': digest,
        'bytes': len(content),
        'documentation': SOURCE_DOCUMENTATION,
        'upstream_game_information': 'Basketball-Reference.com, via FiveThirtyEight',
        'license': 'CC BY 4.0 (FiveThirtyEight repository default for data)',
    }


def parse_games(path: Path, first_season: int = 2010, last_season: int = 2015) -> pd.DataFrame:
    """One home row per regular-season NBA game; neutral sites are excluded.

    Season IDs denote the ending year. This parser is separate from download
    verification to allow small offline fixtures; the CLI always verifies first.
    """
    if not 1947 <= first_season <= last_season <= 2015:
        raise ValueError('Seasons must be ordered and within this archive (1947–2015)')
    raw = pd.read_csv(path, usecols=COLUMNS)
    mask = (
        raw['lg_id'].eq('NBA') & raw['year_id'].between(first_season, last_season)
        & raw['is_playoffs'].eq(0) & raw['game_location'].eq('H')
    )
    games = raw.loc[mask].rename(columns={
        'year_id': 'season', 'date_game': 'date', 'team_id': 'home_team',
        'opp_id': 'away_team', 'pts': 'home_score', 'opp_pts': 'away_score',
    })[['game_id', 'season', 'date', 'home_team', 'away_team', 'home_score', 'away_score']].copy()
    if games.empty or games.isna().any().any():
        raise ValueError('No eligible games or missing required values')
    games['date'] = pd.to_datetime(games['date'], format='%m/%d/%Y', errors='raise')
    for column in ['home_score', 'away_score']:
        games[column] = pd.to_numeric(games[column], errors='raise')
        if not ((games[column] > 0) & (games[column] % 1 == 0)).all():
            raise ValueError('Scores must be positive integers')
    if games['game_id'].duplicated().any():
        raise ValueError('Duplicate home game IDs')
    if games['home_team'].eq(games['away_team']).any():
        raise ValueError('A team cannot play itself')
    if games['home_score'].eq(games['away_score']).any():
        raise ValueError('Completed NBA games cannot end in ties')
    return games.sort_values(['date', 'game_id']).reset_index(drop=True)
