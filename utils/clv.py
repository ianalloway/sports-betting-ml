"""Closing Line Value (CLV) helpers for open-vs-close odds analysis.

CLV answers: did you get a better price than the market's closing line?
Positive probability-point CLV means the close implied a higher win chance
for your side than the open — i.e. you beat the close.

All synthetic generators here are labeled as such; they are demos, not
historical closes from a sportsbook feed.
"""

from __future__ import annotations

from typing import Iterable, Mapping, MutableMapping, Sequence

import numpy as np

from utils.kelly import american_to_implied_prob


def clv_probability_points(open_american: int, close_american: int) -> float:
    """CLV in probability percentage points (close implied − open implied) × 100.

    Positive means the market moved toward your side after you bet the open,
    so your open price beat the close.
    """
    open_implied = american_to_implied_prob(open_american)
    close_implied = american_to_implied_prob(close_american)
    return (close_implied - open_implied) * 100.0


def beat_close(open_american: int, close_american: int) -> bool:
    """Return True when open American odds are strictly better than the close.

    Higher American odds are better for the bettor (+150 > +130, −110 > −150).
    Equal lines do not count as beating the close.
    """
    return open_american > close_american


def annotate_clv(bet: Mapping) -> dict:
    """Copy a bet dict and attach ``clv_pts`` / ``beat_close`` from open/close odds.

    Expects ``open_odds`` and ``close_odds`` as American integers.
    """
    open_odds = int(bet["open_odds"])
    close_odds = int(bet["close_odds"])
    annotated = dict(bet)
    annotated["clv_pts"] = clv_probability_points(open_odds, close_odds)
    annotated["beat_close"] = beat_close(open_odds, close_odds)
    return annotated


def summarize_clv(bets: Sequence[Mapping]) -> dict:
    """Aggregate mean CLV pts, beat-close rate, and counts.

    Empty input returns zeros with ``n_bets == 0``.
    """
    if not bets:
        return {
            "n_bets": 0,
            "mean_clv_pts": 0.0,
            "beat_close_rate": 0.0,
            "n_beat_close": 0,
            "source": None,
        }

    rows = [annotate_clv(b) if "clv_pts" not in b else dict(b) for b in bets]
    clv_values = [float(r["clv_pts"]) for r in rows]
    beat_flags = [bool(r.get("beat_close", r["clv_pts"] > 0)) for r in rows]
    sources = {r.get("odds_source") for r in rows if r.get("odds_source")}

    return {
        "n_bets": len(rows),
        "mean_clv_pts": float(np.mean(clv_values)),
        "beat_close_rate": float(np.mean(beat_flags)),
        "n_beat_close": int(sum(beat_flags)),
        "source": sources.pop() if len(sources) == 1 else ("mixed" if sources else None),
    }


def filter_bets_beating_close(
    bets: Iterable[Mapping],
    *,
    require_beat_close: bool = True,
) -> list[dict]:
    """Keep bets that beat the close (or all annotated bets when filter is off)."""
    annotated = [annotate_clv(b) if "beat_close" not in b else dict(b) for b in bets]
    if not require_beat_close:
        return annotated
    return [b for b in annotated if b["beat_close"]]


def _nudge_american(open_american: int, delta_prob: float) -> int:
    """Move American odds so implied probability shifts by ``delta_prob``.

    Keeps the result a valid non-zero American price (nearest integer).
    """
    open_implied = american_to_implied_prob(open_american)
    target = min(0.95, max(0.05, open_implied + delta_prob))
    # Convert probability to American odds.
    if target >= 0.5:
        american = int(round(-100 * target / (1.0 - target)))
    else:
        american = int(round(100 * (1.0 - target) / target))
    if american == 0:
        american = -100 if open_american < 0 else 100
    return american


def simulate_synthetic_clv_bets(
    n_bets: int = 40,
    *,
    seed: int = 42,
    value_bias: float = 0.015,
) -> list[dict]:
    """Build a labeled synthetic CLV backtest slice (open vs close American odds).

    Closes are simulated — not scraped sportsbook closes. A small positive
    ``value_bias`` makes "value" picks slightly more likely to beat the close,
    which mirrors the demo narrative without claiming live-market CLV.
    """
    rng = np.random.default_rng(seed)
    teams = [
        "Boston Celtics",
        "Milwaukee Bucks",
        "Denver Nuggets",
        "Phoenix Suns",
        "Los Angeles Lakers",
        "Golden State Warriors",
        "Miami Heat",
        "New York Knicks",
        "Oklahoma City Thunder",
        "Dallas Mavericks",
    ]

    bets: list[dict] = []
    for i in range(n_bets):
        team = teams[i % len(teams)]
        opponent = teams[(i + 3) % len(teams)]
        # Mix favorites and dogs around typical NBA moneylines.
        open_odds = int(rng.choice([-180, -150, -130, -110, 105, 120, 140, 165]))
        is_value = bool(rng.random() < 0.55)
        # Value picks: slight drift toward the bet (positive CLV on average).
        # Non-value: slight drift against.
        base_move = value_bias if is_value else -value_bias * 0.5
        noise = float(rng.normal(0.0, 0.012))
        close_odds = _nudge_american(open_odds, base_move + noise)
        # Ensure we sometimes land on equal lines for the zero-CLV edge case.
        if rng.random() < 0.05:
            close_odds = open_odds

        row = {
            "bet_id": i + 1,
            "team": team,
            "opponent": opponent,
            "location": "Home" if i % 2 == 0 else "Away",
            "open_odds": open_odds,
            "close_odds": close_odds,
            "is_value_bet": is_value,
            "model_edge_pct": float(rng.uniform(2.0, 8.0) if is_value else rng.uniform(-2.0, 2.5)),
            "odds_source": "synthetic",
        }
        bets.append(annotate_clv(row))

    return bets


def attach_synthetic_closes(
    bets: Sequence[MutableMapping],
    *,
    seed: int = 7,
    drift_std: float = 0.01,
) -> list[dict]:
    """Attach simulated closes to live/demo value bets for optional beat-close filter.

    Uses each bet's ``odds`` (or ``open_odds``) as the open. Labels
    ``odds_source`` as ``synthetic_close`` so the UI can stay honest.
    """
    rng = np.random.default_rng(seed)
    out: list[dict] = []
    for bet in bets:
        row = dict(bet)
        open_odds = int(row.get("open_odds", row["odds"]))
        # Slight mean-reverting noise; value edge correlates loosely with +CLV.
        edge = float(row.get("edge", 0.0))
        drift = (0.002 * np.sign(edge)) + float(rng.normal(0.0, drift_std))
        close_odds = _nudge_american(open_odds, drift)
        row["open_odds"] = open_odds
        row["close_odds"] = close_odds
        row["odds_source"] = "synthetic_close"
        out.append(annotate_clv(row))
    return out


def clv_chart_records(bets: Sequence[Mapping]) -> list[dict]:
    """Records for a CLV-over-bets chart: bet index, CLV pts, cumulative mean."""
    annotated = [annotate_clv(b) if "clv_pts" not in b else dict(b) for b in bets]
    records: list[dict] = []
    running = 0.0
    for i, row in enumerate(annotated, start=1):
        running += float(row["clv_pts"])
        records.append(
            {
                "bet_index": i,
                "clv_pts": float(row["clv_pts"]),
                "cumulative_mean_clv": running / i,
                "team": row.get("team"),
                "beat_close": bool(row.get("beat_close", row["clv_pts"] > 0)),
            }
        )
    return records
