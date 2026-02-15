"""Kelly Criterion calculator for optimal bet sizing."""


def american_to_decimal(american_odds: int) -> float:
    """Convert American odds to decimal odds."""
    if american_odds > 0:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1


def decimal_to_implied_prob(decimal_odds: float) -> float:
    """Convert decimal odds to implied probability."""
    return 1 / decimal_odds


def american_to_implied_prob(american_odds: int) -> float:
    """Convert American odds to implied probability."""
    decimal = american_to_decimal(american_odds)
    return decimal_to_implied_prob(decimal)


def kelly_criterion(win_prob: float, decimal_odds: float, fraction: float = 0.25) -> float:
    """
    Calculate optimal bet size using Kelly Criterion.
    
    Args:
        win_prob: Model's estimated probability of winning (0-1)
        decimal_odds: Decimal odds offered by bookmaker
        fraction: Kelly fraction (0.25 = quarter Kelly, more conservative)
    
    Returns:
        Optimal bet size as fraction of bankroll (0-1)
    """
    # b = decimal odds - 1 (net odds received on win)
    b = decimal_odds - 1
    # p = probability of winning
    p = win_prob
    # q = probability of losing
    q = 1 - p
    
    # Kelly formula: f* = (bp - q) / b
    kelly = (b * p - q) / b
    
    # Apply fractional Kelly for more conservative sizing
    kelly = kelly * fraction
    
    # Never bet negative (no edge) or more than 100%
    return max(0, min(kelly, 1))


def calculate_edge(model_prob: float, implied_prob: float) -> float:
    """
    Calculate the edge (expected value) of a bet.
    
    Args:
        model_prob: Model's estimated win probability
        implied_prob: Implied probability from odds
    
    Returns:
        Edge as a percentage (positive = +EV)
    """
    return (model_prob - implied_prob) * 100


def find_value_bets(predictions: list, min_edge: float = 3.0) -> list:
    """
    Find value bets where model probability exceeds implied probability.
    
    Args:
        predictions: List of dicts with 'model_prob', 'american_odds', 'team', 'game'
        min_edge: Minimum edge percentage to consider a value bet
    
    Returns:
        List of value bets with calculated edge and Kelly bet size
    """
    value_bets = []
    
    for pred in predictions:
        implied_prob = american_to_implied_prob(pred['american_odds'])
        edge = calculate_edge(pred['model_prob'], implied_prob)
        
        if edge >= min_edge:
            decimal_odds = american_to_decimal(pred['american_odds'])
            kelly_bet = kelly_criterion(pred['model_prob'], decimal_odds)
            
            value_bets.append({
                **pred,
                'implied_prob': implied_prob,
                'edge': edge,
                'kelly_bet': kelly_bet,
                'decimal_odds': decimal_odds
            })
    
    # Sort by edge descending
    value_bets.sort(key=lambda x: x['edge'], reverse=True)
    return value_bets
