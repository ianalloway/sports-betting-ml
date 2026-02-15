"""Integration with The Odds API for live betting odds."""

import os
import requests
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ODDS_API_KEY", "")
BASE_URL = "https://api.the-odds-api.com/v4"


def get_nba_odds(markets: str = "h2h", regions: str = "us") -> Optional[list]:
    """
    Fetch current NBA odds from The Odds API.
    
    Args:
        markets: Betting markets (h2h, spreads, totals)
        regions: Bookmaker regions (us, us2, uk, eu, au)
    
    Returns:
        List of games with odds, or None if API call fails
    """
    if not API_KEY:
        print("Warning: ODDS_API_KEY not set. Using demo data.")
        return get_demo_odds()
    
    url = f"{BASE_URL}/sports/basketball_nba/odds"
    params = {
        "apiKey": API_KEY,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "american"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching odds: {e}")
        return get_demo_odds()


def parse_odds(games: list) -> list:
    """
    Parse odds data into a cleaner format.
    
    Args:
        games: Raw games data from API
    
    Returns:
        List of parsed game odds
    """
    parsed = []
    
    for game in games:
        game_info = {
            "id": game.get("id"),
            "home_team": game.get("home_team"),
            "away_team": game.get("away_team"),
            "commence_time": game.get("commence_time"),
            "bookmakers": []
        }
        
        for bookmaker in game.get("bookmakers", []):
            book_info = {
                "name": bookmaker.get("title"),
                "markets": {}
            }
            
            for market in bookmaker.get("markets", []):
                market_key = market.get("key")
                outcomes = {}
                
                for outcome in market.get("outcomes", []):
                    outcomes[outcome.get("name")] = {
                        "price": outcome.get("price"),
                        "point": outcome.get("point")
                    }
                
                book_info["markets"][market_key] = outcomes
            
            game_info["bookmakers"].append(book_info)
        
        parsed.append(game_info)
    
    return parsed


def get_best_odds(games: list) -> list:
    """
    Get the best available odds for each team across all bookmakers.
    
    Args:
        games: Parsed games data
    
    Returns:
        List of games with best odds for each team
    """
    best_odds = []
    
    for game in games:
        home_team = game["home_team"]
        away_team = game["away_team"]
        
        best_home = {"odds": -10000, "book": None}
        best_away = {"odds": -10000, "book": None}
        
        for book in game.get("bookmakers", []):
            h2h = book.get("markets", {}).get("h2h", {})
            
            home_odds = h2h.get(home_team, {}).get("price")
            away_odds = h2h.get(away_team, {}).get("price")
            
            if home_odds and home_odds > best_home["odds"]:
                best_home = {"odds": home_odds, "book": book["name"]}
            
            if away_odds and away_odds > best_away["odds"]:
                best_away = {"odds": away_odds, "book": book["name"]}
        
        best_odds.append({
            "home_team": home_team,
            "away_team": away_team,
            "commence_time": game["commence_time"],
            "home_odds": best_home,
            "away_odds": best_away
        })
    
    return best_odds


def get_demo_odds() -> list:
    """Return demo odds data when API key is not available."""
    return [
        {
            "id": "demo1",
            "sport_key": "basketball_nba",
            "home_team": "Los Angeles Lakers",
            "away_team": "Golden State Warriors",
            "commence_time": "2026-02-16T02:00:00Z",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "title": "DraftKings",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Los Angeles Lakers", "price": -150},
                                {"name": "Golden State Warriors", "price": 130}
                            ]
                        }
                    ]
                },
                {
                    "key": "fanduel",
                    "title": "FanDuel",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Los Angeles Lakers", "price": -145},
                                {"name": "Golden State Warriors", "price": 125}
                            ]
                        }
                    ]
                }
            ]
        },
        {
            "id": "demo2",
            "sport_key": "basketball_nba",
            "home_team": "Boston Celtics",
            "away_team": "Miami Heat",
            "commence_time": "2026-02-16T00:30:00Z",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "title": "DraftKings",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Boston Celtics", "price": -200},
                                {"name": "Miami Heat", "price": 170}
                            ]
                        }
                    ]
                }
            ]
        },
        {
            "id": "demo3",
            "sport_key": "basketball_nba",
            "home_team": "Denver Nuggets",
            "away_team": "Phoenix Suns",
            "commence_time": "2026-02-16T03:00:00Z",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "title": "DraftKings",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Denver Nuggets", "price": -180},
                                {"name": "Phoenix Suns", "price": 155}
                            ]
                        }
                    ]
                }
            ]
        }
    ]
