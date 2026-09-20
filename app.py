"""
Sports Betting ML - NBA Game Prediction & Value Bet Finder
A Streamlit app for predicting NBA game outcomes and finding value bets.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from utils.kelly import (
    american_to_decimal,
    american_to_implied_prob,
    kelly_criterion,
    calculate_edge
)
from utils.clv import (
    attach_synthetic_closes,
    clv_chart_records,
    filter_bets_beating_close,
    simulate_synthetic_clv_bets,
    summarize_clv,
)
from utils.odds import get_nba_odds, parse_odds, get_best_h2h_odds_for_game
from model.predict import predict_game, get_default_stats, predict_with_confidence, load_model

# Caching API calls and Model Loading
@st.cache_data(ttl=300)
def cached_get_nba_odds():
    return get_nba_odds()

@st.cache_resource
def get_cached_model():
    return load_model()

# Load Model once for the page
cached_model = get_cached_model()

# Page config
st.set_page_config(
    page_title="Sports Betting ML",
    page_icon="🏀",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 10px;
    }
    .value-bet {
        background-color: #1a472a;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .no-value {
        background-color: #4a1a1a;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🏀 Sports Betting ML")
st.markdown("*NBA Game Prediction & Value Bet Finder*")

# Sidebar
st.sidebar.header("Settings")
min_edge = st.sidebar.slider("Minimum Edge (%)", 0.0, 10.0, 3.0, 0.5)
kelly_fraction = st.sidebar.slider("Kelly Fraction", 0.1, 1.0, 0.25, 0.05)
bankroll = st.sidebar.number_input("Bankroll ($)", min_value=100, value=1000, step=100)
require_beat_close = st.sidebar.checkbox(
    "Value bets: require beat-close",
    value=False,
    help=(
        "When enabled, value bets are filtered using simulated closing lines "
        "(not live sportsbook closes). See the CLV Backtest tab."
    ),
)

if st.sidebar.button("Refresh Odds"):
    cached_get_nba_odds.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("""
This app uses machine learning to predict NBA game outcomes and compares
predictions to betting odds to find value bets.

**How it works:**
1. Model predicts win probability
2. Compares to implied odds probability
3. Finds bets where model > odds (positive edge)
4. Uses Kelly Criterion for bet sizing
5. CLV backtest compares open vs close (synthetic closes in this demo)
""")

# Demo team stats (fallback when no real data available)
# In production, fetch from NBA API or stats database
DEMO_TEAM_STATS = {
    'Boston Celtics': {'win_pct': 0.72, 'avg_points_for': 118.5, 'avg_points_against': 109.2, 'point_diff': 9.3},
    'Cleveland Cavaliers': {'win_pct': 0.70, 'avg_points_for': 116.8, 'avg_points_against': 108.5, 'point_diff': 8.3},
    'Oklahoma City Thunder': {'win_pct': 0.68, 'avg_points_for': 117.2, 'avg_points_against': 106.8, 'point_diff': 10.4},
    'Denver Nuggets': {'win_pct': 0.65, 'avg_points_for': 115.5, 'avg_points_against': 110.2, 'point_diff': 5.3},
    'Memphis Grizzlies': {'win_pct': 0.62, 'avg_points_for': 114.8, 'avg_points_against': 111.5, 'point_diff': 3.3},
    'Milwaukee Bucks': {'win_pct': 0.60, 'avg_points_for': 116.2, 'avg_points_against': 112.8, 'point_diff': 3.4},
    'New York Knicks': {'win_pct': 0.58, 'avg_points_for': 112.5, 'avg_points_against': 108.2, 'point_diff': 4.3},
    'Los Angeles Lakers': {'win_pct': 0.55, 'avg_points_for': 114.2, 'avg_points_against': 112.5, 'point_diff': 1.7},
    'Golden State Warriors': {'win_pct': 0.52, 'avg_points_for': 113.8, 'avg_points_against': 113.2, 'point_diff': 0.6},
    'Phoenix Suns': {'win_pct': 0.50, 'avg_points_for': 112.5, 'avg_points_against': 113.0, 'point_diff': -0.5},
    'Miami Heat': {'win_pct': 0.48, 'avg_points_for': 108.5, 'avg_points_against': 110.2, 'point_diff': -1.7},
    'Dallas Mavericks': {'win_pct': 0.52, 'avg_points_for': 115.2, 'avg_points_against': 114.5, 'point_diff': 0.7},
    'Los Angeles Clippers': {'win_pct': 0.45, 'avg_points_for': 110.5, 'avg_points_against': 112.8, 'point_diff': -2.3},
    'Sacramento Kings': {'win_pct': 0.48, 'avg_points_for': 113.2, 'avg_points_against': 114.5, 'point_diff': -1.3},
    'Indiana Pacers': {'win_pct': 0.50, 'avg_points_for': 118.5, 'avg_points_against': 118.2, 'point_diff': 0.3},
    'Minnesota Timberwolves': {'win_pct': 0.55, 'avg_points_for': 110.8, 'avg_points_against': 108.5, 'point_diff': 2.3},
    'Houston Rockets': {'win_pct': 0.52, 'avg_points_for': 112.5, 'avg_points_against': 111.8, 'point_diff': 0.7},
    'San Antonio Spurs': {'win_pct': 0.35, 'avg_points_for': 108.2, 'avg_points_against': 115.5, 'point_diff': -7.3},
    'Detroit Pistons': {'win_pct': 0.30, 'avg_points_for': 106.5, 'avg_points_against': 116.2, 'point_diff': -9.7},
    'Washington Wizards': {'win_pct': 0.25, 'avg_points_for': 105.8, 'avg_points_against': 118.5, 'point_diff': -12.7},
}

# Main content
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Today's Games", "🎯 Value Bets", "📉 CLV Backtest", "📈 Model Info"]
)

with tab1:
    st.header("Today's NBA Games")

    # Fetch odds with error handling
    try:
        with st.spinner("Fetching odds..."):
            raw_odds = cached_get_nba_odds()
            games = parse_odds(raw_odds) if raw_odds else []

        if not games:
            st.warning("No games available. Showing demo data.")
            is_demo = True
        else:
            is_demo = False
    except Exception as e:
        st.error(f"Failed to fetch odds: {str(e)}")
        st.info("Using demo data for display purposes.")
        raw_odds = cached_get_nba_odds()  # This will return demo data on error
        games = parse_odds(raw_odds) if raw_odds else []
        is_demo = True

    # Display games
    for game in games:
        try:
            home_team = game['home_team']
            away_team = game['away_team']

            # Get team stats from demo data, fallback to defaults
            home_stats = DEMO_TEAM_STATS.get(home_team, get_default_stats())
            away_stats = DEMO_TEAM_STATS.get(away_team, get_default_stats())

            # Predict with error handling
            home_prob, away_prob = predict_game(home_stats, away_stats, model=cached_model)

            # Get best odds (fallback defaults are only used when a team
            # has no available price in the parsed bookmaker data).
            best_home_odds, best_away_odds = get_best_h2h_odds_for_game(game)

            # Calculate edge
            home_implied = american_to_implied_prob(best_home_odds)
            away_implied = american_to_implied_prob(best_away_odds)
            home_edge = calculate_edge(home_prob, home_implied)
            away_edge = calculate_edge(away_prob, away_implied)

            # Display game card
            with st.container():
                col1, col2, col3 = st.columns([2, 1, 2])

                # Get prediction with confidence intervals
                pred_ci = predict_with_confidence(home_stats, away_stats, home_team, away_team, model=cached_model)

                with col1:
                    st.subheader(f"🏠 {home_team}")
                    st.metric("Model Probability", f"{home_prob:.1%}")
                    st.caption(f"95% CI: [{pred_ci.home_ci_low:.1%} - {pred_ci.home_ci_high:.1%}]")
                    st.metric("Best Odds", f"{best_home_odds:+d}")
                    st.metric("Implied Probability", f"{home_implied:.1%}")
                    if home_edge > 0:
                        st.success(f"Edge: +{home_edge:.1f}%")
                    else:
                        st.error(f"Edge: {home_edge:.1f}%")

                with col2:
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    st.markdown("### VS")
                    predicted_winner = home_team if home_prob > 0.5 else away_team
                    confidence = max(home_prob, away_prob)
                    st.info(f"Pick: **{predicted_winner}**\n\nConfidence: {confidence:.1%}")

                with col3:
                    st.subheader(f"✈️ {away_team}")
                    st.metric("Model Probability", f"{away_prob:.1%}")
                    st.caption(f"95% CI: [{pred_ci.away_ci_low:.1%} - {pred_ci.away_ci_high:.1%}]")
                    st.metric("Best Odds", f"{best_away_odds:+d}")
                    st.metric("Implied Probability", f"{away_implied:.1%}")
                    if away_edge > 0:
                        st.success(f"Edge: +{away_edge:.1f}%")
                    else:
                        st.error(f"Edge: {away_edge:.1f}%")

                st.markdown("---")

        except Exception as e:
            st.error(f"Error processing game {game.get('home_team', 'Unknown')} vs {game.get('away_team', 'Unknown')}: {str(e)}")
            continue

with tab2:
    st.header("🎯 Value Bets")
    st.markdown(f"*Showing bets with edge >= {min_edge}%*")

    if is_demo:
        st.info("Currently showing demo data. Set ODDS_API_KEY in .env to use live data.")

    value_bets = []

    for game in games:
        try:
            home_team = game['home_team']
            away_team = game['away_team']

            home_stats = DEMO_TEAM_STATS.get(home_team, get_default_stats())
            away_stats = DEMO_TEAM_STATS.get(away_team, get_default_stats())

            home_prob, away_prob = predict_game(home_stats, away_stats, model=cached_model)

            # Get best odds (fallback defaults are only used when a team
            # has no available price in the parsed bookmaker data).
            best_home_odds, best_away_odds = get_best_h2h_odds_for_game(game)

            # Check home team
            home_implied = american_to_implied_prob(best_home_odds)
            home_edge = calculate_edge(home_prob, home_implied)
            if home_edge >= min_edge:
                decimal_odds = american_to_decimal(best_home_odds)
                kelly_bet = kelly_criterion(home_prob, decimal_odds, kelly_fraction)
                value_bets.append({
                    'team': home_team,
                    'opponent': away_team,
                    'location': 'Home',
                    'model_prob': home_prob,
                    'implied_prob': home_implied,
                    'odds': best_home_odds,
                    'edge': home_edge,
                    'kelly_pct': kelly_bet,
                    'bet_amount': kelly_bet * bankroll
                })

            # Check away team
            away_implied = american_to_implied_prob(best_away_odds)
            away_edge = calculate_edge(away_prob, away_implied)
            if away_edge >= min_edge:
                decimal_odds = american_to_decimal(best_away_odds)
                kelly_bet = kelly_criterion(away_prob, decimal_odds, kelly_fraction)
                value_bets.append({
                    'team': away_team,
                    'opponent': home_team,
                    'location': 'Away',
                    'model_prob': away_prob,
                    'implied_prob': away_implied,
                    'odds': best_away_odds,
                    'edge': away_edge,
                    'kelly_pct': kelly_bet,
                    'bet_amount': kelly_bet * bankroll
                })

        except Exception:
            st.warning(f"Error processing value bets for {game.get('home_team', 'Unknown')} vs {game.get('away_team', 'Unknown')}")
            continue

    if value_bets:
        # Sort by edge
        value_bets.sort(key=lambda x: x['edge'], reverse=True)

        if require_beat_close:
            value_bets = filter_bets_beating_close(
                attach_synthetic_closes(value_bets, seed=7),
                require_beat_close=True,
            )
            st.caption(
                "Beat-close filter uses **simulated** closing lines attached to "
                "today's opens — not historical sportsbook closes."
            )

    if value_bets:
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Value Bets Found", len(value_bets))
        with col2:
            total_bet = sum(b['bet_amount'] for b in value_bets)
            st.metric("Total Suggested Bet", f"${total_bet:.2f}")
        with col3:
            avg_edge = np.mean([b['edge'] for b in value_bets])
            st.metric("Average Edge", f"+{avg_edge:.1f}%")

        st.markdown("---")

        # Display value bets
        for bet in value_bets:
            with st.container():
                st.markdown(f"""
                <div class="value-bet">
                    <h3>✅ {bet['team']} ({bet['location']}) vs {bet['opponent']}</h3>
                </div>
                """, unsafe_allow_html=True)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Model Probability", f"{bet['model_prob']:.1%}")
                with col2:
                    st.metric("Odds", f"{bet['odds']:+d}")
                with col3:
                    st.metric("Edge", f"+{bet['edge']:.1f}%")
                with col4:
                    st.metric("Suggested Bet", f"${bet['bet_amount']:.2f}")

                if "clv_pts" in bet:
                    st.caption(
                        f"Synthetic close {bet['close_odds']:+d} · "
                        f"CLV {bet['clv_pts']:+.2f} pts · beat close"
                    )

                st.markdown("---")
    else:
        if require_beat_close:
            st.info(
                f"No value bets with edge >= {min_edge}% that also beat a "
                "simulated close. Try lowering the edge threshold or disabling "
                "the beat-close filter."
            )
        else:
            st.info(f"No value bets found with edge >= {min_edge}%. Try lowering the minimum edge threshold.")

with tab3:
    st.header("📉 CLV Backtest")
    st.markdown(
        "Closing Line Value compares the **open** price you bet to the market's "
        "**closing** line. Positive CLV (probability points) means you beat the close."
    )
    st.info(
        "This demo uses **synthetic** open/close American odds — not scraped "
        "historical sportsbook closes. Treat metrics as a workflow illustration."
    )

    n_clv_bets = st.slider("Synthetic bets in slice", 10, 100, 40, 5)
    value_only = st.checkbox("Only include synthetic value bets", value=False)
    beat_close_only = st.checkbox("Only bets that beat the close", value=False)

    clv_bets = simulate_synthetic_clv_bets(n_bets=n_clv_bets, seed=42)
    if value_only:
        clv_bets = [b for b in clv_bets if b.get("is_value_bet")]
    if beat_close_only:
        clv_bets = filter_bets_beating_close(clv_bets, require_beat_close=True)

    clv_summary = summarize_clv(clv_bets)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Bets in slice", clv_summary["n_bets"])
    with m2:
        st.metric("Mean CLV", f"{clv_summary['mean_clv_pts']:+.2f} pts")
    with m3:
        st.metric("Beat-close rate", f"{clv_summary['beat_close_rate']:.1%}")
    with m4:
        st.metric("Beat close (count)", clv_summary["n_beat_close"])

    st.caption(
        f"Odds source: `{clv_summary['source'] or 'n/a'}` · "
        "CLV pts = (close implied − open implied) × 100."
    )

    if clv_bets:
        chart_df = pd.DataFrame(clv_chart_records(clv_bets))
        fig_clv = px.line(
            chart_df,
            x="bet_index",
            y=["clv_pts", "cumulative_mean_clv"],
            title="CLV over bets (synthetic)",
            labels={"value": "CLV (pts)", "bet_index": "Bet #", "variable": "Series"},
        )
        fig_clv.update_layout(legend_title_text="")
        st.plotly_chart(fig_clv, use_container_width=True)

        table_df = pd.DataFrame(clv_bets)[
            [
                "bet_id",
                "team",
                "opponent",
                "location",
                "open_odds",
                "close_odds",
                "clv_pts",
                "beat_close",
                "is_value_bet",
                "model_edge_pct",
            ]
        ].copy()
        table_df = table_df.rename(
            columns={
                "bet_id": "Bet",
                "team": "Team",
                "opponent": "Opponent",
                "location": "Loc",
                "open_odds": "Open",
                "close_odds": "Close",
                "clv_pts": "CLV pts",
                "beat_close": "Beat close",
                "is_value_bet": "Value bet",
                "model_edge_pct": "Edge %",
            }
        )
        table_df["Open"] = table_df["Open"].map(lambda x: f"{int(x):+d}")
        table_df["Close"] = table_df["Close"].map(lambda x: f"{int(x):+d}")
        table_df["CLV pts"] = table_df["CLV pts"].map(lambda x: f"{x:+.2f}")
        table_df["Edge %"] = table_df["Edge %"].map(lambda x: f"{x:+.1f}")
        st.dataframe(table_df, use_container_width=True, hide_index=True)
    else:
        st.warning("No bets match the current CLV filters.")

with tab4:
    st.header("📈 Model Information")

    st.markdown("""
    ### How the Model Works

    This model uses **XGBoost** (Extreme Gradient Boosting) to predict NBA game outcomes.

    #### Features Used:
    - **Win Percentage**: Team's recent win rate
    - **Points Per Game**: Offensive output
    - **Opponent PPG**: Defensive performance
    - **Point Differential**: Net rating proxy
    - **Home Court Advantage**: ~3 point boost for home teams

    #### Model Performance:
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Holdout accuracy", "~61%")
    with col2:
        st.metric("CV accuracy", "~58%")
    with col3:
        st.metric("Holdout Brier", "~0.24")

    st.caption(
        "Synthetic demo metrics from `python -m model.train` (walk-forward CV + chronological holdout). "
        "Not live betting results — the training script does not compute ROI or Sharpe."
    )

    st.markdown("""
    #### Value Betting Strategy

    The model identifies **value bets** by comparing its predicted probability to the
    **implied probability** from betting odds.

    **Example:**
    - Model predicts Team A wins 60% of the time
    - Odds of +150 imply only 40% win probability
    - Edge = 60% - 40% = **+20%** (strong value bet!)

    #### Kelly Criterion

    The **Kelly Criterion** calculates optimal bet sizing based on edge and odds:

    ```
    f* = (bp - q) / b

    where:
    b = decimal odds - 1
    p = probability of winning
    q = probability of losing
    ```

    We use **fractional Kelly** (default 25%) for more conservative sizing.

    #### Closing Line Value (CLV)

    The **CLV Backtest** tab compares open vs close American odds on a synthetic
    slice. Mean CLV and beat-close rate measure whether picks consistently got
    a better price than the (simulated) close — a standard sports-betting
    evaluation lens, separate from win/loss ROI.
    """)

    # Feature importance chart
    st.subheader("Feature Importance")

    importance_data = pd.DataFrame({
        'Feature': ['Point Diff Difference', 'Win Pct Difference', 'Home Point Diff',
                   'Away Point Diff', 'Home Win Pct', 'Away Win Pct', 'Home PPG',
                   'Away PPG', 'Home Opp PPG', 'Away Opp PPG', 'Home Advantage'],
        'Importance': [0.25, 0.20, 0.12, 0.11, 0.08, 0.07, 0.05, 0.04, 0.03, 0.03, 0.02]
    })

    fig = px.bar(importance_data, x='Importance', y='Feature', orientation='h',
                 title='Feature Importance in Prediction Model')
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Chart values are illustrative placeholders for the demo UI, not loaded from a trained artifact.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>⚠️ <strong>Disclaimer:</strong> This is for educational purposes only.
    Sports betting involves risk. Past performance does not guarantee future results.
    Please gamble responsibly.</p>
    <p>Built by <a href="https://ianalloway.xyz">Ian Alloway</a></p>
</div>
""", unsafe_allow_html=True)
