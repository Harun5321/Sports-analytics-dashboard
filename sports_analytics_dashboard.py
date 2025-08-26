import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# -----------------------------
# Config / Paths
# -----------------------------
MATCHES_PATH = "C:\\Users\\Harun\\Desktop\\Sports_analytics_dashboard\\Datasets\\epl_teams_stats_24.csv"
PLAYERS_PATH = "C:\\Users\\Harun\\Desktop\\Sports_analytics_dashboard\\Datasets\\epl_player_stats_24.csv"
ASSETS_DIR = Path("C:\\Users\\Harun\\Desktop\\Sports_analytics_dashboard\\Datasets\\assets")  # put club logos/headshots here named by club/playe
TOTAL_PL_MATCHES = 38  # EPL season length

st.set_page_config(layout='wide', page_title='EPL 2024 Dashboard + 2025 Projections')

# -----------------------------
# Data loaders (cached)
# -----------------------------
@st.cache_data
def load_team_df(path=MATCHES_PATH):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    # for compatibility ensure canonical names
    if 'goals_for' not in df.columns and 'goals_scored' in df.columns:
        df = df.rename(columns={'goals_scored':'goals_for'})
    if 'goals_against' not in df.columns and 'goals_conceded' in df.columns:
        df = df.rename(columns={'goals_conceded':'goals_against'})
    if 'goal_diff' not in df.columns and 'goal_difference' in df.columns:
        df = df.rename(columns={'goal_difference':'goal_diff'})
    return df

@st.cache_data
def load_players_df(path=PLAYERS_PATH):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

team_df = load_team_df()
players_df = load_players_df()

# -----------------------------
# Helpers
# -----------------------------

def get_logo_path(club: str) -> Path | None:
    if not ASSETS_DIR.exists():
        return None
    # try exact, then sanitized
    candidates = [ASSETS_DIR / f"{club}.png", ASSETS_DIR / f"{club}.jpg"]
    safe = club.lower().replace(' ', '_')
    candidates += [ASSETS_DIR / f"{safe}.png", ASSETS_DIR / f"{safe}.jpg"]
    for p in candidates:
        if p.exists():
            return p
    return None


def kpis_for_club(df: pd.DataFrame, club: str) -> dict | None:
    row = df[df['team'].str.lower() == club.lower()]
    if row.empty:
        return None
    r = row.iloc[0]
    wins, draws, losses = int(r.get('wins', 0)), int(r.get('draws', 0)), int(r.get('losses', 0))
    played = wins + draws + losses if (wins+draws+losses) > 0 else np.nan
    goals_for = int(r.get('goals_for', 0))
    goals_against = int(r.get('goals_against', 0))
    points = int(r.get('points', 0))
    gd = int(r.get('goal_diff', goals_for - goals_against))
    rank = r.get('rank', '—')
    return {
        'team': r['team'], 'rank': rank, 'played': played,
        'wins': wins, 'draws': draws, 'losses': losses,
        'goals_for': goals_for, 'goals_against': goals_against, 'gd': gd, 'points': points,
        'ppg': (points/played) if played and played>0 else np.nan,
        'gpg': (goals_for/played) if played and played>0 else np.nan,
        'cpg': (goals_against/played) if played and played>0 else np.nan,
    }


def simple_points_projection_2025(k: dict) -> float | None:
    """Project 2025 points using 2024 points-per-game × 38."""
    if k is None or np.isnan(k['ppg']):
        return None
    return float(k['ppg'] * TOTAL_PL_MATCHES)


def player_metrics(sr: pd.Series) -> dict:
    def g(name, default=0):
        # try exact, lower, and common variants
        if name in sr.index: return sr[name]
        lower_map = {c.lower(): c for c in sr.index}
        if name.lower() in lower_map: return sr[lower_map[name.lower()]]
        return default
    minutes = g('Minutes', g('minutes', 0))
    goals = g('Goals', g('goals', 0))
    assists = g('Assists', g('assists', 0))
    apps = g('Appearances', g('appearances', 0))
    shots = g('Shots', g('shots', 0))
    sot = g('Shots on Target', g('shots_on_target', 0))
    pos = g('Position', '')
    name = g('Player Name', g('Player', 'Unknown'))
    nationality = g('Nationality', g('nationality', 'N/A'))
    contrib = goals + assists
    per90 = lambda x: (x / minutes * 90) if minutes and minutes>0 else np.nan
    return {
        'player': name,
        'position': pos,
        'nationality': nationality,
        'appearances': apps,
        'minutes': minutes,
        'goals': goals,
        'assists': assists,
        'goal_contributions': contrib,
        'goals_per90': per90(goals),
        'assists_per90': per90(assists),
        'contribs_per90': per90(contrib),
        'shots': shots,
        'shots_on_target': sot,
        'shot_accuracy': (sot/shots) if shots else np.nan,
        'shot_conversion': (goals/shots) if shots else np.nan,
    }


def simple_player_projection_2025(sr: pd.Series) -> dict:
    """Naive 2025 projection based on 2024 per-90 rates times 38 matches.
    Assumes full availability; tune as needed.
    """
    m = player_metrics(sr)
    if not m['minutes'] or m['minutes']<=0:
        return {'predicted_goals_2025': np.nan, 'predicted_assists_2025': np.nan, 'predicted_contribs_2025': np.nan}
    # per90 * (38*90)
    total_minutes = TOTAL_PL_MATCHES * 90
    pg = (m['goals_per90'] or 0) * (total_minutes/90)
    pa = (m['assists_per90'] or 0) * (total_minutes/90)
    return {
        'predicted_goals_2025': pg,
        'predicted_assists_2025': pa,
        'predicted_contribs_2025': pg + pa,
    }


def columns_if_exist(df, cols):
    return [c for c in cols if c in df.columns]

# -----------------------------
# UI: Sidebar
# -----------------------------
st.sidebar.title("Controls")
view = st.sidebar.radio("View", ["Club", "Player", "Compare Players", "Compare Clubs", "Leaderboards"], index=0)
all_teams = sorted(team_df['team'].unique().tolist())
selected_team = st.sidebar.selectbox("Club", all_teams, index=0)

# optional position filter for player views
positions_col = None
if view in ("Player", "Compare Players"):
    # attempt to derive distinct positions
    club_col = [c for c in players_df.columns if c.lower()=="club"]
    club_col = club_col[0] if club_col else [c for c in players_df.columns if 'club' in c.lower()][0]
    pos_col = [c for c in players_df.columns if c.lower()=="position"]
    pos_col = pos_col[0] if pos_col else [c for c in players_df.columns if 'position' in c.lower()][0]
    positions_col = pos_col
    club_players_base = players_df[players_df[club_col].str.lower() == selected_team.lower()].copy()
    pos_values = sorted([p for p in club_players_base[pos_col].dropna().unique().tolist()])
    selected_positions = st.sidebar.multiselect("Positions", options=pos_values, default=pos_values)
else:
    selected_positions = []

st.sidebar.markdown("---")
show_downloads = st.sidebar.checkbox("Enable CSV downloads", value=True)

# -----------------------------
# Header / Logo
# -----------------------------
logo = get_logo_path(selected_team)
cols = st.columns([1,6])
if logo is not None:
    cols[0].image(str(logo))
cols[1].title(f"EPL 2024 Dashboard — {selected_team}")
cols[1].caption("Includes simple 2025 projections (naive per-game/per-90 extrapolation)")

# -----------------------------
# CLUB VIEW
# -----------------------------
if view == "Club":
    k = kpis_for_club(team_df, selected_team)
    if not k:
        st.error("Selected club not found in team data.")
    else:
        kcol1, kcol2, kcol3, kcol4, kcol5 = st.columns(5)
        kcol1.metric("Rank", k['rank'])
        kcol2.metric("Points", k['points'])
        kcol3.metric("Played", int(k['played']) if not np.isnan(k['played']) else '—')
        kcol4.metric("Goal Diff", k['gd'])
        kcol5.metric("PPG", None if np.isnan(k['ppg']) else round(k['ppg'],2))

        sub1, sub2, sub3 = st.columns(3)
        sub1.metric("Wins", k['wins'])
        sub2.metric("Draws", k['draws'])
        sub3.metric("Losses", k['losses'])

        sub4, sub5, sub6 = st.columns(3)
        sub4.metric("Goals For", k['goals_for'])
        sub5.metric("Goals Against", k['goals_against'])
        sub6.metric("Goals/Game", None if np.isnan(k['gpg']) else round(k['gpg'],2))

        # 2025 projection (simple)
        proj = simple_points_projection_2025(k)
        st.subheader("2025 Simple Projection")
        st.write(f"Projected points in 2025 if {selected_team} maintain 2024 PPG: **{proj:.1f}** (over {TOTAL_PL_MATCHES} matches)")

        # League goals bar + highlight selected club
        st.subheader("Goals by Team (Season Total)")
        league_goals = team_df[['team','goals_for']].sort_values('goals_for', ascending=False)
        fig = px.bar(league_goals, x='team', y='goals_for', title='Goals by Team (2024)')
        fig.add_scatter(x=[k['team']], y=[k['goals_for']], mode='markers', name='Selected Club')
        st.plotly_chart(fig, use_container_width=True)

        # Home/Away comparison
        ha_cols = columns_if_exist(team_df, ['home_wins','away_wins','home_goals_scored','away_goals_scored','home_goals_conceded','away_goals_conceded'])
        if ha_cols:
            st.subheader("Home vs Away Performance")
            row = team_df[team_df['team'].str.lower()==selected_team.lower()].iloc[0]
            ha_df = pd.DataFrame({
            'Metric': [
                'Wins',
                'Goals For (Home)',
                'Goals For (Away)',
                'Goals Against (Home)',
                'Goals Against (Away)'
            ],
            'Value': [
                int(row.get('home_wins', 0)),
                int(row.get('home_goals_scored', 0)),
                int(row.get('away_goals_scored', 0)),
                int(row.get('home_goals_conceded', 0)),
                int(row.get('away_goals_conceded', 0))
            ],
            'Type': [
                'Home', 'Home', 'Away', 'Home', 'Away'
            ]
            })

            # Prepare grouped bar: Home vs Away for each metric type
            metrics = ['Wins', 'Goals For', 'Goals Against']
            home_vals = [
            int(row.get('home_wins', 0)),
            int(row.get('home_goals_scored', 0)),
            int(row.get('home_goals_conceded', 0))
            ]
            away_vals = [
            int(row.get('away_wins', 0)),
            int(row.get('away_goals_scored', 0)),
            int(row.get('away_goals_conceded', 0))
            ]
            ha_bar_df = pd.DataFrame({
            'Metric': metrics,
            'Home': home_vals,
            'Away': away_vals
            })
            fig_ha = go.Figure(data=[
            go.Bar(name='Home', x=ha_bar_df['Metric'], y=ha_bar_df['Home']),
            go.Bar(name='Away', x=ha_bar_df['Metric'], y=ha_bar_df['Away']),
            ])
            fig_ha.update_layout(barmode='group', title_text='Home vs Away')
            st.plotly_chart(fig_ha, use_container_width=True)
        else:
            st.info("Home/Away breakdown not available in the team CSV. Add columns like home_wins/away_wins/home_goals_scored/away_goals_scored to enable this chart.")

        # Leaderboards
        st.subheader("Club Leaderboards (Top 5)")
        lcol1, lcol2 = st.columns(2)
        top_goals = team_df[['team','goals_for']].sort_values('goals_for', ascending=False).head(5)
        lcol1.plotly_chart(px.bar(top_goals, x='team', y='goals_for', title='Most Goals (2024)'), use_container_width=True)

        if 'goals_against' in team_df.columns:
            best_def = team_df[['team','goals_against']].sort_values('goals_against').head(5)
            lcol2.plotly_chart(px.bar(best_def, x='team', y='goals_against', title='Fewest Conceded (2024)'), use_container_width=True)

        # Club players table ranked by contributions
        st.subheader(f"Top Players — {selected_team}")
        # filter players to club and (optionally) positions
        club_col = [c for c in players_df.columns if c.lower()=="club"]
        club_col = club_col[0] if club_col else [c for c in players_df.columns if 'club' in c.lower()][0]
        pos_col = positions_col or 'Position'
        club_players = players_df[players_df[club_col].str.lower()==selected_team.lower()].copy()
        if selected_positions:
            club_players = club_players[club_players[pos_col].isin(selected_positions)]
        metrics_rows = [player_metrics(r) for _, r in club_players.iterrows()]
        pm = pd.DataFrame(metrics_rows).sort_values(['goal_contributions','contribs_per90'], ascending=[False, False])
        st.dataframe(pm[['player','position','nationality','appearances','minutes','goals','assists','goal_contributions','contribs_per90']].head(20), use_container_width=True)
        if show_downloads and not pm.empty:
            st.download_button("Download club players (CSV)", pm.to_csv(index=False).encode('utf-8'), file_name=f"{selected_team}_players_2024.csv", mime='text/csv')

# -----------------------------
# PLAYER VIEW
# -----------------------------
if view == "Player":
    club_col = [c for c in players_df.columns if c.lower()=="club"]
    club_col = club_col[0] if club_col else [c for c in players_df.columns if 'club' in c.lower()][0]
    name_col = [c for c in players_df.columns if 'player' in c.lower() and 'name' in c.lower()]
    name_col = name_col[0] if name_col else [c for c in players_df.columns if 'player' in c.lower()][0]
    pos_col = positions_col or 'Position'
    nat_col = [c for c in players_df.columns if 'nationality' in c.lower()]
    nat_col = nat_col[0] if nat_col else None

    pbase = players_df[players_df[club_col].str.lower()==selected_team.lower()].copy()
    if selected_positions:
        pbase = pbase[pbase[pos_col].isin(selected_positions)]

    # quick search box
    q = st.text_input("Search player", "")
    if q:
        pbase = pbase[pbase[name_col].str.contains(q, case=False, na=False)]

    player_list = pbase[name_col].unique().tolist()
    if not player_list:
        st.warning("No players match current filters.")
    else:
        chosen = st.selectbox("Select Player", player_list)
        prow = pbase[pbase[name_col]==chosen].iloc[0]
        pm = player_metrics(prow)
        proj = simple_player_projection_2025(prow)

        # Player profile: Nationality and Position
        st.subheader("Player Profile")
        profile_cols = st.columns(3)
        profile_cols[0].markdown(f"**Name:** {pm['player']}")
        profile_cols[1].markdown(f"**Position:** {pm['position']}")
        profile_cols[2].markdown(f"**Nationality:** {pm['nationality']}")

        # KPI band
        k1,k2,k3,k4,k5 = st.columns(5)
        k1.metric("Goals", int(pm['goals']))
        k2.metric("Assists", int(pm['assists']))
        k3.metric("Contributions", int(pm['goal_contributions']))
        k4.metric("Contribs/90", None if np.isnan(pm['contribs_per90']) else round(pm['contribs_per90'],2))
        k5.metric("Minutes", int(pm['minutes']) if not np.isnan(pm['minutes']) else '—')

        # Shooting
        s1,s2,s3 = st.columns(3)
        s1.metric("Shots", int(pm['shots']))
        s2.metric("Shots on Target", int(pm['shots_on_target']))
        s3.metric("Shot Conversion", '-' if np.isnan(pm['shot_conversion']) else f"{pm['shot_conversion']:.1%}")

        # Simple 2025 projection
        st.subheader("2025 Simple Projection")
        st.write(f"If {chosen} maintains 2024 per-90 rates and full availability, projected **Goals** ≈ {proj['predicted_goals_2025']:.1f}, **Assists** ≈ {proj['predicted_assists_2025']:.1f}, **Contribs** ≈ {proj['predicted_contribs_2025']:.1f}.")

        # Compare within club (bar)
        st.subheader("Club Comparison — Contributions/90")
        club_all = players_df[players_df[club_col].str.lower()==selected_team.lower()].copy()
        club_all_metrics = pd.DataFrame([player_metrics(r) for _,r in club_all.iterrows()])
        club_all_metrics = club_all_metrics.dropna(subset=['contribs_per90']).sort_values('contribs_per90', ascending=False).head(20)
        st.plotly_chart(px.bar(club_all_metrics, x='player', y='contribs_per90', title='Top 20 by Contributions/90'), use_container_width=True)
        if show_downloads and not club_all_metrics.empty:
            st.download_button("Download contributions per90 (CSV)", club_all_metrics.to_csv(index=False).encode('utf-8'), file_name=f"{selected_team}_contribs_per90_2024.csv", mime='text/csv')

# -----------------------------
# COMPARE PLAYERS VIEW (simple visual + table, allow cross-club)
# -----------------------------
if view == "Compare Players":
    club_col = [c for c in players_df.columns if c.lower()=="club"]
    club_col = club_col[0] if club_col else [c for c in players_df.columns if 'club' in c.lower()][0]
    name_col = [c for c in players_df.columns if 'player' in c.lower() and 'name' in c.lower()]
    name_col = name_col[0] if name_col else [c for c in players_df.columns if 'player' in c.lower()][0]
    pos_col = [c for c in players_df.columns if 'position' in c.lower()]
    pos_col = pos_col[0] if pos_col else None
    nat_col = [c for c in players_df.columns if 'nationality' in c.lower()]
    nat_col = nat_col[0] if nat_col else None

    # Allow selection of players from any club
    all_players = players_df[[name_col, club_col] + ([pos_col] if pos_col else []) + ([nat_col] if nat_col else [])].drop_duplicates()
    all_players['display'] = all_players[name_col] + " (" + all_players[club_col] + ")"

    p1_display = st.selectbox("Player 1", all_players['display'].tolist(), index=0)
    p2_display = st.selectbox("Player 2", all_players['display'].tolist(), index=min(1, len(all_players)-1))

    # Get player names and clubs from display
    def parse_display(display):
        # Format: Name (Club)
        if '(' in display and display.endswith(')'):
            name = display[:display.rfind('(')].strip()
            club = display[display.rfind('(')+1:-1].strip()
            return name, club
        return display, None

    p1_name, p1_club = parse_display(p1_display)
    p2_name, p2_club = parse_display(p2_display)

    r1 = players_df[(players_df[name_col]==p1_name) & (players_df[club_col]==p1_club)].iloc[0]
    r2 = players_df[(players_df[name_col]==p2_name) & (players_df[club_col]==p2_club)].iloc[0]
    m1, m2 = player_metrics(r1), player_metrics(r2)

    # Simple bar chart for comparison
    st.subheader("Player Comparison — Key Metrics")
    compare_metrics = [
        ('Goals','goals'), ('Assists','assists'), ('Goal Contributions','goal_contributions'),
        ('Shots','shots'), ('Shots on Target','shots_on_target'), ('Appearances','appearances'), 
    ]
    labels, keys = zip(*[(lab,key) for lab,key in compare_metrics if key in m1 and key in m2])

    bar_df = pd.DataFrame({
        'Metric': labels,
        p1_name: [m1[k] for k in keys],
        p2_name: [m2[k] for k in keys]
    })

    fig_bar = px.bar(bar_df.melt(id_vars='Metric', var_name='Player', value_name='Value'),
                     x='Metric', y='Value', color='Player', barmode='group',
                     title='Player Comparison')
    st.plotly_chart(fig_bar, use_container_width=True)

    # Player profile info
    st.subheader("Player Profiles")
    prof1, prof2 = st.columns(2)
    prof1.markdown(f"**{p1_name}** ({p1_club})")
    if pos_col:
        prof1.markdown(f"**Position:** {r1[pos_col]}")
    if nat_col:
        prof1.markdown(f"**Nationality:** {r1[nat_col]}")
    prof1.markdown(f"**Shot Conversion Rate:** {'-' if np.isnan(m1['shot_conversion']) else f'{m1['shot_conversion']:.1%}'}")
    prof2.markdown(f"**{p2_name}** ({p2_club})")
    if pos_col:
        prof2.markdown(f"**Position:** {r2[pos_col]}")
    if nat_col:
        prof2.markdown(f"**Nationality:** {r2[nat_col]}")
    prof2.markdown(f"**Shot Conversion Rate:** {'-' if np.isnan(m2['shot_conversion']) else f'{m2['shot_conversion']:.1%}'}")

    # Table comparison
    comp_df = pd.DataFrame([
        {'Metric':'Goals', p1_name:m1['goals'], p2_name:m2['goals']},
        {'Metric':'Assists', p1_name:m1['assists'], p2_name:m2['assists']},
        {'Metric':'Contribs/90', p1_name:m1['contribs_per90'], p2_name:m2['contribs_per90']},
        {'Metric':'Shots', p1_name:m1['shots'], p2_name:m2['shots']},
        {'Metric':'Shots on Target', p1_name:m1['shots_on_target'], p2_name:m2['shots_on_target']},
    ])
    st.dataframe(comp_df)


# COMPARE CLUBS VIEW
# -----------------------------
if view == "Compare Clubs":
    clubs = st.multiselect("Select up to 5 clubs", all_teams, default=[selected_team], max_selections=5)
    if not clubs:
        st.info("Select at least one club to compare.")
    else:
        comp = team_df[team_df['team'].isin(clubs)].copy()
        comp['played'] = comp['wins'] + comp['draws'] + comp['losses']
        comp['ppg'] = comp['points'] / comp['played']
        comp['gpg'] = comp['goals_for'] / comp['played']
        comp['cpg'] = comp['goals_against'] / comp['played']
        if 'goal_diff' not in comp.columns:
            comp['goal_diff'] = comp['goals_for'] - comp['goals_against']

        # Comparison table
        st.subheader("Club Comparison Table")
        st.dataframe(comp[['team','points','wins','losses','goal_diff','goals_for','goals_against','ppg','gpg','cpg']])

        # Side-by-side bar charts
        c1,c2,c3 = st.columns(3)
        c1.plotly_chart(px.bar(comp, x='team', y='points', title='Points (2024)'), use_container_width=True)
        c2.plotly_chart(px.bar(comp, x='team', y='wins', title='Wins (2024)'), use_container_width=True)
        c3.plotly_chart(px.bar(comp, x='team', y='losses', title='Losses (2024)'), use_container_width=True)

        c4,c5,c6 = st.columns(3)
        c4.plotly_chart(px.bar(comp, x='team', y='goal_diff', title='Goal Difference (2024)'), use_container_width=True)
        c5.plotly_chart(px.bar(comp, x='team', y='goals_for', title='Goals Scored (2024)'), use_container_width=True)
        c6.plotly_chart(px.bar(comp, x='team', y='goals_against', title='Goals Conceded (2024)'), use_container_width=True)


        # -----------------------------
# Leaderboards
# -----------------------------
if view == "Leaderboards":
    st.title("Leaderboards — Top 10")

    top_goals = team_df[['team', 'goals_for']].sort_values('goals_for', ascending=False).head(10)
    st.subheader("Clubs with Most Goals")
    st.dataframe(top_goals)

    if 'goals_against' in team_df.columns:
        fewest_conceded = team_df[['team', 'goals_against']].sort_values('goals_against').head(10)
        st.subheader("Clubs with Least Goals Conceded")
        st.dataframe(fewest_conceded)

    player_metrics_list = [player_metrics(r) for _, r in players_df.iterrows()]
    player_df_metrics = pd.DataFrame(player_metrics_list)

    top_contribs = player_df_metrics.sort_values('goal_contributions', ascending=False).head(10)
    st.subheader("Players with Most Goal Contributions")
    st.dataframe(top_contribs[['player', 'nationality', 'goal_contributions']])

    top_goals_players = player_df_metrics.sort_values('goals', ascending=False).head(10)
    st.subheader("Players with Most Goals")
    st.dataframe(top_goals_players[['player', 'nationality', 'goals']])

    top_assists_players = player_df_metrics.sort_values('assists', ascending=False).head(10)
    st.subheader("Players with Most Assists")
    st.dataframe(top_assists_players[['player', 'nationality', 'assists']])

    st.dataframe(comp[['team','rank','points','played','wins','draws','losses','goals_for','goals_against','ppg','gpg','cpg']])


# -----------------------------
# Footer
# -----------------------------
