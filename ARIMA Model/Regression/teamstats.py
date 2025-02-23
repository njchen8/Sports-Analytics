from nba_api.stats.endpoints import leaguedashteamstats
import pandas as pd

# Fetch team stats
team_stats = leaguedashteamstats.LeagueDashTeamStats(
    season='2024-25',
    season_type_all_star='Regular Season',
    per_mode_detailed='PerGame',
    measure_type_detailed_defense='Advanced'
)

# Convert to DataFrame
team_stats_df = team_stats.get_data_frames()[0]

# Print available columns for debugging
print("Available columns:", team_stats_df.columns)

# Mapping team names to abbreviations
team_name_to_abbreviation = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN", "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE", "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET", "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "LA Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM", "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS"
}

# Add team abbreviation column
team_stats_df['TEAM_ABBREVIATION'] = team_stats_df['TEAM_NAME'].map(team_name_to_abbreviation)

# Select relevant columns
team_stats_df = team_stats_df[['TEAM_ABBREVIATION', 'OFF_RATING', 'DEF_RATING', 'NET_RATING', 'PACE', 'AST_RATIO', 'REB_PCT']]

# Save to CSV
print(team_stats_df)
team_stats_df.to_csv('team_stats.csv', index=False)
