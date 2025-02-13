import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from datetime import datetime, timedelta

# Load the datasets
nba_game_logs_file = "nba_players_game_logs_2018_25.csv"
parlay_file = "underdog_home_team_props.csv"

# Read NBA game logs dataset
nba_data = pd.read_csv(nba_game_logs_file, low_memory=False)

# Rename columns for consistency
nba_data.rename(columns={
    "GAME_ID": "game_id",
    "PLAYER_NAME": "player",
    "TEAM_ABBREVIATION": "team",
    "SEASON_YEAR": "season",
    "GAME_DATE": "game_date",
    "PTS": "pts",
    "AST": "ast",
    "REB": "reb",
    "MIN": "minutes"
}, inplace=True)

# Keep only relevant columns
nba_data = nba_data[['game_id', 'player', 'team', 'season', 'game_date', 'minutes', 'pts', 'ast', 'reb']]

# Convert necessary columns to appropriate data types
nba_data['season'] = pd.to_numeric(nba_data['season'], errors='coerce').astype('Int64')
nba_data['game_date'] = pd.to_datetime(nba_data['game_date'], errors='coerce')
nba_data['minutes'] = pd.to_numeric(nba_data['minutes'], errors='coerce')
nba_data['pts'] = pd.to_numeric(nba_data['pts'], errors='coerce')
nba_data['ast'] = pd.to_numeric(nba_data['ast'], errors='coerce')
nba_data['reb'] = pd.to_numeric(nba_data['reb'], errors='coerce')

# Drop rows where key columns have missing values
nba_data = nba_data.dropna(subset=['minutes', 'game_date', 'pts', 'ast', 'reb'])

# Normalize player names and team names
nba_data['player'] = nba_data['player'].str.strip().str.lower()
nba_data['team'] = nba_data['team'].str.strip().str.upper()

# Load the parlay dataset
parlay_data = pd.read_csv(parlay_file)

# Normalize player names and stat type
parlay_data['Player'] = parlay_data['Player'].str.strip().str.lower()
parlay_data['Stat Type'] = parlay_data['Stat Type'].str.strip().str.lower()

# Standardize stat names to match NBA data
stat_map = {
    'points': 'pts',
    'assists': 'ast',
    'rebounds': 'reb'
}
parlay_data['Stat Type'] = parlay_data['Stat Type'].replace(stat_map)

# Get unique valid player-stat pairs
valid_props = set(zip(parlay_data['Player'], parlay_data['Stat Type']))

# Compute correlations using past games
correlations = set()  # Use a set to avoid duplicates

# Get today's date and calculate the one-month cutoff
today = nba_data['game_date'].max()
one_month_ago = today - timedelta(days=30)

# Process each team separately
for team, team_data in nba_data.groupby('team'):
    team_players = team_data['player'].unique()

    # Only keep players from the parlay list
    relevant_players = [p for p in team_players if p in parlay_data['Player'].unique()]
    if len(relevant_players) < 2:
        continue

    # Restructure team data for correlation analysis
    pivoted_data = team_data.pivot_table(index='game_id', columns='player', values=['pts', 'ast', 'reb', 'minutes'])

    # Compute correlations over multiple games
    for (p1, p2) in product(relevant_players, repeat=2):
        if p1 >= p2:  # Avoid duplicate pairs (p1, p2) and (p2, p1)
            continue
        for stat1 in ['pts', 'ast', 'reb']:
            for stat2 in ['pts', 'ast', 'reb']:
                try:
                    # Check if the player-stat pair exists in valid props
                    if (p1, stat1) not in valid_props or (p2, stat2) not in valid_props:
                        continue  # Skip if either stat is not in the prop list

                    # Extract historical stats over multiple games
                    p1_stat = pivoted_data[stat1][p1]
                    p2_stat = pivoted_data[stat2][p2]
                    p1_minutes = pivoted_data['minutes'][p1]
                    p2_minutes = pivoted_data['minutes'][p2]

                    # **Ensure both players played in the same game and had at least 5 minutes**
                    common_games = p1_stat.dropna().index.intersection(p2_stat.dropna().index)
                    valid_games = common_games[(p1_minutes.loc[common_games] >= 5) & (p2_minutes.loc[common_games] >= 5)]
                    
                    if len(valid_games) < 10:  # Ensure at least 10 valid games
                        continue

                    # **Ensure at least 5 of the valid games were in the last month**
                    last_month_games = valid_games[valid_games.isin(
                        nba_data[nba_data['game_date'] >= one_month_ago]['game_id']
                    )]
                    if len(last_month_games) < 5:
                        continue

                    # Compute correlation using last 10 valid games
                    last_10_games = valid_games[-10:]
                    p1_last_10 = p1_stat.loc[last_10_games]
                    p2_last_10 = p2_stat.loc[last_10_games]

                    # **Filter out players with zero variance**
                    if p1_last_10.std() == 0 or p2_last_10.std() == 0:
                        continue

                    # Compute correlation
                    corr_value = p1_last_10.corr(p2_last_10)

                    # Avoid NaN or perfect 1.0 correlation due to missing data
                    if not np.isnan(corr_value) and abs(corr_value) < 1.0:
                        correlations.add((p1.title(), stat1, p2.title(), stat2, corr_value, tuple(last_10_games)))

                except KeyError:
                    continue  # Skip if data for both players does not exist in all games

# Convert to DataFrame and sort by absolute correlation
correlations_df = pd.DataFrame(list(correlations), columns=['Player 1', 'Stat 1', 'Player 2', 'Stat 2', 'Correlation', 'Last 10 Games'])
correlations_df = correlations_df.sort_values(by='Correlation', key=abs, ascending=False)

# Save to CSV
correlations_df.drop(columns=['Last 10 Games']).to_csv("top_teammate_correlations_filtered.csv", index=False)

# Display top results
print("Top Correlated Player Props:")
print(correlations_df.head(10))

# Plot correlation for the top correlated pair
if not correlations_df.empty:
    top_pair = correlations_df.iloc[0]
    p1, stat1, p2, stat2 = top_pair['Player 1'], top_pair['Stat 1'], top_pair['Player 2'], top_pair['Stat 2']

    # Extract last 10 games for these two players
    last_10_games = top_pair['Last 10 Games']
    p1_stats = nba_data[(nba_data['player'] == p1.lower()) & (nba_data['game_id'].isin(last_10_games))][['game_id', stat1]]
    p2_stats = nba_data[(nba_data['player'] == p2.lower()) & (nba_data['game_id'].isin(last_10_games))][['game_id', stat2]]

    # Merge on game_id
    merged_stats = pd.merge(p1_stats, p2_stats, on='game_id').sort_values(by='game_id')

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=merged_stats[stat1], y=merged_stats[stat2], hue=merged_stats['game_id'], palette="viridis", edgecolor="black", s=100)
    plt.xlabel(f"{p1} {stat1}")
    plt.ylabel(f"{p2} {stat2}")
    plt.title(f"Correlation: {round(top_pair['Correlation'], 2)}")
    plt.legend(title="Game ID")
    plt.grid()
    plt.show()
