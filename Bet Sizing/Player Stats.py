import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the player game logs dataset
logs_file_path = "nba_players_game_logs_2018_25.csv"
logs_df = pd.read_csv(logs_file_path)

# Convert GAME_DATE to datetime
logs_df["GAME_DATE"] = pd.to_datetime(logs_df["GAME_DATE"], errors="coerce")

# Ensure numerical data types for key stats
stats_columns = ["PTS", "REB", "AST", "MIN"]
for col in stats_columns:
    logs_df[col] = pd.to_numeric(logs_df[col], errors="coerce")

# User input for player name and opponent team
player_name = input("Enter player name (e.g., 'LeBron James'): ").strip().lower()
opponent_team = input("Enter opponent team (e.g., 'BOS'): ").strip().upper()

# Normalize data for case insensitivity
logs_df["PLAYER_NAME"] = logs_df["PLAYER_NAME"].str.strip().str.lower()
logs_df["MATCHUP"] = logs_df["MATCHUP"].str.upper()

# Filter player data
player_df = logs_df[logs_df["PLAYER_NAME"] == player_name].copy()

if player_df.empty:
    print(f"No data found for player: {player_name}")
    exit()

# Remove games where the player played less than 5 minutes
player_df = player_df[player_df["MIN"] >= 5]

# Sort by date (most recent first)
player_df = player_df.sort_values(by="GAME_DATE", ascending=False)

# Filter last 5 games against the specified opponent
player_vs_opponent = player_df[player_df["MATCHUP"].str.contains(opponent_team, na=False)].head(5)

# Filter last 10 overall games
player_last_10 = player_df.head(10)

# Function to plot stats side by side
def plot_combined_stats(player_vs_opponent, player_last_10, stat, title):
    plt.figure(figsize=(12, 5))
    
    # Adjusting data points spacing
    if len(player_vs_opponent) > 0:
        plt.plot(np.linspace(0, 10, len(player_vs_opponent)), player_vs_opponent[stat], marker='o', linestyle='-', label=f"Last 5 vs Opponent ({stat})")
    
    if len(player_last_10) > 0:
        plt.plot(range(10), player_last_10[stat], marker='s', linestyle='-', label=f"Last 10 Overall ({stat})")
    
    plt.xlabel("Games (Scaled for Last 5)")
    plt.ylabel(stat)
    plt.title(title)
    plt.xticks(range(10))
    plt.legend()
    plt.grid()
    plt.show()

# Plot combined stats for PTS, AST, REB
for stat in ["PTS", "AST", "REB"]:
    plot_combined_stats(player_vs_opponent, player_last_10, stat, f"{player_name.title()} - Last 5 vs {opponent_team} & Last 10 Overall ({stat})")

# Calculate and display averages
averages_vs_opponent = player_vs_opponent[["PTS", "AST", "REB"]].mean()
averages_last_10 = player_last_10[["PTS", "AST", "REB"]].mean()

print(f"\n🔹 Averages for {player_name.title()} in Last 5 Games vs {opponent_team}:")
print(averages_vs_opponent.round(2))

print(f"\n🔹 Averages for {player_name.title()} in Last 10 Overall Games:")
print(averages_last_10.round(2))
