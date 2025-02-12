import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the player game logs dataset
logs_file_path = "nba_players_game_logs_2018_25.csv"
logs_df = pd.read_csv(logs_file_path)

# Convert GAME_DATE to datetime
logs_df["GAME_DATE"] = pd.to_datetime(logs_df["GAME_DATE"], errors="coerce")

# Ensure numerical data types for key stats
stats_columns = ["PTS", "REB", "AST"]
for col in stats_columns:
    logs_df[col] = pd.to_numeric(logs_df[col], errors="coerce")

# User input for player name
player_name = input("Enter player name (e.g., 'LeBron James'): ")
opponent_team = input("Enter opponent team (e.g., 'BOS'): ")

# Filter player data
player_df = logs_df[logs_df["PLAYER_NAME"] == player_name].copy()

if player_df.empty:
    print(f"No data found for player: {player_name}")
    exit()

# Sort by date
player_df = player_df.sort_values(by="GAME_DATE")

# Drop NaN values for key stats
player_df = player_df.dropna(subset=stats_columns)

# Apply weights: 
# - Recent matches are weighted higher (exponential decay)
# - Matches vs the same opponent are weighted even more

player_df["Recency_Weight"] = np.exp(-np.arange(len(player_df))[::-1] / (len(player_df) / 3))
player_df["Matchup_Weight"] = np.where(player_df["MATCHUP"].str.contains(opponent_team, na=False), 1.5, 1)

# Final weight = recency weight * matchup weight
player_df["Final_Weight"] = player_df["Recency_Weight"] * player_df["Matchup_Weight"]

# Apply weights to each stat
weighted_stats = {}
for stat in stats_columns:
    weighted_stats[stat] = (player_df[stat] * player_df["Final_Weight"]).sum() / player_df["Final_Weight"].sum()

# Print weighted stats
print(f"\n🔹 Weighted Averages for {player_name} vs {opponent_team}:")
for stat, value in weighted_stats.items():
    print(f"   {stat}: {round(value, 2)}")

# Plot time series for weighted performance
plt.figure(figsize=(12, 5))
for stat in stats_columns:
    plt.plot(player_df["GAME_DATE"], player_df[stat] * player_df["Final_Weight"], label=f"{stat} (Weighted)")

plt.xlabel("Date")
plt.ylabel("Stat Value")
plt.title(f"{player_name}'s Weighted Performance Over Time")
plt.legend()
plt.show()