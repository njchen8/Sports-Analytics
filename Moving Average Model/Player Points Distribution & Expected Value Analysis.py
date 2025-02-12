import pandas as pd
import numpy as np

# ✅ Load Player Stats CSV and Underdog Props CSV
player_stats_file = "nba_players_game_logs_2018_25.csv"
parlay_file = "underdog_player_props.csv"

df_stats = pd.read_csv(player_stats_file, low_memory=False)
df_props = pd.read_csv(parlay_file)

# ✅ Standardize Player Names for Matching
df_stats["PLAYER_NAME"] = df_stats["PLAYER_NAME"].str.lower().str.strip()
df_props["Player"] = df_props["Player"].str.lower().str.strip()

# ✅ Rename "GAMEDATE" to "GAME_DATE" for consistency
df_stats.rename(columns={"GAMEDATE": "GAME_DATE"}, inplace=True)

# ✅ Ensure GAME_DATE is properly parsed
df_stats["GAME_DATE"] = pd.to_datetime(df_stats["GAME_DATE"], errors="coerce", format='%Y-%m-%dT%H:%M:%S')
df_stats = df_stats.dropna(subset=["GAME_DATE"])  # Drop invalid dates

# ✅ Compute Weight for Moving Average (Exponential Decay)
λ = 0.01  # Decay factor
df_stats["Weight"] = np.exp(-λ * (pd.to_datetime("today") - df_stats["GAME_DATE"]).dt.days)

# ✅ Ensure Numeric Columns Before Using Them
df_stats["PTS"] = pd.to_numeric(df_stats["PTS"], errors="coerce")  
df_stats["Weight"] = df_stats["Weight"].astype(float)  # Ensure Weight is float
df_stats = df_stats.dropna(subset=["PTS", "Weight"])

# ✅ Match Players from Prop File to the Database
parlay_players = df_props["Player"].unique()
df_stats = df_stats[df_stats["PLAYER_NAME"].isin(parlay_players)]  # Keep only players in props

# ✅ Debugging - Print Players Missing from the Database
missing_players = [p for p in parlay_players if p not in df_stats["PLAYER_NAME"].unique()]
if missing_players:
    print(f"⚠️ Warning: The following players in the prop file are missing from the database: {missing_players}")

# ✅ Compute Weighted Mean & Variance for Each Player
player_stats = df_stats.groupby("PLAYER_NAME").apply(
    lambda x: pd.Series({
        "Weighted Mean": np.sum(x["PTS"] * x["Weight"]),
        "Weighted Variance": np.sum(x["Weight"] * (x["PTS"] - np.sum(x["PTS"] * x["Weight"])) ** 2),
        "Weighted Std": np.sqrt(np.sum(x["Weight"] * (x["PTS"] - np.sum(x["PTS"] * x["Weight"])) ** 2))
    })
)

# ✅ Ensure At Least 5 Games Exist Per Player
player_counts = df_stats["PLAYER_NAME"].value_counts()
players_below_threshold = player_counts[player_counts < 5].index.tolist()
if players_below_threshold:
    print(f"⚠️ Warning: The following players have fewer than 5 recorded games: {players_below_threshold}")

# ✅ Filter Out Players with Insufficient Data
player_stats = player_stats.loc[~player_stats.index.isin(players_below_threshold)]

# ✅ Print Final Computed Player Stats
print("\n✅ Computed Weighted Averages for Players:")
print(player_stats)
