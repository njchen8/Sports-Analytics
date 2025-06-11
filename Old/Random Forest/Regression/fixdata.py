import pandas as pd
import os

# Define the CSV file path
CSV_FILE = "nba_players_game_logs_2018_25.csv"

# Check if the file exists
if not os.path.exists(CSV_FILE):
    print(f"Error: {CSV_FILE} does not exist.")
    exit()

# Load the game logs data
df = pd.read_csv(CSV_FILE)

# Ensure that the GAME_DATE column is treated as datetime
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors='coerce')

# For each player, sort their games by date (most recent first) and keep only the top 100
trimmed_df = df.sort_values("GAME_DATE", ascending=False)\
               .groupby("PLAYER_NAME", as_index=False)\
               .head(100)

# Optionally, re-sort the final dataset by player and game date in ascending order
trimmed_df = trimmed_df.sort_values(["PLAYER_NAME", "GAME_DATE"])

# Save the trimmed data back to the CSV file (overwriting it)
trimmed_df.to_csv(CSV_FILE, index=False)

print(f"Trimmed data saved to {CSV_FILE}. Total records: {len(trimmed_df)}")
