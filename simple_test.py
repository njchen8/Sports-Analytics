#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

print("SIMPLIFIED OPTIMIZED MODEL - TESTING")
print("Loading data...")

CSV_PLAYERS = Path("nba_players_game_logs_2018_25_clean.csv")
CSV_TEAMS   = Path("team_stats.csv")

df = pd.read_csv(CSV_PLAYERS, parse_dates=["GAME_DATE"])
team_stats = pd.read_csv(CSV_TEAMS) if CSV_TEAMS.exists() else pd.DataFrame()

print(f"Data loaded: {df.shape[0]} games, {len(team_stats)} teams")

lebron_data = df[df["PLAYER_NAME"].str.contains("LeBron", case=False, na=False)]
print(f"LeBron games found: {len(lebron_data)}")
print("Basic test completed successfully!")
