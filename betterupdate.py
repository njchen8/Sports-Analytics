import requests
import pandas as pd
import time
import os

# NBA API Endpoints
NBA_GAME_LOGS_URL = "https://stats.nba.com/stats/playergamelogs"

# Request Headers (Required by NBA.com)
HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nba.com/stats/players/traditional",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true"
}

# The current season
CURRENT_SEASON = "2024-25"
SEASON_TYPE = "Regular Season"
CSV_FILE = "nba_players_game_logs_2018_25.csv"

# Load fixed data and extract player names
fixed_data_df = pd.read_csv("fixed_data.csv", usecols=["Player"])
player_names = fixed_data_df["Player"].unique()

# Load existing game logs
game_logs_df = pd.read_csv(CSV_FILE, low_memory=False) if os.path.exists(CSV_FILE) else pd.DataFrame()

def get_latest_game_date(player_name):
    """Get the latest game date for a player."""
    if not game_logs_df.empty and "GAME_DATE" in game_logs_df.columns:
        player_games = game_logs_df[game_logs_df["PLAYER_NAME"] == player_name]
        if not player_games.empty:
            return pd.to_datetime(player_games["GAME_DATE"], errors='coerce').max()
    return pd.Timestamp("1900-01-01")

def fetch_game_logs(player_name):
    """Fetch new game logs for a player."""
    print(f"Fetching data for {player_name}...")
    
    params = {
        "PlayerOrTeam": "P",
        "Season": CURRENT_SEASON,
        "SeasonType": SEASON_TYPE,
        "LeagueID": "00"
    }
    
    response = requests.get(NBA_GAME_LOGS_URL, headers=HEADERS, params=params)
    
    if response.status_code == 200:
        data = response.json()
        headers = data["resultSets"][0]["headers"]
        rows = data["resultSets"][0]["rowSet"]
        df = pd.DataFrame(rows, columns=headers)
        df["PLAYER_NAME"] = player_name
        df["Season"] = CURRENT_SEASON
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="%Y-%m-%dT%H:%M:%S")
        
        # Fetch the real nickname from existing data
        if "NICKNAME" in game_logs_df.columns:
            existing_nickname = game_logs_df.loc[game_logs_df["PLAYER_NAME"] == player_name, "NICKNAME"].unique()
            if len(existing_nickname) > 0:
                df = df[df["NICKNAME"].isin(existing_nickname)]
        
        return df
    else:
        print(f"Failed to fetch data for {player_name}. HTTP Status Code: {response.status_code}")
        return pd.DataFrame()

# Fetch and append new data
new_games_list = []

for player in player_names:
    latest_game_date = get_latest_game_date(player)
    new_games = fetch_game_logs(player)
    if not new_games.empty:
        new_games = new_games[new_games["GAME_DATE"] > latest_game_date]
        if not new_games.empty:
            new_games_list.append(new_games)
    time.sleep(0.1)  # Avoid rate limiting

# Append new data to CSV if there are new records
if new_games_list:
    new_games_df = pd.concat(new_games_list, ignore_index=True)
    new_games_df.to_csv(CSV_FILE, mode='a', header=not os.path.exists(CSV_FILE), index=False)
    print(f"Appended {len(new_games_df)} new games to {CSV_FILE}")
else:
    print("No new games found.")
