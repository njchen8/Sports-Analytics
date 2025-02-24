import requests
import pandas as pd
import time
import os

# NBA API Endpoints
NBA_GAME_LOGS_URL = "https://stats.nba.com/stats/playergamelogs"
NBA_PLAYERS_URL = "https://stats.nba.com/stats/leaguedashplayerbiostats"

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

# Load fixed data and extract player names (ignore missing PlayerID)
fixed_data_df = pd.read_csv("fixed_data.csv", usecols=["Player"])
player_names = fixed_data_df["Player"].unique()

# Load existing game logs
if os.path.exists(CSV_FILE):
    game_logs_df = pd.read_csv(CSV_FILE, low_memory=False, usecols=["PLAYER_ID", "GAME_ID"])
    existing_games = set(zip(game_logs_df["PLAYER_ID"], game_logs_df["GAME_ID"]))
    print(f"Loaded existing data: {len(existing_games)} games found.")
else:
    existing_games = set()
    print("No existing game logs found. Creating new dataset.")

# Cache for player IDs
player_ids = {}

def get_player_id(player_name):
    """Find the player's ID using the NBA API."""
    if player_name in player_ids:
        return player_ids[player_name]  # Return cached ID if available

    print(f"🔎 Searching for Player ID: {player_name}...")

    params = {
        "LeagueID": "00",
        "Season": CURRENT_SEASON,
        "PerMode": "PerGame"
    }
    
    response = requests.get(NBA_PLAYERS_URL, headers=HEADERS, params=params)
    
    if response.status_code == 200:
        data = response.json()
        players = data["resultSets"][0]["rowSet"]
        player_columns = data["resultSets"][0]["headers"]
        players_df = pd.DataFrame(players, columns=player_columns)
        
        # Find the row matching the player's name
        match = players_df[players_df["PLAYER_NAME"].str.lower() == player_name.lower()]
        if not match.empty:
            player_id = match.iloc[0]["PLAYER_ID"]
            player_ids[player_name] = player_id  # Cache it for later
            print(f"✅ Found Player ID: {player_id} for {player_name}")
            return player_id
        else:
            print(f"❌ Player ID not found for {player_name}. Skipping.")
            return None
    else:
        print(f"❌ Failed to fetch player list. HTTP Status: {response.status_code}")
        return None

def fetch_game_logs(player_name):
    """Fetch new game logs for a specific player."""
    player_id = get_player_id(player_name)
    if player_id is None:
        return pd.DataFrame()  # Skip player if ID is not found

    print(f"\nFetching data for {player_name} (ID: {player_id})...")

    params = {
        "PlayerOrTeam": "P",
        "PlayerID": player_id,  # Fetch only this player's games
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

        # Standardize Date Format
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="%Y-%m-%dT%H:%M:%S")

        # Ensure only this player's logs are added
        df = df[df["PLAYER_ID"] == player_id]
        print(f"  ➡️  Found {len(df)} total games for {player_name}.")

        # Filter out duplicate games
        df = df[~df.apply(lambda row: (row["PLAYER_ID"], row["GAME_ID"]) in existing_games, axis=1)]
        print(f"  ✅  {len(df)} new games after filtering out duplicates.")

        return df
    else:
        print(f"  ❌  Failed to fetch data for {player_name}. HTTP Status Code: {response.status_code}")
        return pd.DataFrame()

# Fetch and append new data
new_games_list = []

for player in player_names:
    new_games = fetch_game_logs(player)

    if not new_games.empty:
        new_games_list.append(new_games)

    time.sleep(0.1)  # Avoid rate limiting

# Append new data to CSV if there are new records
if new_games_list:
    new_games_df = pd.concat(new_games_list, ignore_index=True)
    new_games_df.to_csv(CSV_FILE, mode='a', header=not os.path.exists(CSV_FILE), index=False)
    print(f"\n📌 Appended {len(new_games_df)} new games to {CSV_FILE}")
else:
    print("\n✅ No new games found. Data is already up to date.")
