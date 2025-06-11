import requests
import pandas as pd
import time
import os

# NBA API Endpoint for Player Game Logs
NBA_GAME_LOGS_URL = "https://stats.nba.com/stats/playergamelogs"
NBA_PLAYER_LIST_URL = "https://stats.nba.com/stats/commonallplayers"

# Request Headers (Required by NBA.com)
HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nba.com/stats/players/traditional",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true"
}

# The season range we need
SEASONS = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]

# Ask user for season type
while True:
    season_type = input("Enter season type (Regular Season or Playoffs): ").strip()
    if season_type in ["Regular Season", "Playoffs"]:
        break
    print("Invalid input. Please enter 'Regular Season' or 'Playoffs'.")

# Determine the correct filename
if season_type == "Regular Season":
    csv_file = "nba_players_game_logs_2018_25.csv"
else:
    csv_file = "nba_players_game_logs_2018_25_playoffs.csv"

# Step 1: Get All Players Who Played in 2024-2025 (active players in the current season)
params = {
    "LeagueID": "00",
    "Season": "2024-25",
    "IsOnlyCurrentSeason": "1"
}

response = requests.get(NBA_PLAYER_LIST_URL, headers=HEADERS, params=params)

if response.status_code == 200:
    players_data = response.json()
    player_headers = players_data["resultSets"][0]["headers"]
    player_rows = players_data["resultSets"][0]["rowSet"]

    # Convert to DataFrame
    players_df = pd.DataFrame(player_rows, columns=player_headers)

    # Extract Player IDs & Names (active players only)
    player_ids = players_df[["PERSON_ID", "DISPLAY_FIRST_LAST"]].drop_duplicates()
    player_ids.columns = ["PLAYER_ID", "PLAYER_NAME"]

    print(f"Successfully fetched {len(player_ids)} players!")
else:
    print(f"Failed to fetch player list. HTTP Status Code: {response.status_code}")
    exit()

# Step 2: Prepare CSV file for appending data
header_written = False  # Track if header is already written in CSV

# Check if the CSV file exists
if os.path.exists(csv_file):
    # Read the last row from the existing CSV to find the last processed player
    existing_df = pd.read_csv(csv_file, low_memory=False)
    last_player_name = existing_df["Player_Name"].iloc[-1]
    
    # Find index of the last player in player_ids to start from the next player
    start_index = player_ids[player_ids["PLAYER_NAME"] == last_player_name].index[0] + 1
    print(f"Resuming from player: {last_player_name}")
else:
    start_index = 0
    print("Starting from the first player.")

# Step 3: Fetch Game Logs for Each Player
for player_id, player_name in player_ids.iloc[start_index:].itertuples(index=False):
    print(f"Fetching data for {player_name} (ID: {player_id})...")

    for season in SEASONS:
        params = {
            "PlayerID": player_id,
            "LeagueID": "00",
            "Season": season,
            "SeasonType": season_type  # User-selected season type
        }

        response = requests.get(NBA_GAME_LOGS_URL, headers=HEADERS, params=params)

        if response.status_code == 200:
            data = response.json()
            headers = data["resultSets"][0]["headers"]
            rows = data["resultSets"][0]["rowSet"]

            if rows:
                df = pd.DataFrame(rows, columns=headers)
                df["Player_ID"] = player_id
                df["Player_Name"] = player_name
                df["Season"] = season

                # Append data to CSV as we go
                df.to_csv(csv_file, mode='a', header=not header_written, index=False)
                header_written = True

        # Prevent API blocking
        time.sleep(.1)