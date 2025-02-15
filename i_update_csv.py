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

# The current season
CURRENT_SEASON = "2024-25"

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
    "Season": CURRENT_SEASON,
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

# Step 2: Fetch Game Logs for Each Player and Append New Games for the Current Season
for player_id, player_name in player_ids.itertuples(index=False):
    print(f"Fetching data for {player_name} (ID: {player_id})...")

    # Read the existing CSV to see if there is any data for this player
    if os.path.exists(csv_file):
        existing_df = pd.read_csv(csv_file, low_memory=False)

        if "GAME_ID" in existing_df.columns:
            player_existing_data = existing_df[existing_df["PLAYER_ID"] == player_id]
            
            # Find the latest GAME_DATE for the player
            if not player_existing_data.empty:
                latest_game_date = player_existing_data["GAME_DATE"].max()
            else:
                latest_game_date = "1900-01-01T00:00:00"  # A date far in the past in case no games exist for this player
        else:
            latest_game_date = "1900-01-01T00:00:00"
    else:
        latest_game_date = "1900-01-01T00:00:00"

    # Fetch game logs for the current season and season type
    params = {
        "PlayerID": player_id,
        "LeagueID": "00",
        "Season": CURRENT_SEASON,
        "SeasonType": season_type  # User-selected season type
    }

    response = requests.get(NBA_GAME_LOGS_URL, headers=HEADERS, params=params)

    if response.status_code == 200:
        data = response.json()
        headers = data["resultSets"][0]["headers"]
        rows = data["resultSets"][0]["rowSet"]

        if rows:
            df = pd.DataFrame(rows, columns=headers)
            df["PLAYER_ID"] = player_id
            df["PLAYER_NAME"] = player_name
            df["Season"] = CURRENT_SEASON

            # Convert the GAME_DATE column to datetime, matching the format of the data
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="%Y-%m-%dT%H:%M:%S")

            # Filter the new games based on GAME_DATE being after the latest existing game date
            new_games = df[df["GAME_DATE"] > pd.to_datetime(latest_game_date, format="%Y-%m-%dT%H:%M:%S")]

            # Log how many new games are found for this player
            print(f"New games found for {player_name} (ID: {player_id}): {len(new_games)}")

            if not new_games.empty:
                # Append only new games to CSV without writing the header again
                new_games.to_csv(csv_file, mode='a', header=False, index=False)
                print(f"Appended {len(new_games)} new games for {player_name} (ID: {player_id})")

    # Prevent API blocking
    time.sleep(1)

# Inform user about completion
print(f"Game logs have been successfully fetched and appended to {csv_file}!")
