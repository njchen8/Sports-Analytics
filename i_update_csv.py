import requests
import pandas as pd
import time
import os

# NBA API Endpoints
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

# Get valid season type input
def get_season_type():
    while True:
        season_type = input("Enter season type (Regular Season or Playoffs): ").strip()
        if season_type in ["Regular Season", "Playoffs"]:
            return season_type
        print("Invalid input. Please enter 'Regular Season' or 'Playoffs'.")

SEASON_TYPE = get_season_type()

# Determine the correct filename
CSV_FILE = "nba_players_game_logs_2018_25.csv" if SEASON_TYPE == "Regular Season" else "nba_players_game_logs_2018_25_playoffs.csv"

# Function to fetch all active players for the current season
def fetch_active_players():
    params = {"LeagueID": "00", "Season": CURRENT_SEASON, "IsOnlyCurrentSeason": "1"}
    response = requests.get(NBA_PLAYER_LIST_URL, headers=HEADERS, params=params)

    if response.status_code == 200:
        players_data = response.json()
        player_headers = players_data["resultSets"][0]["headers"]
        player_rows = players_data["resultSets"][0]["rowSet"]

        players_df = pd.DataFrame(player_rows, columns=player_headers)
        player_ids = players_df[["PERSON_ID", "DISPLAY_FIRST_LAST"]].drop_duplicates()
        player_ids.columns = ["PLAYER_ID", "PLAYER_NAME"]

        print(f"Successfully fetched {len(player_ids)} players!")
        return player_ids
    else:
        print(f"Failed to fetch player list. HTTP Status Code: {response.status_code}")
        exit()

# Function to get the latest game date for a player and remove duplicate games
def get_latest_game_date_and_remove_duplicates(player_id, new_data_df):
    if os.path.exists(CSV_FILE):
        existing_df = pd.read_csv(CSV_FILE, low_memory=False)

        if "GAME_ID" in existing_df.columns:
            player_existing_data = existing_df[existing_df["PLAYER_ID"] == player_id]

            if not player_existing_data.empty:
                latest_game_date = player_existing_data["GAME_DATE"].max()
            else:
                latest_game_date = "1900-01-01T00:00:00"

            # Remove games that are already in the CSV
            new_data_df = new_data_df[~new_data_df["GAME_ID"].isin(existing_df["GAME_ID"])]
        else:
            latest_game_date = "1900-01-01T00:00:00"
    else:
        latest_game_date = "1900-01-01T00:00:00"

    return new_data_df, latest_game_date

# Function to fetch game logs for a player
def fetch_game_logs(player_id, player_name):
    print(f"Fetching data for {player_name} (ID: {player_id})...")

    params = {
        "PlayerID": player_id,
        "LeagueID": "00",
        "Season": CURRENT_SEASON,
        "SeasonType": SEASON_TYPE
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

            # Convert GAME_DATE to datetime
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="%Y-%m-%dT%H:%M:%S")

            # Remove duplicate games and get the latest game date
            new_games, latest_game_date = get_latest_game_date_and_remove_duplicates(player_id, df)

            # Filter by latest game date
            filtered_new_games = new_games[new_games["GAME_DATE"] > pd.to_datetime(latest_game_date, format="%Y-%m-%dT%H:%M:%S")]

            print(f"New games found for {player_name} (ID: {player_id}): {len(filtered_new_games)}")

            return filtered_new_games
    else:
        print(f"Failed to fetch data for {player_name} (ID: {player_id}). HTTP Status Code: {response.status_code}")

    return pd.DataFrame()  # Return empty DataFrame if request fails

# Main script execution
if __name__ == "__main__":
    player_list = fetch_active_players()

    for player_id, player_name in player_list.itertuples(index=False):
        new_game_data = fetch_game_logs(player_id, player_name)

        if not new_game_data.empty:
            # Append new games to CSV
            new_game_data.to_csv(CSV_FILE, mode='a', header=not os.path.exists(CSV_FILE), index=False)
            print(f"Appended {len(new_game_data)} new games for {player_name} (ID: {player_id})")

        # Avoid getting blocked by API rate limits
        time.sleep(1)

    print(f"Game logs have been successfully fetched and appended to {CSV_FILE}!")
