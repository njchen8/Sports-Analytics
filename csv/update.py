import requests
import pandas as pd
import time
import os

# NBA API Endpoints
NBA_GAME_LOGS_URL = "https://stats.nba.com/stats/playergamelogs"
NBA_PLAYER_LIST_URL = "https://stats.nba.com/stats/commonallplayers"

# Request headers (NBA.com requires user-agent and referer)
HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nba.com/stats/players/traditional",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true"
}

# Current season
CURRENT_SEASON = "2024-25"

# Ask user for season type
while True:
    season_type = input("Enter season type (Regular Season or Playoffs): ").strip()
    if season_type in ["Regular Season", "Playoffs"]:
        break
    print("Invalid input. Please enter 'Regular Season' or 'Playoffs'.")

# CSV file based on season type
csv_file = "nba_players_game_logs_2018_25.csv" if season_type == "Regular Season" else "nba_players_game_logs_2018_25_playoffs.csv"

# Step 1: Get active players from NBA API
params = {
    "LeagueID": "00",
    "Season": CURRENT_SEASON,
    "IsOnlyCurrentSeason": "1"
}

response = requests.get(NBA_PLAYER_LIST_URL, headers=HEADERS, params=params)
response.raise_for_status()

players_data = response.json()
headers = players_data["resultSets"][0]["headers"]
rows = players_data["resultSets"][0]["rowSet"]

players_df = pd.DataFrame(rows, columns=headers)
player_ids = players_df[["PERSON_ID", "DISPLAY_FIRST_LAST"]].drop_duplicates()
player_ids.columns = ["PLAYER_ID", "PLAYER_NAME"]

print(f"✅ Retrieved {len(player_ids)} players")

# Load existing data if CSV exists
if os.path.exists(csv_file):
    existing_df = pd.read_csv(csv_file, low_memory=False)
    print(f"🔄 Loaded {len(existing_df)} existing rows")
else:
    existing_df = pd.DataFrame()

# Main loop
for player_id, player_name in player_ids.itertuples(index=False):
    print(f"\n🔍 Fetching logs for {player_name} (ID: {player_id})")

    if not existing_df.empty and "PLAYER_ID" in existing_df.columns:
        player_data = existing_df[existing_df["PLAYER_ID"] == player_id]
        if not player_data.empty:
            latest_game_date = pd.to_datetime(player_data["GAME_DATE"].max(), format="%Y-%m-%dT%H:%M:%S")
        else:
            latest_game_date = pd.Timestamp("1900-01-01")
    else:
        latest_game_date = pd.Timestamp("1900-01-01")

    params = {
        "PlayerID": player_id,
        "LeagueID": "00",
        "Season": CURRENT_SEASON,
        "SeasonType": season_type
    }

    try:
        response = requests.get(NBA_GAME_LOGS_URL, headers=HEADERS, params=params)
        response.raise_for_status()
    except Exception as e:
        print(f"  ❌ Request failed for {player_name}: {e}")
        continue

    data = response.json()
    log_headers = data["resultSets"][0]["headers"]
    log_rows = data["resultSets"][0]["rowSet"]

    if not log_rows:
        print("  🟢 No games found for this player.")
        continue

    logs_df = pd.DataFrame(log_rows, columns=log_headers)
    logs_df["PLAYER_ID"] = player_id
    logs_df["PLAYER_NAME"] = player_name
    logs_df["Season"] = CURRENT_SEASON

    # Format GAME_DATE and filter new rows
    logs_df["GAME_DATE"] = pd.to_datetime(logs_df["GAME_DATE"])
    new_games = logs_df[logs_df["GAME_DATE"] > latest_game_date]

    if new_games.empty:
        print("  🟢 No new games to append.")
        continue

    # Format GAME_DATE to match historical CSV
    new_games["GAME_DATE"] = new_games["GAME_DATE"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    # Append with correct header handling
    write_header = not os.path.exists(csv_file)
    new_games.to_csv(csv_file, mode='a', header=write_header, index=False)
    print(f"  ✅ Appended {len(new_games)} new games.")

    
    time.sleep(.1)  # Respect NBA API rate limits

# Final sort step: sort by PLAYER_NAME, then GAME_DATE
if os.path.exists(csv_file):
    final_df = pd.read_csv(csv_file, low_memory=False)
    
    # Ensure GAME_DATE is parsed correctly
    final_df["GAME_DATE"] = pd.to_datetime(final_df["GAME_DATE"], format="%Y-%m-%dT%H:%M:%S")
    
    final_df = final_df.sort_values(by=["PLAYER_NAME", "GAME_DATE"]).reset_index(drop=True)
    
    # Reformat date back to string
    final_df["GAME_DATE"] = final_df["GAME_DATE"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    
    final_df.to_csv(csv_file, index=False)
    print(f"\n✅ Sorted and updated file by PLAYER_NAME and GAME_DATE")

print(f"\n🎉 Game logs successfully updated in: {csv_file}")
