import requests
import pandas as pd

# NBA API Endpoint for Player Game Logs
NBA_GAME_LOGS_URL = "https://stats.nba.com/stats/playergamelogs"

# Request Headers (NBA.com requires these)
HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nba.com/stats/players/traditional",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true"
}

#Replace with the Player ID 
PLAYER_ID = "201939" 

# Query Parameters (Modify Season if needed)
PARAMS = {
    "PlayerID": PLAYER_ID,
    "LeagueID": "00",
    "Season": "2024-25",  
    "SeasonType": "Regular Season"  # Options: 'Regular Season', 'Playoffs'
}

# Fetch Data from API
response = requests.get(NBA_GAME_LOGS_URL, headers=HEADERS, params=PARAMS)

# Check if API request was successful
if response.status_code == 200:
    data = response.json()

    #  Extract Headers (Column Names)
    headers = data["resultSets"][0]["headers"]

    #  Extract Player Game Logs
    rows = data["resultSets"][0]["rowSet"]

    #  Convert to DataFrame
    df = pd.DataFrame(rows, columns=headers)

    #  Save Data to CSV
    df.to_csv("player_game_logs.csv", index=False)
    print(f"✅ Game-by-game stats saved to 'player_game_logs.csv'!")

else:
    print(f"❌ Failed to fetch data. HTTP Status Code: {response.status_code}")
