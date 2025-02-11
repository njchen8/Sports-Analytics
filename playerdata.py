import requests
import pandas as pd

#  Correct NBA API URL
NBA_API_URL = "https://stats.nba.com/stats/leaguedashplayerstats"

#  Request Headers (NBA.com requires these headers)
HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nba.com/stats/players/traditional",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true"
}

#  Query Parameters (Modify Season if needed)
PARAMS = {
    "College": "",
    "Conference": "",
    "Country": "",
    "DateFrom": "",
    "DateTo": "",
    "Division": "",
    "DraftPick": "",
    "DraftYear": "",
    "GameScope": "",
    "GameSegment": "",
    "Height": "",
    "ISTRound": "",
    "LastNGames": "0",
    "LeagueID": "00",
    "Location": "",
    "MeasureType": "Base",
    "Month": "0",
    "OpponentTeamID": "0",
    "Outcome": "",
    "PORound": "0",
    "PaceAdjust": "N",
    "PerMode": "PerGame",
    "Period": "0",
    "PlayerExperience": "",
    "PlayerPosition": "",
    "PlusMinus": "N",
    "Rank": "N",
    "Season": "2023-24",  # Change this for the new season
    "SeasonSegment": "",
    "SeasonType": "Regular Season",
    "ShotClockRange": "",
    "StarterBench": "",
    "TeamID": "0",
    "VsConference": "",
    "VsDivision": "",
    "Weight": ""
}

#  Fetch Data from API
response = requests.get(NBA_API_URL, headers=HEADERS, params=PARAMS)

#  Check if API request was successful
if response.status_code == 200:
    data = response.json()

    #  Extract Headers (Column Names)
    headers = data["resultSets"][0]["headers"]

    #  Extract Player Stats Data
    rows = data["resultSets"][0]["rowSet"]

    #  Convert to DataFrame
    df = pd.DataFrame(rows, columns=headers)

    #  Save Data to CSV
    df.to_csv("nba_player_stats.csv", index=False)
    print(" NBA Player Stats saved to 'nba_player_stats.csv'!")

else:
    print(f"❌ Failed to fetch data. HTTP Status Code: {response.status_code}")
