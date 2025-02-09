import pandas as pd
import requests
from bs4 import BeautifulSoup

# ✅ Define player and year
PLAYER = "lebron-james"  # Format: first-last
YEAR = 2024
URL = f"https://www.basketball-reference.com/players/{PLAYER[0]}/{PLAYER}01/gamelog/{YEAR}/"

# ✅ Fetch HTML data
response = requests.get(URL)
soup = BeautifulSoup(response.text, "html.parser")

# ✅ Extract game log table
table = soup.find("table", {"id": "pgl_basic"})
df_stats = pd.read_html(str(table))[0]

# ✅ Clean and save the data
df_stats = df_stats.dropna(how="all")  # Remove empty rows
df_stats.to_csv(f"{PLAYER}_game_logs_{YEAR}.csv", index=False)
print("✅ Player game logs saved!")

# ✅ Display extracted data
print(df_stats.head())
