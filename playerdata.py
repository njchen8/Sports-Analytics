import time
import sqlite3
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ✅ Configure Selenium WebDriver (Visible Mode)
CHROMEDRIVER_PATH = r"C:\Users\firef\OneDrive\Desktop\Sports Gambling\Sports Model\chromedriver-win64\chromedriver-win64\chromedriver.exe"
chrome_options = Options()
chrome_options.add_argument("--start-maximized")  # Open in full-screen mode
chrome_options.add_experimental_option("detach", True)  # Keep window open after execution

service = Service(CHROMEDRIVER_PATH)
driver = webdriver.Chrome(service=service, options=chrome_options)

# ✅ NBA Stats URL
URL = "https://www.nba.com/stats/players/traditional?PerMode=PerGame&sort=PTS&dir=-1"

# ✅ Open the webpage (Browser remains open)
driver.get(URL)

# ✅ Wait until the player stats table loads
WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "Crom_table__p1iZz")))
time.sleep(5)  # Allow time for JavaScript to load all data

# ✅ Extract Table Data
table = driver.find_element(By.CLASS_NAME, "Crom_table__p1iZz")
rows = table.find_elements(By.TAG_NAME, "tr")

# ✅ Extract column headers
headers = [header.text for header in rows[0].find_elements(By.TAG_NAME, "th")]

# ✅ Extract player data
player_data = []
for row in rows[1:]:  # Skip header row
    cols = row.find_elements(By.TAG_NAME, "td")
    player_stats = [col.text for col in cols]
    if player_stats:
        player_data.append(player_stats)

# ✅ Convert to DataFrame
df = pd.DataFrame(player_data, columns=headers)
print(df.head())

# ✅ Store Data in SQLite Database
DB_NAME = "nba_player_stats.db"

# Connect to database (or create if not exists)
conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()

# Create Table if not exists
cursor.execute("""
    CREATE TABLE IF NOT EXISTS players (
        Rank INTEGER,
        Player TEXT,
        Team TEXT,
        GP INTEGER,
        MIN FLOAT,
        PTS FLOAT,
        REB FLOAT,
        AST FLOAT,
        STL FLOAT,
        BLK FLOAT,
        TOV FLOAT,
        FG_PCT FLOAT,
        FT_PCT FLOAT,
        THREE_PCT FLOAT
    )
""")

# ✅ Insert Data
df.to_sql("players", conn, if_exists="replace", index=False)

# ✅ Close Connection
conn.commit()
conn.close()

print(f"✅ NBA player stats saved to {DB_NAME}")
print("🚀 Keep the browser open for debugging!")
