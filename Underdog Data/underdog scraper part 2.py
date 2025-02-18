import re
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys

# -- Set up Selenium WebDriver
chrome_options = Options()
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-extensions")
chrome_options.add_argument("--incognito")
chrome_options.add_argument("--disable-software-rasterizer")

driver = webdriver.Chrome(options=chrome_options)
NBA_PROPS_URL = "https://underdogfantasy.com/pick-em/higher-lower/all/nba"
driver.get(NBA_PROPS_URL)

input("🚀 Press ENTER to start scraping once you're on the NBA Player Props page...")

# -- Scroll down until all props are loaded
while True:
    prev_height = driver.execute_script("return document.body.scrollHeight")
    driver.find_element(By.TAG_NAME, "body").send_keys(Keys.END)
    time.sleep(3)
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == prev_height:
        break

# -- Wait for player props to load
WebDriverWait(driver, 30).until(
    EC.presence_of_all_elements_located((By.CSS_SELECTOR, "[data-testid='accordion']"))
)

# -- Scrape Player Props
props_data = []

matchup_sections = driver.find_elements(By.CSS_SELECTOR, "[data-testid='accordion']")
print(f"Number of matchup sections found: {len(matchup_sections)}")

for idx, matchup in enumerate(matchup_sections):
    try:
        print(f"--- MATCHUP {idx} ---")
        match_info_element = matchup.find_element(By.CSS_SELECTOR, "[data-testid='match-info']")
        match_info_text = match_info_element.text.strip()
        print("  match_info_text:", match_info_text)

        # Remove "CST" or other time zones
        tz_codes = {"CST", "EST", "PST", "MST", "CDT", "EDT", "PDT", "MDT"}
        found = re.findall(r"\b[A-Z]{3}\b", match_info_text)
        found = [t for t in found if t not in tz_codes]

        if len(found) < 2:
            print("  Not enough teams found, skipping matchup.")
            continue

        # Determine home/away from ' vs ' or ' @ '
        if " VS " in match_info_text.upper():
            home_team, away_team = found[1], found[0]
        elif " @" in match_info_text.upper():
            away_team, home_team = found[1], found[0]
        else:
            home_team, away_team = found[1], found[0]

        print(f"  Found teams => home: {home_team}, away: {away_team}")

        # Grab the player cards
        player_cards = matchup.find_elements(By.CSS_SELECTOR, "[data-testid='over-under-cell']")
        print(f"  Found {len(player_cards)} player cards.")

        player_cards = matchup.find_elements(By.CSS_SELECTOR, "[data-testid='over-under-cell']")

        for card in player_cards:
            try:
                # 1) Get the player's name
                name_el = card.find_element(By.CSS_SELECTOR, "[data-testid='player-name']")
                player_name = name_el.text.strip()

                # 2) Try to locate a team element (this might not exist on Underdog)
                try:
                    team_el = card.find_element(By.CSS_SELECTOR, "[data-testid='player-team']")
                    current_team = team_el.text.strip()
                except:
                    # If that fails, fallback to parse from the card text
                    card_text = card.text
                    teams_in_card = re.findall(r"\b[A-Z]{3}\b", card_text)
                    # Filter out known time zones
                    tz_codes = {"CST","EST","PST","MST","CDT","EDT","PDT","MDT"}
                    teams_in_card = [t for t in teams_in_card if t not in tz_codes]

                    if len(teams_in_card) == 1:
                        current_team = teams_in_card[0]
                    else:
                        # If there's more than one, see if one matches home_team or away_team
                        possible = [t for t in teams_in_card if t in (home_team, away_team)]
                        current_team = possible[0] if possible else "Unknown"

                # 3) Determine the opponent
                if current_team == home_team:
                    opponent_team = away_team
                elif current_team == away_team:
                    opponent_team = home_team
                else:
                    opponent_team = "Unknown"

                # 4) Now extract the player's stats (as before)
                stat_lines = card.find_elements(By.CSS_SELECTOR, "[data-testid='stat-line-container']")
                for stat_line in stat_lines:
                    try:
                        stat_type = stat_line.get_attribute("data-appearance-stat").strip()
                        prop_value = stat_line.find_element(By.CSS_SELECTOR, ".styles__statValue__xmjlQ span").text.strip()

                        props_data.append({
                            "Player": player_name,
                            "Current Team": current_team,
                            "Opponent Team": opponent_team,
                            "Home Team": home_team,
                            "Away Team": away_team,
                            "Stat Type": stat_type,
                            "Prop Line": prop_value
                        })
                    except Exception as e:
                        print(f"⚠️  Skipping a stat line due to error: {e}")

            except Exception as e:
                print(f"⚠️  Skipping a player due to error: {e}")


    except Exception as e:
        print(f"⚠️ Skipping entire matchup {idx} due to error: {e}")

print("Total rows in props_data:", len(props_data))

df = pd.DataFrame(props_data)
df.to_csv("fixed_data.csv", index=False, encoding="utf-8-sig")

print(f"✅ Player props saved to 'fixed_data.csv' with {len(df)} rows.")
driver.quit()