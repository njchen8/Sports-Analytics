from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import pandas as pd
import time
import re

# ✅ Set up Selenium WebDriver
chrome_options = Options()
chrome_options.add_argument("--disable-gpu")  
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-extensions")
chrome_options.add_argument("--incognito")  
chrome_options.add_argument("--disable-software-rasterizer")

# ✅ Initialize WebDriver
driver = webdriver.Chrome(options=chrome_options)

# ✅ Navigate to Underdog NBA Player Props Page
NBA_PROPS_URL = "https://underdogfantasy.com/pick-em/higher-lower/all/nba"
driver.get(NBA_PROPS_URL)
print("✅ Please log in manually if required and ensure you're on the NBA Player Props page.")

# ✅ Wait for user confirmation after logging in
input("🚀 Press ENTER to start scraping once you're on the NBA Player Props page...")

# ✅ Scroll down until all props are loaded
print("🔄 Scrolling down to load all props...")
while True:
    prev_height = driver.execute_script("return document.body.scrollHeight")
    driver.find_element(By.TAG_NAME, "body").send_keys(Keys.END)  # Scroll to the bottom
    time.sleep(3)  # Allow time for elements to load
    new_height = driver.execute_script("return document.body.scrollHeight")
    
    if new_height == prev_height:
        break  # Stop scrolling when no new content loads

# ✅ Wait for player props to load
print("🚀 Waiting for player props to load...")

try:
    WebDriverWait(driver, 30).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "[data-testid='accordion']"))
    )
    print("✅ NBA Player Props Loaded Successfully!")
except Exception as e:
    print(f"❌ Player props did not load: {e}")
    driver.quit()
    exit()

# ✅ Scrape Player Props
props_data = []

# ✅ Locate all matchups
matchup_sections = driver.find_elements(By.CSS_SELECTOR, "[data-testid='accordion']")

for matchup in matchup_sections:
    try:
        # ✅ Extract team info using a 3-character regex match
        match_info = matchup.find_element(By.CSS_SELECTOR, "[data-testid='match-info']").text.strip()
        team_codes = re.findall(r'\b[A-Z]{3}\b', match_info)  # Extract 3-letter team codes
        
        if len(team_codes) >= 2:
            team_one, team_two = team_codes[:2]  # First is the away team, second is home
        else:
            team_one, team_two = "Unknown", "Unknown"

        # ✅ Find all players in the matchup
        player_cards = matchup.find_elements(By.CSS_SELECTOR, "[data-testid='over-under-cell']")

        for card in player_cards:
            try:
                # ✅ Extract player name
                player_name_element = card.find_element(By.CSS_SELECTOR, "[data-testid='player-name']")
                player_name = player_name_element.text.strip() if player_name_element else "Unknown"

                # ✅ Determine player's team
                if team_one in player_name_element.text:
                    current_team, opponent_team = team_one, team_two
                else:
                    current_team, opponent_team = team_two, team_one

                # ✅ Extract all stat types and prop lines
                stat_lines = card.find_elements(By.CSS_SELECTOR, "[data-testid='stat-line-container']")

                for stat_line in stat_lines:
                    try:
                        stat_type = stat_line.get_attribute("data-appearance-stat").strip()
                        prop_value = stat_line.find_element(By.CSS_SELECTOR, ".styles__statValue__xmjlQ span").text.strip()

                        # ✅ Extract "Higher" and "Lower" payouts **for each stat**
                        higher_payout = "1"
                        lower_payout = "1"

                        payout_buttons = stat_line.find_elements(By.XPATH, ".//div[contains(@class, 'styles__lineOption__xTSdA')]")
                        for button in payout_buttons:
                            button_text = button.find_element(By.TAG_NAME, "span").text.strip()
                            payout_element = button.find_elements(By.XPATH, ".//span[contains(@class, 'styles__payoutMultiplierWrapper__sfh5n')]//div[@style='opacity: 1; transform: none;']//span")

                            payout_value = payout_element[0].text.strip() if payout_element else "1"

                            if "Higher" in button_text:
                                higher_payout = payout_value
                            elif "Lower" in button_text:
                                lower_payout = payout_value

                        # ✅ Append to list
                        props_data.append({
                            "Player": player_name,
                            "Current Team": current_team,
                            "Opponent Team": opponent_team,
                            "Stat Type": stat_type,
                            "Prop Line": prop_value,
                            "Higher Payout": higher_payout,
                            "Lower Payout": lower_payout
                        })

                    except Exception as e:
                        print(f"⚠️ Skipping a stat type due to missing data: {e}")

            except Exception as e:
                print(f"⚠️ Skipping a player due to missing data: {e}")

    except Exception as e:
        print(f"⚠️ Skipping a matchup due to missing data: {e}")

# ✅ Convert to DataFrame & Save
df = pd.DataFrame(props_data)

csv_path = "underdog_player_props.csv"
df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"✅ Player props saved to '{csv_path}'")

# ✅ Keep the browser open for debugging
input("Press ENTER to close the browser...")

# ✅ Close the browser
driver.quit()
