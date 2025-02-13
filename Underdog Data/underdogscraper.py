import re
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys

# ✅ Set up Selenium WebDriver
chrome_options = Options()
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-extensions")
chrome_options.add_argument("--incognito")
chrome_options.add_argument("--disable-software-rasterizer")

# ✅ Initialize WebDriver
driver = webdriver.Chrome(options=chrome_options)
NBA_PROPS_URL = "https://underdogfantasy.com/pick-em/higher-lower/all/nba"
driver.get(NBA_PROPS_URL)

input("🚀 Press ENTER to start scraping once you're on the NBA Player Props page...")

# ✅ Scroll down until all props are loaded
while True:
    prev_height = driver.execute_script("return document.body.scrollHeight")
    driver.find_element(By.TAG_NAME, "body").send_keys(Keys.END)
    time.sleep(3)
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == prev_height:
        break  

# ✅ Wait for player props to load
WebDriverWait(driver, 30).until(
    EC.presence_of_all_elements_located((By.CSS_SELECTOR, "[data-testid='accordion']"))
)

# ✅ Scrape Player Props
props_data = []

# ✅ Locate all matchups
matchup_sections = driver.find_elements(By.CSS_SELECTOR, "[data-testid='accordion']")

for matchup in matchup_sections:
    try:
        # ✅ Extract team matchup info
        match_info_element = matchup.find_element(By.CSS_SELECTOR, "[data-testid='match-info']")
        match_info_html = match_info_element.get_attribute("innerHTML")

        # ✅ Extract home team (inside <strong> tags)
        home_team_match = re.search(r"<strong>([A-Z]{3})</strong>", match_info_html)
        home_team = home_team_match.group(1) if home_team_match else "Unknown"

        # ✅ Extract away team (text after "vs" or "@")
        away_team_match = re.findall(r'\b[A-Z]{3}\b', match_info_element.text.strip())
        away_team = away_team_match[1] if len(away_team_match) > 1 else "Unknown"



        # ✅ Find all players in the matchup
        player_cards = matchup.find_elements(By.CSS_SELECTOR, "[data-testid='over-under-cell']")

        for card in player_cards:
            try:
                # ✅ Extract player name
                player_name_element = card.find_element(By.CSS_SELECTOR, "[data-testid='player-name']")
                player_name = player_name_element.text.strip() if player_name_element else "Unknown"

                # ✅ Extract player's team from their matchup text
                player_matchup_text = card.find_element(By.CSS_SELECTOR, "[data-testid='match-info']").text.strip()

                # ✅ Only include home team players
                if not player_matchup_text.startswith(home_team):
                    continue  # Skip players not on the home team

                # ✅ Extract stat types and prop lines
                stat_lines = card.find_elements(By.CSS_SELECTOR, "[data-testid='stat-line-container']")

                for stat_line in stat_lines:
                    try:
                        stat_type = stat_line.get_attribute("data-appearance-stat").strip()
                        prop_value = stat_line.find_element(By.CSS_SELECTOR, ".styles__statValue__xmjlQ span").text.strip()

                        # ✅ Extract "Higher" and "Lower" payouts
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
                            "Current Team": home_team,
                            "Opponent Team": away_team,
                            "Stat Type": stat_type,
                            "Prop Line": prop_value,
                            "Higher Payout": higher_payout,
                            "Lower Payout": lower_payout
                        })

                    except Exception as e:
                        print(f"⚠️ Skipping a stat due to missing data: {e}")

            except Exception as e:
                print(f"⚠️ Skipping a player due to missing data: {e}")

    except Exception as e:
        print(f"⚠️ Skipping a matchup due to missing data: {e}")

# ✅ Convert to DataFrame & Save
df = pd.DataFrame(props_data)
df.to_csv("underdog.csv", index=False, encoding="utf-8-sig")

print(f"✅ Home team player props saved to 'underdog.csv'")

# ✅ Keep the browser open for debugging
input("Press ENTER to close the browser...")

# ✅ Close the browser
driver.quit()
