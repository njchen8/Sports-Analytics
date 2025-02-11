from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import pandas as pd
import time

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

# ✅ Scroll down to load all props
print("🔄 Scrolling down to load all props...")
for _ in range(5):  # Scroll multiple times
    driver.find_element(By.TAG_NAME, "body").send_keys(Keys.PAGE_DOWN)
    time.sleep(1)  # Small delay to load elements

# ✅ Wait for player props to load
print("🚀 Waiting for player props to load...")

try:
    WebDriverWait(driver, 30).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "[data-testid='over-under-list-cell']"))
    )
    print("✅ NBA Player Props Loaded Successfully!")
except Exception as e:
    print(f"❌ Player props did not load: {e}")
    driver.quit()
    exit()

# ✅ Scrape Player Props
props_data = []

# ✅ Locate all player prop elements
player_prop_cells = driver.find_elements(By.CSS_SELECTOR, "[data-testid='over-under-list-cell']")

current_player_name = "Unknown"  # Default player name if not found initially

for cell in player_prop_cells:
    try:
        # ✅ Check if the element is a player name
        player_name_element = cell.find_elements(By.XPATH, ".//preceding::div[@data-testid='player-name'][1]")
        if player_name_element:
            current_player_name = player_name_element[0].text.strip()

        # ✅ Extract stat type (e.g., Points, Assists, etc.)
        stat_type_element = cell.find_element(By.CLASS_NAME, "styles__displayStat__g479A")
        stat_type = stat_type_element.text.strip()

        # ✅ Extract prop line (e.g., 18.5)
        prop_value_element = cell.find_element(By.CLASS_NAME, "styles__statValue__xmjlQ")
        prop_value = prop_value_element.text.strip()

        # ✅ Extract Higher & Lower Odds (Inside each stat container)
        higher_payout = "1"  # Default to 1 if missing
        lower_payout = "1"  # Default to 1 if missing

        payout_buttons = cell.find_elements(By.XPATH, ".//div[contains(@class, 'styles__lineOption__xTSdA')]")
        for button in payout_buttons:
            button_text = button.find_element(By.TAG_NAME, "span").text.strip()

            # ✅ Extract payout multiplier if available
            payout_element = button.find_elements(By.XPATH, ".//span[contains(@class, 'styles__payoutMultiplierWrapper__sfh5n')]//div[@style='opacity: 1; transform: none;']//span")

            if payout_element:
                payout_value = payout_element[0].text.strip()
            else:
                payout_value = "1"  # Default if missing

            if "Higher" in button_text:
                higher_payout = payout_value
            elif "Lower" in button_text:
                lower_payout = payout_value

        # ✅ Append to data list
        props_data.append({
            "Player": current_player_name,
            "Stat Type": stat_type,
            "Prop Line": prop_value,
            "Higher Payout": higher_payout,
            "Lower Payout": lower_payout
        })

    except Exception as e:
        print(f"⚠️ Skipping an element due to missing data: {e}")

# ✅ Convert to DataFrame & Save
df = pd.DataFrame(props_data)

csv_path = "underdog_player_props.csv"
df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"✅ Player props saved to '{csv_path}'")

# ✅ Keep the browser open for debugging
input("Press ENTER to close the browser...")

# ✅ Close the browser
driver.quit()
