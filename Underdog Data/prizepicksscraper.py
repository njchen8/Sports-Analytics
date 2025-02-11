import time
import pandas as pd
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys

# ✅ Initialize undetected ChromeDriver with bot evasion
chrome_options = uc.ChromeOptions()
chrome_options.add_argument("--disable-blink-features=AutomationControlled")  # Bypass bot detection
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--disable-extensions")
chrome_options.add_argument("--start-maximized")
chrome_options.add_argument("--disable-infobars")
chrome_options.add_argument("--incognito")

# ✅ Use real Chrome profile (Modify with your correct path)
chrome_options.add_argument(f"--user-data-dir=C:\\Users\\firef\\AppData\\Local\\Google\\Chrome\\User Data")
chrome_options.add_argument("--profile-directory=Default")

# ✅ Launch Browser
driver = uc.Chrome(options=chrome_options)
PRIZEPICKS_URL = "https://app.prizepicks.com/board"
driver.get(PRIZEPICKS_URL)

print("🚀 Please **log in manually** if required and solve the CAPTCHA.")
input("✅ Press ENTER to start scraping once you're on the player props page...")

# ✅ Scroll down to load all player props
print("🔄 Scrolling down to load all props...")
for _ in range(5):  # Scroll multiple times
    driver.find_element(By.TAG_NAME, "body").send_keys(Keys.PAGE_DOWN)
    time.sleep(1)  # Small delay to load elements

# ✅ Wait for player props to load dynamically
print("🚀 Waiting for player props to load...")

try:
    WebDriverWait(driver, 30).until(
        EC.presence_of_all_elements_located((By.XPATH, "//div[contains(@class, 'grid') and contains(@class, 'grid-rows')]"))
    )
    print("✅ Player props loaded successfully!")
except Exception as e:
    print(f"❌ Error loading player props: {e}")
    driver.quit()
    exit()

# ✅ Scrape Player Props
props_data = []

player_rows = driver.find_elements(By.XPATH, "//div[contains(@class, 'grid') and contains(@class, 'grid-rows')]")
for row in player_rows:
    try:
        # Extract Player Name
        player_name = row.find_element(By.XPATH, ".//h3[@id='test-player-name']").text.strip()

        # Extract Prop Line (e.g., 9.5)
        prop_value = row.find_element(By.XPATH, ".//div[contains(@class, 'heading-md')]").text.strip()

        # Extract Stat Type (if available, adjust if incorrect)
        try:
            stat_type = row.find_element(By.XPATH, ".//span[contains(@class, 'text-soClean-140')]").text.strip()
        except:
            stat_type = "Unknown"

        # Extract Payouts (Modify if structure changes)
        payouts = row.find_elements(By.XPATH, ".//button[contains(@class, 'option-button')]")
        higher_payout = payouts[0].text.strip() if len(payouts) > 0 else 1
        lower_payout = payouts[1].text.strip() if len(payouts) > 1 else 1

        # ✅ Append to Data List
        props_data.append({
            "Player": player_name,
            "Stat Type": stat_type,
            "Prop Line": prop_value,
            "Higher Payout": higher_payout,
            "Lower Payout": lower_payout
        })

    except Exception as e:
        print(f"⚠ Skipping row due to missing data: {e}")

# ✅ Convert to DataFrame & Save
df = pd.DataFrame(props_data)
csv_path = "prizepicks_player_props.csv"
df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"✅ Player props saved to '{csv_path}'")

# ✅ Show extracted data OR print full HTML for debugging
if not df.empty:
    print(df.head())  # Show first few rows
else:
    print("⚠ No player props found. Printing full HTML for debugging:")
    print(driver.page_source)  # Debugging output

# ✅ Keep the browser open for debugging
input("Press ENTER to close the browser...")

# ✅ Close the browser
driver.quit()
