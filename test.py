from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import time

# ✅ Enter your Underdog Fantasy login credentials
UNDERDOG_EMAIL = "njchen8@gmail.com"  # Replace with your email
UNDERDOG_PASSWORD = "Fortniteballs8!"  # Replace with your password

# ✅ Update with your ChromeDriver path
CHROMEDRIVER_PATH = r"C:\Users\firef\OneDrive\Desktop\Sports Gambling\Sports Model\chromedriver-win64\chromedriver-win64\chromedriver.exe"

# ✅ Set up Selenium WebDriver
chrome_options = Options()
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--enable-unsafe-swiftshader")  
chrome_options.add_argument("--disable-usb-keyboard-detect")  

# ✅ Open Chrome in visible mode (disable headless for debugging)
service = Service(CHROMEDRIVER_PATH)
driver = webdriver.Chrome(service=service, options=chrome_options)

# ✅ Open Underdog Fantasy Login Page
driver.get("https://underdogfantasy.com/login")
time.sleep(3)  # Allow the page to load

# ✅ Locate and enter email
try:
    email_field = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='email_input']"))
    )
    email_field.send_keys(UNDERDOG_EMAIL)
    print("✅ Entered Email.")
except Exception as e:
    print(f"❌ Error entering email: {e}")
    driver.quit()
    exit()

# ✅ Locate and enter password
try:
    password_field = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='password_input']"))
    )
    password_field.send_keys(UNDERDOG_PASSWORD)
    password_field.send_keys(Keys.RETURN)  # Press Enter to login
    print("✅ Entered Password and submitted login.")
except Exception as e:
    print(f"❌ Error entering password: {e}")
    driver.quit()
    exit()

# ✅ Wait for login confirmation and handle loading screens
try:
    # Wait for the URL to change (ensures login transition is complete)
    WebDriverWait(driver, 20).until(
        lambda d: d.current_url != "https://underdogfantasy.com/login"
    )
    print(f"🔍 Redirected to: {driver.current_url}")

    # Wait for the "Account" button to appear (final confirmation of login)
    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.XPATH, "//div[contains(text(), 'Account')]"))
    )
    print("✅ Login Successful!")

    # Take a screenshot for verification
    driver.save_screenshot("logged_in_screenshot.png")
    print("📸 Screenshot saved as 'logged_in_screenshot.png'")

except Exception as e:
    print(f"❌ Login failed: {e}")
    driver.save_screenshot("login_failed_screenshot.png")  # Save failed login screenshot
    print("📸 Screenshot saved as 'login_failed_screenshot.png'")
    driver.quit()
    exit()

# ✅ Wait for player props to load dynamically
try:
    WebDriverWait(driver, 20).until(
        EC.presence_of_all_elements_located((By.CLASS_NAME, "player-prop-container"))  # Update if needed
    )
    print("✅ Player props loaded successfully!")
except Exception as e:
    print(f"⚠ Warning: Player props did not load in time. Error: {e}")

# ✅ Get page source and parse with BeautifulSoup
soup = BeautifulSoup(driver.page_source, "html.parser")

# ✅ Try multiple methods to locate player props
player_props = []
for prop in soup.find_all("div", class_="player-prop-container"):  # Update with correct class
    try:
        player_name = prop.find("span", class_="player-name").text.strip()
        stat_type = prop.find("span", class_="prop-type").text.strip()
        prop_value = prop.find("span", class_="prop-value").text.strip()
        
        player_props.append({
            "Player": player_name,
            "Stat": stat_type,
            "Prop Line": prop_value
        })
    except AttributeError:
        print("⚠ Warning: Skipping an element due to missing data.")

# ✅ Convert to DataFrame
df = pd.DataFrame(player_props)

# ✅ Save as CSV
csv_path = "underdog_player_props.csv"
df.to_csv(csv_path, index=False)
print(f"✅ Player prop data saved to '{csv_path}'")

# ✅ Show extracted data OR print full HTML for debugging
if not df.empty:
    print(df)
else:
    print("⚠ No player props found. Printing full HTML for debugging:")
    print(soup.prettify())  # Prints the entire page's HTML

# ✅ Keep the browser open for debugging
input("Press ENTER to close the browser...")

# ✅ Close the browser
driver.quit()