import pandas as pd

# File path
file_path = r'nba_players_game_logs_2018_25.csv'

# Load the dataset
df = pd.read_csv(file_path, low_memory=False)

# Identify the correct date column dynamically
possible_date_columns = ['GAME_DATE']  # Add more if needed
date_column = next((col for col in possible_date_columns if col in df.columns), None)

if date_column is None:
    print("Error: No valid date column found. Check column names again.")
    print(df.columns)
    exit()

# Convert the date column while handling multiple formats
def parse_mixed_dates(date):
    if isinstance(date, str):
        if "T" in date:  # Check if it's in ISO format (YYYY-MM-DDTHH:MM:SS)
            return pd.to_datetime(date, errors='coerce')
        else:  # Standard YYYY-MM-DD format
            return pd.to_datetime(date, format='%Y-%m-%d', errors='coerce')
    return pd.NaT  # Return NaT (Not a Time) if it's not a valid string

df[date_column] = df[date_column].apply(parse_mixed_dates)

# Drop rows where the date is completely invalid
df = df.dropna(subset=[date_column])

# Sort by player name and standardized date
df = df.sort_values(by=['PLAYER_NAME', date_column])

# Remove duplicate rows
df = df.drop_duplicates()

# Save cleaned dataset
output_file = r'nba_players_game_logs_cleaned.csv'
df.to_csv(output_file, index=False)

print(f"✅ Cleaned data saved to: {output_file}")
