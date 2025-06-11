import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from datetime import datetime, timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX

props_file_path = "fixed_data.csv"
logs_file_path = "nba_players_game_logs_cleaned.csv"

props_df = pd.read_csv(props_file_path, low_memory=False)
logs_df = pd.read_csv(logs_file_path, low_memory=False)
props_df = props_df[~props_df['Stat Type'].str.contains('1_')]

# Identify the correct date column dynamically
possible_date_columns = ['GAME_DATE']  # Add more if needed
date_column = next((col for col in possible_date_columns if col in logs_df.columns), None)

if date_column is None:
    print("Error: No valid date column found. Check column names again.")
    print(logs_df.columns)
    exit()

# Convert the date column while handling multiple formats
def parse_mixed_dates(date):
    if isinstance(date, str):
        if "T" in date:  # Check if it's in ISO format (YYYY-MM-DDTHH:MM:SS)
            return pd.to_datetime(date, errors='coerce')
        else:  # Standard YYYY-MM-DD format
            return pd.to_datetime(date, format='%Y-%m-%d', errors='coerce')
    return pd.NaT  # Return NaT (Not a Time) if it's not a valid string

logs_df[date_column] = logs_df[date_column].apply(parse_mixed_dates)

# Drop rows with invalid GAME_DATE
logs_df = logs_df.dropna(subset=[date_column])

# Sort the logs_df by GAME_DATE in descending order (most recent first)
logs_df = logs_df.sort_values(by=date_column, ascending=False)

# Debug: Print first and last few rows after sorting
print("First few rows of logs_df after sorting:")
print(logs_df.head())

print("Last few rows of logs_df after sorting:")
print(logs_df.tail())

# Drop duplicates based on GAME_ID and PLAYER_ID
logs_df = logs_df.drop_duplicates(subset=["GAME_ID", "PLAYER_ID"])

# Check if players in props_df exist in logs_df
players_in_props = props_df['Player'].unique()
players_in_logs = logs_df['PLAYER_NAME'].unique()

missing_players = set(players_in_props) - set(players_in_logs)
if missing_players:
    print(f"Players in props_df not found in logs_df: {missing_players}")
else:
    print("All players in props_df exist in logs_df.")

# Ensure key stat columns are numeric
stats_columns = ['PTS', 'REB', 'AST']
for col in stats_columns:
    logs_df[col] = pd.to_numeric(logs_df[col], errors='coerce')

# Calculate pts_rebs_asts accurately
logs_df['PTS_REB_AST'] = logs_df['PTS'] + logs_df['REB'] + logs_df['AST']

# Filter out invalid data: remove outliers and games with less than 5 minutes
logs_df['MIN'] = pd.to_numeric(logs_df['MIN'], errors='coerce')
logs_df = logs_df.dropna(subset=['MIN'])
logs_df = logs_df[logs_df['MIN'] >= 5]
for col in stats_columns + ['PTS_REB_AST']:
    mean = logs_df[col].mean()
    std = logs_df[col].std()
    logs_df = logs_df[(logs_df[col] >= mean - 3 * std) & (logs_df[col] <= mean + 3 * std)]

# Normalize stat type mappings
stat_mapping = {
    "points": "PTS",
    "rebounds": "REB",
    "assists": "AST",
    "pts_rebs_asts": "PTS_REB_AST",
    "period_1_points": "PTS"  # add mapping if needed
}

# Suppress ARIMA warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def predict_stat_with_variance(player_name, stat, player_data, prop_row):
    """
    Predicts a player's expected value and variance for a given stat.
    - Uses a weighted mean from the last 25 most recent games.
    - Uses ARIMA on the last 15 most recent games.
    - If ARIMA forecast is more than 1 standard deviation away from weighted mean, discard it.
    """

    if stat not in player_data.columns:
        print(f"Stat {stat} not found for {player_name}")
        return np.nan, np.nan, "Stat Not Found", np.nan, np.nan, np.nan

    # ✅ **Ensure GAME_DATE is correctly parsed**
    player_data['GAME_DATE'] = pd.to_datetime(player_data['GAME_DATE'], errors='coerce')
    player_data[stat] = pd.to_numeric(player_data[stat], errors='coerce')

    # ✅ **Drop rows where GAME_DATE is NaT**
    player_data = player_data.dropna(subset=['GAME_DATE'])

    # ✅ **Sort entire dataset by GAME_DATE (most recent first)**
    player_data = player_data.sort_values(by="GAME_DATE", ascending=False)

    # ✅ **Find the latest year in the dataset**
    latest_year = player_data['GAME_DATE'].dt.year.max()
    print(f"Latest year in dataset: {latest_year}")

    # ✅ **Filter player-specific data**
    player_df = player_data[player_data['PLAYER_NAME'] == player_name]

    total_games = len(player_df)
    print(f"Player: {player_name}, Total Games Available: {total_games}")

    if player_df.empty:
        print(f"No data available for {player_name}")
        return np.nan, np.nan, "No Data", np.nan, np.nan, np.nan

    # ✅ **Select only the most recent games in the latest year**
    recent_games = player_df[player_df['GAME_DATE'].dt.year == latest_year]

    # ✅ **If player has no games in the latest year, use the most recent 25 available**
    if recent_games.empty:
        print(f"No games found for {player_name} in {latest_year}. Using most recent available games.")
        recent_games = player_df.iloc[:25]  # Take last 25 available games

    else:
        recent_games = recent_games.iloc[:25]  # Take last 25 games in latest year

    # ✅ **Ensure at least 5 games exist for meaningful prediction**
    if len(recent_games) < 5:
        mean_val = recent_games[stat].mean()
        return round(mean_val, 2), 0, f"{recent_games['GAME_DATE'].min().date()} to {recent_games['GAME_DATE'].max().date()}", round(mean_val, 2), round(mean_val, 2), round(mean_val, 2)

    # ✅ **Calculate weighted mean**
    recent_games['recent_weight'] = np.exp(-np.arange(len(recent_games))[::-1] / 3)
    weighted_mean = np.average(recent_games[stat], weights=recent_games['recent_weight'])
    std_dev = recent_games[stat].std()

    # ✅ **Take the 15 most recent games for ARIMA**
    arima_df = recent_games.iloc[:15]

    if arima_df.empty or len(arima_df) < 5:
        print(f"Not enough data for ARIMA for {player_name}. Using weighted mean.")
        return round(weighted_mean, 2), 0, f"{recent_games['GAME_DATE'].min().date()} to {recent_games['GAME_DATE'].max().date()}", round(weighted_mean, 2), round(weighted_mean, 2), round(weighted_mean, 2)

    # ✅ **Run ARIMA Forecasting**
    try:
        model = SARIMAX(arima_df[stat], order=(1, 0, 1), enforce_stationarity=False, enforce_invertibility=False)
        best_model = model.fit(disp=False)
        arima_forecast = best_model.forecast(steps=1).iloc[0]

        # ✅ **Discard ARIMA forecast if it's more than 1 SD away from the weighted mean**
        if abs(arima_forecast - weighted_mean) > std_dev:
            print(f"ARIMA forecast deviated too much for {player_name}, stat {stat}. Discarding ARIMA forecast.")
            arima_forecast = weighted_mean  # Default to weighted mean

        print(f"SARIMAX forecast succeeded for {player_name}, stat {stat}: {arima_forecast}")

    except Exception as e:
        print(f"SARIMAX failed for {player_name}, stat {stat}. Error: {e}")
        arima_forecast = weighted_mean  # Default to weighted mean on failure

    # ✅ **Final weighted combination**
    weighted_combination = 0.5 * weighted_mean + 0.5 * arima_forecast

    return (
        round(arima_forecast, 2), 
        round(std_dev ** 2, 2), 
        f"{recent_games['GAME_DATE'].min().date()} to {recent_games['GAME_DATE'].max().date()}",
        round(weighted_mean, 2), 
        round(arima_forecast, 2), 
        round(weighted_combination, 2)
    )


predictions = []

for _, row in props_df.iterrows():
    player_name = row['Player']
    stat_type = row['Stat Type'].lower()
    
    stat_col = stat_mapping.get(stat_type)
    if stat_col is None:
        print(f"Skipping {player_name}: Stat type '{stat_type}' not found in mapping.")
        continue
    
    if not logs_df[logs_df['PLAYER_NAME'] == player_name].empty:
        predicted_value, variance_value, date_range, mean_val, arima_val, weighted_combination = predict_stat_with_variance(player_name, stat_col, logs_df, row)
        
        print(f"Player: {player_name}, Stat: {stat_col}, Predicted Value: {predicted_value}, ARIMA Forecast: {arima_val}, Date Range: {date_range}")
        
        if date_range:
            # Determine the upcoming opponent based on prop data and player's team
            player_team = logs_df[logs_df['PLAYER_NAME'] == player_name].sort_values(by="GAME_DATE").iloc[-1]['TEAM_ABBREVIATION']
            opponent_team = row['Opponent Team'] if row['Current Team'] == player_team else row['Current Team']
            predictions.append({
                "Player": player_name,
                "Current Team": player_team,
                "Opponent Team": opponent_team,
                "Stat Type": stat_type,
                "Prop Line": row["Prop Line"],
                "Predicted Value": predicted_value,
                "Variance": variance_value,
                "Date Range": date_range,
                "Weighted Mean": mean_val,
                "ARIMA Forecast": arima_val,
                "Weighted Combination": weighted_combination
            })

# Convert predictions to DataFrame and save to CSV
if predictions:
    predictions_df = pd.DataFrame(predictions)
    if not predictions_df.empty:
        predictions_df = predictions_df.sort_values(by=["ARIMA Forecast", "Weighted Mean"], ascending=[False, False])
        predictions_df.to_csv("player_expected_values.csv", index=False)
        print("Expected values and variances saved to player_expected_values.csv")
    else:
        print("No valid predictions were generated")
else:
    print("No predictions were generated")