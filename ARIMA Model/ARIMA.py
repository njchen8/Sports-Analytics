import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from datetime import datetime, timedelta

# Load datasets with low_memory=False to avoid dtype warnings
props_file_path = "underdog.csv"
logs_file_path = "nba_players_game_logs_2018_25.csv"

props_df = pd.read_csv(props_file_path, low_memory=False)
logs_df = pd.read_csv(logs_file_path, low_memory=False)

# Convert GAME_DATE to datetime with specific format
logs_df['GAME_DATE'] = pd.to_datetime(logs_df['GAME_DATE'], format='%Y-%m-%dT%H:%M:%S', errors='coerce')

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
    "pts_rebs_asts": "PTS_REB_AST"
}

# Suppress ARIMA warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def predict_stat_with_variance(player_name, stat, player_data, prop_row):
    """
    Predicts a player's expected value and variance for a given stat using the last 25 valid matches.
    Variance is calculated on the past 25 matches filtered only by MIN >= 5 (without the 3 SD filter).
    ARIMA modeling, weighted mean, and forecasts are computed on the same data after applying the 3 SD filter.
    """

    if stat not in player_data.columns:
        return np.nan, np.nan, "", np.nan, np.nan, np.nan

    # Filter data for the player and sort by GAME_DATE
    player_df = player_data[player_data['PLAYER_NAME'] == player_name].copy()
    player_df = player_df.sort_values(by="GAME_DATE")

    # Ensure the last 25 matches span a reasonable time period
    if len(player_df) > 25:
        time_span = (player_df['GAME_DATE'].iloc[-1] - player_df['GAME_DATE'].iloc[-25]).days
        if time_span > 180:  # Ensure matches are within the last 6 months
            player_df = player_df[player_df['GAME_DATE'] >= player_df['GAME_DATE'].iloc[-25]]
    player_df = player_df.tail(25)

    # Drop NaNs for the specific stat and filter for games with at least 5 minutes
    player_df = player_df.dropna(subset=[stat])
    player_df = player_df[player_df['MIN'] >= 5]

    if player_df.empty:
        return np.nan, np.nan, "", np.nan, np.nan, np.nan

    # Save a copy for variance calculation (only MIN filter applied)
    variance_df = player_df.copy()

    # --- Now apply the 3 SD filter for ARIMA/weighted mean calculations ---
    mean_val = player_df[stat].mean()
    std_val = player_df[stat].std()
    filtered_df = player_df[(player_df[stat] >= mean_val - 3 * std_val) &
                             (player_df[stat] <= mean_val + 3 * std_val)].copy()

    if filtered_df.empty:
        return np.nan, np.nan, "", np.nan, np.nan, np.nan

    # Calculate date range from the filtered data
    date_range = f"{filtered_df['GAME_DATE'].min().date()} to {filtered_df['GAME_DATE'].max().date()}"
    if 'NaT' in date_range:
        return np.nan, np.nan, "", np.nan, np.nan, np.nan

    # Check if the player has played in the last 2 weeks (using filtered data)
    last_game_date = filtered_df['GAME_DATE'].max()
    if (datetime.now() - last_game_date) > timedelta(days=14):
        return np.nan, np.nan, "", np.nan, np.nan, np.nan

    # Identify the opponent team correctly
    player_team = filtered_df.iloc[-1]['TEAM_ABBREVIATION']
    teams = {prop_row['Current Team'], prop_row['Opponent Team']}
    if player_team in teams:
        opponent_team = (teams - {player_team}).pop()
    else:
        return np.nan, np.nan, "", np.nan, np.nan, np.nan

    # Calculate weights for the filtered data using .loc to avoid SettingWithCopyWarning
    filtered_df.loc[:, 'recent_weight'] = np.exp(-np.arange(len(filtered_df))[::-1] / 3)
    filtered_df.loc[:, 'home_weight'] = np.where(filtered_df['MATCHUP'].str.contains('@'), 0.8, 1.2)
    filtered_df.loc[:, 'same_team_weight'] = np.where(filtered_df['MATCHUP'].str.contains(opponent_team), 1.5, 1.0)
    filtered_df.loc[:, 'combined_weight'] = (
        filtered_df.loc[:, 'recent_weight'] *
        filtered_df.loc[:, 'home_weight'] *
        filtered_df.loc[:, 'same_team_weight']
    )

    # Calculate weighted mean from the filtered data
    weighted_mean = np.average(filtered_df[stat], weights=filtered_df['combined_weight'])

    # Calculate weighted variance using the unfiltered data (variance_df) with MIN filter only
    variance_df.loc[:, 'recent_weight'] = np.exp(-np.arange(len(variance_df))[::-1] / 3)
    variance_df.loc[:, 'home_weight'] = np.where(variance_df['MATCHUP'].str.contains('@'), 0.8, 1.2)
    variance_df.loc[:, 'same_team_weight'] = np.where(variance_df['MATCHUP'].str.contains(opponent_team), 1.5, 1.0)
    variance_df.loc[:, 'combined_weight'] = (
        variance_df.loc[:, 'recent_weight'] *
        variance_df.loc[:, 'home_weight'] *
        variance_df.loc[:, 'same_team_weight']
    )
    variance_value = np.average((variance_df[stat] - weighted_mean) ** 2, weights=variance_df['combined_weight'])

    # Apply ARIMA model if sufficient data is available using the filtered data
    if len(filtered_df) >= 10:
        best_model = None
        best_aic = float('inf')
        # Create a temporary index for ARIMA model
        filtered_df = filtered_df.copy()  # Additional copy to be safe
        filtered_df.loc[:, 'game_index'] = np.arange(len(filtered_df))
        for p in range(1, 4):
            for d in range(1, 3):
                for q in range(1, 4):
                    try:
                        model = ARIMA(filtered_df.set_index('game_index')[stat], 
                                      order=(p, d, q),
                                      enforce_stationarity=False,
                                      enforce_invertibility=False)
                        fit = model.fit()
                        residuals = fit.resid
                        mse = np.mean(residuals ** 2)
                        if mse < best_aic:
                            best_aic = mse
                            best_model = fit
                    except Exception:
                        continue
        if best_model:
            arima_forecast = best_model.forecast(steps=1).iloc[0]
            # Discard ARIMA predictions more than 1 SD away from the weighted mean
            if abs(arima_forecast - weighted_mean) > std_val:
                arima_forecast = weighted_mean
                print("more than SD away")
        else:
            arima_forecast = weighted_mean
            print("no model")
    else:
        arima_forecast = weighted_mean
        print("not enough data")

    # Calculate weighted combination of weighted mean and ARIMA forecast
    weighted_combination = 0.5 * weighted_mean + 0.5 * arima_forecast

    return round(arima_forecast, 2), round(variance_value, 2), date_range, weighted_mean, arima_forecast, round(weighted_combination, 2)




# Create a dictionary to store predictions for each player and stat
predictions = []

for _, row in props_df.iterrows():
    player_name = row['Player']
    stat_type = row['Stat Type'].lower()

    stat_col = stat_mapping.get(stat_type)
    if stat_col is None:
        continue

    if not logs_df[logs_df['PLAYER_NAME'] == player_name].empty:
        predicted_value, variance_value, date_range, mean_val, arima_val, weighted_combination = predict_stat_with_variance(player_name, stat_col, logs_df, row)

        if date_range:
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

# Convert to DataFrame and save
predictions_df = pd.DataFrame(predictions)

# Sort predictions by ARIMA forecast and weighted mean
predictions_df = predictions_df.sort_values(by=["ARIMA Forecast", "Weighted Mean"], ascending=[False, False])

if not predictions_df.empty:
    predictions_df.to_csv("player_expected_values.csv", index=False)
    print("Expected values and variances saved to player_expected_values.csv")
else:
    print("No valid predictions were generated")