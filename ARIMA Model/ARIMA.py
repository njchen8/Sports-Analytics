import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# Load datasets with low_memory=False to avoid dtype warnings
props_file_path = "underdog_home_team_props.csv"
logs_file_path = "nba_players_game_logs_2018_25.csv"

props_df = pd.read_csv(props_file_path, low_memory=False)
logs_df = pd.read_csv(logs_file_path, low_memory=False)

# Convert GAME_DATE to datetime with specific format
logs_df['GAME_DATE'] = pd.to_datetime(logs_df['GAME_DATE'], format='%Y-%m-%dT%H:%M:%S', errors='coerce')

# Ensure key stat columns are numeric
stats_columns = ['PTS', 'REB', 'AST']
for col in stats_columns:
    logs_df[col] = pd.to_numeric(logs_df[col], errors='coerce')

# Filter out invalid data: remove outliers and games with less than 5 minutes
logs_df['MIN'] = pd.to_numeric(logs_df['MIN'], errors='coerce')
logs_df = logs_df.dropna(subset=['MIN'])
logs_df = logs_df[logs_df['MIN'] >= 5]
for col in stats_columns:
    mean = logs_df[col].mean()
    std = logs_df[col].std()
    logs_df = logs_df[(logs_df[col] >= mean - 3 * std) & (logs_df[col] <= mean + 3 * std)]

# Normalize stat type mappings
stat_mapping = {
    "points": "PTS",
    "rebounds": "REB",
    "assists": "AST",
    "pts_rebs_asts": ["PTS", "REB", "AST"]
}

def predict_stat_with_variance(player_name, stat, player_data, prop_row):
    """
    Predicts a player's expected value and variance for a given stat using the last 25 valid matches.
    """

    if stat not in player_data.columns:
        return np.nan, np.nan, ""

    # Filter data for the player
    player_df = player_data[player_data['PLAYER_NAME'] == player_name].copy()
    player_df = player_df.sort_values(by="GAME_DATE")

    # Ensure the last 25 matches span a reasonable time period
    if len(player_df) > 25:
        time_span = (player_df['GAME_DATE'].iloc[-1] - player_df['GAME_DATE'].iloc[-25]).days
        if time_span > 180:  # Ensure matches are within the last 6 months
            player_df = player_df[player_df['GAME_DATE'] >= player_df['GAME_DATE'].iloc[-25]]
    player_df = player_df.tail(25)

    # Drop NaNs for the specific stat
    player_df = player_df.dropna(subset=[stat])

    if player_df.empty:
        return np.nan, np.nan, ""

    # Determine player's current team from the most recent game
    most_recent_game = player_df.iloc[-1]
    player_team = most_recent_game['TEAM_ABBREVIATION']

    # Calculate date range
    date_range = f"{player_df['GAME_DATE'].min().date()} to {player_df['GAME_DATE'].max().date()}"
    if 'NaT' in date_range:
        return np.nan, np.nan, ""

    # Identify the opponent team
    if player_team == prop_row['Current Team']:
        opponent_team = prop_row['Opponent Team']
    else:
        opponent_team = prop_row['Current Team']

    # Calculate weights based on recency and correct opponent team
    player_df['recent_weight'] = np.exp(-np.arange(len(player_df))[::-1] / 3)
    player_df['home_weight'] = np.where(player_df['MATCHUP'].str.contains('@'), 0.8, 1.2)
    player_df['same_team_weight'] = np.where(player_df['MATCHUP'].str.contains(opponent_team), 1.5, 1.0)
    player_df['combined_weight'] = player_df['recent_weight'] * player_df['home_weight'] * player_df['same_team_weight']

    # Calculate weighted mean and variance
    weighted_mean = np.average(player_df[stat], weights=player_df['combined_weight'])
    weighted_variance = np.average((player_df[stat] - weighted_mean) ** 2, weights=player_df['combined_weight'])

    # Apply ARIMA model if sufficient data
    if len(player_df) >= 10:
        best_model = None
        best_aic = float('inf')
        for p in range(1, 4):
            for d in range(1, 3):
                for q in range(1, 4):
                    try:
                        model = ARIMA(player_df.set_index('GAME_DATE')[stat].asfreq('D', method='pad'), order=(p, d, q))
                        fit = model.fit()
                        residuals = fit.resid
                        mse = np.mean(residuals**2)
                        if mse < best_aic:
                            best_aic = mse
                            best_model = fit
                    except Exception as e:
                        continue
        arima_forecast = best_model.forecast(steps=1).iloc[0] if best_model else weighted_mean
    else:
        arima_forecast = weighted_mean

    # Final prediction: weighted combination of historical performance and ARIMA forecast
    final_prediction = 0.6 * weighted_mean + 0.4 * arima_forecast

    return round(final_prediction, 2), round(weighted_variance, 2), date_range

# Create a dictionary to store predictions for each player and stat
predictions = []

for _, row in props_df.iterrows():
    player_name = row['Player']
    stat_type = row['Stat Type'].lower()

    stat_cols = stat_mapping.get(stat_type)
    if stat_cols is None:
        continue

    if isinstance(stat_cols, list):  # For combined stats like pts_rebs_asts
        predicted_total = 0
        variance_total = 0
        date_range = ""
        for stat_col in stat_cols:
            pred_value, var_value, dr = predict_stat_with_variance(player_name, stat_col, logs_df, row)
            if not np.isnan(pred_value):
                predicted_total += pred_value
            if not np.isnan(var_value):
                variance_total += var_value
            date_range = dr

        if date_range:
            predictions.append({
                "Player": player_name,
                "Current Team": row['Current Team'],
                "Opponent Team": row['Opponent Team'],
                "Stat Type": "pts_rebs_asts",
                "Prop Line": row["Prop Line"],
                "Predicted Value": round(predicted_total, 2),
                "Variance": round(variance_total, 2),
                "Date Range": date_range
            })
    else:
        predicted_value, variance_value, date_range = predict_stat_with_variance(player_name, stat_cols, logs_df, row)

        if date_range:
            predictions.append({
                "Player": player_name,
                "Current Team": row['Current Team'],
                "Opponent Team": row['Opponent Team'],
                "Stat Type": stat_type,
                "Prop Line": row["Prop Line"],
                "Predicted Value": predicted_value,
                "Variance": variance_value,
                "Date Range": date_range
            })

# Convert to DataFrame and save
predictions_df = pd.DataFrame(predictions)

if not predictions_df.empty:
    predictions_df.to_csv("player_expected_values.csv", index=False)
    print("Expected values and variances saved to player_expected_values.csv")
else:
    print("No valid predictions were generated.")
