import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# Load datasets with low_memory=False to avoid dtype warnings
props_file_path = "underdog_home_team_props.csv"
logs_file_path = "nba_players_game_logs_2018_25.csv"

props_df = pd.read_csv(props_file_path, low_memory=False)
logs_df = pd.read_csv(logs_file_path, low_memory=False)

# Convert GAME_DATE to datetime
logs_df['GAME_DATE'] = pd.to_datetime(logs_df['GAME_DATE'], errors='coerce')

# Ensure key stat columns are numeric
stats_columns = ['PTS', 'REB', 'AST']
for col in stats_columns:
    logs_df[col] = pd.to_numeric(logs_df[col], errors='coerce')

# Filter out invalid data: remove outliers and games with less than 5 minutes
# Convert 'MIN' column to numeric, forcing errors to become NaN
logs_df['MIN'] = pd.to_numeric(logs_df['MIN'], errors='coerce')

# Drop rows where 'MIN' is NaN before filtering
logs_df = logs_df.dropna(subset=['MIN'])

# Now apply the filter
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

def predict_stat_with_variance(player_name, opponent_team, stat, player_data):
    """
    Predicts a player's expected value and variance for a given stat using the last 25 valid matches.
    """

    if stat not in player_data.columns:
        return np.nan, np.nan

    # Filter data for the player
    player_df = player_data[player_data['PLAYER_NAME'] == player_name].copy()
    player_df = player_df.sort_values(by="GAME_DATE").tail(25)

    # Drop NaNs for the specific stat
    player_df = player_df.dropna(subset=[stat])

    if player_df.empty:
        return np.nan, np.nan

    # Calculate weights based on recency and home/away performance
    player_df['recent_weight'] = np.exp(-np.arange(len(player_df))[::-1] / 5)
    player_df['home_weight'] = np.where(player_df['MATCHUP'].str.contains('@'), 0.8, 1.2)
    player_df['combined_weight'] = player_df['recent_weight'] * player_df['home_weight']

    # Calculate weighted mean and variance
    weighted_mean = np.average(player_df[stat], weights=player_df['combined_weight'])
    weighted_variance = np.average((player_df[stat] - weighted_mean) ** 2, weights=player_df['combined_weight'])

    # Apply ARIMA model if sufficient data
    if len(player_df) >= 10:
        arima_model = ARIMA(player_df[stat], order=(2, 1, 2))
        arima_fit = arima_model.fit()
        arima_forecast = arima_fit.forecast(steps=1).iloc[0]
    else:
        arima_forecast = weighted_mean

    # Final prediction: weighted combination of historical performance and ARIMA forecast
    final_prediction = 0.6 * weighted_mean + 0.4 * arima_forecast

    return round(final_prediction, 2), round(weighted_variance, 2)

# Create a dictionary to store predictions for each player and stat
predictions = []

for _, row in props_df.iterrows():
    player_name = row['Player']
    opponent_team = row['Opponent Team']
    stat_type = row['Stat Type'].lower()

    stat_cols = stat_mapping.get(stat_type)
    if stat_cols is None:
        continue

    if isinstance(stat_cols, list):  # For combined stats like pts_rebs_asts
        predicted_total = 0
        variance_total = 0
        for stat_col in stat_cols:
            pred_value, var_value = predict_stat_with_variance(player_name, opponent_team, stat_col, logs_df)
            if not np.isnan(pred_value):
                predicted_total += pred_value
            if not np.isnan(var_value):
                variance_total += var_value

        predictions.append({
            "Player": player_name,
            "Opponent": opponent_team,
            "Stat Type": "pts_rebs_asts",
            "Prop Line": row["Prop Line"],
            "Predicted Value": round(predicted_total, 2),
            "Variance": round(variance_total, 2)
        })
    else:
        predicted_value, variance_value = predict_stat_with_variance(player_name, opponent_team, stat_cols, logs_df)

        predictions.append({
            "Player": player_name,
            "Opponent": opponent_team,
            "Stat Type": stat_type,
            "Prop Line": row["Prop Line"],
            "Predicted Value": predicted_value,
            "Variance": variance_value
        })

# Convert to DataFrame
predictions_df = pd.DataFrame(predictions)

# Save predictions
if not predictions_df.empty:
    predictions_df.to_csv("optimized_player_props.csv", index=False)
    print("Predictions successfully saved to optimized_player_props.csv")
else:
    print("No valid predictions were generated.")
