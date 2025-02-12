import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# Load datasets with low_memory=False to avoid dtype warnings
props_file_path = "underdog_player_props.csv"
logs_file_path = "nba_players_game_logs_2018_25.csv"

props_df = pd.read_csv(props_file_path, low_memory=False)
logs_df = pd.read_csv(logs_file_path, low_memory=False)

# Convert GAME_DATE to datetime
logs_df['GAME_DATE'] = pd.to_datetime(logs_df['GAME_DATE'], errors='coerce')

# Ensure key stat columns are numeric
stats_columns = ['PTS', 'REB', 'AST']
for col in stats_columns:
    logs_df[col] = pd.to_numeric(logs_df[col], errors='coerce')

# Drop NaN values in key stats
logs_df = logs_df.dropna(subset=stats_columns)

# Normalize stat type mappings
stat_mapping = {
    "points": "PTS",
    "rebounds": "REB",
    "assists": "AST",
    "pts_rebs_asts": ["PTS", "REB", "AST"]  # Sum of these stats
}

def predict_stat_with_variance(player_name, opponent_team, stat, player_data, recent_weight=0.8, matchup_weight=1.2, forecast_weight=0.6):
    """
    Predicts a player's expected value and variance for a given stat in an upcoming game.
    """
    
    if stat not in player_data.columns:
        return np.nan, np.nan  

    # Filter data for the player
    player_df = player_data[player_data['PLAYER_NAME'] == player_name].copy()

    # Sort by date
    player_df = player_df.sort_values(by="GAME_DATE")

    # Drop NaNs for the specific stat
    player_df = player_df.dropna(subset=[stat])

    if player_df.empty:
        return np.nan, np.nan  

    # Weighted recent performance
    recent_games = player_df.tail(10).copy()
    recent_games['Weight'] = np.exp(-np.arange(len(recent_games))[::-1] / (len(recent_games) / 3))  

    # Weighted past matchups
    matchup_games = player_df[player_df['MATCHUP'].str.contains(opponent_team, na=False)].copy()
    matchup_games['Weight'] = np.exp(-np.arange(len(matchup_games))[::-1] / (len(matchup_games) / 3)) * matchup_weight  

    # Compute weighted averages
    weighted_recent = np.average(recent_games[stat], weights=recent_games['Weight']) if not recent_games.empty else np.nan
    weighted_matchup = np.average(matchup_games[stat], weights=matchup_games['Weight']) if not matchup_games.empty else np.nan

    # Compute variance
    var_recent = np.average((recent_games[stat] - weighted_recent) ** 2, weights=recent_games['Weight']) if not recent_games.empty else np.nan
    var_matchup = np.average((matchup_games[stat] - weighted_matchup) ** 2, weights=matchup_games['Weight']) if not matchup_games.empty else np.nan

    # ARIMA trend prediction
    arima_forecast = np.nan  
    if len(player_df) > 10:  
        arima_model = ARIMA(player_df[stat], order=(3,1,2))
        arima_fit = arima_model.fit()
        arima_forecast = arima_fit.forecast(steps=1).iloc[0]

    # Combine predictions
    final_prediction = (
        (recent_weight * weighted_recent if not np.isnan(weighted_recent) else 0) +
        (matchup_weight * weighted_matchup if not np.isnan(weighted_matchup) else 0) +
        (forecast_weight * arima_forecast if not np.isnan(arima_forecast) else 0)
    ) / (recent_weight + matchup_weight + forecast_weight)

    # Combine variance (weighted sum of variances)
    final_variance = (
        (recent_weight * var_recent if not np.isnan(var_recent) else 0) +
        (matchup_weight * var_matchup if not np.isnan(var_matchup) else 0)
    ) / (recent_weight + matchup_weight)  # ARIMA variance not included

    return round(final_prediction, 2), round(final_variance, 2)

# Create a dictionary to store predictions for each player and stat
predictions = []

for _, row in props_df.iterrows():
    player_name = row['Player']
    opponent_team = row['Opponent Team']
    stat_type = row['Stat Type'].lower()  # Normalize case
    
    # Get corresponding game log column(s)
    stat_cols = stat_mapping.get(stat_type)

    if stat_cols is None:
        continue  # Skip unsupported stats

    if isinstance(stat_cols, list):  # Handle "pts_rebs_asts"
        predicted_total = 0
        variance_total = 0
        for stat_col in stat_cols:
            pred_value, var_value = predict_stat_with_variance(player_name, opponent_team, stat_col, logs_df)
            if not np.isnan(pred_value):
                predicted_total += pred_value
            if not np.isnan(var_value):
                variance_total += var_value  # Sum variances
            
        predictions.append({
            "Player": player_name,
            "Opponent": opponent_team,
            "Stat Type": "pts_rebs_asts",
            "Prop Line": row["Prop Line"],
            "Predicted Value": round(predicted_total, 2),
            "Variance": round(variance_total, 2)
        })

    else:  # Normal single-stat case
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

# Ensure the DataFrame is not empty before saving
if not predictions_df.empty:
    predictions_df.to_csv("predicted_player_props.csv", index=False)
    print("Predictions successfully saved to predicted_player_props.csv")
else:
    print("No valid predictions were generated. Please check the input data.")
