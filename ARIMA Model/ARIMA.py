import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from datetime import datetime, timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load datasets with low_memory=False to avoid dtype warnings
props_file_path = "fixed_data.csv"
logs_file_path = "nba_players_game_logs_2018_25.csv"

props_df = pd.read_csv(props_file_path, low_memory=False)
logs_df = pd.read_csv(logs_file_path, low_memory=False)
props_df = props_df[~props_df['Stat Type'].str.contains('1_')]

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
    "pts_rebs_asts": "PTS_REB_AST",
    "period_1_points": "PTS"  # add mapping if needed
}

# Suppress ARIMA warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def extract_opponent(matchup):
    """
    Extract the opponent team abbreviation from the matchup string.
    If the string contains ' vs. ', assume home game and return the part after it.
    Otherwise, if it contains '@', return the part after '@'.
    """
    if " vs. " in matchup:
        return matchup.split(" vs. ")[1].strip()
    elif "@" in matchup:
        return matchup.split("@")[1].strip()
    else:
        return None

def predict_stat_with_variance(player_name, stat, player_data, prop_row):
    """
    Predicts a player's expected value and variance for a given stat using the last 25 valid matches.
    This version builds exogenous variables for ARIMA (is_home and is_opponent) without relying on '@'
    detection—instead, it uses the 'vs. ' indicator for home games and extracts the opponent with a helper.
    
    For the upcoming game, the prop file provides Home Team and Away Team.
    """
    if stat not in player_data.columns:
        return np.nan, np.nan, "", np.nan, np.nan, np.nan

    # Filter and sort data for the player
    player_df = player_data[player_data['PLAYER_NAME'] == player_name].copy()
    player_df = player_df.sort_values(by="GAME_DATE")

    # Ensure the last 25 matches are within a reasonable time period
    if len(player_df) > 25:
        time_span = (player_df['GAME_DATE'].iloc[-1] - player_df['GAME_DATE'].iloc[-25]).days
        if time_span > 180:  # keep matches within the last 6 months
            player_df = player_df[player_df['GAME_DATE'] >= player_df['GAME_DATE'].iloc[-25]]
    player_df = player_df.tail(25)

    # Drop NaNs for the stat and require at least 5 minutes of play
    player_df = player_df.dropna(subset=[stat])
    player_df = player_df[player_df['MIN'] >= 5]

    if player_df.empty:
        return np.nan, np.nan, "", np.nan, np.nan, np.nan

    # Copy for variance calculation (only MIN filter applied)
    variance_df = player_df.copy()

    # Apply 3 SD filter for ARIMA and weighted mean calculations
    mean_val = player_df[stat].mean()
    std_val = player_df[stat].std()
    filtered_df = player_df[(player_df[stat] >= mean_val - 3 * std_val) &
                             (player_df[stat] <= mean_val + 3 * std_val)].copy()

    if filtered_df.empty:
        return np.nan, np.nan, "", np.nan, np.nan, np.nan

    # Date range for the filtered data
    date_range = f"{filtered_df['GAME_DATE'].min().date()} to {filtered_df['GAME_DATE'].max().date()}"
    if 'NaT' in date_range:
        return np.nan, np.nan, "", np.nan, np.nan, np.nan

    # Check if the player has played recently (last 2 weeks)
    last_game_date = filtered_df['GAME_DATE'].max()
    if (datetime.now() - last_game_date) > timedelta(days=14):
        return np.nan, np.nan, "", np.nan, np.nan, np.nan

    # Determine upcoming matchup context using prop data
    player_team = filtered_df.iloc[-1]['TEAM_ABBREVIATION']
    teams = {prop_row['Current Team'], prop_row['Opponent Team']}
    if player_team in teams:
        opponent_team = (teams - {player_team}).pop()
    else:
        return np.nan, np.nan, "", np.nan, np.nan, np.nan

    # Upcoming game home/away from prop data: if Current Team equals Home Team then game is at home.
    upcoming_home = 1 if prop_row["Current Team"] == prop_row["Home Team"] else 0

    # Compute weights for weighted mean and variance using the matchup string,
    # but determine home/away via the presence of " vs. " (home) rather than "@".
    filtered_df.loc[:, 'recent_weight'] = np.exp(-np.arange(len(filtered_df))[::-1] / 3)
    
    # For historical games, if the MATCHUP string contains " vs. ", assume it was a home game.
    filtered_df.loc[:, 'location_weight'] = np.where(filtered_df['MATCHUP'].str.contains(" vs. "), 
                                                       1.2 if upcoming_home == 1 else 0.8,
                                                       0.8 if upcoming_home == 1 else 1.2)
    
    # Extract opponent from MATCHUP and assign weight if it matches the upcoming opponent.
    filtered_df.loc[:, 'extracted_opponent'] = filtered_df['MATCHUP'].apply(extract_opponent)
    filtered_df.loc[:, 'opponent_weight'] = np.where(filtered_df['extracted_opponent'] == opponent_team, 1.5, 1.0)
    
    # Combined weight for historical games
    filtered_df.loc[:, 'combined_weight'] = (
        filtered_df['recent_weight'] *
        filtered_df['location_weight'] *
        filtered_df['opponent_weight']
    )
    weighted_mean = np.average(filtered_df[stat], weights=filtered_df['combined_weight'])
    
    # Compute variance using variance_df with similar weights
    variance_df.loc[:, 'recent_weight'] = np.exp(-np.arange(len(variance_df))[::-1] / 3)
    variance_df.loc[:, 'location_weight'] = np.where(variance_df['MATCHUP'].str.contains(" vs. "), 
                                                       1.2 if upcoming_home == 1 else 0.8,
                                                       0.8 if upcoming_home == 1 else 1.2)
    variance_df.loc[:, 'extracted_opponent'] = variance_df['MATCHUP'].apply(extract_opponent)
    variance_df.loc[:, 'opponent_weight'] = np.where(variance_df['extracted_opponent'] == opponent_team, 1.5, 1.0)
    variance_df.loc[:, 'combined_weight'] = (
        variance_df['recent_weight'] *
        variance_df['location_weight'] *
        variance_df['opponent_weight']
    )
    variance_value = np.average((variance_df[stat] - weighted_mean) ** 2, weights=variance_df['combined_weight'])
    
    # ARIMA modeling with exogenous variables for matchup context.
    from statsmodels.tsa.statespace.sarimax import SARIMAX

# ----------------------------
# SARIMAX modeling with exogenous variables for matchup context.
# ----------------------------
    if len(filtered_df) >= 5:
        # Make a copy and assign a new index
        model_df = filtered_df.copy()
        model_df['game_index'] = np.arange(len(model_df))
        model_df = model_df.set_index('game_index')  # Now both endog and exog share the same index

        # Determine home/away using both indicators:
        # Home game if MATCHUP contains " vs. ", away if it contains "@".
        model_df['is_home'] = model_df['MATCHUP'].apply(
            lambda x: 1 if " vs. " in x else (0 if "@" in x else 0)
        )
        # Extract opponent using helper function
        model_df['extracted_opponent'] = model_df['MATCHUP'].apply(extract_opponent)
        # is_opponent: 1 if the extracted opponent equals the upcoming opponent, else 0.
        model_df['is_opponent'] = (model_df['extracted_opponent'] == opponent_team).astype(int)
        
        # Prepare exogenous data (they now share the same index as the stat series)
        exog_data = model_df[['is_home', 'is_opponent']]
        # Upcoming game exogenous variables: use prop data for home status and assume opponent=1.
        upcoming_exog = np.array([[upcoming_home, 1]])
        
        try:
            model = SARIMAX(
                model_df[stat],
                order=(1, 0, 1),
                exog=exog_data,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            best_model = model.fit(disp=False)
            arima_forecast = best_model.forecast(steps=1, exog=upcoming_exog).iloc[0]
            if abs(arima_forecast - weighted_mean) > std_val:
                arima_forecast = weighted_mean
                print("Forecast deviated too much; defaulting to weighted mean")
            else:
                print("SARIMAX forecast succeeded")
        except Exception as e:
            print("SARIMAX model failed; using weighted mean. Error:", e)
            arima_forecast = weighted_mean
    else:
        arima_forecast = weighted_mean
        print("Not enough data for SARIMAX; using weighted mean")


    weighted_combination = 0.5 * weighted_mean + 0.5 * arima_forecast
    
    return (round(arima_forecast, 2), round(variance_value, 2), date_range,
            weighted_mean, arima_forecast, round(weighted_combination, 2))

# Create a list to store predictions
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
predictions_df = pd.DataFrame(predictions)
predictions_df = predictions_df.sort_values(by=["ARIMA Forecast", "Weighted Mean"], ascending=[False, False])

if not predictions_df.empty:
    predictions_df.to_csv("player_expected_values.csv", index=False)
    print("Expected values and variances saved to player_expected_values.csv")
else:
    print("No valid predictions were generated")