import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm

def preprocess_nba_data(game_logs_path, prop_lines_path):
    """Loads and preprocesses NBA game logs and prop lines for regression modeling."""
    
    # Load the datasets
    game_logs = pd.read_csv(game_logs_path)
    prop_lines = pd.read_csv(prop_lines_path)

    # Convert 'MIN_SEC' to numerical minutes
    def convert_min_to_float(min_sec):
        try:
            minutes, seconds = map(int, min_sec.split(':'))
            return minutes + seconds / 60
        except:
            return 0  # Default to 0 for any errors

    game_logs["MINUTES_PLAYED"] = game_logs["MIN_SEC"].apply(convert_min_to_float)

    # Filter out games where player played less than 5 minutes
    filtered_games = game_logs[game_logs["MINUTES_PLAYED"] >= 5]

    # Convert GAME_DATE to datetime format
    filtered_games["GAME_DATE"] = pd.to_datetime(filtered_games["GAME_DATE"])

    # Sort by player and date
    filtered_games = filtered_games.sort_values(by=["PLAYER_ID", "GAME_DATE"], ascending=[True, False])

    # Keep the last 50 games per player
    filtered_games = filtered_games.groupby("PLAYER_ID").head(50)

    # Create indicator variables
    filtered_games["Home"] = filtered_games["MATCHUP"].apply(lambda x: 1 if "vs." in x else 0)
    filtered_games["GAME_DATE_LAG"] = filtered_games.groupby("PLAYER_ID")["GAME_DATE"].shift(1)
    filtered_games["Back_to_Back"] = ((filtered_games["GAME_DATE"] - filtered_games["GAME_DATE_LAG"]).dt.days == 1).astype(int)
    filtered_games["OPPONENT_LAG"] = filtered_games.groupby("PLAYER_ID")["MATCHUP"].shift(1)
    filtered_games["Same_Opponent"] = (filtered_games["MATCHUP"] == filtered_games["OPPONENT_LAG"]).astype(int)

    return filtered_games, prop_lines

def weighted_moving_average(series, alpha=0.2):
    """Applies an exponential weighted moving average for smoother predictions."""
    return series.ewm(alpha=alpha, adjust=False).mean()

def run_regression_and_predict(filtered_games, prop_lines):
    """Trains regression models, predicts expected values, computes probabilities, and sorts results."""
    
    # Map Stat Types to Dataset Columns
    stat_type_mapping = {
        "points": "PTS",
        "assists": "AST",
        "rebounds": "REB"
    }

    # Filter only valid stat types
    prop_lines_filtered = prop_lines[prop_lines["Stat Type"].isin(stat_type_mapping.keys())]

    predictions = []

    for _, row in prop_lines_filtered.iterrows():
        player_name = row["Player"]
        stat_type = row["Stat Type"]
        prop_team = row["Opponent Team"]
        home_vs_away = 1 if row["Home Team"] == row["Current Team"] else 0

        # Map stat type to correct column name
        stat_column = stat_type_mapping[stat_type]

        # Filter player's past games
        player_data = filtered_games[filtered_games["PLAYER_NAME"] == player_name].copy()

        if player_data.empty or stat_column not in player_data.columns:
            continue

        # Ensure numerical conversion
        player_data[stat_column] = pd.to_numeric(player_data[stat_column], errors="coerce")

        # Apply weighted moving average for smoothing
        player_data[stat_column] = weighted_moving_average(player_data[stat_column])

        # Select relevant features
        X = player_data[["Home", "Back_to_Back", "Same_Opponent", "MINUTES_PLAYED"]].astype(float)
        X = sm.add_constant(X)  # Add intercept

        # Target variable
        y = player_data[stat_column].astype(float)

        if len(y) < 10:  # Ensure enough data points for regression
            continue

        # Fit regression model
        model = sm.OLS(y, X, missing="drop").fit()

        # Predict expected value based on prop line conditions
        input_features = np.array([1, home_vs_away, 0, 0, player_data["MINUTES_PLAYED"].mean()], dtype=float)
        expected_value = model.predict(input_features.reshape(1, -1))[0]

        # Get residual standard deviation (sigma) from full dataset variance
        full_sigma = player_data[stat_column].std() + 1  # Add small variance correction

        # Compute probability of going over the prop line
        prop_line = row["Prop Line"]
        prob_over = 1 - norm.cdf(prop_line, loc=expected_value, scale=full_sigma)
        prob_under = norm.cdf(prop_line, loc=expected_value, scale=full_sigma)

        # Best bet decision
        best_bet = "Over" if prob_over > 0.5 else "Under"

        predictions.append({
            "Player": player_name,
            "Stat Type": stat_type,
            "Prop Team": prop_team,
            "Home vs. Away": home_vs_away,
            "Expected Value": expected_value,
            "Prop Line": prop_line,
            "Difference": expected_value - prop_line,
            "Probability Over": prob_over,
            "Probability Under": prob_under,
            "Best Bet": best_bet
        })

    predictions_df = pd.DataFrame(predictions)

    # Sort predictions by the highest probability (certainty of bet)
    predictions_df["Max Probability"] = predictions_df[["Probability Over", "Probability Under"]].max(axis=1)
    predictions_df = predictions_df.sort_values(by="Max Probability", ascending=False)

    return predictions_df

# Example usage
game_logs_path = "nba_players_game_logs_cleaned.csv"
prop_lines_path = "fixed_data.csv"

filtered_games, prop_lines = preprocess_nba_data(game_logs_path, prop_lines_path)
sorted_predictions_df = run_regression_and_predict(filtered_games, prop_lines)

# Save or display the results
sorted_predictions_df.to_csv("sorted_predicted_props.csv", index=False)
print(sorted_predictions_df)
