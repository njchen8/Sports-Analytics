import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import poisson
from scipy.optimize import minimize

def preprocess_nba_data(game_logs_path, prop_lines_path):
    """Loads and preprocesses NBA game logs and prop lines for regression modeling."""
    
    game_logs = pd.read_csv(game_logs_path)
    prop_lines = pd.read_csv(prop_lines_path)

    def convert_min_to_float(min_sec):
        try:
            minutes, seconds = map(int, min_sec.split(':'))
            return minutes + seconds / 60
        except:
            return 0  

    game_logs["MINUTES_PLAYED"] = game_logs["MIN_SEC"].apply(convert_min_to_float)
    filtered_games = game_logs[game_logs["MINUTES_PLAYED"] >= 5]
    filtered_games["GAME_DATE"] = pd.to_datetime(filtered_games["GAME_DATE"])
    filtered_games = filtered_games.sort_values(by=["PLAYER_ID", "GAME_DATE"], ascending=[True, False])
    filtered_games = filtered_games.groupby("PLAYER_ID").head(50)

    filtered_games["Home"] = filtered_games["MATCHUP"].apply(lambda x: 1 if "vs." in x else 0)
    filtered_games["GAME_DATE_LAG"] = filtered_games.groupby("PLAYER_ID")["GAME_DATE"].shift(1)
    filtered_games["Back_to_Back"] = ((filtered_games["GAME_DATE"] - filtered_games["GAME_DATE_LAG"]).dt.days == 1).astype(int)
    filtered_games["OPPONENT_LAG"] = filtered_games.groupby("PLAYER_ID")["MATCHUP"].shift(1)
    filtered_games["Same_Opponent"] = (filtered_games["MATCHUP"] == filtered_games["OPPONENT_LAG"]).astype(int)
    
    return filtered_games, prop_lines

def optimize_decay_rate(player_data, stat_column):
    """Finds the optimal alpha decay rate for EWMA by maximizing log-likelihood."""
    def negative_log_likelihood(alpha):
        smoothed_series = player_data[stat_column].ewm(alpha=alpha, adjust=False).mean()
        poisson_likelihood = poisson.logpmf(player_data[stat_column], smoothed_series)
        return -np.sum(poisson_likelihood)  # Minimize negative log-likelihood

    result = minimize(negative_log_likelihood, x0=0.2, bounds=[(0.05, 0.5)])  # Search optimal alpha
    return result.x[0]

def weighted_moving_average(series, alpha=None):
    """Applies an exponential weighted moving average with an optimized decay rate."""
    if alpha is None:  # Use default if not optimized
        alpha = 0.2
    return series.ewm(alpha=alpha, adjust=False).mean()

def run_poisson_regression_and_predict(filtered_games, prop_lines):
    """Trains Poisson regression models, predicts expected values, and computes probabilities using Poisson distribution."""
    
    stat_type_mapping = {
        "points": "PTS",
        "assists": "AST",
        "rebounds": "REB"
    }

    prop_lines_filtered = prop_lines[prop_lines["Stat Type"].isin(stat_type_mapping.keys())]
    predictions = []

    for _, row in prop_lines_filtered.iterrows():
        player_name = row["Player"]
        stat_type = row["Stat Type"]
        prop_team = row["Opponent Team"]
        home_vs_away = 1 if row["Home Team"] == row["Current Team"] else 0
        stat_column = stat_type_mapping[stat_type]

        player_data = filtered_games[filtered_games["PLAYER_NAME"] == player_name].copy()
        if player_data.empty or stat_column not in player_data.columns:
            continue

        player_data[stat_column] = pd.to_numeric(player_data[stat_column], errors="coerce")
        alpha_optimal = optimize_decay_rate(player_data, stat_column)
        player_data[stat_column] = weighted_moving_average(player_data[stat_column], alpha=alpha_optimal)
        X = player_data[["Home", "Back_to_Back", "Same_Opponent", "MINUTES_PLAYED"]].astype(float)
        X = sm.add_constant(X)
        y = player_data[stat_column].astype(int)

        if len(y) < 25:
            continue

        model = sm.GLM(y, X, family=sm.families.NegativeBinomial()).fit()
        input_features = np.array([1, home_vs_away, 0, 0, player_data["MINUTES_PLAYED"].mean()], dtype=float)
        expected_value = model.predict(input_features.reshape(1, -1))[0]

        empirical_variance = player_data[stat_column].var()
        adjusted_lambda = expected_value + (empirical_variance - expected_value) / expected_value

        prop_line = row["Prop Line"]
        prob_over = 1 - poisson.cdf(prop_line, mu=adjusted_lambda)
        prob_under = poisson.cdf(prop_line, mu=adjusted_lambda)

        predictions.append({
            "Player": player_name,
            "Stat Type": stat_type,
            "Prop Team": prop_team,
            "Home vs. Away": home_vs_away,
            "Expected Value": expected_value,
            "Prop Line": prop_line,
            "Probability Over": prob_over,
            "Probability Under": prob_under,
        })

    predictions_df = pd.DataFrame(predictions)
    predictions_df["Max Probability"] = predictions_df[["Probability Over", "Probability Under"]].max(axis=1)
    predictions_df = predictions_df.sort_values(by="Max Probability", ascending=False)
    
    return predictions_df

# Example usage
game_logs_path = "nba_players_game_logs_cleaned.csv"
prop_lines_path = "fixed_data.csv"

filtered_games, prop_lines = preprocess_nba_data(game_logs_path, prop_lines_path)
sorted_predictions_df = run_poisson_regression_and_predict(filtered_games, prop_lines)

sorted_predictions_df.to_csv("sorted_predicted_props_poisson.csv", index=False)
print(sorted_predictions_df.head(20))
