import pandas as pd
import numpy as np
import joblib
import os
import itertools
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler

# File paths
data_file = "nba_players_game_logs_5_seasons.csv"
model_file = "optimized_team_joint_prob_model.pkl"
scaler_file = "optimized_team_scaler.pkl"

print("Loading dataset...")
df = pd.read_csv(data_file)

# Select relevant columns
selected_columns = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "MATCHUP", "PTS", "REB", "AST"]
df_selected = df[selected_columns]

# Extract numerical features
data = df_selected[["PTS", "REB", "AST"]].values

# Standardize data
print("Scaling data...")
scaler = StandardScaler()

if os.path.exists(scaler_file):
    print("Loading saved scaler...")
    scaler = joblib.load(scaler_file)
    data_scaled = scaler.transform(data)
else:
    print("Fitting scaler...")
    data_scaled = scaler.fit_transform(data)
    joblib.dump(scaler, scaler_file)

# Train or load model
if os.path.exists(model_file):
    print("Loading pre-trained optimized probability model...")
    joint_model = joblib.load(model_file)
else:
    print("Fitting Multivariate Gaussian Model...")
    mean_vector = np.mean(data_scaled, axis=0)
    cov_matrix = np.cov(data_scaled.T)
    joint_model = multivariate_normal(mean=mean_vector, cov=cov_matrix)
    
    joblib.dump(joint_model, model_file)
    print("Model training complete and saved.")

# Function to compute weighted probability based on game type
def get_weighted_probability(player, related_player, stat, value, pick, game_type, is_teammate):
    """
    Computes probability based on game type:
    - **Same Team:** Joint Distribution (Strong Correlation)
    - **Same Game:** Joint Distribution (Weaker Correlation)
    - **Different Matches:** Independent Distribution (No Correlation)
    """
    print(f"\n📊 Checking probability for {player} (vs/with {related_player}) - {game_type}...")

    # General probability from joint model
    num_samples = 5000
    samples_scaled = joint_model.rvs(size=num_samples)
    samples_original = scaler.inverse_transform(samples_scaled)

    stat_index = 0 if stat == "PTS" else 1 if stat == "REB" else 2
    if pick == "over":
        general_prob = np.mean(samples_original[:, stat_index] >= value)
    else:
        general_prob = np.mean(samples_original[:, stat_index] <= value)

    if game_type == "different_match":
        print(f"🎯 Independent Probability: {general_prob:.2%}")
        return general_prob

    # Compute same-team adjustments
    player_games = df_selected[df_selected["PLAYER_NAME"] == player]
    related_games = df_selected[df_selected["PLAYER_NAME"] == related_player] if related_player else None
    same_games = pd.merge(player_games, related_games, on="MATCHUP", suffixes=("_player", "_related")) if related_games is not None else None

    same_team_prob = None
    if game_type in ["same_team", "same_game"] and same_games is not None and not same_games.empty:
        stat_col = f"{stat}_player"
        past_values = same_games[stat_col]
        if pick == "over":
            same_team_prob = np.mean(past_values >= value)
        else:
            same_team_prob = np.mean(past_values <= value)

    # Weighted combination based on game type
    weight_team = 0.70 if game_type == "same_team" else 0.50  # More weight for same-team
    weight_general = 1 - weight_team

    final_prob = (
        (same_team_prob if same_team_prob is not None else general_prob) * weight_team +
        general_prob * weight_general
    )

    print(f"📊 P({player} {stat} {pick} {value}) weighted = {final_prob:.2%}")
    return final_prob

# Function to compute probability for a parlay
def compute_parlay_probability(parlay_bets):
    """
    Computes the probability of hitting a parlay using weighted probabilities.
    """
    total_prob = 1
    for bet in parlay_bets:
        prob = get_weighted_probability(
            bet["player"], bet["related_player"], bet["stat"], bet["value"], bet["pick"], bet["game_type"], bet["is_teammate"]
        )
        total_prob *= prob

    return total_prob

# Function to compute total parlay odds
def compute_parlay_odds(parlay_bets):
    """
    Computes the total odds for a given parlay based on individual leg odds.
    """
    decimal_odds = [bet["odds"] for bet in parlay_bets]
    total_odds = np.prod(decimal_odds)
    return total_odds

# Function to find the best parlay using all possible bets
def find_best_parlay(all_possible_bets):
    """
    Finds the best +EV parlay using weighted probabilities and individual leg odds.
    """
    best_ev = -float("inf")
    best_parlay = None
    best_prob = 0
    best_odds = 0

    # Generate all valid parlay combinations (2 to 4 legs)
    for r in range(2, min(5, len(all_possible_bets) + 1)):
        for parlay_bets in itertools.combinations(all_possible_bets, r):
            parlay_prob = compute_parlay_probability(parlay_bets)
            parlay_odds = compute_parlay_odds(parlay_bets)

            # Compute Expected Value (EV)
            payout = (parlay_odds - 1) * 100
            risk = 100
            ev = (parlay_prob * payout) - ((1 - parlay_prob) * risk)

            # Track best +EV parlay
            if ev > best_ev:
                best_ev = ev
                best_parlay = parlay_bets
                best_prob = parlay_prob
                best_odds = parlay_odds

    print(f"\n🔥 Best Parlay Found:\n📊 **Probability: {best_prob:.2%}** 🎰 **Odds: {best_odds:.2f}** 💰 **EV: ${best_ev:.2f}**")
    return best_parlay, best_ev
