import pandas as pd
import numpy as np
import joblib
import os
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler

# File paths
data_file = "nba_players_game_logs_5_seasons.csv"
model_file = "team_joint_prob_model.pkl"
scaler_file = "team_scaler.pkl"

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
    print("Loading pre-trained team joint probability model...")
    joint_model = joblib.load(model_file)
else:
    print("Fitting Multivariate Gaussian Model...")
    mean_vector = np.mean(data_scaled, axis=0)
    cov_matrix = np.cov(data_scaled.T)
    joint_model = multivariate_normal(mean=mean_vector, cov=cov_matrix)
    
    joblib.dump(joint_model, model_file)
    print("Model training complete and saved.")

# Function to compute probability for any given parlay
def compute_parlay_probability(parlay_bets):
    """
    Computes the probability of hitting a parlay using conditional probability.
    
    :param parlay_bets: List of bets in format:
        [{"player": "LeBron James", "stat": "PTS", "value": 25.5, "pick": "over"}, ...]
    :return: Final parlay probability
    """
    num_samples = 5000
    samples_scaled = joint_model.rvs(size=num_samples)
    samples_original = scaler.inverse_transform(samples_scaled)

    # Compute probabilities based on given conditions
    total_prob = 1
    for bet in parlay_bets:
        player = bet["player"]
        stat = bet["stat"]
        value = bet["value"]
        pick = bet["pick"]

        player_data = df_selected[df_selected["PLAYER_NAME"] == player]
        if player_data.empty:
            continue

        stat_index = 0 if stat == "PTS" else 1 if stat == "REB" else 2
        if pick == "over":
            prob = np.mean(samples_original[:, stat_index] >= value)
        else:
            prob = np.mean(samples_original[:, stat_index] <= value)

        total_prob *= prob

    return total_prob

# Function to find the best parlay based on user inputted possible bets
def find_best_parlay(all_possible_bets, underdog_odds):
    """
    Finds the best +EV parlay using all possible bets provided.
    
    :param all_possible_bets: List of all individual bets.
    :param underdog_odds: Decimal odds for the parlay bet.
    """
    best_ev = -float("inf")
    best_parlay = None
    best_prob = 0

    # Generate all possible parlay combinations (1 to 4 legs recommended for realism)
    for r in range(2, min(5, len(all_possible_bets) + 1)):  # Parlays from 2 to 4 legs
        for parlay_bets in itertools.combinations(all_possible_bets, r):
            parlay_prob = compute_parlay_probability(parlay_bets)

            # Convert sportsbook odds to implied probability
            implied_prob = 1 / underdog_odds

            # Compute Expected Value (EV)
            payout = (underdog_odds - 1) * 100
            risk = 100
            ev = (parlay_prob * payout) - ((1 - parlay_prob) * risk)

            # Track best +EV parlay
            if ev > best_ev:
                best_ev = ev
                best_parlay = parlay_bets
                best_prob = parlay_prob

    # Print the best parlay found
    print("\n🔥 **Best Parlay Found:**")
    for bet in best_parlay:
        print(f"{bet['player']} {bet['stat']} {bet['pick']} {bet['value']}")

    print(f"\n📊 **Parlay Probability: {best_prob:.2%}**")
    print(f"🎰 **Sportsbook Implied Probability: {1/underdog_odds:.2%}**")
    print(f"💰 **Expected Value (EV): ${best_ev:.2f}**")

    if best_ev > 0:
        print("✅ **+EV Bet! Consider placing this wager.**")
    else:
        print("❌ **-EV Bet. Avoid this bet.**")

    return best_parlay, best_ev

# **Example Usage - Input All Possible Bets**
all_possible_bets = [
    {"player": "Thanasis Antetokounmpo", "stat": "PTS", "value": 10, "pick": "over"},
    {"player": "Thanasis Antetokounmpo", "stat": "REB", "value": 8, "pick": "under"},
    {"player": "Thanasis Antetokounmpo", "stat": "AST", "value": 5, "pick": "over"},
    {"player": "Ryan Arcidiacono", "stat": "AST", "value": 8, "pick": "under"},
    {"player": "Ryan Arcidiacono", "stat": "PTS", "value": 5, "pick": "over"},
    {"player": "Ryan Arcidiacono", "stat": "REB", "value": 6, "pick": "under"},
    {"player": "Udoka Azubuike", "stat": "PTS", "value": 2, "pick": "over"},
    {"player": "Udoka Azubuike", "stat": "REB", "value": 3, "pick": "under"},
]

# Assume Underdog odds for this parlay is +750 (8.5 in decimal odds)
underdog_odds = 8.5

# Find the best possible parlay
find_best_parlay(all_possible_bets, underdog_odds)
