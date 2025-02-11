import pandas as pd
import numpy as np
import joblib
import os
import itertools
import requests
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler
import re

# File paths
player_stats_file = "nba_player_stats.csv"
parlay_file = "underdog_player_props.csv"
model_file = "optimized_team_joint_prob_model.pkl"
scaler_file = "optimized_team_scaler.pkl"

print("Loading player stats dataset...")
df_stats = pd.read_csv(player_stats_file)

# Select relevant columns
selected_columns = ["PLAYER_NAME", "TEAM_ABBREVIATION", "PTS", "REB", "AST"]
df_selected = df_stats[selected_columns]

# Load parlay legs dataset
print("Loading player props dataset...")
parlay_df = pd.read_csv(parlay_file)

# Clean payout columns
parlay_df["Higher Payout"] = parlay_df["Higher Payout"].astype(str).str.replace("x", "").astype(float)
parlay_df["Lower Payout"] = parlay_df["Lower Payout"].astype(str).str.replace("x", "").astype(float)

# Function to clean and extract the first valid float from "Prop Line"
def extract_first_float(value):
    if isinstance(value, str):  
        value = value.replace("\n", "").strip()  
        match = re.search(r"\d+(\.\d+)?", value)  
        if match:
            return float(match.group())  
    return None  

parlay_df["Prop Line"] = parlay_df["Prop Line"].apply(extract_first_float)
parlay_df = parlay_df.dropna(subset=["Prop Line"])

# Extract numerical features for ML model
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

# Function to compute probability for a specific player prop
def get_player_probability(player, stat, value, pick):
    player_data = df_selected[df_selected["PLAYER_NAME"] == player]
    
    # Automatically remove bets with missing player data
    if player_data.empty:
        return None  

    stat_index = 0 if stat == "Points" else 1 if stat == "Rebounds" else 2
    num_samples = 5000
    samples_scaled = joint_model.rvs(size=num_samples)
    samples_original = scaler.inverse_transform(samples_scaled)

    if pick == "over":
        prob = np.mean(samples_original[:, stat_index] >= value)
    else:
        prob = np.mean(samples_original[:, stat_index] <= value)

    return prob

# Function to filter bets before computation
def filter_bets():
    filtered_bets = []
    for _, row in parlay_df.iterrows():
        probability = get_player_probability(row["Player"], row["Stat Type"], row["Prop Line"], "over")

        # Automatically discard bets with missing probability (i.e., missing player data)
        if probability is None:
            continue  

        ev = (probability * (row["Higher Payout"] - 1) * 100) - ((1 - probability) * 100)
        
        # Keep bets with probability between 0.50 and 0.90
        if 0.50 <= probability <= 0.90 and ev > 0:
            filtered_bets.append({
                "Player": row["Player"],
                "Stat Type": row["Stat Type"],
                "Prop Line": row["Prop Line"],
                "Pick": "over",
                "Odds": row["Higher Payout"],
                "Probability": probability,
                "EV": ev
            })

    # Ensure we retain at least 15 bets
    return filtered_bets[:15] if len(filtered_bets) > 15 else filtered_bets

# Function to identify the best +EV parlay with pruning
def find_best_parlay():
    all_possible_bets = filter_bets()

    if not all_possible_bets:
        print("❌ No valid bets found. Exiting...")
        return None, None

    best_ev = -float("inf")
    best_parlay = None

    # Stage 1: Compute 2-leg parlays, keep top 10
    two_leg_parlays = []
    for parlay_bets in itertools.combinations(all_possible_bets, 2):
        ev = sum(bet["EV"] for bet in parlay_bets)  
        two_leg_parlays.append((parlay_bets, ev))
    two_leg_parlays = sorted(two_leg_parlays, key=lambda x: x[1], reverse=True)[:10]  

    # Stage 2: Compute 3-leg parlays from top 10, keep top 5
    three_leg_parlays = []
    for parlay_bets, _ in two_leg_parlays:
        for new_bet in all_possible_bets:
            if new_bet not in parlay_bets:
                new_parlay = parlay_bets + (new_bet,)
                ev = sum(bet["EV"] for bet in new_parlay)
                three_leg_parlays.append((new_parlay, ev))
    three_leg_parlays = sorted(three_leg_parlays, key=lambda x: x[1], reverse=True)[:5]  

    # If no valid 3-leg parlays exist, return a message instead of crashing
    if not three_leg_parlays:
        print("❌ No valid 3-leg parlays found. Returning the best 2-leg parlay instead.")
        return two_leg_parlays[0] if two_leg_parlays else (None, None)

    best_parlay, best_ev = max(three_leg_parlays, key=lambda x: x[1])

    print(f"\n🔥 Best Parlay Found:\n🎰 **EV: ${best_ev:.2f}**")
    return best_parlay, best_ev

# Run optimization
best_parlay, best_ev = find_best_parlay()
