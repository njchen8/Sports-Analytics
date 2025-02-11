import pandas as pd
import numpy as np
import joblib
import os
import itertools
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import re

# File paths
player_stats_file = "nba_players_game_logs_2018_25.csv"
parlay_file = "underdog_player_props.csv"
model_file = "team_joint_prob_nn.keras"
scaler_file = "team_scaler.pkl"
encoder_file = "matchup_encoder.pkl"

print("Loading player stats dataset...")
df_stats = pd.read_csv(player_stats_file, low_memory=False)

# Select relevant columns
selected_columns = ["PLAYER_NAME", "TEAM_ABBREVIATION", "MATCHUP", "GAME_DATE", "PTS", "REB", "AST"]
df_selected = df_stats[selected_columns].copy()

# Convert necessary columns to numeric and datetime
df_selected[["PTS", "REB", "AST"]] = df_selected[["PTS", "REB", "AST"]].apply(pd.to_numeric, errors='coerce')
df_selected = df_selected.dropna(subset=["PTS", "REB", "AST"])
df_selected["GAME_DATE"] = pd.to_datetime(df_selected["GAME_DATE"])

# Compute game importance using Exponential Decay
λ = 0.01  
latest_date = df_selected["GAME_DATE"].max()
df_selected["Game_Age"] = (latest_date - df_selected["GAME_DATE"]).dt.days
df_selected["Weight"] = np.exp(-λ * df_selected["Game_Age"])

# One-Hot Encode Opponent Matchups
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
opponent_encoded = encoder.fit_transform(df_selected[["MATCHUP"]])

# Standardize numerical stats
scaler = StandardScaler()
numerical_data = df_selected[["PTS", "REB", "AST"]].values
data_scaled = scaler.fit_transform(numerical_data)

# Merge opponent features with player stats
final_data = np.hstack((data_scaled, opponent_encoded))

# Save scaler and encoder
joblib.dump(scaler, scaler_file)
joblib.dump(encoder, encoder_file)

print("Building TensorFlow neural network with opponent matchups...")

# **Build & Train Model**
inputs = Input(shape=(final_data.shape[1],))
x = Dense(128, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
outputs = Dense(3, activation='linear')(x)

team_model = Model(inputs=inputs, outputs=outputs)
team_model.compile(optimizer='adam', loss='mse')

team_model.fit(final_data, data_scaled, sample_weight=df_selected["Weight"].values, epochs=2, batch_size=32, verbose=1)

# **Save in New Keras Format**
team_model.save(model_file)
print("Model training complete and saved.")

# ============================ CONNECT TO UNDERDOG PROPS ============================ #

print("Loading player props dataset...")
parlay_df = pd.read_csv(parlay_file)

# Clean payout columns
parlay_df["Higher Payout"] = parlay_df["Higher Payout"].astype(str).str.replace("x", "").astype(float)
parlay_df["Lower Payout"] = parlay_df["Lower Payout"].astype(str).str.replace("x", "").astype(float)

# Extract numeric values from "Prop Line"
def extract_first_float(value):
    if isinstance(value, str):  
        value = value.replace("\n", "").strip()  
        match = re.search(r"\d+(\.\d+)?", value)  
        if match:
            return float(match.group())  
    return None  

parlay_df["Prop Line"] = parlay_df["Prop Line"].apply(extract_first_float)
parlay_df = parlay_df.dropna(subset=["Prop Line"])

def get_player_probability(player, stat, value, pick):
    player_data = df_selected[df_selected["PLAYER_NAME"] == player]

    if player_data.empty:
        return None  

    # Get the latest available game stats for this player
    latest_game = player_data.iloc[-1][["PTS", "REB", "AST"]].values.reshape(1, -1)
    
    # Standardize the player's latest game stats
    latest_game_scaled = scaler.transform(latest_game)

    # Get opponent matchup encoding
    player_matchup = pd.DataFrame([[player_data["MATCHUP"].iloc[-1]]], columns=["MATCHUP"])
    matchup_vector = encoder.transform(player_matchup)  # Ensure it's a 2D array

    # Concatenate **1 row of stats** with **1 row of matchup vector**
    model_input = np.hstack((latest_game_scaled, matchup_vector))

    # Ensure input is 2D (batch size of 1)
    samples = team_model.predict(model_input)

    stat_index = 0 if stat == "PTS" else 1 if stat == "REB" else 2

    # Compute probability
    if pick == "over":
        prob = np.mean(samples[:, stat_index] >= value)
    else:
        prob = np.mean(samples[:, stat_index] <= value)
    print(prob)
    return prob


from scipy.stats import norm

# **Compute Over/Under Decision, Expected Value (EV), and Confidence Interval**
def evaluate_bet_ev(probability, odds, stake=100):
    """
    Compares the neural network probability with the fair probability from betting odds.
    Determines whether to bet Over or Under and calculates confidence intervals.
    """
    fair_prob = 1 / odds  # Convert betting odds to probability
    EV_odds = (fair_prob * (odds - 1) * stake) - ((1 - fair_prob) * stake)  # Market EV
    EV_model = (probability * (odds - 1) * stake) - ((1 - probability) * stake)  # Model EV

    # Determine bet direction
    bet_decision = "Over" if EV_model > EV_odds else "Under"

    # Compute Variance & Confidence Interval
    variance = probability * (1 - probability)
    N = 5000  # Monte Carlo sample size
    standard_error = np.sqrt(variance / N)
    ci_lower = max(0, probability - 1.96 * standard_error)
    ci_upper = min(1, probability + 1.96 * standard_error)

    return {
        "EV_Model": EV_model,
        "EV_Odds": EV_odds,
        "Bet Decision": bet_decision,
        "Variance": variance,
        "Confidence Interval": (ci_lower, ci_upper),
        "Probability": probability,
        "Fair Probability": fair_prob
    }


def filter_bets():
    filtered_bets = []
    missing_players = []

    for _, row in parlay_df.iterrows():
        probability = get_player_probability(row["Player"], row["Stat Type"], row["Prop Line"], "over")

        if probability is None:
            missing_players.append(row["Player"])
            continue  

        # Compute EV
        bet_info = evaluate_bet_ev(probability, row["Higher Payout"])

        # **Loosened Filters: Probability ≥ 0.15 & EV_Model ≥ -25**
        if probability >= 0.15 and bet_info["EV_Model"] > -25:
            filtered_bets.append({
                "Player": row["Player"],
                "Stat Type": row["Stat Type"],
                "Prop Line": row["Prop Line"],
                "Pick": bet_info["Bet Decision"],
                "Odds": row["Higher Payout"],
                "Probability": probability,
                "EV_Model": bet_info["EV_Model"],
                "EV_Odds": bet_info["EV_Odds"],
                "Variance": bet_info["Variance"],
                "Confidence Interval": bet_info["Confidence Interval"]
            })

    # **Ensure at Least 3 Bets Exist for Each Parlay Type**
    if len(filtered_bets) < 3:
        print(f"⚠️ Not enough filtered bets ({len(filtered_bets)}). Adding extra bets...")
        
        # Sort all bets by EV_Model to prioritize best bets
        all_bets_sorted = sorted(parlay_df.iterrows(), key=lambda row: evaluate_bet_ev(
            get_player_probability(row[1]["Player"], row[1]["Stat Type"], row[1]["Prop Line"], "over"), 
            row[1]["Higher Payout"]
        )["EV_Model"], reverse=True)

        # Add highest EV bets until we reach at least 3
        for i, (_, row) in enumerate(all_bets_sorted):
            if len(filtered_bets) >= 3:
                break  # Stop once we have 3 bets
            probability = get_player_probability(row["Player"], row["Stat Type"], row["Prop Line"], "over")
            if probability is not None:
                bet_info = evaluate_bet_ev(probability, row["Higher Payout"])
                filtered_bets.append({
                    "Player": row["Player"],
                    "Stat Type": row["Stat Type"],
                    "Prop Line": row["Prop Line"],
                    "Pick": bet_info["Bet Decision"],
                    "Odds": row["Higher Payout"],
                    "Probability": probability,
                    "EV_Model": bet_info["EV_Model"],
                    "EV_Odds": bet_info["EV_Odds"],
                    "Variance": bet_info["Variance"],
                    "Confidence Interval": bet_info["Confidence Interval"]
                })

    return filtered_bets



# **Ensures a Parlay Includes Players from at Least 2 Different Teams**
def is_valid_parlay(parlay_bets):
    teams = {df_selected[df_selected["PLAYER_NAME"] == bet["Player"]]["TEAM_ABBREVIATION"].values[0] for bet in parlay_bets}
    return len(teams) >= 2  # ✅ Ensures players come from at least 2 teams


# **Compute the Best Parlays with Neural Network Odds & Sorting**
def find_best_parlays():
    all_possible_bets = filter_bets()

    if not all_possible_bets:
        print("❌ No valid bets found, selecting closest alternatives...")
        return None

    best_parlays = {"2-leg": [], "3-leg": [], "4-leg": []}

    for r in range(2, 5):
        parlays = []
        for parlay_bets in itertools.combinations(all_possible_bets, r):
            if not is_valid_parlay(parlay_bets):
                continue  # Skip parlays that don't meet team diversity requirement

            parlay_ev_model = sum(bet["EV_Model"] for bet in parlay_bets)
            parlay_ev_odds = sum(bet["EV_Odds"] for bet in parlay_bets)
            parlay_odds = np.prod([bet["Odds"] for bet in parlay_bets])

            # Compute probability of the parlay occurring
            parlay_probability = np.prod([bet["Probability"] for bet in parlay_bets])
            
            # Compute Neural Network Odds (If probability is too small, set a high value)
            parlay_nn_odds = round(1 / max(parlay_probability, 0.0001), 2)

            # Compute variance of the parlay
            parlay_variance = np.sum([bet["Variance"] for bet in parlay_bets])

            # Compute confidence interval for the parlay
            standard_error = np.sqrt(parlay_variance / 5000)
            ci_lower = max(0, parlay_probability - 1.96 * standard_error)
            ci_upper = min(1, parlay_probability + 1.96 * standard_error)

            parlays.append({
                "Parlay Bets": parlay_bets,
                "Parlay EV Model": parlay_ev_model,
                "Parlay EV Odds": parlay_ev_odds,
                "Parlay Probability": parlay_probability,
                "Neural Network Odds": parlay_nn_odds,
                "Variance": parlay_variance,
                "Confidence Interval": (ci_lower, ci_upper),
                "Parlay Odds": parlay_odds
            })

        # **Sort and Keep Top 5 Parlays**
        best_parlays[f"{r}-leg"] = sorted(parlays, key=lambda x: x["Parlay EV Model"], reverse=True)[:5]

    # **Print Results**
    for k, parlays in best_parlays.items():
        print(f"\n🔥 Top 5 {k.capitalize()} Parlays 🔥")
        for i, parlay in enumerate(parlays):
            parlay_details = [(bet["Player"], bet["Stat Type"], bet["Pick"]) for bet in parlay["Parlay Bets"]]
            print(f"{i+1}. EV_Model: ${parlay['Parlay EV Model']:.2f} | EV_Odds: ${parlay['Parlay EV Odds']:.2f} | "
                  f"Probability: {parlay['Parlay Probability']:.2%} | Neural Network Odds: {parlay['Neural Network Odds']} | "
                  f"Odds: {parlay['Parlay Odds']:.2f} | Confidence Interval: {parlay['Confidence Interval']} | {parlay_details}")

    return best_parlays


# **Run Optimization**
top_parlays = find_best_parlays()
