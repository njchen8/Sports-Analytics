import pandas as pd
import numpy as np
import itertools

import pandas as pd

# Load predicted player props
predictions_file_path = "predicted_player_props.csv"
pred_df = pd.read_csv(predictions_file_path)

# Ensure numeric types
pred_df["Prop Line"] = pd.to_numeric(pred_df["Prop Line"], errors="coerce")
pred_df["Predicted Value"] = pd.to_numeric(pred_df["Predicted Value"], errors="coerce")
pred_df["Variance"] = pd.to_numeric(pred_df["Variance"], errors="coerce")

# Compute Expected Value (EV)
pred_df["Edge"] = pred_df["Predicted Value"] - pred_df["Prop Line"]
pred_df["Bet Odds"] = np.where(pred_df["Edge"] > 0, 1.9, 2.1)  # Example odds structure
pred_df["Implied Probability"] = 1 / pred_df["Bet Odds"]
pred_df["EV"] = pred_df["Edge"] * pred_df["Bet Odds"] - (1 - pred_df["Implied Probability"])

# Sort first by team (Opponent), then by best single-leg bet (highest EV)
top_single_bets = pred_df.sort_values(by=["Opponent", "EV"], ascending=[True, False])

# Save the results
top_single_bets.to_csv("top_single_bets_sorted.csv", index=False)

print("\n✅ Top single bets sorted by team and EV saved to: top_single_bets_sorted.csv")



# Ensure numeric types
pred_df["Prop Line"] = pd.to_numeric(pred_df["Prop Line"], errors="coerce")
pred_df["Predicted Value"] = pd.to_numeric(pred_df["Predicted Value"], errors="coerce")
pred_df["Variance"] = pd.to_numeric(pred_df["Variance"], errors="coerce")

# Define betting parameters
kelly_fraction = 0.5  # Fractional Kelly to reduce risk
bankroll = 1000  # Example starting bankroll

# Compute Expected Value (EV) for each bet
pred_df["Edge"] = pred_df["Predicted Value"] - pred_df["Prop Line"]
pred_df["Bet Odds"] = np.where(pred_df["Edge"] > 0, 1.9, 2.1)  # Example odds structure
pred_df["Implied Probability"] = 1 / pred_df["Bet Odds"]

# Calculate expected return
pred_df["EV"] = pred_df["Edge"] * pred_df["Bet Odds"] - (1 - pred_df["Implied Probability"])

# Compute Kelly Criterion for bet sizing
pred_df["Kelly Fraction"] = (pred_df["Edge"] / (pred_df["Variance"] + 1e-6))
pred_df["Optimal Bet"] = bankroll * kelly_fraction * pred_df["Kelly Fraction"]

# Ensure bet sizes are reasonable (e.g., not more than 10% of bankroll)
pred_df["Optimal Bet"] = pred_df["Optimal Bet"].clip(0, bankroll * 0.1)

# Select the top single bets sorted by matchup and highest EV
top_single_bets = pred_df.sort_values(by=["Opponent", "EV"], ascending=[True, False])

# Function to generate the best 2-leg parlays
def generate_best_2_leg_parlays(bets_df, top_n=20):
    parlays = []
    
    # Unique teams in the dataset
    teams = bets_df["Opponent"].unique()

    # Filter to top 50 bets to improve speed
    bets_df = bets_df.sort_values(by="EV", ascending=False).head(50)

    for parlay in itertools.combinations(bets_df.iterrows(), 2):  # Only 2-leg parlays
        parlay_bets = [p[1] for p in parlay]  # Extract bet rows
        
        # Ensure at least 2 different teams
        teams_in_parlay = {bet["Opponent"] for bet in parlay_bets}
        if len(teams_in_parlay) >= 2:
            total_odds = np.prod([bet["Bet Odds"] for bet in parlay_bets])
            parlay_ev = np.mean([bet["EV"] for bet in parlay_bets])  # Average EV across legs
            parlay_var = np.mean([bet["Variance"] for bet in parlay_bets])  # Average variance
            
            parlays.append({
                "Parlay Legs": [(bet["Player"], bet["Stat Type"]) for bet in parlay_bets],
                "Total Odds": round(total_odds, 2),
                "Parlay EV": round(parlay_ev, 4),
                "Variance": round(parlay_var, 4)
            })

    # Convert to DataFrame and select top N parlays
    parlays_df = pd.DataFrame(parlays).sort_values(by="Parlay EV", ascending=False).head(top_n)
    
    return parlays_df

# Generate top 20 2-leg parlays
top_2_leg_parlays = generate_best_2_leg_parlays(pred_df, top_n=20)

# Save results
top_single_bets.to_csv("top_single_bets.csv", index=False)
top_2_leg_parlays.to_csv("top_2_leg_parlays.csv", index=False)


print("\n✅ Top single bets saved to: top_single_bets.csv")
print("✅ Top 2-leg parlays saved to: top_2_leg_parlays.csv")
