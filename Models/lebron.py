#!/usr/bin/env python3
# bayes_negbin_player_props.py
# --------------------------------------------------------------
import pandas as pd
import numpy as np
from pathlib import Path
import pymc as pm
import arviz as az
from datetime import timedelta
import matplotlib.pyplot as plt
from pytensor.tensor import reshape  # Add this import

# --------------------------------------------------------------
# 1  CONFIG & DATA
# --------------------------------------------------------------
CSV_PLAYERS = Path("nba_players_game_logs_2018_25_clean.csv")
CSV_TEAMS   = Path("team_stats.csv")
OUT         = Path("lebron_preds.csv")

ROLLS   = [5, 10]
TARGETS = ["PTS", "REB", "AST"]

# --------------------------------------------------------------
# 1-A  LOAD DATASETS
# --------------------------------------------------------------
df = pd.read_csv(CSV_PLAYERS, parse_dates=["GAME_DATE"])
t_df = pd.read_csv(CSV_TEAMS)

df.sort_values(["PLAYER_ID", "GAME_DATE"], inplace=True)

# Filter for LeBron James (replace with LeBron's actual PLAYER_ID)
LEBRON_ID = 2544  # Example PLAYER_ID for LeBron James
df = df[df["PLAYER_ID"] == LEBRON_ID].reset_index(drop=True)  # Reset index after filtering

# Filter for 2024-25 season only
season_2024_25_df = df[df["SEASON_YEAR"] == "2024-25"].copy()

if season_2024_25_df.empty:
    print("No 2024-25 season data found for LeBron James")
    exit()

# Sort by game date to ensure chronological order
season_2024_25_df = season_2024_25_df.sort_values("GAME_DATE").reset_index(drop=True)

# Split into first 3/4 for training and last 1/4 for testing
total_games = len(season_2024_25_df)
train_size = int(total_games * 0.75)

print(f"Total games in 2024-25 season: {total_games}")
print(f"Training on first {train_size} games")
print(f"Testing on last {total_games - train_size} games")

train_df = season_2024_25_df.iloc[:train_size].copy()
test_df = season_2024_25_df.iloc[train_size:].copy()

# --------------------------------------------------------------
# 2  FEATURE ENGINEERING
# --------------------------------------------------------------
# Add home_flag
df["home_flag"] = df["MATCHUP"].apply(lambda x: 1 if "vs" in x else 0)

# Add one-hot encoded opponent columns (using all available data to get all possible opponents)
opp_dummies = pd.get_dummies(df["OPP_TEAM"], prefix="opp")
df = pd.concat([df, opp_dummies], axis=1)

# Now apply the same opponent columns to our train/test splits
train_df["home_flag"] = train_df["MATCHUP"].apply(lambda x: 1 if "vs" in x else 0)
test_df["home_flag"] = test_df["MATCHUP"].apply(lambda x: 1 if "vs" in x else 0)

# Add opponent dummy columns to train and test sets
for col in opp_dummies.columns:
    train_df[col] = 0
    test_df[col] = 0

# Activate appropriate opponent features
for idx, row in train_df.iterrows():
    opponent = row["OPP_TEAM"]
    if f"opp_{opponent}" in opp_dummies.columns:
        train_df.loc[idx, f"opp_{opponent}"] = 1

for idx, row in test_df.iterrows():
    opponent = row["OPP_TEAM"]
    if f"opp_{opponent}" in opp_dummies.columns:
        test_df.loc[idx, f"opp_{opponent}"] = 1

# Merge team stats for the player's team
team_cols = ["OFF_RATING", "DEF_RATING", "NET_RATING", "PACE", "AST_RATIO", "REB_PCT"]
t_own = t_df.rename(columns={col: f"TEAM_{col}" for col in team_cols})
train_df = train_df.merge(t_own, on="TEAM_ABBREVIATION", how="left")
test_df = test_df.merge(t_own, on="TEAM_ABBREVIATION", how="left")

# Merge team stats for the opponent's team
t_opp = t_df.rename(columns={"TEAM_ABBREVIATION": "OPP_TEAM", **{col: f"OPP_{col}" for col in team_cols}})
train_df = train_df.merge(t_opp, on="OPP_TEAM", how="left")
test_df = test_df.merge(t_opp, on="OPP_TEAM", how="left")

# Create player_idx column (will be 0 for LeBron since he's the only player)
player2idx = {LEBRON_ID: 0}
train_df["player_idx"] = 0
test_df["player_idx"] = 0

# --------------------------------------------------------------
# 2-B  DESIGN-MATRIX FEATURE LIST
# --------------------------------------------------------------
feature_cols = (
    [f"{t.lower()}_roll_{w}" for t in TARGETS for w in ROLLS] +  # Rolling features
    ["rest_days", "is_b2b", "home_flag"] +                      # Additional features
    list(opp_dummies.columns) +                                 # One-hot opponent features
    [f"TEAM_{c}" for c in team_cols] +                         # Team stats
    [f"OPP_{c}"  for c in team_cols]                           # Opponent stats
)

# Filter feature_cols to only include columns that exist in both train and test sets
available_cols = set(train_df.columns) & set(test_df.columns)
feature_cols = [col for col in feature_cols if col in available_cols]

print(f"Using {len(feature_cols)} features for modeling")

# ---- Build numeric DataFrames, coerce non-numerics, fill NaNs ----
X_train_df = train_df[feature_cols].copy()
X_test_df  = test_df[feature_cols].copy()

X_train_df = X_train_df.apply(pd.to_numeric, errors="coerce")
X_test_df  = X_test_df.apply(pd.to_numeric, errors="coerce")

meds = X_train_df.median()
X_train_df = X_train_df.fillna(meds)
X_test_df  = X_test_df.fillna(meds)

X_train = X_train_df.to_numpy(dtype="float64")
X_test  = X_test_df.to_numpy(dtype="float64")

# ---- Standardize ----
if X_train.ndim == 1:
    X_train = X_train.reshape(-1, 1)

means = X_train.mean(axis=0)
stds  = X_train.std(axis=0) + 1e-6
X_train_std = (X_train - means) / stds
X_test_std  = (X_test - means) / stds

# --------------------------------------------------------------
# 3  DYNAMICALLY ACTIVATE opp_* FEATURES
# --------------------------------------------------------------
def activate_opp_features(df, opponent):
    """Activate only the relevant opp_* feature for the given opponent."""
    opp_cols = [col for col in df.columns if col.startswith("opp_")]
    df[opp_cols] = 0  # Deactivate all opp_* features
    if f"opp_{opponent}" in df.columns:
        df[f"opp_{opponent}"] = 1  # Activate the relevant opponent feature
    return df

# --------------------------------------------------------------
# 4  BAYESIAN NEGATIVE BINOMIAL MODEL
# --------------------------------------------------------------
if __name__ == '__main__':
    pred_frames = []

    for target in TARGETS:
        print(f"--- Fitting {target} ---")
        y_train = train_df[target].values.astype("int64")

        with pm.Model() as model:
            x_data = pm.Data("x_data", X_train_std)
            player_idx = pm.Data("player_idx", train_df["player_idx"].values)

            mu_a = pm.Normal("mu_a", mu=0, sigma=2)
            sigma_a = pm.HalfNormal("sigma_a", sigma=2)
            a = pm.Normal("a", mu=mu_a, sigma=sigma_a, shape=len(player2idx))

            beta = pm.Normal("beta", mu=0, sigma=1, shape=X_train_std.shape[1])
            phi = pm.HalfNormal("phi", sigma=2)

            # Debugging: Print shapes of key variables
            print(f"Shape of player_idx: {train_df['player_idx'].values.shape}")
            print(f"Shape of x_data: {X_train_std.shape}")
            print(f"Shape of y_train: {y_train.shape}")

            # Ensure theta matches y_train shape
            theta = pm.math.exp(a[player_idx] + pm.math.dot(x_data, beta))

            # Define the Negative Binomial distribution
            pm.NegativeBinomial("y", mu=theta, alpha=phi, observed=y_train)

            # Increase target_accept to reduce divergences
            trace = pm.sample(500, tune=500, target_accept=0.99, chains=4, cores=1)

            # Posterior predictive for TEST SET
            test_df_copy = test_df.copy()
            
            # Prepare test features
            X_test_final = test_df_copy[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(meds)
            X_test_std_final = (X_test_final.to_numpy(dtype="float64") - means) / stds
            
            # Update the model with test data
            pm.set_data({"x_data": X_test_std_final, "player_idx": test_df_copy["player_idx"].values}, model=model)

            # Debug shapes during posterior predictive sampling
            print(f"Shape of X_test_std_final: {X_test_std_final.shape}")
            print(f"Shape of player_idx (test): {test_df_copy['player_idx'].values.shape}")

            # Sample posterior predictive
            ppc = pm.sample_posterior_predictive(trace, var_names=["y"], random_seed=42)
            draws = ppc.posterior_predictive["y"]

            pred_df = test_df_copy[["GAME_ID", "PLAYER_ID", "GAME_DATE", target]].copy()
            pred_df[f"{target}_mean"] = draws.mean(axis=(0, 1))  # Average over chains and draws
            pred_df[f"{target}_lo"] = np.percentile(draws, 5, axis=(0, 1))
            pred_df[f"{target}_hi"] = np.percentile(draws, 95, axis=(0, 1))
            pred_frames.append(pred_df)

    # --------------------------------------------------------------
    # 5  MERGE & SAVE
    # --------------------------------------------------------------
    preds = pred_frames[0]
    for pf in pred_frames[1:]:
        preds = preds.merge(pf, on=["GAME_ID", "PLAYER_ID", "GAME_DATE"])

    preds.to_csv(OUT, index=False)
    print(f"\nPosterior predictive summaries saved to {OUT}")

    # --------------------------------------------------------------
    # 6  PLOT RESULTS
    # --------------------------------------------------------------
    for target in TARGETS:
        plt.figure(figsize=(10, 6))
        plt.plot(preds["GAME_DATE"], preds[target], label="Actual", marker="o")
        plt.plot(preds["GAME_DATE"], preds[f"{target}_mean"], label="Predicted", marker="x")
        plt.fill_between(
            preds["GAME_DATE"],
            preds[f"{target}_lo"],
            preds[f"{target}_hi"],
            color="gray",
            alpha=0.3,
            label="95% CI"
        )
        plt.title(f"LeBron James - {target} Predictions")
        plt.xlabel("Game Date")
        plt.ylabel(target)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
