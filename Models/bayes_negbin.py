#!/usr/bin/env python3
# bayes_negbin_player_props.py
# --------------------------------------------------------------
import pandas as pd
import numpy as np
from pathlib import Path
import pymc as pm
import arviz as az
from datetime import timedelta

# --------------------------------------------------------------
# 1  CONFIG & DATA
# --------------------------------------------------------------
CSV_PLAYERS = Path("nba_players_game_logs_2018_25_clean.csv")
CSV_TEAMS   = Path("team_stats.csv")
OUT         = Path("bayes_props_preds.csv")

ROLLS   = [5, 10]
TARGETS = ["PTS", "REB", "AST"]

# --------------------------------------------------------------
# 1-A  LOAD DATASETS
# --------------------------------------------------------------
df = pd.read_csv(CSV_PLAYERS, parse_dates=["GAME_DATE"])
t_df = pd.read_csv(CSV_TEAMS)

df.sort_values(["PLAYER_ID", "GAME_DATE"], inplace=True)

# Define test_mask to split data into training and testing sets
test_mask = df["SEASON_YEAR"] == "2024-25"  # Adjust column name if necessary

# --------------------------------------------------------------
# 2  FEATURE ENGINEERING
# --------------------------------------------------------------
# Add home_flag
df["home_flag"] = df["MATCHUP"].apply(lambda x: 1 if "vs" in x else 0)

# Add one-hot encoded opponent columns
opp_dummies = pd.get_dummies(df["OPP_TEAM"], prefix="opp")
df = pd.concat([df, opp_dummies], axis=1)

# Merge team stats for the player's team
team_cols = ["OFF_RATING", "DEF_RATING", "NET_RATING", "PACE", "AST_RATIO", "REB_PCT"]
t_own = t_df.rename(columns={col: f"TEAM_{col}" for col in team_cols})
df = df.merge(t_own, on="TEAM_ABBREVIATION", how="left")

# Merge team stats for the opponent's team
t_opp = t_df.rename(columns={"TEAM_ABBREVIATION": "OPP_TEAM", **{col: f"OPP_{col}" for col in team_cols}})
df = df.merge(t_opp, on="OPP_TEAM", how="left")

# Create player_idx column
player2idx = {pid: i for i, pid in enumerate(df["PLAYER_ID"].unique())}
df["player_idx"] = df["PLAYER_ID"].map(player2idx)

# Filter training data to only include the past year
most_recent_date = df["GAME_DATE"].max()
one_year_ago = most_recent_date - timedelta(days=365)

# For each player, include only games from the past year or all available games if less than a year
train_df = df.loc[~test_mask].groupby("PLAYER_ID", group_keys=False).apply(
    lambda group: group[group["GAME_DATE"] >= one_year_ago] if group["GAME_DATE"].min() < one_year_ago else group
).reset_index(drop=True)

test_df = df.loc[test_mask].reset_index(drop=True)

# Ensure all necessary columns are in train_df and test_df
train_df = train_df[list(df.columns)]
test_df = test_df[list(df.columns)]

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
            x_data = pm.MutableData("x_data", X_train_std)
            player_idx = pm.MutableData("player_idx", train_df["player_idx"].values)

            mu_a = pm.Normal("mu_a", mu=0, sigma=2)
            sigma_a = pm.HalfNormal("sigma_a", sigma=2)
            a = pm.Normal("a", mu=mu_a, sigma=sigma_a, shape=len(player2idx))

            beta = pm.Normal("beta", mu=0, sigma=1, shape=X_train_std.shape[1])
            phi = pm.HalfNormal("phi", sigma=2)

            theta = pm.math.exp(a[player_idx] + pm.math.dot(x_data, beta))
            pm.NegativeBinomial("y", mu=theta, alpha=phi, observed=y_train)

            # Reduce chains and iterations for debugging
            trace = pm.sample(500, tune=500, target_accept=0.9, chains=2, cores=1)

            # Posterior predictive for TEST SET
            for opponent in test_df["OPP_TEAM"].unique():
                test_df = activate_opp_features(test_df, opponent)
                X_test_std = (test_df[feature_cols].to_numpy(dtype="float64") - means) / stds

                pm.set_data({"x_data": X_test_std, "player_idx": test_df["player_idx"].values}, model=model)
                ppc = pm.sample_posterior_predictive(trace, var_names=["y"], random_seed=42)
                draws = ppc["y"]

                pred_df = test_df[["GAME_ID", "PLAYER_ID", "GAME_DATE"]].copy()
                pred_df[f"{target}_mean"] = draws.mean(axis=0)
                pred_df[f"{target}_lo"] = np.percentile(draws, 5, axis=0)
                pred_df[f"{target}_hi"] = np.percentile(draws, 95, axis=0)
                pred_frames.append(pred_df)

    # --------------------------------------------------------------
    # 5  MERGE & SAVE
    # --------------------------------------------------------------
    preds = pred_frames[0]
    for pf in pred_frames[1:]:
        preds = preds.merge(pf, on=["GAME_ID", "PLAYER_ID", "GAME_DATE"])

    preds.to_csv(OUT, index=False)
    print(f"\nPosterior predictive summaries saved to {OUT}")
