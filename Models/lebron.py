#!/usr/bin/env python3
# bayes_negbin_player_props.py - ONLINE LEARNING VERSION
# --------------------------------------------------------------
import pandas as pd
import numpy as np
from pathlib import Path
import pymc as pm
import arviz as az
from datetime import timedelta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# --------------------------------------------------------------
# 1  CONFIG & DATA
# --------------------------------------------------------------
CSV_PLAYERS = Path("nba_players_game_logs_2018_25_clean.csv")
CSV_TEAMS   = Path("team_stats.csv")
OUT         = Path("lebron_preds_online.csv")

ROLLS   = [1, 3, 5, 10]
TARGETS = ["PTS", "REB", "AST"]

# Time weighting parameters
DECAY_RATE = 0.05  # Higher = more emphasis on recent games (0.01-0.1 typical range)

# Online learning parameters
RETRAIN_FREQUENCY = 3  # Retrain model every N games (1 = every game, 3 = every 3 games)
FAST_SAMPLING = True   # Use fewer samples for online updates

# --------------------------------------------------------------
# 1-A  LOAD DATASETS
# --------------------------------------------------------------
df = pd.read_csv(CSV_PLAYERS, parse_dates=["GAME_DATE"])
team_stats = pd.read_csv(CSV_TEAMS)

# Filter for LeBron James only - 2024-25 season
LEBRON_ID = 2544
season_2024_25_df = df[(df["PLAYER_ID"] == LEBRON_ID) & (df["SEASON_YEAR"] == "2024-25")].copy()

if season_2024_25_df.empty:
    print("No 2024-25 season data found for LeBron James")
    exit()

# Sort by game date to ensure chronological order
season_2024_25_df = season_2024_25_df.sort_values("GAME_DATE").reset_index(drop=True)

# Split into first 3/4 for initial training and last 1/4 for online testing
total_games = len(season_2024_25_df)
initial_train_size = int(total_games * 0.75)

print(f"Total games in 2024-25 season: {total_games}")
print(f"Initial training: first {initial_train_size} games")
print(f"Online testing: last {total_games - initial_train_size} games")
print(f"Retrain frequency: every {RETRAIN_FREQUENCY} games")

# --------------------------------------------------------------
# 2  HELPER FUNCTIONS FOR ONLINE LEARNING
# --------------------------------------------------------------

def prepare_features(data_df, opponent_stats, team_stat_cols, season_full_df, feature_cols):
    """Prepare features for a subset of data"""
    # Add basic features
    data_df["home_flag"] = (data_df["MATCHUP"].str.contains("vs")).astype(int)
    
    # Merge opponent stats
    data_df = data_df.merge(opponent_stats, on="OPP_TEAM", how="left")
    
    # Fill missing team stats
    for col in team_stat_cols:
        if col in data_df.columns:
            data_df[col] = data_df[col].fillna(data_df[col].mean())
    
    # Calculate rolling averages using full season data up to this point
    for target in TARGETS:
        for roll in ROLLS:
            col_name = f"{target.lower()}_roll_{roll}"
            # Get the indices in the full season dataframe
            indices = data_df.index
            data_df[col_name] = season_full_df.loc[indices, col_name]
    
    # Add rest days (simplified)
    data_df["rest_days"] = 1
    
    # Filter to feature columns
    feature_data = data_df[feature_cols].fillna(method='ffill').fillna(0).astype(np.float32).values
    
    return data_df, feature_data

def calculate_time_weights(n_games, decay_rate):
    """Calculate exponential time weights"""
    game_indices = np.arange(n_games)
    time_weights = np.exp(-decay_rate * (n_games - 1 - game_indices))
    time_weights = time_weights * n_games / time_weights.sum()
    return time_weights

def fit_model_fast(X_train_std, y_train, time_weights, target, expected_mean, n_basic, n_team_stats, n_rolling):
    """Fast model fitting for online updates"""
    
    sampling_params = {
        'draws': 150 if FAST_SAMPLING else 250,
        'tune': 200 if FAST_SAMPLING else 300,
        'chains': 2,
        'cores': 1,
        'target_accept': 0.80 if FAST_SAMPLING else 0.85,
        'max_treedepth': 6 if FAST_SAMPLING else 8
    }
    
    with pm.Model() as model:
        # Informative intercept
        intercept = pm.Normal("intercept", mu=np.log(expected_mean), sigma=0.3)
        
        # Feature coefficients
        beta_basic = pm.Normal("beta_basic", mu=0, sigma=0.3, shape=n_basic)
        beta_team = pm.Normal("beta_team", mu=0, sigma=0.5, shape=n_team_stats)
        beta_rolling = pm.Normal("beta_rolling", mu=0, sigma=0.8, shape=n_rolling)
        
        # Combine all betas
        beta = pm.math.concatenate([beta_basic, beta_team, beta_rolling])
        
        # Dispersion parameter
        phi = pm.HalfNormal("phi", sigma=0.8)

        # Linear predictor
        linear_pred = intercept + pm.math.dot(X_train_std, beta)
        mu = pm.math.exp(pm.math.clip(linear_pred, -10, 10))
        
        # Time-weighted likelihood
        likelihood_vals = pm.logp(pm.NegativeBinomial.dist(mu=mu, alpha=phi), y_train)
        weighted_likelihood = pm.math.sum(time_weights * likelihood_vals)
        pm.Potential("time_weighted_likelihood", weighted_likelihood)

        # Sample
        trace = pm.sample(
            **sampling_params,
            return_inferencedata=True,
            progressbar=False,
            init='adapt_diag'
        )
    
    return model, trace

# --------------------------------------------------------------
# 3  FEATURE ENGINEERING SETUP
# --------------------------------------------------------------

# Merge opponent team stats setup
opponent_stats = team_stats[["TEAM_ABBREVIATION", "OFF_RATING", "DEF_RATING", "NET_RATING", 
                           "PACE", "AST_RATIO", "REB_PCT"]].copy()
opponent_stats.columns = ["OPP_TEAM", "OPP_OFF_RATING", "OPP_DEF_RATING", "OPP_NET_RATING", 
                         "OPP_PACE", "OPP_AST_RATIO", "OPP_REB_PCT"]

team_stat_cols = ["OPP_OFF_RATING", "OPP_DEF_RATING", "OPP_NET_RATING", "OPP_PACE", 
                  "OPP_AST_RATIO", "OPP_REB_PCT"]

# Pre-calculate ALL rolling averages for the full season
for target in TARGETS:
    for roll in ROLLS:
        col_name = f"{target.lower()}_roll_{roll}"
        season_2024_25_df[col_name] = season_2024_25_df[target].rolling(window=roll, min_periods=1).mean()

# Define feature columns
feature_cols = (
    ["home_flag", "rest_days"] +
    [col for col in team_stat_cols] +
    [f"{target.lower()}_roll_{roll}" for target in TARGETS for roll in ROLLS]
)

print(f"Using {len(feature_cols)} features for online learning:")
print(f"  Basic: home_flag, rest_days")
print(f"  Team stats: {len(team_stat_cols)} opponent features")  
print(f"  Rolling: {len(TARGETS) * len(ROLLS)} rolling averages")

# --------------------------------------------------------------
# 4  ONLINE LEARNING MAIN LOOP
# --------------------------------------------------------------

if __name__ == '__main__':
    all_predictions = []
    
    for target in TARGETS:
        print(f"\n{'='*60}")
        print(f"ONLINE LEARNING FOR {target}")
        print(f"{'='*60}")
        
        target_predictions = []
        
        # Target-specific parameters
        target_means = {"PTS": 25, "REB": 7, "AST": 7}
        expected_mean = target_means.get(target, 15)
        
        # Feature breakdown
        n_basic = 2
        n_team_stats = len(team_stat_cols)
        n_rolling = len(TARGETS) * len(ROLLS)
        
        # Initial training data
        current_train_end = initial_train_size
        
        # Iterate through each test game
        for test_game_idx in range(initial_train_size, total_games):
            
            print(f"\nPredicting game {test_game_idx + 1}/{total_games} (Test game {test_game_idx - initial_train_size + 1})")
            
            # Current training data (all games up to test game)
            current_train_df = season_2024_25_df.iloc[:current_train_end].copy()
            
            # Current test game (single game)
            current_test_df = season_2024_25_df.iloc[[test_game_idx]].copy()
            
            # Prepare features
            current_train_df, X_train = prepare_features(
                current_train_df, opponent_stats, team_stat_cols, 
                season_2024_25_df, feature_cols
            )
            
            current_test_df, X_test = prepare_features(
                current_test_df, opponent_stats, team_stat_cols,
                season_2024_25_df, feature_cols
            )
            
            # Standardize features
            means = np.mean(X_train, axis=0)
            stds = np.std(X_train, axis=0) + 1e-8
            X_train_std = (X_train - means) / stds
            X_test_std = (X_test - means) / stds
            
            # Time weights for current training data
            time_weights = calculate_time_weights(len(current_train_df), DECAY_RATE)
            
            # Training target
            y_train = current_train_df[target].values.astype(np.int32)
            
            # Decide whether to retrain model
            should_retrain = (
                test_game_idx == initial_train_size or  # First test game
                (test_game_idx - initial_train_size) % RETRAIN_FREQUENCY == 0  # Every N games
            )
            
            if should_retrain:
                print(f"  🔄 Retraining model on {len(current_train_df)} games...")
                
                # Fit model
                model, trace = fit_model_fast(
                    X_train_std, y_train, time_weights, target, expected_mean,
                    n_basic, n_team_stats, n_rolling
                )
                
                # Store model for next predictions (if not retraining every game)
                current_model = model
                current_trace = trace
                current_means = means
                current_stds = stds
                
            else:
                print(f"  ⚡ Using cached model (last retrained {(test_game_idx - initial_train_size) % RETRAIN_FREQUENCY} games ago)")
                # Use previously fitted model but update standardization
                model = current_model
                trace = current_trace
                # Update test standardization with new means/stds
                X_test_std = (X_test - current_means) / current_stds
            
            # Make prediction for this single game
            with model:
                # Use the updated test features
                intercept_samples = trace.posterior['intercept'].values.flatten()
                beta_basic_samples = trace.posterior['beta_basic'].values.reshape(-1, n_basic)
                beta_team_samples = trace.posterior['beta_team'].values.reshape(-1, n_team_stats)
                beta_rolling_samples = trace.posterior['beta_rolling'].values.reshape(-1, n_rolling)
                phi_samples = trace.posterior['phi'].values.flatten()
                
                # Combine betas
                beta_samples = np.concatenate([beta_basic_samples, beta_team_samples, beta_rolling_samples], axis=1)
                
                # Predict
                n_samples = len(intercept_samples)
                predictions = []
                
                for i in range(n_samples):
                    linear_pred = intercept_samples[i] + np.dot(X_test_std[0], beta_samples[i])
                    mu_pred = np.exp(np.clip(linear_pred, -10, 10))
                    
                    # Sample from negative binomial
                    pred_sample = np.random.negative_binomial(phi_samples[i], phi_samples[i] / (mu_pred + phi_samples[i]))
                    predictions.append(pred_sample)
                
                predictions = np.array(predictions)
            
            # Store prediction
            actual_value = current_test_df[target].iloc[0]
            game_date = current_test_df["GAME_DATE"].iloc[0]
            
            pred_mean = np.mean(predictions)
            pred_lo = np.percentile(predictions, 10)
            pred_hi = np.percentile(predictions, 90)
            
            target_predictions.append({
                'GAME_DATE': game_date,
                'game_num': test_game_idx + 1,
                f'{target}': actual_value,
                f'{target}_mean': pred_mean,
                f'{target}_lo': pred_lo,
                f'{target}_hi': pred_hi,
                'training_games': len(current_train_df)
            })
            
            # Print prediction vs actual
            error = abs(actual_value - pred_mean)
            print(f"  📊 Predicted: {pred_mean:.1f} [{pred_lo:.0f}-{pred_hi:.0f}] | Actual: {actual_value} | Error: {error:.1f}")
            
            # Update training set to include this game for next iteration
            current_train_end = test_game_idx + 1
        
        # Convert to DataFrame
        target_df = pd.DataFrame(target_predictions)
        all_predictions.append(target_df)
        
        # Performance summary
        actual = target_df[target].values
        predicted = target_df[f'{target}_mean'].values
        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(np.mean((actual - predicted)**2))
        
        print(f"\n{target} ONLINE PERFORMANCE:")
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  Coverage: {np.mean((actual >= target_df[f'{target}_lo']) & (actual <= target_df[f'{target}_hi']))*100:.1f}%")

    # --------------------------------------------------------------
    # 5  COMBINE RESULTS AND SAVE
    # --------------------------------------------------------------
    
    # Merge all predictions
    final_preds = all_predictions[0]
    for pred_df in all_predictions[1:]:
        final_preds = final_preds.merge(
            pred_df, on=['GAME_DATE', 'game_num', 'training_games'], how='outer'
        )
    
    final_preds.to_csv(OUT, index=False)
    print(f"\n📁 Online predictions saved to {OUT}")

    # --------------------------------------------------------------
    # 6  VISUALIZATION
    # --------------------------------------------------------------
    
    fig, axes = plt.subplots(len(TARGETS), 1, figsize=(15, 4*len(TARGETS)))
    if len(TARGETS) == 1:
        axes = [axes]

    for i, target in enumerate(TARGETS):
        ax = axes[i]
        
        # Plot actual vs predicted
        ax.plot(final_preds["game_num"], final_preds[target], 'o-', 
               label="Actual", alpha=0.9, linewidth=2.5, markersize=7, color='darkblue')
        ax.plot(final_preds["game_num"], final_preds[f"{target}_mean"], 's-', 
               label="Predicted", alpha=0.9, linewidth=2.5, markersize=7, color='darkred')
        ax.fill_between(
            final_preds["game_num"],
            final_preds[f"{target}_lo"],
            final_preds[f"{target}_hi"],
            alpha=0.25,
            label="80% Prediction Interval",
            color='gray'
        )
        
        ax.set_title(f"LeBron {target} - Online Learning (Retrain every {RETRAIN_FREQUENCY} games)", 
                    fontsize=16, pad=20)
        ax.set_xlabel("Game Number", fontsize=14)
        ax.set_ylabel(target, fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------------
    # 7  FINAL PERFORMANCE SUMMARY
    # --------------------------------------------------------------
    
    print("\n" + "="*80)
    print("ONLINE LEARNING PERFORMANCE SUMMARY")
    print("="*80)
    
    for target in TARGETS:
        if f"{target}_mean" in final_preds.columns:
            actual = final_preds[target].values
            predicted = final_preds[f"{target}_mean"].values
            
            mae = np.mean(np.abs(actual - predicted))
            rmse = np.sqrt(np.mean((actual - predicted)**2))
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100
            
            # Coverage
            lo = final_preds[f"{target}_lo"].values
            hi = final_preds[f"{target}_hi"].values
            coverage = np.mean((actual >= lo) & (actual <= hi)) * 100
            
            # Correlation
            correlation = np.corrcoef(actual, predicted)[0, 1]
            
            print(f"{target:>3} | MAE: {mae:5.2f} | RMSE: {rmse:5.2f} | MAPE: {mape:5.1f}% | Coverage: {coverage:5.1f}% | Corr: {correlation:.3f}")

    print("="*80)
    print("ONLINE LEARNING ADVANTAGES:")
    print(f"• Model updates every {RETRAIN_FREQUENCY} games with new information")
    print("• Time-weighted learning emphasizes recent performance")
    print("• Captures evolving player form and opponent adjustments")
    print("• Rolling features automatically update with each game")
    print("• More realistic evaluation of real-world prediction accuracy")
    print("="*80)
