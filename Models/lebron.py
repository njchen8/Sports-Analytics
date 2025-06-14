#!/usr/bin/env python3
# bayes_negbin_player_props.py - ONLINE LEARNING VERSION WITH COMPREHENSIVE VALIDATION
# --------------------------------------------------------------
import pandas as pd
import numpy as np
from pathlib import Path
import pymc as pm
import arviz as az
from datetime import timedelta
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
warnings.filterwarnings('ignore')

# --------------------------------------------------------------
# 1  CONFIG & DATA
# --------------------------------------------------------------
CSV_PLAYERS = Path("nba_players_game_logs_2018_25_clean.csv")
CSV_TEAMS   = Path("team_stats.csv")
OUT         = Path("lebron_preds_online.csv")
BASELINE_OUT = Path("lebron_baseline_comparison.csv")

ROLLS   = [1, 3, 5, 10]
TARGETS = ["PTS", "REB", "AST"]

# Time weighting parameters
DECAY_RATE = 0.05  # Higher = more emphasis on recent games (0.01-0.1 typical range)

# Online learning parameters
RETRAIN_FREQUENCY = 1  # Retrain model every N games (1 = every game, 3 = every 3 games)
FAST_SAMPLING = True   # Use fewer samples for online updates

# --------------------------------------------------------------
# 1-A  LOAD DATASETS
# --------------------------------------------------------------
df = pd.read_csv(CSV_PLAYERS, parse_dates=["GAME_DATE"])
team_stats = pd.read_csv(CSV_TEAMS)

# Filter for LeBron James only - 2024-25 season
LEBRON_ID = 1628398  # Updated to correct LeBron ID
season_2024_25_df = df[(df["PLAYER_ID"] == LEBRON_ID) & (df["SEASON_YEAR"] == "2024-25")].copy()

if season_2024_25_df.empty:
    print("No 2024-25 season data found for LeBron James")
    # Try to find LeBron by name
    lebron_df = df[df["PLAYER_NAME"].str.contains("LeBron", case=False, na=False)]
    if not lebron_df.empty:
        LEBRON_ID = lebron_df["PLAYER_ID"].iloc[0]
        print(f"Found LeBron James with Player ID: {LEBRON_ID}")
        season_2024_25_df = df[(df["PLAYER_ID"] == LEBRON_ID) & (df["SEASON_YEAR"] == "2024-25")].copy()
    
    if season_2024_25_df.empty:
        print("Still no data found. Available players:")
        print(df[["PLAYER_ID", "PLAYER_NAME"]].drop_duplicates().head(20))
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
# 2  BASELINE MODELS FOR ONLINE COMPARISON
# --------------------------------------------------------------
def create_online_baseline_predictions(train_history, target):
    """Create baseline predictions for current game based on training history"""
    baselines = {}
    
    if len(train_history) == 0:
        # Fallback if no training history
        return {f'MA_{w}': 15 for w in [3, 5, 10]}, {}
    
    # 1. Simple Moving Averages
    for window in [3, 5, 10, 20]:
        if len(train_history) >= window:
            baseline_pred = train_history[target].tail(window).mean()
        else:
            baseline_pred = train_history[target].mean()
        baselines[f'MA_{window}'] = baseline_pred
    
    # 2. Exponentially Weighted Moving Average
    alpha = 0.3
    if len(train_history) >= 2:
        ewm_pred = train_history[target].ewm(alpha=alpha).mean().iloc[-1]
    else:
        ewm_pred = train_history[target].mean()
    baselines['EWMA'] = ewm_pred
    
    # 3. Linear Trend (if enough data)
    if len(train_history) >= 5:
        recent_games = train_history.tail(15)
        x = np.arange(len(recent_games))
        y = recent_games[target].values
        try:
            slope, intercept = np.polyfit(x, y, 1)
            trend_pred = slope * len(recent_games) + intercept
            baselines['Linear_Trend'] = trend_pred
        except:
            baselines['Linear_Trend'] = train_history[target].mean()
    
    return baselines

def calculate_comprehensive_metrics(actual, predicted, prediction_intervals=None):
    """Calculate comprehensive evaluation metrics"""
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Handle edge cases
    if len(actual) == 0 or len(predicted) == 0:
        return {}
    
    metrics = {}
    
    # Basic accuracy metrics
    metrics['MAE'] = mean_absolute_error(actual, predicted)
    metrics['RMSE'] = np.sqrt(mean_squared_error(actual, predicted))
    metrics['MAPE'] = np.mean(np.abs((actual - predicted) / np.maximum(actual, 1e-8))) * 100
    
    # R-squared (handle edge case where variance is 0)
    if np.var(actual) > 1e-8:
        metrics['R2'] = r2_score(actual, predicted)
        metrics['Explained_Variance'] = 1 - np.var(actual - predicted) / np.var(actual)
    else:
        metrics['R2'] = 0
        metrics['Explained_Variance'] = 0
    
    # Correlation metrics (handle edge case where std is 0)
    if len(actual) > 1 and np.std(actual) > 1e-8 and np.std(predicted) > 1e-8:
        pearson_r, pearson_p = stats.pearsonr(actual, predicted)
        spearman_r, spearman_p = stats.spearmanr(actual, predicted)
        metrics['Pearson_R'] = pearson_r
        metrics['Pearson_P'] = pearson_p
        metrics['Spearman_R'] = spearman_r
        metrics['Spearman_P'] = spearman_p
    else:
        metrics['Pearson_R'] = 0
        metrics['Pearson_P'] = 1
        metrics['Spearman_R'] = 0
        metrics['Spearman_P'] = 1
    
    # Directional accuracy (for betting/DFS)
    if len(actual) > 1:
        actual_changes = np.diff(actual)
        pred_changes = np.diff(predicted)
        # Handle case where changes are all zero
        if np.sum(np.abs(actual_changes)) > 1e-8:
            directional_accuracy = np.mean(np.sign(actual_changes) == np.sign(pred_changes))
            metrics['Directional_Accuracy'] = directional_accuracy
        else:
            metrics['Directional_Accuracy'] = 0.5  # Random guess when no movement
    
    # Bias metrics
    metrics['Mean_Bias'] = np.mean(predicted - actual)
    metrics['Median_Bias'] = np.median(predicted - actual)
    
    # Prediction interval metrics (if available) - FIX HERE
    if prediction_intervals is not None:
        lower, upper = prediction_intervals
        # Convert to numpy arrays if they're lists
        lower = np.array(lower)
        upper = np.array(upper)
        
        coverage = np.mean((actual >= lower) & (actual <= upper))
        avg_width = np.mean(upper - lower)
        
        metrics['Coverage'] = coverage * 100
        metrics['Avg_Interval_Width'] = avg_width
        
        # Sharpness vs Coverage trade-off
        if avg_width > 0:
            metrics['Coverage_Width_Ratio'] = coverage / (avg_width / np.mean(actual))
        else:
            metrics['Coverage_Width_Ratio'] = 0
    
    # Quantile-based metrics
    residuals = actual - predicted
    metrics['Residual_Q25'] = np.percentile(residuals, 25)
    metrics['Residual_Q75'] = np.percentile(residuals, 75)
    metrics['Residual_IQR'] = metrics['Residual_Q75'] - metrics['Residual_Q25']
    
    return metrics

# --------------------------------------------------------------
# 3  HELPER FUNCTIONS FOR ONLINE LEARNING
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
# 4  FEATURE ENGINEERING SETUP
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
# 5  ONLINE LEARNING MAIN LOOP WITH BASELINES
# --------------------------------------------------------------

if __name__ == '__main__':
    all_predictions = []
    all_baseline_predictions = {}
    comparison_results = []
    
    for target in TARGETS:
        print(f"\n{'='*60}")
        print(f"ONLINE LEARNING FOR {target} WITH BASELINE COMPARISON")
        print(f"{'='*60}")
        
        target_predictions = []
        baseline_predictions = {model: [] for model in ['MA_3', 'MA_5', 'MA_10', 'EWMA', 'Linear_Trend']}
        bayesian_predictions = []
        actual_values = []
        
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
            
            # Get actual value for this test game
            actual_value = current_test_df[target].iloc[0]
            actual_values.append(actual_value)
            
            # CREATE BASELINE PREDICTIONS
            baselines = create_online_baseline_predictions(current_train_df, target)
            
            for model_name, pred_value in baselines.items():
                if model_name in baseline_predictions:
                    baseline_predictions[model_name].append(pred_value)
            
            # BAYESIAN MODEL PREDICTION
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
                print(f"  🔄 Retraining Bayesian model on {len(current_train_df)} games...")
                
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
                print(f"  ⚡ Using cached Bayesian model (last retrained {(test_game_idx - initial_train_size) % RETRAIN_FREQUENCY} games ago)")
                # Use previously fitted model but update standardization
                model = current_model
                trace = current_trace
                # Update test standardization with new means/stds
                X_test_std = (X_test - current_means) / current_stds
            
            # Make Bayesian prediction for this single game
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
            
            # Store Bayesian prediction
            game_date = current_test_df["GAME_DATE"].iloc[0]
            
            pred_mean = np.mean(predictions)
            pred_lo = np.percentile(predictions, 10)
            pred_hi = np.percentile(predictions, 90)
            
            bayesian_predictions.append(pred_mean)
            
            target_predictions.append({
                'GAME_DATE': game_date,
                'game_num': test_game_idx + 1,
                f'{target}': actual_value,
                f'{target}_mean': pred_mean,
                f'{target}_lo': pred_lo,
                f'{target}_hi': pred_hi,
                'training_games': len(current_train_df)
            })
            
            # Print prediction vs actual for all models
            error = abs(actual_value - pred_mean)
            print(f"  📊 Actual: {actual_value}")
            print(f"     Bayesian: {pred_mean:.1f} [{pred_lo:.0f}-{pred_hi:.0f}]")
            for model_name in ['MA_5', 'EWMA']:
                if model_name in baselines:
                    print(f"     {model_name}: {baselines[model_name]:.1f}")
            
            # Update training set to include this game for next iteration
            current_train_end = test_game_idx + 1
        
        # Convert to DataFrame
        target_df = pd.DataFrame(target_predictions)
        all_predictions.append(target_df)
        
        # Store baseline predictions for this target
        all_baseline_predictions[target] = baseline_predictions
        
        # --------------------------------------------------------------
        # 6  COMPREHENSIVE MODEL COMPARISON FOR THIS TARGET
        # --------------------------------------------------------------
        print(f"\n--- {target} MODEL COMPARISON ---")
        
        # Calculate metrics for all models
        all_model_predictions = dict(baseline_predictions)
        all_model_predictions['Bayesian'] = bayesian_predictions
        
        target_comparison = []
        for model_name, predictions in all_model_predictions.items():
            if len(predictions) == len(actual_values):
                if model_name == 'Bayesian':
                    # Include prediction intervals for Bayesian - FIX HERE
                    intervals = (
                        [p[f'{target}_lo'] for p in target_predictions], 
                        [p[f'{target}_hi'] for p in target_predictions]
                    )
                    metrics = calculate_comprehensive_metrics(actual_values, predictions, intervals)
                else:
                    metrics = calculate_comprehensive_metrics(actual_values, predictions)
                
                metrics['Target'] = target
                metrics['Model'] = model_name
                target_comparison.append(metrics)
        
        # Convert to DataFrame and rank
        if target_comparison:
            comparison_df = pd.DataFrame(target_comparison)
            
            # Rank models
            for col in ['MAE', 'RMSE', 'MAPE']:
                if col in comparison_df.columns:
                    comparison_df[f'{col}_Rank'] = comparison_df[col].rank()
            for col in ['R2', 'Pearson_R']:
                if col in comparison_df.columns:
                    comparison_df[f'{col}_Rank'] = comparison_df[col].rank(ascending=False)
            
            # Average rank
            rank_cols = [col for col in comparison_df.columns if col.endswith('_Rank')]
            if rank_cols:
                comparison_df['Avg_Rank'] = comparison_df[rank_cols].mean(axis=1)
                
                print("\nModel Performance Ranking (lower average rank = better):")
                display_cols = ['MAE', 'RMSE', 'R2', 'Pearson_R', 'Avg_Rank']
                available_cols = [col for col in display_cols if col in comparison_df.columns]
                print(comparison_df.sort_values('Avg_Rank')[available_cols].round(3))
                
                # Check if Bayesian is best
                bayesian_row = comparison_df[comparison_df['Model'] == 'Bayesian']
                if not bayesian_row.empty:
                    bayesian_rank = bayesian_row['Avg_Rank'].iloc[0]
                    best_rank = comparison_df['Avg_Rank'].min()
                    
                    if bayesian_rank == best_rank:
                        print(f"✅ Bayesian model is BEST for {target}!")
                    else:
                        best_model = comparison_df.loc[comparison_df['Avg_Rank'].idxmin(), 'Model']
                        print(f"⚠️  Best model for {target}: {best_model} (rank {best_rank:.2f}) vs Bayesian (rank {bayesian_rank:.2f})")
        
        comparison_results.extend(target_comparison)

    # --------------------------------------------------------------
    # 7  COMBINE RESULTS AND SAVE
    # --------------------------------------------------------------
    
    if all_predictions:
        # Merge all predictions
        final_preds = all_predictions[0]
        for pred_df in all_predictions[1:]:
            final_preds = final_preds.merge(
                pred_df, on=['GAME_DATE', 'game_num', 'training_games'], how='outer'
            )
        
        final_preds.to_csv(OUT, index=False)
        print(f"\n📁 Online predictions saved to {OUT}")
        
        # Save comparison results
        if comparison_results:
            comparison_df = pd.DataFrame(comparison_results)
            comparison_df.to_csv(BASELINE_OUT, index=False)
            print(f"📁 Model comparison report saved to {BASELINE_OUT}")

        # --------------------------------------------------------------
        # 8  ENHANCED VISUALIZATION
        # --------------------------------------------------------------
        
        fig, axes = plt.subplots(len(TARGETS), 2, figsize=(20, 4*len(TARGETS)))
        if len(TARGETS) == 1:
            axes = axes.reshape(1, -1)

        for i, target in enumerate(TARGETS):
            # Left plot: Predictions vs Actual
            ax1 = axes[i, 0]
            
            ax1.plot(final_preds["game_num"], final_preds[target], 'o-', 
                   label="Actual", alpha=0.9, linewidth=2.5, markersize=7, color='darkblue')
            ax1.plot(final_preds["game_num"], final_preds[f"{target}_mean"], 's-', 
                   label="Bayesian", alpha=0.9, linewidth=2.5, markersize=7, color='darkred')
            
            # Add best baseline
            if target in all_baseline_predictions and 'MA_5' in all_baseline_predictions[target]:
                ax1.plot(final_preds["game_num"], all_baseline_predictions[target]['MA_5'], '^-',
                       label="MA_5 Baseline", alpha=0.7, linewidth=2, markersize=5, color='green')
            
            ax1.fill_between(
                final_preds["game_num"],
                final_preds[f"{target}_lo"],
                final_preds[f"{target}_hi"],
                alpha=0.25,
                label="80% Prediction Interval",
                color='gray'
            )
            
            ax1.set_title(f"LeBron {target} - Online Learning vs Baselines", fontsize=14)
            ax1.set_xlabel("Game Number", fontsize=12)
            ax1.set_ylabel(target, fontsize=12)
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # Right plot: Residuals analysis
            ax2 = axes[i, 1]
            
            residuals = final_preds[target] - final_preds[f"{target}_mean"]
            ax2.scatter(final_preds[f"{target}_mean"], residuals, alpha=0.7, color='darkred')
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax2.set_xlabel(f"Predicted {target}", fontsize=12)
            ax2.set_ylabel("Residuals", fontsize=12)
            ax2.set_title(f"{target} Residuals vs Predicted", fontsize=12)
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # --------------------------------------------------------------
        # 9  FINAL COMPREHENSIVE SUMMARY
        # --------------------------------------------------------------
        
        print("\n" + "="*80)
        print("COMPREHENSIVE ONLINE LEARNING VALIDATION SUMMARY")
        print("="*80)
        
        if comparison_results:
            # Overall model ranking across all targets
            comparison_df = pd.DataFrame(comparison_results)
            overall_comparison = comparison_df.groupby('Model').agg({
                'MAE': 'mean',
                'RMSE': 'mean', 
                'R2': 'mean',
                'Pearson_R': 'mean',
                'MAPE': 'mean'
            }).round(3)
            
            # Rank models
            for col in ['MAE', 'RMSE', 'MAPE']:
                if col in overall_comparison.columns:
                    overall_comparison[f'{col}_Rank'] = overall_comparison[col].rank()
            for col in ['R2', 'Pearson_R']:
                if col in overall_comparison.columns:
                    overall_comparison[f'{col}_Rank'] = overall_comparison[col].rank(ascending=False)
            
            rank_cols = [col for col in overall_comparison.columns if col.endswith('_Rank')]
            if rank_cols:
                overall_comparison['Overall_Rank'] = overall_comparison[rank_cols].mean(axis=1)
                
                print("OVERALL MODEL RANKING (averaged across all targets):")
                display_cols = ['MAE', 'RMSE', 'R2', 'Pearson_R', 'Overall_Rank']
                available_cols = [col for col in display_cols if col in overall_comparison.columns]
                print(overall_comparison.sort_values('Overall_Rank')[available_cols])
                
                print("\n" + "="*80)
                print("KEY INSIGHTS:")
                
                # Check if Bayesian model is best
                if 'Bayesian' in overall_comparison.index:
                    bayesian_rank = overall_comparison.loc['Bayesian', 'Overall_Rank']
                    best_rank = overall_comparison['Overall_Rank'].min()
                    
                    if bayesian_rank == best_rank:
                        print("✅ Bayesian online learning OUTPERFORMS all baseline approaches!")
                    else:
                        best_model = overall_comparison['Overall_Rank'].idxmin()
                        print(f"⚠️  Best model is {best_model} (rank {best_rank:.2f}) vs Bayesian (rank {bayesian_rank:.2f})")
                    
                    # Specific advantages
                    if 'R2' in overall_comparison.columns:
                        bayesian_r2 = overall_comparison.loc['Bayesian', 'R2']
                        print(f"📊 Bayesian R² = {bayesian_r2:.3f} (explains {bayesian_r2*100:.1f}% of variance)")
                
                # Coverage analysis for Bayesian
                bayesian_metrics = [r for r in comparison_results if r['Model'] == 'Bayesian']
                if bayesian_metrics and 'Coverage' in bayesian_metrics[0]:
                    avg_coverage = np.mean([m['Coverage'] for m in bayesian_metrics if 'Coverage' in m])
                    print(f"🎯 Average prediction interval coverage: {avg_coverage:.1f}% (target: 80%)")
                
                print(f"\n💡 Online Bayesian approach provides:")
                print(f"   • Model updates every {RETRAIN_FREQUENCY} games with new information")
                print(f"   • Time-weighted learning (decay rate: {DECAY_RATE})")
                print("   • Uncertainty quantification with prediction intervals")
                print("   • Incorporation of team stats and rolling performance")
                print("   • Real-world applicable prediction pipeline")
                print("="*80)
