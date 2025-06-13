#!/usr/bin/env python3
# hybrid_bayesian_arima_player_props.py - BAYESIAN FIRST, THEN ARIMA ON RESIDUALS
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
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
import itertools
warnings.filterwarnings('ignore')

# --------------------------------------------------------------
# 1  CONFIG & DATA
# --------------------------------------------------------------
CSV_PLAYERS = Path("nba_players_game_logs_2018_25_clean.csv")
CSV_TEAMS   = Path("team_stats.csv")
OUT         = Path("hybrid_bayesian_arima_preds.csv")
BASELINE_OUT = Path("hybrid_baseline_comparison.csv")

TARGETS = ["PTS", "REB", "AST"]

# Time weighting parameters for Bayesian model
DECAY_RATE = 0.05

# Online learning parameters
RETRAIN_FREQUENCY = 3  # Retrain models every N games
FAST_SAMPLING = True   # Use fewer samples for online updates

# ARIMA parameters
ARIMA_MAX_P = 3  # Maximum AR order
ARIMA_MAX_D = 1  # Maximum differencing
ARIMA_MAX_Q = 3  # Maximum MA order
MIN_ARIMA_HISTORY = 10  # Minimum games needed for ARIMA fitting

# --------------------------------------------------------------
# 1-A  LOAD DATASETS
# --------------------------------------------------------------
df = pd.read_csv(CSV_PLAYERS, parse_dates=["GAME_DATE"])
team_stats = pd.read_csv(CSV_TEAMS)

# Filter for LeBron James
LEBRON_ID = 2544
season_2024_25_df = df[(df["PLAYER_ID"] == LEBRON_ID) & (df["SEASON_YEAR"] == "2024-25")].copy()

if season_2024_25_df.empty:
    print("No 2024-25 season data found for LeBron James")
    lebron_df = df[df["PLAYER_NAME"].str.contains("LeBron", case=False, na=False)]
    if not lebron_df.empty:
        LEBRON_ID = lebron_df["PLAYER_ID"].iloc[0]
        print(f"Found LeBron James with Player ID: {LEBRON_ID}")
        season_2024_25_df = df[(df["PLAYER_ID"] == LEBRON_ID) & (df["SEASON_YEAR"] == "2024-25")].copy()
    
    if season_2024_25_df.empty:
        print("Still no data found. Available players:")
        print(df[["PLAYER_ID", "PLAYER_NAME"]].drop_duplicates().head(20))
        exit()

# Sort by game date
season_2024_25_df = season_2024_25_df.sort_values("GAME_DATE").reset_index(drop=True)

# Split data
total_games = len(season_2024_25_df)
initial_train_size = int(total_games * 0.75)

print(f"Total games in 2024-25 season: {total_games}")
print(f"Initial training: first {initial_train_size} games")
print(f"Online testing: last {total_games - initial_train_size} games")

# --------------------------------------------------------------
# 2  ARIMA HELPER FUNCTIONS FOR RESIDUALS
# --------------------------------------------------------------
def find_best_arima_order(series, max_p=3, max_d=1, max_q=3):
    """Find best ARIMA order using AIC"""
    best_aic = np.inf
    best_order = (1, 0, 1)
    
    # Test stationarity
    try:
        adf_result = adfuller(series.dropna())
        is_stationary = adf_result[1] <= 0.05
        d_range = [0] if is_stationary else [1]
    except:
        d_range = [0, 1]
    
    # Grid search for best parameters
    for p in range(max_p + 1):
        for d in d_range:
            for q in range(max_q + 1):
                try:
                    model = ARIMA(series, order=(p, d, q))
                    fitted_model = model.fit()
                    
                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_order = (p, d, q)
                        
                except:
                    continue
    
    return best_order

def fit_arima_to_residuals(residuals_series, order=None):
    """Fit ARIMA model to Bayesian residuals"""
    try:
        if order is None:
            order = find_best_arima_order(residuals_series)
        
        model = ARIMA(residuals_series, order=order)
        fitted_model = model.fit()
        
        return fitted_model, order
    except Exception as e:
        print(f"ARIMA fitting to residuals failed: {e}")
        return None, (1, 0, 1)

def create_baseline_predictions(train_history, target):
    """Create baseline predictions without lagged features"""
    baselines = {}
    
    if len(train_history) == 0:
        return {f'MA_{w}': 15 for w in [3, 5, 10]}
    
    # Simple Moving Averages (no lagged features)
    for window in [3, 5, 10, 20]:
        if len(train_history) >= window:
            baseline_pred = train_history[target].tail(window).mean()
        else:
            baseline_pred = train_history[target].mean()
        baselines[f'MA_{window}'] = baseline_pred
    
    # Exponentially Weighted Moving Average
    alpha = 0.3
    if len(train_history) >= 2:
        ewm_pred = train_history[target].ewm(alpha=alpha).mean().iloc[-1]
    else:
        ewm_pred = train_history[target].mean()
    baselines['EWMA'] = ewm_pred
    
    # Linear Trend
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
    
    if len(actual) == 0 or len(predicted) == 0:
        return {}
    
    metrics = {}
    
    # Basic accuracy metrics
    metrics['MAE'] = mean_absolute_error(actual, predicted)
    metrics['RMSE'] = np.sqrt(mean_squared_error(actual, predicted))
    metrics['MAPE'] = np.mean(np.abs((actual - predicted) / np.maximum(actual, 1e-8))) * 100
    
    # R-squared
    if np.var(actual) > 1e-8:
        metrics['R2'] = r2_score(actual, predicted)
        metrics['Explained_Variance'] = 1 - np.var(actual - predicted) / np.var(actual)
    else:
        metrics['R2'] = 0
        metrics['Explained_Variance'] = 0
    
    # Correlation metrics
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
    
    # Directional accuracy
    if len(actual) > 1:
        actual_changes = np.diff(actual)
        pred_changes = np.diff(predicted)
        if np.sum(np.abs(actual_changes)) > 1e-8:
            directional_accuracy = np.mean(np.sign(actual_changes) == np.sign(pred_changes))
            metrics['Directional_Accuracy'] = directional_accuracy
        else:
            metrics['Directional_Accuracy'] = 0.5
    
    # Bias metrics
    metrics['Mean_Bias'] = np.mean(predicted - actual)
    metrics['Median_Bias'] = np.median(predicted - actual)
    
    # Prediction interval metrics
    if prediction_intervals is not None:
        lower, upper = prediction_intervals
        lower = np.array(lower)
        upper = np.array(upper)
        
        coverage = np.mean((actual >= lower) & (actual <= upper))
        avg_width = np.mean(upper - lower)
        
        metrics['Coverage'] = coverage * 100
        metrics['Avg_Interval_Width'] = avg_width
        
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
# 3  HELPER FUNCTIONS (WITHOUT LAGGED FEATURES)
# --------------------------------------------------------------

def prepare_features_no_lags(data_df, opponent_stats, team_stat_cols):
    """Prepare features WITHOUT lagged indicators"""
    # Add basic features
    data_df["home_flag"] = (data_df["MATCHUP"].str.contains("vs")).astype(int)
    
    # Merge opponent stats
    data_df = data_df.merge(opponent_stats, on="OPP_TEAM", how="left")
    
    # Fill missing team stats
    for col in team_stat_cols:
        if col in data_df.columns:
            data_df[col] = data_df[col].fillna(data_df[col].mean())
    
    # Add rest days (simplified)
    data_df["rest_days"] = 1
    
    # Only use contextual features - NO ROLLING AVERAGES OR LAGS
    feature_cols = ["home_flag", "rest_days"] + team_stat_cols
    
    # Filter to feature columns that exist
    feature_cols = [col for col in feature_cols if col in data_df.columns]
    feature_data = data_df[feature_cols].fillna(method='ffill').fillna(0).astype(np.float32).values
    
    return data_df, feature_data, feature_cols

def calculate_time_weights(n_games, decay_rate):
    """Calculate exponential time weights"""
    game_indices = np.arange(n_games)
    time_weights = np.exp(-decay_rate * (n_games - 1 - game_indices))
    time_weights = time_weights * n_games / time_weights.sum()
    return time_weights

def fit_bayesian_model_no_lags(X_train_std, y_train, time_weights, target, expected_mean):
    """Fit Bayesian model without lagged features"""
    
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
        intercept = pm.Normal("intercept", mu=np.log(expected_mean), sigma=0.5)
        
        # Feature coefficients (fewer features now)
        n_features = X_train_std.shape[1]
        beta = pm.Normal("beta", mu=0, sigma=0.5, shape=n_features)
        
        # Dispersion parameter
        phi = pm.HalfNormal("phi", sigma=1.0)

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

def get_bayesian_predictions_for_training(model, trace, X_train_std):
    """Get Bayesian predictions for training data to calculate residuals"""
    intercept_samples = trace.posterior['intercept'].values.flatten()
    beta_samples = trace.posterior['beta'].values.reshape(-1, X_train_std.shape[1])
    phi_samples = trace.posterior['phi'].values.flatten()
    
    n_samples = len(intercept_samples)
    n_games = X_train_std.shape[0]
    
    # Get mean predictions for each training game
    train_predictions = []
    
    for game_idx in range(n_games):
        game_preds = []
        for sample_idx in range(n_samples):
            linear_pred = intercept_samples[sample_idx] + np.dot(X_train_std[game_idx], beta_samples[sample_idx])
            mu_pred = np.exp(np.clip(linear_pred, -10, 10))
            game_preds.append(mu_pred)  # Use mean of distribution, not random sample
        
        train_predictions.append(np.mean(game_preds))
    
    return np.array(train_predictions)

# --------------------------------------------------------------
# 4  FEATURE ENGINEERING SETUP (NO LAGS)
# --------------------------------------------------------------

# Merge opponent team stats setup
opponent_stats = team_stats[["TEAM_ABBREVIATION", "OFF_RATING", "DEF_RATING", "NET_RATING", 
                           "PACE", "AST_RATIO", "REB_PCT"]].copy()
opponent_stats.columns = ["OPP_TEAM", "OPP_OFF_RATING", "OPP_DEF_RATING", "OPP_NET_RATING", 
                         "OPP_PACE", "OPP_AST_RATIO", "OPP_REB_PCT"]

team_stat_cols = ["OPP_OFF_RATING", "OPP_DEF_RATING", "OPP_NET_RATING", "OPP_PACE", 
                  "OPP_AST_RATIO", "OPP_REB_PCT"]

print("NEW APPROACH: Bayesian model first, then ARIMA on residuals")
print("Features used (NO LAGGED INDICATORS):")
print("  Basic: home_flag, rest_days")
print(f"  Team stats: {len(team_stat_cols)} opponent features")  
print("  ARIMA will model patterns in Bayesian residuals")

# --------------------------------------------------------------
# 5  HYBRID ONLINE LEARNING: BAYESIAN → RESIDUALS → ARIMA
# --------------------------------------------------------------

if __name__ == '__main__':
    all_predictions = []
    all_baseline_predictions = {}
    comparison_results = []
    
    for target in TARGETS:
        print(f"\n{'='*70}")
        print(f"BAYESIAN → ARIMA RESIDUALS FOR {target}")
        print(f"{'='*70}")
        
        target_predictions = []
        baseline_predictions = {model: [] for model in ['MA_3', 'MA_5', 'MA_10', 'EWMA', 'Linear_Trend']}
        bayesian_predictions = []
        arima_corrections = []
        hybrid_predictions = []
        actual_values = []
        residuals_history = []  # Track residuals for ARIMA
        
        # Target-specific parameters
        target_means = {"PTS": 25, "REB": 7, "AST": 7}
        expected_mean = target_means.get(target, 15)
        
        # Initial training data
        current_train_end = initial_train_size
        
        # Track ARIMA models for residuals
        current_arima_model = None
        arima_order = (1, 0, 1)
        
        # Iterate through each test game
        for test_game_idx in range(initial_train_size, total_games):
            
            print(f"\nPredicting game {test_game_idx + 1}/{total_games} (Test game {test_game_idx - initial_train_size + 1})")
            
            # Current training data
            current_train_df = season_2024_25_df.iloc[:current_train_end].copy()
            current_test_df = season_2024_25_df.iloc[[test_game_idx]].copy()
            
            # Get actual value
            actual_value = current_test_df[target].iloc[0]
            actual_values.append(actual_value)
            
            # CREATE BASELINE PREDICTIONS
            baselines = create_baseline_predictions(current_train_df, target)
            for model_name, pred_value in baselines.items():
                if model_name in baseline_predictions:
                    baseline_predictions[model_name].append(pred_value)
            
            # STEP 1: BAYESIAN COMPONENT (contextual features only)
            current_train_df_processed, X_train, feature_cols = prepare_features_no_lags(
                current_train_df, opponent_stats, team_stat_cols
            )
            current_test_df_processed, X_test, _ = prepare_features_no_lags(
                current_test_df, opponent_stats, team_stat_cols
            )
            
            # Standardize features
            means = np.mean(X_train, axis=0)
            stds = np.std(X_train, axis=0) + 1e-8
            X_train_std = (X_train - means) / stds
            X_test_std = (X_test - means) / stds
            
            # Time weights
            time_weights = calculate_time_weights(len(current_train_df), DECAY_RATE)
            y_train = current_train_df[target].values.astype(np.int32)
            
            # Retrain Bayesian model
            should_retrain = (
                test_game_idx == initial_train_size or
                (test_game_idx - initial_train_size) % RETRAIN_FREQUENCY == 0
            )
            
            if should_retrain:
                print(f"  🔄 Retraining Bayesian model on {len(current_train_df)} games...")
                
                bayesian_model, bayesian_trace = fit_bayesian_model_no_lags(
                    X_train_std, y_train, time_weights, target, expected_mean
                )
                
                # STEP 2: GET BAYESIAN PREDICTIONS FOR TRAINING DATA TO CALCULATE RESIDUALS
                print(f"  📊 Calculating Bayesian residuals for ARIMA...")
                train_bayesian_preds = get_bayesian_predictions_for_training(
                    bayesian_model, bayesian_trace, X_train_std
                )
                
                # Calculate residuals: actual - bayesian_prediction
                current_residuals = y_train - train_bayesian_preds
                residuals_history = current_residuals.tolist()  # Update full residuals history
                
                # Store for reuse
                current_bayesian_model = bayesian_model
                current_bayesian_trace = bayesian_trace
                current_means = means
                current_stds = stds
                
            else:
                print(f"  ⚡ Using cached Bayesian model")
                bayesian_model = current_bayesian_model
                bayesian_trace = current_bayesian_trace
                X_test_std = (X_test - current_means) / current_stds
                
                # Add latest residual to history (from previous prediction)
                # We need to calculate the Bayesian prediction for the latest training game
                if len(residuals_history) < len(current_train_df):
                    latest_game_X = X_train_std[-1:, :]  # Last training game features
                    latest_bayesian_pred = get_bayesian_predictions_for_training(
                        bayesian_model, bayesian_trace, latest_game_X
                    )[0]
                    latest_residual = y_train[-1] - latest_bayesian_pred
                    residuals_history.append(latest_residual)
            
            # STEP 3: BAYESIAN PREDICTION FOR TEST GAME
            with bayesian_model:
                intercept_samples = bayesian_trace.posterior['intercept'].values.flatten()
                beta_samples = bayesian_trace.posterior['beta'].values.reshape(-1, X_test_std.shape[1])
                phi_samples = bayesian_trace.posterior['phi'].values.flatten()
                
                n_samples = len(intercept_samples)
                bayesian_preds = []
                
                for i in range(n_samples):
                    linear_pred = intercept_samples[i] + np.dot(X_test_std[0], beta_samples[i])
                    mu_pred = np.exp(np.clip(linear_pred, -10, 10))
                    pred_sample = np.random.negative_binomial(phi_samples[i], phi_samples[i] / (mu_pred + phi_samples[i]))
                    bayesian_preds.append(pred_sample)
                
                bayesian_pred_mean = np.mean(bayesian_preds)
                bayesian_pred_lo = np.percentile(bayesian_preds, 10)
                bayesian_pred_hi = np.percentile(bayesian_preds, 90)
            
            # STEP 4: ARIMA ON RESIDUALS
            arima_correction = 0  # Default: no correction
            
            if len(residuals_history) >= MIN_ARIMA_HISTORY:
                print(f"  🔄 Fitting ARIMA to {len(residuals_history)} Bayesian residuals...")
                
                try:
                    residuals_series = pd.Series(residuals_history)
                    
                    # Fit ARIMA to residuals
                    arima_model, arima_order = fit_arima_to_residuals(residuals_series, arima_order)
                    
                    if arima_model is not None:
                        # Forecast next residual
                        arima_forecast = arima_model.forecast(steps=1)
                        arima_correction = float(arima_forecast[0])
                        current_arima_model = arima_model
                        
                        print(f"    ARIMA order: {arima_order}, residual correction: {arima_correction:.2f}")
                    else:
                        arima_correction = np.mean(residuals_history[-5:]) if len(residuals_history) >= 5 else 0
                        print(f"    ARIMA failed, using recent residual mean: {arima_correction:.2f}")
                        
                except Exception as e:
                    arima_correction = np.mean(residuals_history[-5:]) if len(residuals_history) >= 5 else 0
                    print(f"    ARIMA error: {e}, using recent residual mean: {arima_correction:.2f}")
            else:
                arima_correction = np.mean(residuals_history) if residuals_history else 0
                print(f"    Insufficient residual history, using mean: {arima_correction:.2f}")
            
            # STEP 5: HYBRID PREDICTION (Bayesian + ARIMA correction)
            hybrid_pred = bayesian_pred_mean + arima_correction
            
            # Store predictions
            bayesian_predictions.append(bayesian_pred_mean)
            arima_corrections.append(arima_correction)
            hybrid_predictions.append(hybrid_pred)
            
            # Store detailed results
            game_date = current_test_df["GAME_DATE"].iloc[0]
            
            target_predictions.append({
                'GAME_DATE': game_date,
                'game_num': test_game_idx + 1,
                f'{target}': actual_value,
                f'{target}_bayesian': bayesian_pred_mean,
                f'{target}_arima_correction': arima_correction,
                f'{target}_hybrid': hybrid_pred,
                f'{target}_lo': bayesian_pred_lo,
                f'{target}_hi': bayesian_pred_hi,
                'training_games': len(current_train_df)
            })
            
            # Print predictions
            print(f"  📊 Actual: {actual_value}")
            print(f"     Bayesian: {bayesian_pred_mean:.1f}")
            print(f"     ARIMA correction: {arima_correction:+.2f}")
            print(f"     Hybrid: {hybrid_pred:.1f} [{bayesian_pred_lo:.0f}-{bayesian_pred_hi:.0f}]")
            print(f"     Best Baseline (MA_5): {baselines.get('MA_5', 0):.1f}")
            
            # Update training set
            current_train_end = test_game_idx + 1
        
        # Convert to DataFrame
        target_df = pd.DataFrame(target_predictions)
        all_predictions.append(target_df)
        all_baseline_predictions[target] = baseline_predictions
        
        # --------------------------------------------------------------
        # 6  COMPREHENSIVE MODEL COMPARISON
        # --------------------------------------------------------------
        print(f"\n--- {target} MODEL COMPARISON ---")
        
        # Prepare all models for comparison
        all_model_predictions = dict(baseline_predictions)
        all_model_predictions['Bayesian'] = bayesian_predictions
        all_model_predictions['ARIMA_Correction'] = arima_corrections  # Show ARIMA corrections separately
        all_model_predictions['Hybrid'] = hybrid_predictions
        
        target_comparison = []
        for model_name, predictions in all_model_predictions.items():
            if len(predictions) == len(actual_values) and model_name != 'ARIMA_Correction':  # Skip correction-only
                if model_name == 'Bayesian':
                    # Include prediction intervals for Bayesian
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
        
        # Display comparison
        if target_comparison:
            comparison_df = pd.DataFrame(target_comparison)
            
            # Rank models
            for col in ['MAE', 'RMSE', 'MAPE']:
                if col in comparison_df.columns:
                    comparison_df[f'{col}_Rank'] = comparison_df[col].rank()
            for col in ['R2', 'Pearson_R']:
                if col in comparison_df.columns:
                    comparison_df[f'{col}_Rank'] = comparison_df[col].rank(ascending=False)
            
            rank_cols = [col for col in comparison_df.columns if col.endswith('_Rank')]
            if rank_cols:
                comparison_df['Avg_Rank'] = comparison_df[rank_cols].mean(axis=1)
                
                print("\nModel Performance Ranking:")
                display_cols = ['MAE', 'RMSE', 'R2', 'Pearson_R', 'Avg_Rank']
                available_cols = [col for col in display_cols if col in comparison_df.columns]
                print(comparison_df.sort_values('Avg_Rank')[available_cols].round(3))
                
                # Check best model
                best_model = comparison_df.loc[comparison_df['Avg_Rank'].idxmin(), 'Model']
                best_rank = comparison_df['Avg_Rank'].min()
                
                print(f"\n🏆 Best model for {target}: {best_model} (rank {best_rank:.2f})")
                
                # Hybrid performance
                hybrid_row = comparison_df[comparison_df['Model'] == 'Hybrid']
                if not hybrid_row.empty:
                    hybrid_rank = hybrid_row['Avg_Rank'].iloc[0]
                    bayesian_rank = comparison_df[comparison_df['Model'] == 'Bayesian']['Avg_Rank'].iloc[0]
                    improvement = bayesian_rank - hybrid_rank
                    print(f"🔗 Hybrid model rank: {hybrid_rank:.2f} (improvement over Bayesian: {improvement:+.2f})")
        
        comparison_results.extend(target_comparison)

    # --------------------------------------------------------------
    # 7  SAVE RESULTS AND VISUALIZE
    # --------------------------------------------------------------
    
    if all_predictions:
        # Combine all predictions
        final_preds = all_predictions[0]
        for pred_df in all_predictions[1:]:
            final_preds = final_preds.merge(
                pred_df, on=['GAME_DATE', 'game_num', 'training_games'], how='outer'
            )
        
        final_preds.to_csv(OUT, index=False)
        print(f"\n📁 Hybrid predictions saved to {OUT}")
        
        # Save comparison
        if comparison_results:
            comparison_df = pd.DataFrame(comparison_results)
            comparison_df.to_csv(BASELINE_OUT, index=False)
            print(f"📁 Comparison report saved to {BASELINE_OUT}")

        # --------------------------------------------------------------
        # 8  ENHANCED VISUALIZATION
        # --------------------------------------------------------------
        
        fig, axes = plt.subplots(len(TARGETS), 2, figsize=(20, 4*len(TARGETS)))
        if len(TARGETS) == 1:
            axes = axes.reshape(1, -1)

        for i, target in enumerate(TARGETS):
            # Left plot: All model predictions
            ax1 = axes[i, 0]
            
            ax1.plot(final_preds["game_num"], final_preds[target], 'o-', 
                   label="Actual", alpha=0.9, linewidth=2.5, markersize=7, color='darkblue')
            ax1.plot(final_preds["game_num"], final_preds[f"{target}_hybrid"], 's-', 
                   label="Hybrid (Bayesian + ARIMA)", alpha=0.9, linewidth=2.5, markersize=7, color='purple')
            ax1.plot(final_preds["game_num"], final_preds[f"{target}_bayesian"], '^-', 
                   label="Bayesian Only", alpha=0.7, linewidth=2, markersize=5, color='darkred')
            
            # Show ARIMA corrections as bars
            ax1_twin = ax1.twinx()
            ax1_twin.bar(final_preds["game_num"], final_preds[f"{target}_arima_correction"], 
                        alpha=0.3, color='orange', width=0.6, label="ARIMA Correction")
            ax1_twin.set_ylabel("ARIMA Correction", fontsize=10, color='orange')
            ax1_twin.tick_params(axis='y', labelcolor='orange')
            
            # Add best baseline
            if target in all_baseline_predictions and 'MA_5' in all_baseline_predictions[target]:
                ax1.plot(final_preds["game_num"], all_baseline_predictions[target]['MA_5'], 'd-',
                       label="MA_5", alpha=0.6, linewidth=1.5, markersize=4, color='green')
            
            ax1.fill_between(
                final_preds["game_num"],
                final_preds[f"{target}_lo"],
                final_preds[f"{target}_hi"],
                alpha=0.2,
                label="80% Pred. Interval",
                color='gray'
            )
            
            ax1.set_title(f"LeBron {target} - Bayesian → ARIMA Residual Correction", fontsize=14)
            ax1.set_xlabel("Game Number", fontsize=12)
            ax1.set_ylabel(target, fontsize=12)
            ax1.legend(loc='upper left', fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # Right plot: Residual analysis
            ax2 = axes[i, 1]
            
            # Show both Bayesian and Hybrid residuals
            bayesian_residuals = final_preds[target] - final_preds[f"{target}_bayesian"]
            hybrid_residuals = final_preds[target] - final_preds[f"{target}_hybrid"]
            
            ax2.scatter(final_preds[f"{target}_bayesian"], bayesian_residuals, 
                       alpha=0.6, color='darkred', label='Bayesian Residuals', s=40)
            ax2.scatter(final_preds[f"{target}_hybrid"], hybrid_residuals, 
                       alpha=0.6, color='purple', label='Hybrid Residuals', s=40)
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax2.set_xlabel(f"Predicted {target}", fontsize=12)
            ax2.set_ylabel("Residuals", fontsize=12)
            ax2.set_title(f"{target} Residuals: Bayesian vs Hybrid", fontsize=12)
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # --------------------------------------------------------------
        # 9  FINAL COMPREHENSIVE SUMMARY
        # --------------------------------------------------------------
        
        print("\n" + "="*80)
        print("BAYESIAN → ARIMA RESIDUAL CORRECTION MODEL SUMMARY")
        print("="*80)
        
        if comparison_results:
            # Overall rankings
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
                
                print("OVERALL MODEL RANKING:")
                display_cols = ['MAE', 'RMSE', 'R2', 'Pearson_R', 'Overall_Rank']
                available_cols = [col for col in display_cols if col in overall_comparison.columns]
                print(overall_comparison.sort_values('Overall_Rank')[available_cols])
                
                print("\n" + "="*80)
                print("KEY INSIGHTS:")
                
                # Check hybrid vs bayesian improvement
                if 'Hybrid' in overall_comparison.index and 'Bayesian' in overall_comparison.index:
                    hybrid_rank = overall_comparison.loc['Hybrid', 'Overall_Rank']
                    bayesian_rank = overall_comparison.loc['Bayesian', 'Overall_Rank']
                    improvement = bayesian_rank - hybrid_rank
                    
                    if improvement > 0:
                        print(f"✅ ARIMA residual correction IMPROVES Bayesian model!")
                        print(f"   Hybrid rank: {hybrid_rank:.2f} vs Bayesian rank: {bayesian_rank:.2f}")
                        print(f"   Improvement: {improvement:+.2f} rank positions")
                    else:
                        print(f"⚠️  ARIMA correction doesn't improve Bayesian model")
                        print(f"   Hybrid rank: {hybrid_rank:.2f} vs Bayesian rank: {bayesian_rank:.2f}")
                    
                    # R² comparison
                    if 'R2' in overall_comparison.columns:
                        hybrid_r2 = overall_comparison.loc['Hybrid', 'R2']
                        bayesian_r2 = overall_comparison.loc['Bayesian', 'R2']
                        r2_improvement = hybrid_r2 - bayesian_r2
                        print(f"📊 R² improvement: {bayesian_r2:.3f} → {hybrid_r2:.3f} ({r2_improvement:+.3f})")
                
                # Check against best overall
                best_model = overall_comparison['Overall_Rank'].idxmin()
                best_rank = overall_comparison['Overall_Rank'].min()
                
                if 'Hybrid' in overall_comparison.index:
                    hybrid_rank = overall_comparison.loc['Hybrid', 'Overall_Rank']
                    if hybrid_rank == best_rank:
                        print(f"🏆 Hybrid model is BEST OVERALL!")
                    else:
                        print(f"🏆 Best model: {best_model} (rank {best_rank:.2f}) vs Hybrid (rank {hybrid_rank:.2f})")
                
                print(f"\n💡 Bayesian → ARIMA approach:")
                print(f"   • Step 1: Bayesian model captures contextual effects")
                print(f"   • Step 2: Calculate residuals = actual - bayesian_prediction")
                print(f"   • Step 3: ARIMA models temporal patterns in residuals")
                print(f"   • Step 4: Final prediction = bayesian + arima_correction")
                print(f"   • This separates contextual vs temporal modeling concerns")
                print("="*80)
