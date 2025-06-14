#!/usr/bin/env python3
# multi_player_evaluation.py - Comprehensive Model Testing Across All Players
# --------------------------------------------------------------
# Tests the optimized research model on multiple players to evaluate:
# - Overall model accuracy and effectiveness
# - Performance consistency across different player types
# - Comparison with baseline methods
# - Statistical significance of improvements
# --------------------------------------------------------------

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
warnings.filterwarnings('ignore')

# Import our RNN + TFT model functions
from rnn_tft_predictor import (
    calculate_lstm_features,
    calculate_gru_features,
    calculate_bilstm_features,
    calculate_attention_features,
    calculate_tft_features,
    calculate_baseline_features,
    calculate_rnn_ensemble_prediction,
    prepare_rnn_tft_features,
    transform_to_normal_scores,
    calculate_metrics,
    extract_opponent_from_matchup
)

# --------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------
CSV_PLAYERS = Path("nba_players_game_logs_2018_25_clean.csv")
CSV_TEAMS   = Path("nba_team_stats_monthly_2018_current.csv")  # Updated to monthly team stats
RESULTS_FILE = Path("multi_player_rnn_tft_results.csv")
SUMMARY_FILE = Path("rnn_tft_model_evaluation_summary.txt")

TARGETS = ["PTS", "REB", "AST"]

# RNN + TFT Configuration
SEQUENCE_LENGTHS = [2, 3, 4]           # Multiple sequence lengths
USE_LSTM = True                        # Enable LSTM features
USE_GRU = True                         # Enable GRU features
USE_BI_LSTM = True                     # Enable Bidirectional LSTM
USE_ATTENTION = True                   # Enable attention mechanism
USE_TFT = True                         # Enable Temporal Fusion Transformer
NORMAL_SCORE_TRANSFORM = True          # Transform to normal scores

# Test parameters
MIN_GAMES_FOR_PLAYER = 50      # Reduced to get more players
MIN_GAMES_FOR_PREDICTION = 25  # Minimum games before making predictions
MAX_PLAYERS_TO_TEST = 80       # Reduced for faster testing
ENSEMBLE_METHODS = True        # Use ensemble methods
ENSEMBLE_WEIGHTS = [0.35, 0.25, 0.25, 0.15]  # TFT, LSTM, Attention, Baseline

print("MULTI-PLAYER MODEL EVALUATION")
print("="*60)
print("Testing optimized research model across multiple NBA players")
print(f"Targets: {TARGETS}")
print(f"Max players to test: {MAX_PLAYERS_TO_TEST}")
print(f"Minimum games per player: {MIN_GAMES_FOR_PLAYER}")

# --------------------------------------------------------------
# DATA LOADING AND PLAYER SELECTION
# --------------------------------------------------------------

def load_data_and_select_players():
    """Load data and select diverse set of players for testing"""
    print(f"\nLOADING DATA AND SELECTING PLAYERS")
    print("-" * 40)
    
    # Load datasets
    df = pd.read_csv(CSV_PLAYERS, parse_dates=["GAME_DATE"])
    team_stats = pd.read_csv(CSV_TEAMS) if CSV_TEAMS.exists() else pd.DataFrame()
    
    print(f"Dataset loaded: {df.shape[0]:,} games")
    print(f"Players in dataset: {df['PLAYER_ID'].nunique():,}")
    print(f"Date range: {df['GAME_DATE'].min()} to {df['GAME_DATE'].max()}")
    
    # Calculate player game counts and basic stats
    player_stats = df.groupby(['PLAYER_ID', 'PLAYER_NAME']).agg({
        'PTS': ['count', 'mean', 'std'],
        'REB': ['mean', 'std'],
        'AST': ['mean', 'std'],
        'GAME_DATE': ['min', 'max']
    }).round(2)
    player_stats.columns = ['_'.join(col).strip() for col in player_stats.columns]
    player_stats = player_stats.reset_index()
    player_stats.rename(columns={'PTS_count': 'total_games'}, inplace=True)
    
    # Filter players with sufficient games and recent activity
    recent_cutoff = df['GAME_DATE'].max() - pd.DateOffset(years=2)
    recent_players = df[df['GAME_DATE'] >= recent_cutoff]['PLAYER_ID'].unique()
    
    eligible_players = player_stats[
        (player_stats['total_games'] >= MIN_GAMES_FOR_PLAYER) &
        (player_stats['PLAYER_ID'].isin(recent_players))
    ].copy()
    
    print(f"Players with {MIN_GAMES_FOR_PLAYER}+ games and recent activity: {len(eligible_players)}")
    
    if len(eligible_players) == 0:
        print(f"❌ No players found with {MIN_GAMES_FOR_PLAYER}+ games!")
        print("Top 10 players by game count:")
        top_by_games = player_stats.nlargest(10, 'total_games')[['PLAYER_NAME', 'total_games', 'PTS_mean']]
        print(top_by_games)
        return df, team_stats, pd.DataFrame()
    
    # Select diverse set of players across different performance levels
    if len(eligible_players) <= MAX_PLAYERS_TO_TEST:
        # Use all eligible players if we have few enough
        selected_players = eligible_players.copy()
        print(f"Using all {len(selected_players)} eligible players")
    else:
        # Select top players by games played
        top_players = eligible_players.nlargest(min(15, len(eligible_players)), 'total_games')
        
        # Add some medium-volume players for diversity if available
        remaining_players = eligible_players[~eligible_players['PLAYER_ID'].isin(top_players['PLAYER_ID'])]
        
        if len(remaining_players) > 0:
            # Sample remaining players
            medium_count = min(5, len(remaining_players), MAX_PLAYERS_TO_TEST - len(top_players))
            if medium_count > 0:
                medium_players = remaining_players.sample(medium_count, random_state=42)
                selected_players = pd.concat([top_players, medium_players])
            else:
                selected_players = top_players
        else:
            selected_players = top_players
        
        # Ensure we don't exceed the limit
        selected_players = selected_players.head(MAX_PLAYERS_TO_TEST)
    
    print(f"✅ Selected {len(selected_players)} players for testing")
    if len(selected_players) > 0:
        print(f"Games per player range: {selected_players['total_games'].min():.0f} - {selected_players['total_games'].max():.0f}")
        print(f"Average PTS range: {selected_players['PTS_mean'].min():.1f} - {selected_players['PTS_mean'].max():.1f}")
        
        # Show the selected players
        print("\nSelected players:")
        display_players = selected_players[['PLAYER_NAME', 'total_games', 'PTS_mean', 'REB_mean', 'AST_mean']].copy()
        display_players.columns = ['Player', 'Games', 'PPG', 'RPG', 'APG']
        for i, row in display_players.head(10).iterrows():
            print(f"  {row['Player']:<25} {row['Games']:>3.0f} games  {row['PPG']:>5.1f} PPG  {row['RPG']:>4.1f} RPG  {row['APG']:>4.1f} APG")
        if len(display_players) > 10:
            print(f"  ... and {len(display_players) - 10} more players")
    
    return df, team_stats, selected_players

# --------------------------------------------------------------
# SINGLE PLAYER EVALUATION
# --------------------------------------------------------------

def evaluate_single_player(df, team_stats, player_id, player_name):
    """Evaluate model performance for a single player"""
    
    # Get player data
    player_data = df[df["PLAYER_ID"] == player_id].copy()
    player_data = player_data.sort_values("GAME_DATE").reset_index(drop=True)
    
    if len(player_data) < MIN_GAMES_FOR_PLAYER:
        return None
    
    # Split data: use last 20% of games as test set
    n_games = len(player_data)
    test_size = max(10, int(n_games * 0.2))  # At least 10 games for testing
    train_size = n_games - test_size
    
    if train_size < MIN_GAMES_FOR_PREDICTION:
        return None
    
    train_data = player_data.iloc[:train_size].copy()
    test_data = player_data.iloc[train_size:].copy()
    
    player_results = {
        'player_id': player_id,
        'player_name': player_name,
        'total_games': n_games,
        'train_games': len(train_data),
        'test_games': len(test_data)
    }
      # Evaluate each target
    for target in TARGETS:
        try:
            # Storage for predictions
            rnn_tft_predictions = []
            baseline_predictions = {
                "Simple_MA5": [], "Simple_MA10": [], "Exponential": [], 
                "Linear_Trend": [], "Median": [], "Last_Value": []
            }
            actual_values = []
            
            # Start with training data
            current_data = train_data.copy()
            
            # Online learning loop
            for test_idx in range(len(test_data)):
                current_game = test_data.iloc[test_idx]
                actual_value = current_game[target]
                actual_values.append(actual_value)
                
                if len(current_data) < MIN_GAMES_FOR_PREDICTION:
                    # Not enough data - use simple baseline
                    rnn_tft_predictions.append(actual_value)
                    for model_name in baseline_predictions.keys():
                        baseline_predictions[model_name].append(actual_value)
                    current_data = pd.concat([current_data, current_game.to_frame().T], ignore_index=True)
                    continue
                
                # Generate baseline predictions using simple methods
                target_history = current_data[target].values
                
                # Simple baselines
                baseline_predictions["Simple_MA5"].append(np.mean(target_history[-5:]) if len(target_history) >= 5 else np.mean(target_history))
                baseline_predictions["Simple_MA10"].append(np.mean(target_history[-10:]) if len(target_history) >= 10 else np.mean(target_history))
                baseline_predictions["Exponential"].append(calculate_exponential_smoothing(target_history))
                baseline_predictions["Linear_Trend"].append(calculate_linear_trend(target_history))
                baseline_predictions["Median"].append(np.median(target_history[-7:]) if len(target_history) >= 7 else np.median(target_history))
                baseline_predictions["Last_Value"].append(target_history[-1])
                
                # Generate RNN + TFT prediction
                try:
                    # Get team stats (simplified - no leakage check for multi-player evaluation)
                    team_stats = create_default_team_stats()
                    
                    # Prepare RNN + TFT features
                    features = prepare_rnn_tft_features(current_data, target, team_stats, current_game)
                    
                    # Make RNN + TFT ensemble prediction
                    rnn_tft_pred = calculate_rnn_ensemble_prediction(features, target_history, target)
                    rnn_tft_predictions.append(rnn_tft_pred)
                    
                except Exception as e:
                    # Fallback to simple moving average
                    fallback_pred = np.mean(target_history[-5:]) if len(target_history) >= 5 else np.mean(target_history)
                    rnn_tft_predictions.append(fallback_pred)
                
                # Online learning: add game to training data
                current_data = pd.concat([current_data, current_game.to_frame().T], ignore_index=True)
                current_data = current_data.sort_values("GAME_DATE").reset_index(drop=True)
              # Calculate metrics
            if len(actual_values) > 0:
                # RNN + TFT model metrics
                rnn_tft_metrics = calculate_metrics(actual_values, rnn_tft_predictions)
                
                # Baseline metrics
                baseline_metrics = {}
                best_baseline_rmse = np.inf
                best_baseline_name = ""
                
                for model_name, predictions in baseline_predictions.items():
                    if len(predictions) == len(actual_values):
                        metrics = calculate_metrics(actual_values, predictions)
                        baseline_metrics[model_name] = metrics
                        
                        if metrics['rmse'] < best_baseline_rmse:
                            best_baseline_rmse = metrics['rmse']
                            best_baseline_name = model_name
                
                # Store results
                player_results[f'{target}_rnn_tft_rmse'] = rnn_tft_metrics['rmse']
                player_results[f'{target}_rnn_tft_mae'] = rnn_tft_metrics['mae']
                player_results[f'{target}_rnn_tft_mape'] = rnn_tft_metrics['mape']
                player_results[f'{target}_rnn_tft_r2'] = rnn_tft_metrics['r2']
                player_results[f'{target}_rnn_tft_bias'] = rnn_tft_metrics['mean_error']
                
                player_results[f'{target}_best_baseline'] = best_baseline_name
                player_results[f'{target}_best_baseline_rmse'] = best_baseline_rmse
                player_results[f'{target}_rmse_improvement'] = best_baseline_rmse - rnn_tft_metrics['rmse']
                player_results[f'{target}_improvement_pct'] = (best_baseline_rmse - rnn_tft_metrics['rmse']) / best_baseline_rmse * 100
                  # Performance categorization
                if rnn_tft_metrics['rmse'] < best_baseline_rmse:
                    improvement_pct = (best_baseline_rmse - rnn_tft_metrics['rmse']) / best_baseline_rmse * 100
                    if improvement_pct > 15:
                        player_results[f'{target}_performance'] = 'Major Improvement'
                    elif improvement_pct > 7:
                        player_results[f'{target}_performance'] = 'Significant Improvement'
                    else:
                        player_results[f'{target}_performance'] = 'Minor Improvement'
                else:
                    player_results[f'{target}_performance'] = 'Baseline Better'
        
        except Exception as e:
            print(f"Error evaluating {player_name} for {target}: {e}")
            player_results[f'{target}_rnn_tft_rmse'] = np.nan
            player_results[f'{target}_improvement_pct'] = np.nan
            player_results[f'{target}_performance'] = 'Error'
    
    return player_results

# --------------------------------------------------------------
# MAIN EVALUATION FUNCTION
# --------------------------------------------------------------

def run_multi_player_evaluation():
    """Run comprehensive evaluation across multiple players"""
    
    # Load data and select players
    df, team_stats, selected_players = load_data_and_select_players()
    
    print(f"\nSTARTING EVALUATION")
    print("-" * 40)
    
    all_results = []
    start_time = time.time()
    
    # Progress bar for player evaluation
    for idx, (_, player_row) in enumerate(tqdm(selected_players.iterrows(), 
                                              total=len(selected_players), 
                                              desc="Evaluating players")):
        
        player_id = player_row['PLAYER_ID']
        player_name = player_row['PLAYER_NAME']
        
        # Evaluate player
        result = evaluate_single_player(df, team_stats, player_id, player_name)
        
        if result is not None:
            all_results.append(result)
        
        # Progress update every 10 players
        if (idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Completed {idx + 1}/{len(selected_players)} players in {elapsed:.1f}s")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save detailed results
    results_df.to_csv(RESULTS_FILE, index=False)
    print(f"\nDetailed results saved to: {RESULTS_FILE}")
    
    return results_df

# --------------------------------------------------------------
# RESULTS ANALYSIS AND REPORTING
# --------------------------------------------------------------

def analyze_and_report_results(results_df):
    """Analyze results and create comprehensive report"""
    
    print(f"\nANALYZING RESULTS")
    print("-" * 40)
    
    n_players = len(results_df)
    
    # Summary statistics for each target
    summary_report = []
    summary_report.append("MULTI-PLAYER MODEL EVALUATION SUMMARY")
    summary_report.append("=" * 60)
    summary_report.append(f"Total players evaluated: {n_players}")
    summary_report.append(f"Average games per player: {results_df['total_games'].mean():.1f}")
    summary_report.append(f"Average test games per player: {results_df['test_games'].mean():.1f}")
    summary_report.append("")
    
    overall_stats = {}
    
    for target in TARGETS:
        summary_report.append(f"TARGET: {target}")
        summary_report.append("-" * 30)
        
        # Filter out error cases
        valid_results = results_df[~results_df[f'{target}_optimized_r2'].isna()]
        n_valid = len(valid_results)
        
        if n_valid == 0:
            summary_report.append("No valid results for this target")
            continue
          # Performance statistics
        opt_r2_mean = valid_results[f'{target}_optimized_r2'].mean()
        opt_r2_std = valid_results[f'{target}_optimized_r2'].std()
        opt_r2_median = valid_results[f'{target}_optimized_r2'].median()
        opt_r2_min = valid_results[f'{target}_optimized_r2'].min()
        opt_r2_max = valid_results[f'{target}_optimized_r2'].max()
        opt_mae_mean = valid_results[f'{target}_optimized_mae'].mean()
        opt_mae_std = valid_results[f'{target}_optimized_mae'].std()
        opt_correlation_mean = valid_results[f'{target}_optimized_correlation'].mean()
        opt_directional_acc_mean = valid_results[f'{target}_optimized_directional_acc'].mean()
        
        baseline_r2_mean = valid_results[f'{target}_best_baseline_r2'].mean()
        baseline_r2_std = valid_results[f'{target}_best_baseline_r2'].std()
        improvement_mean = valid_results[f'{target}_improvement'].mean()
        improvement_std = valid_results[f'{target}_improvement'].std()
        improvement_median = valid_results[f'{target}_improvement'].median()
        
        # Count performance categories
        performance_counts = valid_results[f'{target}_performance'].value_counts()
        improvement_rate = (performance_counts.get('Major Improvement', 0) + 
                          performance_counts.get('Significant Improvement', 0) + 
                          performance_counts.get('Minor Improvement', 0)) / n_valid        # Statistical significance test
        improvements = valid_results[f'{target}_improvement'].values
        from scipy import stats as scipy_stats
        t_stat, p_value = scipy_stats.ttest_1samp(improvements, 0)
        
        # Additional statistical tests
        positive_improvements = sum(1 for imp in improvements if imp > 0)
        negative_improvements = sum(1 for imp in improvements if imp < 0)
        improvement_ratio = positive_improvements / n_valid if n_valid > 0 else 0
        
        # Effect size (Cohen's d)
        cohens_d = improvement_mean / improvement_std if improvement_std > 0 else 0
        
        # Confidence interval for improvement
        confidence_level = 0.95
        alpha = 1 - confidence_level
        t_critical = scipy_stats.t.ppf(1 - alpha/2, n_valid - 1)
        margin_error = t_critical * (improvement_std / np.sqrt(n_valid))
        ci_lower = improvement_mean - margin_error
        ci_upper = improvement_mean + margin_error
        
        # Quartile analysis
        improvements_q25 = np.percentile(improvements, 25)
        improvements_q75 = np.percentile(improvements, 75)
        
        # Performance vs game volume correlation
        games_correlation = valid_results[['total_games', f'{target}_improvement']].corr().iloc[0,1]
        summary_report.append(f"Valid evaluations: {n_valid}/{n_players} ({n_valid/n_players:.1%})")
        summary_report.append(f"")
        summary_report.append(f"OPTIMIZED MODEL PERFORMANCE:")
        summary_report.append(f"  Average R²: {opt_r2_mean:.4f} ± {opt_r2_std:.4f}")
        summary_report.append(f"  Median R²: {opt_r2_median:.4f}")
        summary_report.append(f"  R² Range: [{opt_r2_min:.4f}, {opt_r2_max:.4f}]")
        summary_report.append(f"  Average MAE: {opt_mae_mean:.3f} ± {opt_mae_std:.3f}")
        summary_report.append(f"  Average Correlation: {opt_correlation_mean:.4f}")
        summary_report.append(f"  Average Directional Accuracy: {opt_directional_acc_mean:.1%}")
        summary_report.append(f"")
        summary_report.append(f"BASELINE COMPARISON:")
        summary_report.append(f"  Average baseline R²: {baseline_r2_mean:.4f} ± {baseline_r2_std:.4f}")
        summary_report.append(f"  Average improvement: {improvement_mean:.4f} ± {improvement_std:.4f}")
        summary_report.append(f"  Median improvement: {improvement_median:.4f}")
        summary_report.append(f"  Improvement quartiles: Q1={improvements_q25:.4f}, Q3={improvements_q75:.4f}")
        summary_report.append(f"  Players improved: {positive_improvements}/{n_valid} ({improvement_ratio:.1%})")
        summary_report.append(f"  Players worse: {negative_improvements}/{n_valid} ({negative_improvements/n_valid:.1%})")
        summary_report.append(f"")
        summary_report.append(f"EFFECT SIZE & PRACTICAL SIGNIFICANCE:")
        summary_report.append(f"  Cohen's d (effect size): {cohens_d:.3f}")
        if abs(cohens_d) >= 0.8:
            summary_report.append(f"  Effect size interpretation: LARGE effect")
        elif abs(cohens_d) >= 0.5:
            summary_report.append(f"  Effect size interpretation: MEDIUM effect")
        elif abs(cohens_d) >= 0.2:
            summary_report.append(f"  Effect size interpretation: SMALL effect")
        else:
            summary_report.append(f"  Effect size interpretation: NEGLIGIBLE effect")
        summary_report.append(f"  95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
        summary_report.append(f"")
        summary_report.append(f"PERFORMANCE BREAKDOWN:")
        for category, count in performance_counts.items():
            summary_report.append(f"  {category}: {count} ({count/n_valid:.1%})")
        summary_report.append(f"")
        summary_report.append(f"CORRELATION ANALYSIS:")
        summary_report.append(f"  Games vs Improvement correlation: {games_correlation:.3f}")
        if abs(games_correlation) > 0.5:
            summary_report.append(f"  -> Model performance STRONGLY correlates with game volume")
        elif abs(games_correlation) > 0.3:
            summary_report.append(f"  -> Model performance MODERATELY correlates with game volume")
        else:
            summary_report.append(f"  -> Model performance shows WEAK correlation with game volume")
        summary_report.append(f"")
        summary_report.append(f"STATISTICAL SIGNIFICANCE:")
        summary_report.append(f"  t-statistic: {t_stat:.4f}")
        summary_report.append(f"  p-value: {p_value:.8f}")
        summary_report.append(f"  Degrees of freedom: {n_valid - 1}")
        if p_value < 0.001:
            summary_report.append("  Result: *** HIGHLY SIGNIFICANT *** improvement (p < 0.001)")
        elif p_value < 0.01:
            summary_report.append("  Result: ** VERY SIGNIFICANT ** improvement (p < 0.01)")
        elif p_value < 0.05:
            summary_report.append("  Result: * SIGNIFICANT * improvement (p < 0.05)")
        elif p_value < 0.10:
            summary_report.append("  Result: MARGINALLY significant improvement (p < 0.10)")
        else:
            summary_report.append("  Result: NOT statistically significant")
        summary_report.append("")
          # Store for overall analysis
        overall_stats[target] = {
            'n_valid': n_valid,
            'opt_r2_mean': opt_r2_mean,
            'opt_r2_std': opt_r2_std,
            'opt_mae_mean': opt_mae_mean,
            'improvement_mean': improvement_mean,
            'improvement_std': improvement_std,
            'improvement_ratio': improvement_ratio,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'correlation_mean': opt_correlation_mean,
            'directional_acc_mean': opt_directional_acc_mean,
            'games_correlation': games_correlation,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
      # Overall summary
    summary_report.append("OVERALL MODEL EFFECTIVENESS")
    summary_report.append("-" * 40)
    
    total_improvements = sum(stat_data['improvement_mean'] for stat_data in overall_stats.values())
    avg_improvement_rate = np.mean([stat_data['improvement_ratio'] for stat_data in overall_stats.values()])
    avg_r2 = np.mean([stat_data['opt_r2_mean'] for stat_data in overall_stats.values()])
    avg_correlation = np.mean([stat_data['correlation_mean'] for stat_data in overall_stats.values()])
    avg_directional_acc = np.mean([stat_data['directional_acc_mean'] for stat_data in overall_stats.values()])
    significant_targets = sum(1 for stat_data in overall_stats.values() if stat_data['p_value'] < 0.05)
    highly_significant_targets = sum(1 for stat_data in overall_stats.values() if stat_data['p_value'] < 0.01)
    large_effect_targets = sum(1 for stat_data in overall_stats.values() if abs(stat_data['cohens_d']) >= 0.5)
    
    summary_report.append(f"AGGREGATE PERFORMANCE METRICS:")
    summary_report.append(f"  Average R² across all targets: {avg_r2:.4f}")
    summary_report.append(f"  Average R² improvement: {total_improvements/len(TARGETS):.4f}")
    summary_report.append(f"  Average improvement success rate: {avg_improvement_rate:.1%}")
    summary_report.append(f"  Average correlation: {avg_correlation:.4f}")
    summary_report.append(f"  Average directional accuracy: {avg_directional_acc:.1%}")
    summary_report.append(f"")
    summary_report.append(f"STATISTICAL SIGNIFICANCE SUMMARY:")
    summary_report.append(f"  Significant targets (p < 0.05): {significant_targets}/{len(TARGETS)}")
    summary_report.append(f"  Highly significant targets (p < 0.01): {highly_significant_targets}/{len(TARGETS)}")
    summary_report.append(f"  Large effect size targets (|d| >= 0.5): {large_effect_targets}/{len(TARGETS)}")
    summary_report.append("")
      # Model recommendation
    if avg_improvement_rate > 0.6 and significant_targets >= 2 and avg_r2 > 0.02:
        summary_report.append("🟢 RECOMMENDATION: STRONG MODEL - DEPLOY FOR PRODUCTION")
        summary_report.append("   ✓ High success rate across multiple targets")
        summary_report.append("   ✓ Statistically significant improvements")
        summary_report.append("   ✓ Practical R² improvements")
        deployment_recommendation = "DEPLOY"
    elif avg_improvement_rate > 0.5 and (significant_targets >= 1 or avg_r2 > 0.01):
        summary_report.append("🟡 RECOMMENDATION: GOOD MODEL - CONDITIONAL DEPLOYMENT")
        summary_report.append("   ✓ Moderate success rate with some significant targets")
        summary_report.append("   ⚠ Monitor performance and consider target-specific tuning")
        deployment_recommendation = "CONDITIONAL"
    elif avg_improvement_rate > 0.3 and avg_r2 > 0:
        summary_report.append("🟠 RECOMMENDATION: MODEST MODEL - LIMITED USE")
        summary_report.append("   ⚠ Some improvement but inconsistent performance")
        summary_report.append("   ⚠ Consider ensemble with other models or further optimization")
        deployment_recommendation = "LIMITED"
    else:
        summary_report.append("🔴 RECOMMENDATION: WEAK MODEL - DO NOT DEPLOY")
        summary_report.append("   ❌ Insufficient improvement over baselines")
        summary_report.append("   ❌ Requires significant redesign before production use")
        deployment_recommendation = "DO_NOT_DEPLOY"
    summary_report.append("")
    summary_report.append("DETAILED STATISTICS BY TARGET:")
    summary_report.append("-" * 40)
    
    for target in TARGETS:
        if target in overall_stats:
            stat_data = overall_stats[target]
            summary_report.append(f"{target}:")
            summary_report.append(f"  R² = {stat_data['opt_r2_mean']:.4f} ± {stat_data['opt_r2_std']:.4f}")
            summary_report.append(f"  MAE = {stat_data['opt_mae_mean']:.3f}")
            summary_report.append(f"  Improvement = {stat_data['improvement_mean']:.4f} ± {stat_data['improvement_std']:.4f}")
            summary_report.append(f"  Success Rate = {stat_data['improvement_ratio']:.1%}")
            summary_report.append(f"  Correlation = {stat_data['correlation_mean']:.4f}")
            summary_report.append(f"  Directional Accuracy = {stat_data['directional_acc_mean']:.1%}")
            summary_report.append(f"  Effect Size (Cohen's d) = {stat_data['cohens_d']:.3f}")
            summary_report.append(f"  95% CI = [{stat_data['ci_lower']:.4f}, {stat_data['ci_upper']:.4f}]")
            summary_report.append(f"  p-value = {stat_data['p_value']:.6f}")
            summary_report.append("")
    
    # Add model diagnostics
    summary_report.append("MODEL DIAGNOSTICS & INSIGHTS:")
    summary_report.append("-" * 40)
    
    # Best and worst performing targets
    best_target = max(overall_stats.keys(), key=lambda t: overall_stats[t]['improvement_mean'])
    worst_target = min(overall_stats.keys(), key=lambda t: overall_stats[t]['improvement_mean'])
    
    summary_report.append(f"Best performing target: {best_target}")
    summary_report.append(f"  -> Improvement: {overall_stats[best_target]['improvement_mean']:.4f}")
    summary_report.append(f"  -> Success rate: {overall_stats[best_target]['improvement_ratio']:.1%}")
    summary_report.append(f"Worst performing target: {worst_target}")
    summary_report.append(f"  -> Improvement: {overall_stats[worst_target]['improvement_mean']:.4f}")
    summary_report.append(f"  -> Success rate: {overall_stats[worst_target]['improvement_ratio']:.1%}")
    summary_report.append("")
    
    # Consistency analysis
    improvement_consistency = np.std([stat_data['improvement_mean'] for stat_data in overall_stats.values()])
    summary_report.append(f"Cross-target consistency (std of improvements): {improvement_consistency:.4f}")
    if improvement_consistency < 0.01:
        summary_report.append("  -> Model shows HIGHLY CONSISTENT performance across targets")
    elif improvement_consistency < 0.02:
        summary_report.append("  -> Model shows GOOD consistency across targets")
    else:
        summary_report.append("  -> Model shows VARIABLE performance across targets")
    
    summary_report.append("")
    summary_report.append(f"FINAL DEPLOYMENT RECOMMENDATION: {deployment_recommendation}")
    
    # Save summary report
    with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_report))
    
    # Print key findings
    print('\n'.join(summary_report))
    print(f"\nFull summary saved to: {SUMMARY_FILE}")
    
    return overall_stats, deployment_recommendation

# --------------------------------------------------------------
# HELPER FUNCTIONS FOR RNN + TFT EVALUATION
# --------------------------------------------------------------

def calculate_exponential_smoothing(values, alpha=0.3):
    """Calculate exponential smoothing prediction"""
    if len(values) == 0:
        return 0
    
    smoothed = values[0]
    for val in values[1:]:
        smoothed = alpha * val + (1 - alpha) * smoothed
    return smoothed

def calculate_linear_trend(values):
    """Calculate linear trend prediction"""
    if len(values) < 3:
        return np.mean(values) if len(values) > 0 else 0
    
    try:
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)
        return slope * len(values) + intercept
    except:
        return np.mean(values)

def create_default_team_stats():
    """Create default team stats"""
    teams = ["ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW", 
             "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK", 
             "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS"]
    
    data = []
    for team in teams:
        data.append({
            'TEAM_ABBREVIATION': team,
            'OFF_RATING': 113.5,
            'DEF_RATING': 113.5,
            'NET_RATING': 0.0,
            'PACE': 100.0,
            'AST_RATIO': 19.0,
            'REB_PCT': 0.5
        })
    
    return pd.DataFrame(data)

def get_team_stats_for_date(game_date, team_stats_df):
    """Get team stats for a specific date (no data leakage)"""
    if team_stats_df.empty:
        return create_default_team_stats()
    
    try:
        if isinstance(game_date, str):
            game_date = pd.to_datetime(game_date)
        
        # Find appropriate month data
        available_data = team_stats_df[
            (team_stats_df['DATE_FROM'] <= game_date) &
            (team_stats_df['DATE_TO'] >= game_date)
        ]
        
        if available_data.empty:
            # Get most recent data before game date
            available_data = team_stats_df[team_stats_df['DATE_TO'] < game_date]
            if available_data.empty:
                return create_default_team_stats()
            available_data = available_data.sort_values('DATE_TO').groupby('TEAM_ABBREVIATION').tail(1)
        
        return available_data
    except:
        return create_default_team_stats()

# --------------------------------------------------------------
# DYNAMIC TEAM STATS (NO DATA LEAKAGE)
# --------------------------------------------------------------

def get_season_from_date(game_date):
    """Determine NBA season from game date"""
    try:
        if isinstance(game_date, str):
            game_date = pd.to_datetime(game_date)
        
        year = game_date.year
        month = game_date.month
        
        # NBA season runs from October to June
        if month >= 10:  # October, November, December
            season_start_year = year
        else:  # January through September
            season_start_year = year - 1
        
        return f"{season_start_year}-{str(season_start_year + 1)[-2:]}"
        
    except Exception as e:
        print(f"Error determining season from date {game_date}: {e}")
        return "2023-24"  # Default fallback

def get_team_stats_for_date(game_date, team_stats_df):
    """Get team stats for a specific date (no data leakage)"""
    if team_stats_df.empty:
        return create_default_team_stats()
    
    try:
        if isinstance(game_date, str):
            game_date = pd.to_datetime(game_date)
        
        # Find appropriate month data
        available_data = team_stats_df[
            (team_stats_df['DATE_FROM'] <= game_date) &
            (team_stats_df['DATE_TO'] >= game_date)
        ]
        
        if available_data.empty:
            # Get most recent data before game date
            available_data = team_stats_df[team_stats_df['DATE_TO'] < game_date]
            if available_data.empty:
                return create_default_team_stats()
            available_data = available_data.sort_values('DATE_TO').groupby('TEAM_ABBREVIATION').tail(1)
        
        return available_data
    except:
        return create_default_team_stats()

def create_default_team_stats():
    """Create default team stats"""
    teams = ["ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW", 
             "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK", 
             "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS"]
    
    data = []
    for team in teams:
        data.append({
            'TEAM_ABBREVIATION': team,
            'OFF_RATING': 113.5,
            'DEF_RATING': 113.5,
            'NET_RATING': 0.0,
            'PACE': 100.0,
            'AST_RATIO': 19.0,
            'REB_PCT': 0.5
        })
    
    return pd.DataFrame(data)

def test_single_player_rnn_tft(train_data, test_data, team_stats, target, player_name):
    """
    Test RNN + TFT model on a single player and compare to baselines
    """
    try:
        rnn_tft_predictions = []
        baseline_predictions = []
        actuals = []
        
        # Test each game in the test set
        for idx in range(len(test_data)):
            current_test_game = test_data.iloc[idx]
            game_date = current_test_game['GAME_DATE']
            
            # Combine training data with test data up to current game
            current_data = pd.concat([train_data, test_data.iloc[:idx]]).reset_index(drop=True)
            
            if len(current_data) < MIN_GAMES_FOR_PREDICTION:
                continue
            
            # Get team stats for this date
            current_team_stats = get_team_stats_for_date(game_date, team_stats)
            
            # Prepare RNN + TFT features
            features = prepare_rnn_tft_features(
                current_data, target, current_team_stats, current_test_game
            )
            
            # Get target history
            target_history = current_data[target].values
            
            # RNN + TFT prediction
            rnn_tft_pred = calculate_rnn_ensemble_prediction(features, target_history, target)
            
            # Simple baseline prediction (5-game moving average)
            baseline_pred = np.mean(target_history[-5:]) if len(target_history) >= 5 else np.mean(target_history)
            
            # Store results
            actual = current_test_game[target]
            rnn_tft_predictions.append(rnn_tft_pred)
            baseline_predictions.append(baseline_pred)
            actuals.append(actual)
        
        if len(rnn_tft_predictions) > 0:
            # Calculate errors
            rnn_tft_errors = [abs(p - a) for p, a in zip(rnn_tft_predictions, actuals)]
            baseline_errors = [abs(p - a) for p, a in zip(baseline_predictions, actuals)]
            
            # Calculate metrics
            rnn_tft_mae = np.mean(rnn_tft_errors)
            baseline_mae = np.mean(baseline_errors)
            improvement = (baseline_mae - rnn_tft_mae) / baseline_mae * 100 if baseline_mae > 0 else 0
            
            print(f"   RNN+TFT MAE: {rnn_tft_mae:.3f}")
            print(f"   Baseline MAE: {baseline_mae:.3f}")
            print(f"   Improvement: {improvement:+.1f}%")
            print(f"   Games tested: {len(rnn_tft_predictions)}")
            
            return {
                'player_name': player_name,
                'target': target,
                'games_tested': len(rnn_tft_predictions),
                'rnn_tft_mae': rnn_tft_mae,
                'baseline_mae': baseline_mae,
                'improvement_pct': improvement,
                'rnn_tft_errors': rnn_tft_errors,
                'baseline_errors': baseline_errors
            }
        else:
            return None
            
    except Exception as e:
        print(f"   ❌ Error testing {player_name}: {e}")
        return None

def test_single_player_comprehensive(train_data, test_data, team_stats, target, player_name):
    """
    Test RNN + TFT model on a single player with comprehensive baseline comparison
    """
    try:
        rnn_tft_predictions = []
        rnn_tft_errors = []
        baseline_wins = {}
        rnn_tft_wins = 0
        actuals = []
        games_tested = 0
        
        # Test each game in the test set
        for idx in range(len(test_data)):
            current_test_game = test_data.iloc[idx]
            game_date = current_test_game['GAME_DATE']
            
            # Combine training data with test data up to current game
            current_data = pd.concat([train_data, test_data.iloc[:idx]]).reset_index(drop=True)
            
            if len(current_data) < MIN_GAMES_FOR_PREDICTION:
                continue
            
            # Get team stats for this date
            current_team_stats = get_team_stats_for_date(game_date, team_stats)
            
            try:
                # Prepare RNN + TFT features
                features = prepare_rnn_tft_features(
                    current_data, target, current_team_stats, current_test_game
                )
                
                # Get target history
                target_history = current_data[target].values
                
                # RNN + TFT prediction
                rnn_tft_pred = calculate_rnn_ensemble_prediction(features, target_history, target)
                
                # Get actual value
                actual = current_test_game[target]
                
                # Calculate RNN+TFT error
                rnn_tft_error = abs(rnn_tft_pred - actual)
                
                # Calculate all baseline predictions and their errors
                all_baselines = calculate_multiple_baselines(target_history, target)
                baseline_errors = {}
                for baseline_name, baseline_pred in all_baselines.items():
                    if not np.isnan(baseline_pred) and not np.isinf(baseline_pred):
                        baseline_errors[baseline_name] = abs(baseline_pred - actual)
                
                # Find the most accurate prediction (including RNN+TFT)
                all_errors = {'RNN+TFT': rnn_tft_error}
                all_errors.update(baseline_errors)
                
                if all_errors:
                    most_accurate = min(all_errors.items(), key=lambda x: x[1])[0]
                    
                    if most_accurate == 'RNN+TFT':
                        rnn_tft_wins += 1
                    else:
                        baseline_wins[most_accurate] = baseline_wins.get(most_accurate, 0) + 1
                
                # Store results
                rnn_tft_predictions.append(rnn_tft_pred)
                rnn_tft_errors.append(rnn_tft_error)
                actuals.append(actual)
                games_tested += 1
                
            except Exception as inner_e:
                print(f"     ⚠️  Error on game {idx+1}: {inner_e}")
                continue
        
        if games_tested > 0:
            # Calculate overall metrics
            rnn_tft_mae = np.mean(rnn_tft_errors)
            
            # Calculate comparison with best average baseline
            avg_baseline_error = []
            target_history_full = train_data[target].values
            for actual in actuals:
                # Use 5-game moving average as representative baseline
                baseline_pred = np.mean(target_history_full[-5:]) if len(target_history_full) >= 5 else np.mean(target_history_full)
                avg_baseline_error.append(abs(baseline_pred - actual))
            
            baseline_mae = np.mean(avg_baseline_error) if avg_baseline_error else rnn_tft_mae
            improvement = (baseline_mae - rnn_tft_mae) / baseline_mae * 100 if baseline_mae > 0 else 0
            
            print(f"   RNN+TFT MAE: {rnn_tft_mae:.3f}")
            print(f"   Baseline MAE: {baseline_mae:.3f}")
            print(f"   Improvement: {improvement:+.1f}%")
            print(f"   Games tested: {games_tested}")
            print(f"   RNN+TFT wins: {rnn_tft_wins} ({rnn_tft_wins/games_tested*100:.1f}%)")
            
            # Show baseline wins
            if baseline_wins:
                print(f"   Baseline wins: {dict(sorted(baseline_wins.items(), key=lambda x: x[1], reverse=True))}")
            
            return {
                'player_name': player_name,
                'target': target,
                'games_tested': games_tested,
                'rnn_tft_mae': rnn_tft_mae,
                'baseline_mae': baseline_mae,
                'improvement_pct': improvement,
                'baseline_wins': baseline_wins,
                'rnn_tft_wins': rnn_tft_wins,
                'rnn_tft_errors': rnn_tft_errors
            }
        else:
            return None
            
    except Exception as e:
        print(f"   ❌ Error testing {player_name}: {e}")
        return None

def calculate_multiple_baselines(target_history, target):
    """
    Calculate multiple baseline predictions for comparison
    """
    baselines = {}
    
    if len(target_history) == 0:
        return baselines
    
    # 1. Simple Moving Averages
    for window in [3, 5, 7, 10]:
        if len(target_history) >= window:
            baselines[f'MA_{window}'] = np.mean(target_history[-window:])
    
    # 2. Weighted Moving Average (recent games weighted more)
    if len(target_history) >= 5:
        weights = np.array([0.1, 0.15, 0.2, 0.25, 0.3])  # Recent games weighted more
        recent_5 = target_history[-5:]
        baselines['WMA_5'] = np.average(recent_5, weights=weights)
    
    # 3. Exponential Moving Average
    if len(target_history) >= 3:
        alpha = 0.3
        ema = target_history[0]
        for val in target_history[1:]:
            ema = alpha * val + (1 - alpha) * ema
        baselines['EMA'] = ema
    
    # 4. Median (robust to outliers)
    for window in [5, 7, 10]:
        if len(target_history) >= window:
            baselines[f'Median_{window}'] = np.median(target_history[-window:])
    
    # 5. Linear Trend
    if len(target_history) >= 5:
        x = np.arange(len(target_history[-7:]))
        y = target_history[-7:]
        try:
            slope, intercept = np.polyfit(x, y, 1)
            trend_pred = slope * len(x) + intercept
            baselines['Linear_Trend'] = max(0, trend_pred)  # Ensure non-negative
        except:
            baselines['Linear_Trend'] = np.mean(target_history[-5:])
    
    # 6. Season Average (if enough data)
    if len(target_history) >= 15:
        baselines['Season_Avg'] = np.mean(target_history)
    
    # 7. Last Game (naive baseline)
    baselines['Last_Game'] = target_history[-1]
    
    # 8. Best 5 Game Average
    if len(target_history) >= 10:
        sorted_recent = sorted(target_history[-10:], reverse=True)
        baselines['Best_5_Avg'] = np.mean(sorted_recent[:5])
    
    return baselines

def find_best_baseline_for_comparison(target_history, actual_next, target):
    """
    Find which baseline would have performed best for this prediction
    """
    baselines = calculate_multiple_baselines(target_history, target)
    
    if not baselines:
        return None, None
    
    # Calculate errors for each baseline
    baseline_errors = {}
    for name, pred in baselines.items():
        if not np.isnan(pred) and not np.isinf(pred):
            baseline_errors[name] = abs(pred - actual_next)
    
    if not baseline_errors:
        return None, None
    
    # Find best baseline
    best_baseline = min(baseline_errors.items(), key=lambda x: x[1])
    return best_baseline[0], baselines

# --------------------------------------------------------------
# COMPREHENSIVE BASELINE ANALYSIS
# --------------------------------------------------------------

def print_comprehensive_baseline_analysis(all_results):
    """
    Print comprehensive analysis including baseline win percentages and RNN+TFT wins
    """
    print(f"\n{'='*70}")
    print("COMPREHENSIVE BASELINE COMPARISON ANALYSIS")
    print(f"{'='*70}")
    
    # Aggregate baseline wins and RNN+TFT wins across all players and targets
    total_baseline_wins = {}
    total_rnn_tft_wins = 0
    total_games = 0
    total_players = len(all_results)
    
    for result in all_results:
        if 'baseline_wins' in result:
            for baseline_name, wins in result['baseline_wins'].items():
                total_baseline_wins[baseline_name] = total_baseline_wins.get(baseline_name, 0) + wins
                total_games += wins
        
        if 'rnn_tft_wins' in result:
            total_rnn_tft_wins += result['rnn_tft_wins']
            total_games += result['rnn_tft_wins']
    
    print(f"\n📊 PREDICTION ACCURACY WIN RATES:")
    print(f"   (Which method predicted most accurately for each game)")
    print("-" * 50)
    
    if total_games > 0:
        # Create combined results including RNN+TFT
        all_wins = {'RNN+TFT': total_rnn_tft_wins}
        all_wins.update(total_baseline_wins)
        
        # Sort by win count
        sorted_methods = sorted(all_wins.items(), key=lambda x: x[1], reverse=True)
        
        for i, (method_name, wins) in enumerate(sorted_methods, 1):
            percentage = (wins / total_games * 100) if total_games > 0 else 0
            if method_name == 'RNN+TFT':
                print(f"   {i:2d}. 🤖 {method_name:<13} {wins:4d} wins ({percentage:5.1f}%)")
            else:
                print(f"   {i:2d}. {method_name:<15} {wins:4d} wins ({percentage:5.1f}%)")
        
        print(f"\n   Total game predictions analyzed: {total_games}")
        
        # Best performing method overall
        if sorted_methods:
            best_method, best_wins = sorted_methods[0]
            best_percentage = (best_wins / total_games * 100) if total_games > 0 else 0
            if best_method == 'RNN+TFT':
                print(f"\n🏆 MOST ACCURATE METHOD: 🤖 {best_method}")
            else:
                print(f"\n🏆 MOST ACCURATE METHOD: {best_method}")
            print(f"   Won {best_wins} out of {total_games} predictions ({best_percentage:.1f}%)")
        
        # RNN+TFT specific performance
        rnn_tft_percentage = (total_rnn_tft_wins / total_games * 100) if total_games > 0 else 0
        print(f"\n🤖 RNN+TFT ACCURACY PERFORMANCE:")
        print(f"   Games won: {total_rnn_tft_wins} out of {total_games} ({rnn_tft_percentage:.1f}%)")
        
        if rnn_tft_percentage > 50:
            print(f"   🎉 RNN+TFT is the most accurate method overall!")
        elif rnn_tft_percentage > 30:
            print(f"   ✅ RNN+TFT shows strong competitive performance")
        elif rnn_tft_percentage > 15:
            print(f"   ⚠️  RNN+TFT shows moderate competitive performance")
        else:
            print(f"   ❌ RNN+TFT needs significant improvement")
    
    # Analysis by target
    print(f"\n📈 PERFORMANCE BY TARGET:")
    print("-" * 30)
    
    targets_analysis = {}
    for result in all_results:
        target = result['target']
        if target not in targets_analysis:
            targets_analysis[target] = {
                'players': 0,
                'rnn_tft_wins': 0,
                'total_comparisons': 0,
                'avg_improvement': [],
                'total_games': 0
            }
        
        targets_analysis[target]['players'] += 1
        if 'improvement_pct' in result:
            targets_analysis[target]['avg_improvement'].append(result['improvement_pct'])
            if result['improvement_pct'] > 0:
                targets_analysis[target]['rnn_tft_wins'] += 1
            targets_analysis[target]['total_comparisons'] += 1
        
        if 'games_tested' in result:
            targets_analysis[target]['total_games'] += result['games_tested']
    
    for target, analysis in targets_analysis.items():
        avg_improvement = np.mean(analysis['avg_improvement']) if analysis['avg_improvement'] else 0
        win_rate = (analysis['rnn_tft_wins'] / analysis['total_comparisons'] * 100) if analysis['total_comparisons'] > 0 else 0
        
        print(f"\n   {target}:")
        print(f"     Players tested: {analysis['players']}")
        print(f"     Total games: {analysis['total_games']}")
        print(f"     RNN+TFT player win rate: {win_rate:.1f}% ({analysis['rnn_tft_wins']}/{analysis['total_comparisons']})")
        print(f"     Avg improvement: {avg_improvement:+.1f}%")
    
    # Overall RNN+TFT performance
    print(f"\n🤖 RNN + TFT OVERALL PERFORMANCE:")
    print("-" * 40)
    
    total_improvements = [r['improvement_pct'] for r in all_results if 'improvement_pct' in r and not np.isnan(r['improvement_pct'])]
    if total_improvements:
        avg_improvement = np.mean(total_improvements)
        positive_improvements = sum(1 for imp in total_improvements if imp > 0)
        total_tests = len(total_improvements)
        success_rate = (positive_improvements / total_tests * 100) if total_tests > 0 else 0
        
        print(f"   Average improvement: {avg_improvement:+.1f}%")
        print(f"   Success rate: {success_rate:.1f}% ({positive_improvements}/{total_tests} players)")
        print(f"   Best improvement: {max(total_improvements):+.1f}%")
        print(f"   Worst performance: {min(total_improvements):+.1f}%")
        
        # Performance categories
        excellent = sum(1 for imp in total_improvements if imp > 10)
        good = sum(1 for imp in total_improvements if 5 < imp <= 10)
        moderate = sum(1 for imp in total_improvements if 0 < imp <= 5)
        poor = sum(1 for imp in total_improvements if imp <= 0)
        
        print(f"\n   Performance Breakdown:")
        print(f"     Excellent (>10% improvement): {excellent} players ({excellent/total_tests*100:.1f}%)")
        print(f"     Good (5-10% improvement): {good} players ({good/total_tests*100:.1f}%)")
        print(f"     Moderate (0-5% improvement): {moderate} players ({moderate/total_tests*100:.1f}%)")
        print(f"     Poor (≤0% improvement): {poor} players ({poor/total_tests*100:.1f}%)")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print("-" * 20)
    
    if total_games > 0 and sorted_methods:
        best_method = sorted_methods[0][0]
        if best_method == 'RNN+TFT':
            print(f"   • 🎉 RNN+TFT is performing excellently - continue current approach")
            print(f"   • Consider deploying model for production use")
        else:
            print(f"   • Consider incorporating {best_method} features more heavily into RNN+TFT")
            print(f"   • {best_method} shows most consistent accuracy across players")
            if rnn_tft_percentage > 25:
                print(f"   • RNN+TFT is competitive but needs optimization")
            else:
                print(f"   • RNN+TFT needs significant architectural improvements")
    
    if total_improvements:
        if avg_improvement > 5:
            print(f"   • RNN+TFT shows strong overall performance ({avg_improvement:+.1f}% avg improvement)")
            print(f"   • Model is ready for production use")
        elif avg_improvement > 0:
            print(f"   • RNN+TFT shows modest improvement ({avg_improvement:+.1f}% avg)")
            print(f"   • Consider feature engineering optimization")
        else:
            print(f"   • RNN+TFT needs improvement (avg: {avg_improvement:+.1f}%)")
            print(f"   • Recommend sticking with best baseline methods")
    
    print(f"\n{'='*70}")

# --------------------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------------------

if __name__ == "__main__":
    try:
        print("\n🚀 STARTING MULTI-PLAYER RNN + TFT EVALUATION")
        print("-" * 50)
        
        # Load data and select players
        df, team_stats, selected_players = load_data_and_select_players()
        
        if len(selected_players) == 0:
            print("❌ No eligible players found. Exiting.")
            exit(1)
        
        # Run comprehensive evaluation
        all_results = []
        
        for target in TARGETS:
            print(f"\n{'='*60}")
            print(f"EVALUATING {target}")
            print(f"{'='*60}")
            
            # Test each selected player
            for idx, player_row in selected_players.iterrows():
                player_id = player_row['PLAYER_ID']
                player_name = player_row['PLAYER_NAME']
                
                print(f"\n🏀 Testing {player_name} for {target}")
                print("-" * 40)
                
                # Get player's game data
                player_data = df[df['PLAYER_ID'] == player_id].copy()
                player_data = player_data.sort_values('GAME_DATE').reset_index(drop=True)
                
                if len(player_data) < MIN_GAMES_FOR_PREDICTION:
                    print(f"   ⚠️  Insufficient data ({len(player_data)} games)")
                    continue
                
                # Split into train/test
                available_seasons = sorted(player_data['SEASON_YEAR'].unique())
                if len(available_seasons) < 2:
                    print(f"   ⚠️  Need at least 2 seasons of data")
                    continue
                
                test_season = available_seasons[-1]
                train_data = player_data[player_data['SEASON_YEAR'] != test_season]
                test_data = player_data[player_data['SEASON_YEAR'] == test_season]
                
                if len(test_data) < 5:
                    print(f"   ⚠️  Insufficient test data ({len(test_data)} games)")
                    continue
                
                print(f"   📊 Train: {len(train_data)} games, Test: {len(test_data)} games")
                
                # Test the player with comprehensive baseline comparison
                player_result = test_single_player_comprehensive(
                    train_data, test_data, team_stats, target, player_name
                )
                
                if player_result:
                    all_results.append(player_result)
                    print(f"   ✅ {player_name} completed successfully")
                else:
                    print(f"   ❌ {player_name} failed")
        
        # Print comprehensive baseline analysis
        if all_results:
            print_comprehensive_baseline_analysis(all_results)
            
            # Save detailed results
            results_df = pd.DataFrame(all_results)
            results_df.to_csv(RESULTS_FILE, index=False)
            print(f"\n💾 Detailed results saved to: {RESULTS_FILE}")
        else:
            print("❌ No results to analyze")
            
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
