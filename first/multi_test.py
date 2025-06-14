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

# Import our optimized model functions
from optimized_research_predictor import (
    prepare_comprehensive_features,
    calculate_ensemble_prediction,
    calculate_optimized_weighted_prediction,
    create_enhanced_baselines,
    calculate_prediction_metrics,
    extract_opponent_from_matchup
)

# --------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------
CSV_PLAYERS = Path("nba_players_game_logs_2018_25_clean.csv")
CSV_TEAMS   = Path("team_stats.csv")
RESULTS_FILE = Path("multi_player_results.csv")
SUMMARY_FILE = Path("model_evaluation_summary.txt")

TARGETS = ["PTS", "REB", "AST"]

# Test parameters
MIN_GAMES_FOR_PLAYER = 50      # Minimum games a player must have
MIN_GAMES_FOR_PREDICTION = 12  # Minimum games before making predictions
MAX_PLAYERS_TO_TEST = 50       # Limit for reasonable runtime
ENSEMBLE_METHODS = True        # Use ensemble methods
ENSEMBLE_WEIGHTS = [0.4, 0.3, 0.2, 0.1]

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
    
    # Filter players with sufficient games
    eligible_players = player_stats[player_stats['total_games'] >= MIN_GAMES_FOR_PLAYER].copy()
    print(f"Players with {MIN_GAMES_FOR_PLAYER}+ games: {len(eligible_players)}")
    
    # Select diverse set of players across different performance levels
    # Sort by total games and select top players + random sample for diversity
    top_players = eligible_players.nlargest(25, 'total_games')
    
    # Add some medium-volume players for diversity
    medium_players = eligible_players[
        (eligible_players['total_games'] >= 100) & 
        (eligible_players['total_games'] < eligible_players['total_games'].quantile(0.8))
    ].sample(min(15, len(eligible_players) - 25), random_state=42)
    
    # Combine and limit
    selected_players = pd.concat([top_players, medium_players]).drop_duplicates()
    selected_players = selected_players.head(MAX_PLAYERS_TO_TEST)
    
    print(f"Selected {len(selected_players)} players for testing")
    print(f"Games per player range: {selected_players['total_games'].min():.0f} - {selected_players['total_games'].max():.0f}")
    print(f"Average PTS range: {selected_players['PTS_mean'].min():.1f} - {selected_players['PTS_mean'].max():.1f}")
    
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
            optimized_predictions = []
            baseline_predictions = {
                "MA_5": [], "MA_10": [], "EWMA": [], 
                "Linear_Trend": [], "Weighted_Avg": []
            }
            actual_values = []
            
            # Start with training data
            current_data = train_data.copy()
            
            # Online learning loop
            for test_idx in range(len(test_data)):
                current_game = test_data.iloc[test_idx]
                actual_value = current_game[target]
                actual_values.append(actual_value)
                
                # Generate baseline predictions
                baselines = create_enhanced_baselines(current_data, target, len(current_data))
                for model_name in baseline_predictions.keys():
                    baseline_predictions[model_name].append(baselines.get(model_name, actual_value))
                
                # Generate optimized prediction
                features = prepare_comprehensive_features(current_data, target, len(current_data), team_stats)
                target_history = current_data[target].values
                
                if ENSEMBLE_METHODS and len(features) > 0:
                    optimized_pred = calculate_ensemble_prediction(features, target_history, target)
                else:
                    optimized_pred = calculate_optimized_weighted_prediction(features, target_history, target)
                
                optimized_predictions.append(optimized_pred)
                
                # Online learning: add game to training data
                current_data = pd.concat([current_data, current_game.to_frame().T], ignore_index=True)
                current_data = current_data.sort_values("GAME_DATE").reset_index(drop=True)
            
            # Calculate metrics
            if len(actual_values) > 0:
                # Optimized model metrics
                opt_metrics = calculate_prediction_metrics(actual_values, optimized_predictions)
                
                # Baseline metrics
                baseline_metrics = {}
                best_baseline_r2 = -np.inf
                best_baseline_name = ""
                
                for model_name, predictions in baseline_predictions.items():
                    if len(predictions) == len(actual_values):
                        metrics = calculate_prediction_metrics(actual_values, predictions)
                        baseline_metrics[model_name] = metrics
                        
                        if metrics['R2'] > best_baseline_r2:
                            best_baseline_r2 = metrics['R2']
                            best_baseline_name = model_name
                
                # Store results
                player_results[f'{target}_optimized_r2'] = opt_metrics['R2']
                player_results[f'{target}_optimized_mae'] = opt_metrics['MAE']
                player_results[f'{target}_optimized_correlation'] = opt_metrics['Correlation']
                player_results[f'{target}_optimized_directional_acc'] = opt_metrics['Directional_Accuracy']
                player_results[f'{target}_optimized_bias'] = opt_metrics['Bias']
                
                player_results[f'{target}_best_baseline'] = best_baseline_name
                player_results[f'{target}_best_baseline_r2'] = best_baseline_r2
                player_results[f'{target}_improvement'] = opt_metrics['R2'] - best_baseline_r2
                
                # Performance categorization
                if opt_metrics['R2'] > best_baseline_r2:
                    if opt_metrics['R2'] - best_baseline_r2 > 0.1:
                        player_results[f'{target}_performance'] = 'Major Improvement'
                    elif opt_metrics['R2'] - best_baseline_r2 > 0.05:
                        player_results[f'{target}_performance'] = 'Significant Improvement'
                    else:
                        player_results[f'{target}_performance'] = 'Minor Improvement'
                else:
                    player_results[f'{target}_performance'] = 'Baseline Better'
        
        except Exception as e:
            print(f"Error evaluating {player_name} for {target}: {e}")
            player_results[f'{target}_optimized_r2'] = np.nan
            player_results[f'{target}_improvement'] = np.nan
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
# EXECUTION
# --------------------------------------------------------------

if __name__ == "__main__":
    try:
        print("Starting multi-player evaluation...")
        
        # Run evaluation
        results_df = run_multi_player_evaluation()
        if len(results_df) > 0:
            # Analyze and report
            overall_stats, deployment_rec = analyze_and_report_results(results_df)
            
            print(f"\nEVALUATION COMPLETE!")
            print(f"Evaluated {len(results_df)} players successfully")
            print(f"Results saved to: {RESULTS_FILE}")
            print(f"Summary saved to: {SUMMARY_FILE}")
            print(f"Final recommendation: {deployment_rec}")
        else:
            print("No players could be evaluated successfully.")
            
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()