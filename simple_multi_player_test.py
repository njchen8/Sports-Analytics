#!/usr/bin/env python3
# simple_multi_player_test.py - Quick Multi-Player Model Test
# Testing the optimized research model on multiple players

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from tqdm import tqdm
from datetime import datetime
warnings.filterwarnings('ignore')

print("SIMPLE MULTI-PLAYER MODEL TEST")
print("="*50)
print(f"Started at: {datetime.now()}")

try:
    # Import our functions
    from optimized_research_predictor import (
        prepare_comprehensive_features,
        calculate_ensemble_prediction,
        calculate_optimized_weighted_prediction,
        create_enhanced_baselines,
        calculate_prediction_metrics,
        extract_opponent_from_matchup,
        TARGETS, ENSEMBLE_METHODS, MIN_GAMES_FOR_PREDICTION
    )
    print("✓ Model functions imported successfully")
    
    # Load data
    df = pd.read_csv("nba_players_game_logs_2018_25_clean.csv", parse_dates=["GAME_DATE"])
    team_stats = pd.read_csv("team_stats.csv") if Path("team_stats.csv").exists() else pd.DataFrame()
    print(f"✓ Data loaded: {len(df):,} games, {len(team_stats)} teams")
    
    # Find eligible players (minimum 30 games)
    player_stats = df.groupby(['PLAYER_ID', 'PLAYER_NAME']).agg({
        'GAME_DATE': 'count',
        'SEASON_YEAR': 'nunique'
    }).rename(columns={'GAME_DATE': 'total_games'}).reset_index()
    
    eligible_players = player_stats[
        (player_stats['total_games'] >= 30) & 
        (player_stats['SEASON_YEAR'] >= 2)  # Need multiple seasons
    ].head(20)  # Test on first 20 eligible players
    
    print(f"✓ Testing {len(eligible_players)} players")
    
    # Test each player
    results = []
    successful_tests = 0
    
    for idx, row in tqdm(eligible_players.iterrows(), total=len(eligible_players), desc="Testing players"):
        try:
            player_id = row['PLAYER_ID']
            player_name = row['PLAYER_NAME']
            
            # Get player data
            player_data = df[df["PLAYER_ID"] == player_id].copy()
            player_data = player_data.sort_values("GAME_DATE").reset_index(drop=True)
            
            if len(player_data) < 30:
                continue
                
            # Add opponent team if missing
            if 'OPP_TEAM' not in player_data.columns:
                player_data['OPP_TEAM'] = player_data['MATCHUP'].apply(extract_opponent_from_matchup)
            
            # Simple train/test split: last 10 games for testing
            train_data = player_data.iloc[:-10].copy()
            test_data = player_data.iloc[-10:].copy()
            
            if len(train_data) < MIN_GAMES_FOR_PREDICTION:
                continue
            
            # Test each target
            for target in TARGETS:
                try:
                    # Simple single prediction test
                    current_data = train_data.copy()
                    test_game = test_data.iloc[0]
                    actual_value = test_game[target]
                    
                    # Generate features and prediction
                    features = prepare_comprehensive_features(current_data, target, len(current_data), team_stats)
                    target_history = current_data[target].values
                    
                    if ENSEMBLE_METHODS:
                        prediction = calculate_ensemble_prediction(features, target_history, target)
                    else:
                        prediction = calculate_optimized_weighted_prediction(features, target_history, target)
                    
                    # Calculate error
                    error = abs(prediction - actual_value)
                    relative_error = error / max(actual_value, 1)
                    
                    # Generate baseline (simple moving average)
                    baseline_pred = np.mean(current_data[target].tail(5))
                    baseline_error = abs(baseline_pred - actual_value)
                    
                    # Store result
                    results.append({
                        'Player': player_name,
                        'Target': target,
                        'Games': len(current_data),
                        'Actual': actual_value,
                        'Prediction': prediction,
                        'Error': error,
                        'Relative_Error': relative_error,
                        'Baseline_Pred': baseline_pred,
                        'Baseline_Error': baseline_error,
                        'Improvement': baseline_error - error,
                        'Better_Than_Baseline': baseline_error > error
                    })
                    
                except Exception as e:
                    print(f"Error with {player_name} {target}: {e}")
                    continue
            
            successful_tests += 1
            
        except Exception as e:
            print(f"Error with player {row.get('PLAYER_NAME', 'Unknown')}: {e}")
            continue
    
    # Analyze results
    if len(results) > 0:
        results_df = pd.DataFrame(results)
        
        print(f"\nRESULTS SUMMARY")
        print("-"*40)
        print(f"Successfully tested: {successful_tests} players")
        print(f"Total predictions made: {len(results_df)}")
        
        # Overall performance by target
        for target in TARGETS:
            target_results = results_df[results_df['Target'] == target]
            if len(target_results) > 0:
                wins = target_results['Better_Than_Baseline'].sum()
                total = len(target_results)
                avg_error = target_results['Error'].mean()
                avg_improvement = target_results['Improvement'].mean()
                
                print(f"\n{target}:")
                print(f"   Tests: {total}")
                print(f"   Wins vs Baseline: {wins}/{total} ({wins/total:.1%})")
                print(f"   Average Error: {avg_error:.2f}")
                print(f"   Average Improvement: {avg_improvement:+.2f}")
        
        # Save results
        results_df.to_csv("simple_multi_player_results.csv", index=False)
        print(f"\nResults saved to: simple_multi_player_results.csv")
        
        # Overall conclusion
        total_wins = results_df['Better_Than_Baseline'].sum()
        total_tests = len(results_df)
        win_rate = total_wins / total_tests
        
        print(f"\nOVERALL CONCLUSION:")
        print(f"Win Rate: {total_wins}/{total_tests} ({win_rate:.1%})")
        
        if win_rate > 0.6:
            print("✅ Model shows strong performance - suitable for deployment")
        elif win_rate > 0.5:
            print("⚠️ Model shows moderate performance - needs refinement")
        else:
            print("❌ Model underperforms - requires significant improvement")
    else:
        print("❌ No successful tests completed")

except Exception as e:
    print(f"❌ Critical error: {e}")
    import traceback
    traceback.print_exc()

print(f"\nCompleted at: {datetime.now()}")
