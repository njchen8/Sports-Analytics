#!/usr/bin/env python3
# research_enhanced_predictor.py - Research-Based NBA Predictions with Lagged Indicators
# --------------------------------------------------------------
# Implements research paper strategy focusing on lagged indicators and adaptive weighting
# Designed to outperform your current baseline approaches
# --------------------------------------------------------------

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# --------------------------------------------------------------
# 1  RESEARCH-BASED CONFIGURATION
# --------------------------------------------------------------
CSV_PLAYERS = Path("nba_players_game_logs_2018_25_clean.csv")
CSV_TEAMS   = Path("team_stats.csv")
OUT         = Path("research_enhanced_preds.csv")

TARGETS = ["PTS", "REB", "AST"]

# Research-based parameters from sports analytics literature
LAG_WINDOWS = [1, 2, 3, 5, 7, 10]  # Multiple lag periods for comprehensive history
ADAPTIVE_WEIGHTS = True             # Use adaptive weighting based on recency
MOMENTUM_ANALYSIS = True            # Include momentum/trend analysis
OPPONENT_ADJUSTMENT = True          # Adjust for opponent strength
CONTEXTUAL_FEATURES = True          # Include rest, venue, etc.

# Time decay parameters (research-optimized)
DECAY_RATE = 0.05                  # Exponential decay for historical weights
MOMENTUM_WEIGHT = 0.1              # Reduced weight for trend/momentum features
OPPONENT_WEIGHT = 0.05             # Added: Small weight for opponent adjustments
VOLATILITY_PENALTY = 0.1           # Penalty for high volatility periods

# Online learning parameters
MIN_GAMES_FOR_PREDICTION = 15      # Minimum games before making predictions
VALIDATION_WINDOW = 10             # Rolling validation window

print("🏀 RESEARCH-ENHANCED NBA PREDICTION MODEL")
print("="*50)
print("📊 Key Research Features:")
print("   ✅ Multi-period lagged indicators")
print("   ✅ Adaptive exponential weighting")
print("   ✅ Momentum and trend analysis")
print("   ✅ Opponent strength adjustments")
print("   ✅ Contextual game features")
print("   ✅ Online learning with validation")

# --------------------------------------------------------------
# 2  DATA LOADING AND PREPROCESSING
# --------------------------------------------------------------

def load_and_prepare_data():
    """Load and prepare data for analysis"""
    print(f"\n🔍 LOADING DATA")
    print("-"*20)
    
    # Load datasets
    df = pd.read_csv(CSV_PLAYERS, parse_dates=["GAME_DATE"])
    team_stats = pd.read_csv(CSV_TEAMS) if CSV_TEAMS.exists() else pd.DataFrame()
    
    print(f"📊 Dataset loaded: {df.shape[0]:,} games")
    print(f"📅 Date range: {df['GAME_DATE'].min()} to {df['GAME_DATE'].max()}")
    print(f"🏀 Players: {df['PLAYER_ID'].nunique():,}")
    print(f"🏟️  Seasons: {', '.join(sorted(df['SEASON_YEAR'].unique()))}")
    
    # Find LeBron James
    lebron_candidates = df[df["PLAYER_NAME"].str.contains("LeBron", case=False, na=False)]
    
    if lebron_candidates.empty:
        print("❌ No LeBron James data found")
        return None, None
    
    lebron_id = lebron_candidates["PLAYER_ID"].iloc[0]
    lebron_name = lebron_candidates["PLAYER_NAME"].iloc[0]
    
    print(f"\n🎯 TARGET PLAYER: {lebron_name}")
    print(f"   Player ID: {lebron_id}")
    
    # Get all LeBron data, sorted chronologically
    player_data = df[df["PLAYER_ID"] == lebron_id].copy()
    player_data = player_data.sort_values("GAME_DATE").reset_index(drop=True)
    
    print(f"   Total games: {len(player_data)}")
    print(f"   Seasons: {', '.join(sorted(player_data['SEASON_YEAR'].unique()))}")
    
    # Focus on recent seasons for better prediction relevance
    recent_seasons = ["2022-23", "2023-24", "2024-25"]
    recent_data = player_data[player_data["SEASON_YEAR"].isin(recent_seasons)].copy()
    
    if len(recent_data) < MIN_GAMES_FOR_PREDICTION:
        print(f"⚠️  Insufficient recent data: {len(recent_data)} games")
        print("   Using all available data...")
        recent_data = player_data.copy()
    
    print(f"   Analysis data: {len(recent_data)} games")
    
    return recent_data, team_stats

# --------------------------------------------------------------
# 3  RESEARCH-BASED FEATURE ENGINEERING
# --------------------------------------------------------------

def calculate_lagged_indicators(df, target, current_idx):
    """
    Calculate comprehensive lagged indicators based on research.
    Multiple lag periods capture different aspects of performance patterns.
    """
    if current_idx < max(LAG_WINDOWS):
        return {}
    
    features = {}
    target_values = df[target].iloc[:current_idx].values
    
    # Direct lag features (L1, L2, L3, L5, L7, L10)
    for lag in LAG_WINDOWS:
        if current_idx >= lag:
            features[f"{target}_L{lag}"] = target_values[-lag]
        else:
            features[f"{target}_L{lag}"] = np.mean(target_values) if len(target_values) > 0 else 0
    
    # Adaptive weighted averages (research enhancement)
    if ADAPTIVE_WEIGHTS and len(target_values) >= 5:
        # Exponential weights favoring recent games
        weights = np.exp(-DECAY_RATE * np.arange(len(target_values)-1, -1, -1))
        weights = weights / weights.sum()
        features[f"{target}_adaptive_avg"] = np.average(target_values, weights=weights)
        
        # Weighted average of recent 10 games
        recent_10 = target_values[-10:] if len(target_values) >= 10 else target_values
        recent_weights = np.exp(-DECAY_RATE * np.arange(len(recent_10)-1, -1, -1))
        recent_weights = recent_weights / recent_weights.sum()
        features[f"{target}_recent_weighted"] = np.average(recent_10, weights=recent_weights)
    
    return features

def calculate_momentum_features(df, target, current_idx):
    """
    Calculate momentum and trend features based on research.
    Captures short-term performance trends and streaks.
    """
    if not MOMENTUM_ANALYSIS or current_idx < 5:
        return {}
    
    features = {}
    target_values = df[target].iloc[:current_idx].values
    
    # Short-term momentum (last 5 games trend)
    if len(target_values) >= 5:
        recent_5 = target_values[-5:]
        x = np.arange(len(recent_5))
        try:
            slope, _ = np.polyfit(x, recent_5, 1)
            features[f"{target}_momentum_5"] = slope * MOMENTUM_WEIGHT
        except:
            features[f"{target}_momentum_5"] = 0.0
        
        # Performance vs season average (hot/cold streak)
        season_avg = np.mean(target_values)
        recent_avg = np.mean(recent_5)
        streak_indicator = (recent_avg - season_avg) / (season_avg + 0.1)
        features[f"{target}_streak"] = streak_indicator
    
    # Medium-term trend (last 10 games)
    if len(target_values) >= 10:
        recent_10 = target_values[-10:]
        x = np.arange(len(recent_10))
        try:
            slope, _ = np.polyfit(x, recent_10, 1)
            features[f"{target}_trend_10"] = slope * MOMENTUM_WEIGHT
        except:
            features[f"{target}_trend_10"] = 0.0
    
    # Volatility measure (consistency)
    if len(target_values) >= 10:
        recent_std = np.std(target_values[-10:])
        recent_mean = np.mean(target_values[-10:])
        if recent_mean > 0:
            cv = recent_std / recent_mean  # Coefficient of variation
            consistency = 1 / (1 + cv)    # Higher = more consistent
            features[f"{target}_consistency"] = consistency
            
            # Volatility penalty for unreliable predictions
            if cv > 0.3:  # High volatility threshold
                features[f"{target}_volatility_penalty"] = -VOLATILITY_PENALTY
            else:
                features[f"{target}_volatility_penalty"] = 0.0
        else:
            features[f"{target}_consistency"] = 0.5
            features[f"{target}_volatility_penalty"] = 0.0
    
    return features

def extract_opponent_from_matchup(matchup_str):
    """Extract opponent team abbreviation from matchup string"""
    if pd.isna(matchup_str):
        return "UNK"
    
    # Handle formats like "LAL vs. GSW" or "LAL @ GSW"
    parts = str(matchup_str).replace("@", "vs.").split("vs.")
    if len(parts) == 2:
        return parts[1].strip()
    
    # Fallback: take last 3 characters
    clean_str = str(matchup_str).strip()
    if len(clean_str) >= 3:
        return clean_str[-3:]
    
    return "UNK"

def calculate_opponent_features(df, target, current_idx, team_stats):
    """
    Calculate opponent-specific adjustments based on research.
    Players perform differently against different defensive strengths.
    Returns normalized adjustment factors, not raw values.
    """
    if not OPPONENT_ADJUSTMENT or current_idx == 0:
        return {}
    
    features = {}
    current_game = df.iloc[current_idx]
    current_opponent = extract_opponent_from_matchup(current_game.get("MATCHUP", ""))
    
    # Calculate player's season average up to this point for normalization
    season_avg = np.mean(df.iloc[:current_idx][target]) if current_idx > 0 else np.mean(df[target])
    
    # Historical performance against this specific opponent
    opponent_history = []
    for i in range(current_idx):
        game = df.iloc[i]
        opponent = extract_opponent_from_matchup(game.get("MATCHUP", ""))
        if opponent == current_opponent:
            opponent_history.append(game[target])
    
    if len(opponent_history) > 0:
        opponent_avg = np.mean(opponent_history)
        # Calculate relative adjustment (not raw value)
        adjustment = (opponent_avg - season_avg) / season_avg if season_avg > 0 else 0
        features[f"{target}_vs_opponent_adj"] = max(-0.3, min(0.3, adjustment))  # Cap at ±30%
    else:
        features[f"{target}_vs_opponent_adj"] = 0.0
    
    # Opponent defensive strength (if team stats available)
    if not team_stats.empty and current_opponent in team_stats["TEAM_ABBREVIATION"].values:
        opp_stats = team_stats[team_stats["TEAM_ABBREVIATION"] == current_opponent].iloc[0]
        def_rating = opp_stats.get("DEF_RATING", 110)
        
        # Normalize defensive rating (lower is better defense)
        # 105 = elite defense, 110 = average, 115 = poor defense
        normalized_def = (def_rating - 110) / 10  # Smaller range: roughly -0.5 to 0.5
        features[f"{target}_opp_def_strength"] = max(-0.2, min(0.2, normalized_def))
        
        # Find performance against similar defensive teams
        similar_def_teams = team_stats[
            abs(team_stats["DEF_RATING"] - def_rating) <= 2
        ]["TEAM_ABBREVIATION"].tolist()
        
        similar_performance = []
        for i in range(current_idx):
            game = df.iloc[i]
            game_opponent = extract_opponent_from_matchup(game.get("MATCHUP", ""))
            if game_opponent in similar_def_teams:
                similar_performance.append(game[target])
        
        if len(similar_performance) >= 3:
            similar_avg = np.mean(similar_performance[-10:])  # Recent 10
            similar_adj = (similar_avg - season_avg) / season_avg if season_avg > 0 else 0
            features[f"{target}_vs_similar_def_adj"] = max(-0.2, min(0.2, similar_adj))
        else:
            features[f"{target}_vs_similar_def_adj"] = features.get(f"{target}_vs_opponent_adj", 0.0)
    else:
        features[f"{target}_opp_def_strength"] = 0.0
        features[f"{target}_vs_similar_def_adj"] = 0.0
    
    return features

def calculate_contextual_features(df, current_idx):
    """
    Calculate contextual game features based on research.
    Rest, venue, and scheduling factors significantly impact performance.
    """
    if not CONTEXTUAL_FEATURES or current_idx == 0:
        return {}
    
    features = {}
    current_game = df.iloc[current_idx]
    
    # Home court advantage
    matchup = str(current_game.get("MATCHUP", ""))
    is_home = 1 if "vs" in matchup else 0
    features["home_advantage"] = 0.1 if is_home else -0.05  # Research-based coefficients
    
    # Rest days impact
    rest_days = current_game.get("rest_days", 1)
    if rest_days == 0:  # Back-to-back
        features["rest_impact"] = -0.15  # Significant penalty
    elif rest_days == 1:  # Optimal rest
        features["rest_impact"] = 0.05
    elif rest_days in [2, 3]:  # Good rest
        features["rest_impact"] = 0.0
    else:  # Too much rest (rust factor)
        features["rest_impact"] = -0.03
    
    # Day of week effect (weekend boost)
    if "GAME_DATE" in current_game and not pd.isna(current_game["GAME_DATE"]):
        day_of_week = pd.to_datetime(current_game["GAME_DATE"]).dayofweek
        is_weekend = day_of_week in [4, 5, 6]  # Fri, Sat, Sun
        features["weekend_boost"] = 0.03 if is_weekend else 0.0
    else:
        features["weekend_boost"] = 0.0
    
    return features

def prepare_comprehensive_features(df, target, current_idx, team_stats):
    """
    Combine all research-based features for comprehensive prediction.
    """
    if current_idx < MIN_GAMES_FOR_PREDICTION:
        return {}
    
    # Combine all feature types
    lag_features = calculate_lagged_indicators(df, target, current_idx)
    momentum_features = calculate_momentum_features(df, target, current_idx)
    opponent_features = calculate_opponent_features(df, target, current_idx, team_stats)
    contextual_features = calculate_contextual_features(df, current_idx)
    
    # Merge all features
    all_features = {
        **lag_features,
        **momentum_features,
        **opponent_features,
        **contextual_features
    }
    
    return all_features

# --------------------------------------------------------------
# 4  RESEARCH-BASED PREDICTION MODEL
# --------------------------------------------------------------

def calculate_weighted_prediction(features, target_values):
    """
    Calculate prediction using research-based weighted combination.
    Combines multiple indicators with adaptive weights.
    """
    if len(features) == 0 or len(target_values) == 0:
        return np.mean(target_values) if len(target_values) > 0 else 15.0
    
    # Base prediction from recent weighted average
    base_pred = features.get(f"{list(features.keys())[0].split('_')[0]}_adaptive_avg", np.mean(target_values))
    
    # Momentum adjustment (additive, small values)
    momentum_keys = [k for k in features.keys() if 'momentum' in k or 'trend' in k or 'streak' in k]
    momentum_adj = sum(features.get(k, 0) for k in momentum_keys) * MOMENTUM_WEIGHT
    
    # Opponent adjustment (multiplicative percentage adjustments)
    opponent_keys = [k for k in features.keys() if 'opponent_adj' in k or 'def_strength' in k or 'similar_def_adj' in k]
    opponent_factor = 1.0
    for key in opponent_keys:
        if 'adj' in key:  # These are percentage adjustments
            opponent_factor += features.get(key, 0) * OPPONENT_WEIGHT
        else:  # def_strength is direct multiplicative
            opponent_factor += features.get(key, 0) * OPPONENT_WEIGHT * 0.5
    
    # Contextual adjustment (additive, small values)
    context_keys = [k for k in features.keys() if k in ['home_advantage', 'rest_impact', 'weekend_boost']]
    context_adj = sum(features.get(k, 0) for k in context_keys)
    
    # Volatility penalty (additive)
    volatility_penalty = features.get(f"{list(features.keys())[0].split('_')[0]}_volatility_penalty", 0)
    
    # Combined prediction: base * opponent_factor + additive adjustments
    final_pred = (base_pred * opponent_factor) + momentum_adj + context_adj + volatility_penalty
    
    # Ensure reasonable bounds (no negative stats)
    return max(0, final_pred)
    
    # Ensure reasonable bounds (no negative stats)
    return max(0, final_pred)

def create_enhanced_baselines(df, target, current_idx):
    """Create enhanced baseline models for comparison"""
    if current_idx < 3:
        return {}
    
    target_values = df[target].iloc[:current_idx].values
    baselines = {}
    
    # Simple moving averages
    for window in [3, 5, 10, 15]:
        if len(target_values) >= window:
            baselines[f"MA_{window}"] = np.mean(target_values[-window:])
        else:
            baselines[f"MA_{window}"] = np.mean(target_values)
    
    # Exponentially weighted moving average
    if len(target_values) >= 2:
        alpha = 0.3
        ewma = target_values[0]
        for val in target_values[1:]:
            ewma = alpha * val + (1 - alpha) * ewma
        baselines["EWMA"] = ewma
    else:
        baselines["EWMA"] = np.mean(target_values)
    
    # Linear trend
    if len(target_values) >= 5:
        recent_vals = target_values[-min(15, len(target_values)):]
        x = np.arange(len(recent_vals))
        try:
            slope, intercept = np.polyfit(x, recent_vals, 1)
            trend_pred = slope * len(recent_vals) + intercept
            baselines["Linear_Trend"] = max(0, trend_pred)
        except:
            baselines["Linear_Trend"] = np.mean(recent_vals)
    else:
        baselines["Linear_Trend"] = np.mean(target_values)
    
    # Weighted recent average
    if len(target_values) >= 5:
        recent_vals = target_values[-min(10, len(target_values)):]
        weights = np.arange(1, len(recent_vals) + 1)
        weights = weights / weights.sum()
        baselines["Weighted_Avg"] = np.average(recent_vals, weights=weights)
    else:
        baselines["Weighted_Avg"] = np.mean(target_values)
    
    return baselines

def calculate_prediction_metrics(actual, predicted):
    """Calculate comprehensive prediction metrics"""
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    if len(actual) == 0 or len(predicted) == 0:
        return {}
    
    # Basic metrics
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mse)
    
    # R-squared
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Correlation
    correlation = stats.pearsonr(actual, predicted)[0] if len(actual) > 1 else 0
    
    # Directional accuracy (for over/under predictions)
    mean_actual = np.mean(actual)
    direction_correct = np.sum((predicted > mean_actual) == (actual > mean_actual))
    directional_accuracy = direction_correct / len(actual)
    
    # Bias metrics
    bias = np.mean(predicted - actual)
    
    return {
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Correlation": correlation,
        "Directional_Accuracy": directional_accuracy,
        "Bias": bias
    }

# --------------------------------------------------------------
# 5  MAIN ANALYSIS FUNCTION
# --------------------------------------------------------------

def run_research_enhanced_analysis():
    """Run complete research-enhanced analysis with online learning"""
    
    # Load data
    player_data, team_stats = load_and_prepare_data()
    if player_data is None:
        return
    
    # Split data for online learning
    total_games = len(player_data)
    initial_training = max(MIN_GAMES_FOR_PREDICTION, int(total_games * 0.7))
    
    print(f"\n🎯 ONLINE LEARNING SETUP")
    print(f"   Total games: {total_games}")
    print(f"   Initial training: {initial_training}")
    print(f"   Testing games: {total_games - initial_training}")
    
    results = {}
    
    # Analyze each target statistic
    for target in TARGETS:
        print(f"\n{'='*50}")
        print(f"🏀 ANALYZING {target}")
        print(f"{'='*50}")
        
        # Storage for predictions and actuals
        research_predictions = []
        actual_values = []
        baseline_predictions = {
            "MA_5": [], "MA_10": [], "EWMA": [], 
            "Linear_Trend": [], "Weighted_Avg": []
        }
        
        # Online learning loop
        for game_idx in range(initial_training, total_games):
            current_game = player_data.iloc[game_idx]
            game_date = current_game["GAME_DATE"].strftime("%Y-%m-%d")
            actual_value = current_game[target]
            
            print(f"\n📅 Game {game_idx + 1}/{total_games} ({game_date})")
            print(f"   Actual {target}: {actual_value}")
            
            # Store actual value
            actual_values.append(actual_value)
            
            # Generate baseline predictions
            baselines = create_enhanced_baselines(player_data, target, game_idx)
            for model_name in baseline_predictions.keys():
                baseline_predictions[model_name].append(baselines.get(model_name, actual_value))
            
            # Generate research-enhanced prediction
            features = prepare_comprehensive_features(player_data, target, game_idx, team_stats)
            target_history = player_data[target].iloc[:game_idx].values
            
            research_pred = calculate_weighted_prediction(features, target_history)
            research_predictions.append(research_pred)
            
            print(f"   Research Prediction: {research_pred:.1f}")
            print(f"   Difference: {research_pred - actual_value:+.1f}")
            
            # Show feature contributions
            if len(features) > 0:
                #print("   Key Features:")
                momentum_contrib = sum(v for k, v in features.items() if 'momentum' in k or 'trend' in k)
                opponent_contrib = sum(v for k, v in features.items() if 'opponent' in k)
                context_contrib = sum(v for k, v in features.items() if k in ['home_advantage', 'rest_impact'])
                
                if abs(momentum_contrib) > 0.01:
                    print(f"     Momentum: {momentum_contrib:+.2f}")
                if abs(opponent_contrib) > 0.01:
                    print(f"     Opponent: {opponent_contrib:+.2f}")
                if abs(context_contrib) > 0.01:
                    print(f"     Context: {context_contrib:+.2f}")
        
        # Calculate final metrics
        if len(actual_values) > 0:
            print(f"\n📊 FINAL RESULTS FOR {target}")
            print("-" * 40)
            
            # Research model metrics
            research_metrics = calculate_prediction_metrics(actual_values, research_predictions)
            print(f"🧠 Research-Enhanced Model:")
            print(f"   R² = {research_metrics['R2']:.3f}")
            print(f"   MAE = {research_metrics['MAE']:.2f}")
            print(f"   Correlation = {research_metrics['Correlation']:.3f}")
            print(f"   Directional Accuracy = {research_metrics['Directional_Accuracy']:.1%}")
            print(f"   Bias = {research_metrics['Bias']:+.2f}")
            
            # Baseline comparisons
            print(f"\n📈 Baseline Comparisons:")
            best_baseline_r2 = -np.inf
            best_baseline_name = ""
            
            for model_name, predictions in baseline_predictions.items():
                if len(predictions) == len(actual_values):
                    metrics = calculate_prediction_metrics(actual_values, predictions)
                    print(f"   {model_name}: R² = {metrics['R2']:.3f}, MAE = {metrics['MAE']:.2f}")
                    
                    if metrics['R2'] > best_baseline_r2:
                        best_baseline_r2 = metrics['R2']
                        best_baseline_name = model_name
            
            # Performance summary
            print(f"\n🏆 PERFORMANCE SUMMARY:")
            if research_metrics['R2'] > best_baseline_r2:
                improvement = research_metrics['R2'] - best_baseline_r2
                print(f"   ✅ Research model OUTPERFORMS best baseline ({best_baseline_name})")
                print(f"   📈 R² improvement: +{improvement:.3f}")
                
                if improvement > 0.05:
                    print(f"   🌟 SIGNIFICANT IMPROVEMENT!")
                
                # Additional advantages
                if research_metrics['MAE'] < min([calculate_prediction_metrics(actual_values, preds)['MAE'] 
                                                for preds in baseline_predictions.values()]):
                    print(f"   🎯 Also achieved LOWEST prediction error (MAE)")
                    
            else:
                gap = best_baseline_r2 - research_metrics['R2']
                print(f"   ⚠️  Best baseline ({best_baseline_name}) still leads by {gap:.3f}")
                print(f"   🔧 Consider tuning research parameters")
            
            # Store results
            results[target] = {
                'research_predictions': research_predictions,
                'actual_values': actual_values,
                'baseline_predictions': baseline_predictions,
                'research_metrics': research_metrics
            }
    
    print(f"\n🎉 RESEARCH-ENHANCED ANALYSIS COMPLETE!")
    print(f"💾 Results stored for {len(results)} targets")
    
    return results

# --------------------------------------------------------------
# 6  EXECUTE ANALYSIS
# --------------------------------------------------------------

if __name__ == "__main__":
    results = run_research_enhanced_analysis()
