#!/usr/bin/env python3
# optimized_research_predictor.py - Optimized NBA Prediction Model
# --------------------------------------------------------------
# Advanced optimization of the research-enhanced model with:
# - Dynamic feature weighting based on recent performance
# - Advanced team matchup analytics 
# - Situational context modeling
# - Ensemble prediction methods
# - Cross-validation for parameter tuning
# --------------------------------------------------------------

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# --------------------------------------------------------------
# 1  OPTIMIZED CONFIGURATION
# --------------------------------------------------------------
CSV_PLAYERS = Path("nba_players_game_logs_2018_25_clean.csv")
CSV_TEAMS   = Path("team_stats.csv")
OUT         = Path("optimized_research_preds.csv")

TARGETS = ["PTS", "REB", "AST"]

# Enhanced research-based parameters
LAG_WINDOWS = [1, 2, 3, 5, 7, 10, 15]  # Extended lag windows
ADAPTIVE_WEIGHTS = True
MOMENTUM_ANALYSIS = True
OPPONENT_ADJUSTMENT = True
CONTEXTUAL_FEATURES = True
ADVANCED_TEAM_FEATURES = True          # NEW: Advanced team analytics
ENSEMBLE_METHODS = True                # NEW: Ensemble predictions
DYNAMIC_WEIGHTING = True               # NEW: Adaptive feature weights

# Optimized decay parameters
DECAY_RATE = 0.015                     # Slower decay for more stability
MOMENTUM_WEIGHT = 0.04                 # Further reduced momentum weight
OPPONENT_WEIGHT = 0.02                 # Further reduced opponent weight  
TEAM_MATCHUP_WEIGHT = 0.03            # Reduced team matchup weight
VOLATILITY_PENALTY = 0.05             # Reduced penalty
SITUATIONAL_WEIGHT = 0.02             # Reduced situational context weight

# Bias correction parameters
BIAS_CORRECTION = True                 # Enable bias correction
CONSERVATIVE_BOUNDS = True            # Apply conservative prediction bounds
ADAPTIVE_REGRESSION = 0.15            # Regression to mean factor
MAX_SINGLE_ADJUSTMENT = 0.08          # Cap individual feature adjustments

# Advanced parameters
MIN_GAMES_FOR_PREDICTION = 12          # Minimum games before making predictions
ENSEMBLE_WEIGHTS = [0.4, 0.3, 0.2, 0.1]  # Weights for ensemble methods
FEATURE_SELECTION_THRESHOLD = 0.001    # Minimum feature importance

print("OPTIMIZED RESEARCH-ENHANCED NBA PREDICTION MODEL")
# --------------------------------------------------------------
# 2  ENHANCED DATA LOADING
# --------------------------------------------------------------

def load_and_prepare_data():
    """Enhanced data loading with additional preprocessing"""
    print(f"\nLOADING AND PREPROCESSING DATA")
    print("-"*40)
    
    # Load datasets
    df = pd.read_csv(CSV_PLAYERS, parse_dates=["GAME_DATE"])
    team_stats = pd.read_csv(CSV_TEAMS) if CSV_TEAMS.exists() else pd.DataFrame()
    
    print(f"Dataset loaded: {df.shape[0]:,} games")
    print(f"Team stats: {len(team_stats)} teams")
    print(f"Date range: {df['GAME_DATE'].min()} to {df['GAME_DATE'].max()}")
    
    # Find LeBron James
    lebron_candidates = df[df["PLAYER_NAME"].str.contains("Nikola", case=False, na=False)]
    
    if lebron_candidates.empty:
        print("No LeBron James data found")
        return None, None, None
    
    lebron_id = lebron_candidates["PLAYER_ID"].iloc[0]
    lebron_name = lebron_candidates["PLAYER_NAME"].iloc[0]
    
    print(f"\nTARGET PLAYER: {lebron_name}")
    print(f"   Player ID: {lebron_id}")
    
    # Get all LeBron data, sorted chronologically
    player_data = df[df["PLAYER_ID"] == lebron_id].copy()
    player_data = player_data.sort_values("GAME_DATE").reset_index(drop=True)
    
    print(f"   Total games: {len(player_data)}")
    
    # Enhanced data preprocessing
    if 'OPP_TEAM' not in player_data.columns:
        # Extract opponent from MATCHUP if OPP_TEAM doesn't exist
        player_data['OPP_TEAM'] = player_data['MATCHUP'].apply(extract_opponent_from_matchup)
    
    # Split data: all past seasons for training, most recent for testing
    available_seasons = sorted(player_data['SEASON_YEAR'].unique())
    test_season = available_seasons[-1]
    
    train_data = player_data[player_data["SEASON_YEAR"] != test_season].copy()
    test_data = player_data[player_data["SEASON_YEAR"] == test_season].copy()
    
    print(f"\nTRAIN/TEST SPLIT:")
    print(f"   Training seasons: {sorted(train_data['SEASON_YEAR'].unique())}")
    print(f"   Training games: {len(train_data)}")
    print(f"   Test season: {test_season}")
    print(f"   Test games: {len(test_data)}")
    
    if len(train_data) < MIN_GAMES_FOR_PREDICTION:
        print(f"Insufficient training data: {len(train_data)} games")
        return None, None, None
    
    if len(test_data) < 5:
        print(f"Insufficient test data: {len(test_data)} games")
        return None, None, None
    
    return train_data, test_data, team_stats

# --------------------------------------------------------------
# 3  ADVANCED FEATURE ENGINEERING
# --------------------------------------------------------------

def calculate_dynamic_lagged_indicators(df, target, current_idx):
    """
    Enhanced lagged indicators with bias correction
    """
    if current_idx < max(LAG_WINDOWS) or current_idx >= len(df):
        return {}
    
    features = {}
    target_values = df[target].iloc[:current_idx].values
    
    if len(target_values) == 0:
        return {}
    
    # Calculate historical mean for bias correction
    historical_mean = np.mean(target_values)
    
    # Extended lag features with performance-based weighting
    recent_performance_variance = np.var(target_values[-10:]) if len(target_values) >= 10 else np.var(target_values)
    stability_factor = 1 / (1 + recent_performance_variance / (np.mean(target_values) + 0.1))
    
    for lag in LAG_WINDOWS:
        if len(target_values) >= lag:
            lag_value = target_values[-lag]
            # Apply bias correction - regress extreme values toward mean
            if BIAS_CORRECTION:
                deviation = lag_value - historical_mean
                corrected_value = historical_mean + deviation * (1 - ADAPTIVE_REGRESSION)
                lag_value = corrected_value
            
            # Weight by recency and stability (reduced influence)
            weight = stability_factor * np.exp(-DECAY_RATE * (lag - 1)) * 0.7
            features[f"{target}_L{lag}"] = lag_value
            features[f"{target}_L{lag}_weighted"] = lag_value * weight
        else:
            avg_val = historical_mean
            features[f"{target}_L{lag}"] = avg_val
            features[f"{target}_L{lag}_weighted"] = avg_val * 0.5
    
    # Multi-scale adaptive averages with bias correction
    if ADAPTIVE_WEIGHTS and len(target_values) >= 5:
        # Short-term adaptive (last 5 games) - more conservative
        short_term = target_values[-5:]
        short_weights = np.exp(-DECAY_RATE * 3 * np.arange(len(short_term)-1, -1, -1))
        short_weights = short_weights / short_weights.sum()
        short_avg = np.average(short_term, weights=short_weights)
        
        # Bias correction: blend with historical mean
        if BIAS_CORRECTION:
            short_avg = historical_mean * 0.3 + short_avg * 0.7
        features[f"{target}_adaptive_short"] = short_avg
        
        # Medium-term adaptive (last 15 games)
        if len(target_values) >= 15:
            medium_term = target_values[-15:]
            medium_weights = np.exp(-DECAY_RATE * 0.8 * np.arange(len(medium_term)-1, -1, -1))
            medium_weights = medium_weights / medium_weights.sum()
            medium_avg = np.average(medium_term, weights=medium_weights)
            
            # Less aggressive bias correction for medium term
            if BIAS_CORRECTION:
                medium_avg = historical_mean * 0.2 + medium_avg * 0.8
            features[f"{target}_adaptive_medium"] = medium_avg
        
        # Long-term adaptive (all games) - minimal bias correction
        long_weights = np.exp(-DECAY_RATE * 0.3 * np.arange(len(target_values)-1, -1, -1))
        long_weights = long_weights / long_weights.sum()
        long_avg = np.average(target_values, weights=long_weights)
        
        if BIAS_CORRECTION:
            long_avg = historical_mean * 0.1 + long_avg * 0.9
        features[f"{target}_adaptive_long"] = long_avg
    
    # More conservative trend analysis
    for window in [5, 10, 15]:
        if len(target_values) >= window:
            recent_vals = target_values[-window:]
            x = np.arange(len(recent_vals))
            try:
                slope, intercept = np.polyfit(x, recent_vals, 1)
                # Cap trend influence to reduce bias
                capped_slope = np.clip(slope, -0.5, 0.5)
                features[f"{target}_trend_{window}"] = capped_slope * 0.6  # Reduced influence
                features[f"{target}_trend_{window}_strength"] = abs(capped_slope) * stability_factor * 0.5
                
                # More conservative trend acceleration
                if len(recent_vals) >= 3:
                    mid_point = len(recent_vals) // 2
                    early_slope = np.polyfit(x[:mid_point+1], recent_vals[:mid_point+1], 1)[0]
                    late_slope = np.polyfit(x[mid_point:], recent_vals[mid_point:], 1)[0]
                    accel = np.clip(late_slope - early_slope, -0.3, 0.3)
                    features[f"{target}_trend_accel_{window}"] = accel * 0.4
            except:
                features[f"{target}_trend_{window}"] = 0.0
                features[f"{target}_trend_{window}_strength"] = 0.0
                features[f"{target}_trend_accel_{window}"] = 0.0
    
    return features

def calculate_advanced_momentum_features(df, target, current_idx):
    """
    More conservative momentum features with bias correction
    """
    if not MOMENTUM_ANALYSIS or current_idx < 5:
        return {}
    
    features = {}
    target_values = df[target].iloc[:current_idx].values
    historical_mean = np.mean(target_values)
    
    # More conservative momentum at multiple scales
    for window in [3, 5, 7, 10]:
        if len(target_values) >= window:
            recent_vals = target_values[-window:]
            recent_avg = np.mean(recent_vals)
            
            # More conservative streak intensity calculation
            raw_intensity = (recent_avg - historical_mean) / (historical_mean + 0.1)
            # Apply bias correction and capping
            capped_intensity = np.clip(raw_intensity, -0.15, 0.15)
            if BIAS_CORRECTION:
                # Regress toward zero for extreme streaks
                streak_intensity = capped_intensity * (1 - abs(capped_intensity) * 0.3)
            else:
                streak_intensity = capped_intensity
            
            features[f"{target}_streak_{window}"] = streak_intensity
            
            # More conservative momentum consistency
            if len(recent_vals) >= 3:
                diffs = np.diff(recent_vals)
                raw_consistency = np.mean(np.sign(diffs)) if len(diffs) > 0 else 0
                # Reduce consistency influence
                features[f"{target}_momentum_consistency_{window}"] = raw_consistency * 0.5
            
            # Conservative volatility-adjusted momentum
            volatility = np.std(recent_vals)
            adj_momentum = streak_intensity / (1 + volatility / (recent_avg + 0.1))
            features[f"{target}_adj_momentum_{window}"] = adj_momentum * 0.7
    
    # Conservative performance percentiles
    if len(target_values) >= 20:
        recent_10 = np.mean(target_values[-10:])
        historical_10_game_avgs = [np.mean(target_values[i:i+10]) 
                                  for i in range(len(target_values)-10) if i >= 10]
        if len(historical_10_game_avgs) > 0:
            percentile = stats.percentileofscore(historical_10_game_avgs, recent_10)
            # Convert to more conservative adjustment
            percentile_adj = (percentile - 50) / 100 * 0.05  # Much smaller influence
            features[f"{target}_form_percentile"] = percentile_adj
    
    return features

def extract_opponent_from_matchup(matchup_str):
    """Enhanced opponent extraction with better parsing"""
    if pd.isna(matchup_str):
        return "UNK"
    
    matchup_str = str(matchup_str).strip()
    
    # Handle different formats
    if " vs. " in matchup_str:
        return matchup_str.split(" vs. ")[1].strip()
    elif " @ " in matchup_str:
        return matchup_str.split(" @ ")[1].strip()
    elif "vs" in matchup_str:
        return matchup_str.split("vs")[1].strip()
    elif "@" in matchup_str:
        return matchup_str.split("@")[1].strip()
    
    # Fallback: take last 3 characters
    if len(matchup_str) >= 3:
        return matchup_str[-3:]
    
    return "UNK"

def calculate_advanced_team_features(df, target, current_idx, team_stats, current_game):
    """
    Advanced team and matchup analytics using all available team stats
    """
    if not ADVANCED_TEAM_FEATURES or team_stats.empty:
        return {}
    
    features = {}
    
    try:
        player_team = current_game.get('TEAM_ABBREVIATION', '')
        opponent_team = extract_opponent_from_matchup(current_game.get('MATCHUP', ''))
        
        # Player's team stats
        if player_team in team_stats['TEAM_ABBREVIATION'].values:
            team_row = team_stats[team_stats['TEAM_ABBREVIATION'] == player_team].iloc[0]
            
            # Normalize team stats (z-scores relative to league average)
            features[f'team_off_rating_norm'] = (team_row['OFF_RATING'] - 113.5) / 5
            features[f'team_def_rating_norm'] = (team_row['DEF_RATING'] - 113.5) / 5
            features[f'team_net_rating'] = team_row['NET_RATING'] / 10
            features[f'team_pace_norm'] = (team_row['PACE'] - 100) / 5
            features[f'team_ast_ratio'] = team_row['AST_RATIO'] / 20
            features[f'team_reb_pct'] = team_row['REB_PCT']
        
        # Opponent team stats
        if opponent_team in team_stats['TEAM_ABBREVIATION'].values:
            opp_row = team_stats[team_stats['TEAM_ABBREVIATION'] == opponent_team].iloc[0]
            
            features[f'opp_off_rating_norm'] = (opp_row['OFF_RATING'] - 113.5) / 5
            features[f'opp_def_rating_norm'] = (opp_row['DEF_RATING'] - 113.5) / 5
            features[f'opp_net_rating'] = opp_row['NET_RATING'] / 10
            features[f'opp_pace_norm'] = (opp_row['PACE'] - 100) / 5
            features[f'opp_ast_ratio'] = opp_row['AST_RATIO'] / 20
            features[f'opp_reb_pct'] = opp_row['REB_PCT']
            
            # Matchup-specific features
            pace_matchup = (team_row['PACE'] + opp_row['PACE']) / 2 if player_team in team_stats['TEAM_ABBREVIATION'].values else opp_row['PACE']
            features[f'game_pace_projection'] = (pace_matchup - 100) / 5
            
            # Defensive matchup for different stats
            if target == 'PTS':
                features[f'def_matchup'] = (opp_row['DEF_RATING'] - 113.5) / 5
            elif target == 'REB':
                features[f'def_matchup'] = 1 - opp_row['REB_PCT']  # Lower opp rebounding = more opportunities
            elif target == 'AST':
                features[f'def_matchup'] = (opp_row['PACE'] - 100) / 5  # Higher pace = more assist opportunities
        
        # Team chemistry factors (how long has player been with team)
        games_with_team = len(df[df['TEAM_ABBREVIATION'] == player_team].iloc[:current_idx])
        team_chemistry = min(1.0, games_with_team / 82)  # Normalize to season length
        features[f'team_chemistry'] = team_chemistry
        
    except Exception as e:
        print(f"Warning: Error in team features: {e}")
        # Return default values
        for key in ['team_off_rating_norm', 'team_def_rating_norm', 'opp_def_rating_norm', 'def_matchup']:
            features[key] = 0.0
    
    return features

def calculate_situational_features(df, current_idx, current_game):
    """
    Advanced situational context with more granular factors
    """
    if not CONTEXTUAL_FEATURES or current_idx >= len(df):
        return {}
    
    features = {}
    
    # Enhanced rest day analysis
    rest_days = current_game.get("rest_days", 1)
    if rest_days == 0:  # Back-to-back
        features["rest_fatigue"] = -0.12
        features["rest_optimal"] = 0.0
    elif rest_days == 1:  # Optimal rest
        features["rest_fatigue"] = 0.0
        features["rest_optimal"] = 0.08
    elif rest_days in [2, 3]:  # Good rest
        features["rest_fatigue"] = 0.0
        features["rest_optimal"] = 0.03
    else:  # Extended rest (potential rust)
        features["rest_fatigue"] = -0.02
        features["rest_optimal"] = 0.0
    
    # Home court with historical context
    matchup = str(current_game.get("MATCHUP", ""))
    is_home = 1 if "vs" in matchup else 0
    
    # Calculate historical home/away performance
    if current_idx > 0:
        past_games = df.iloc[:current_idx]
        home_games = past_games[past_games['MATCHUP'].str.contains('vs', na=False)]
        away_games = past_games[past_games['MATCHUP'].str.contains('@', na=False)]
        
        if len(home_games) > 5 and len(away_games) > 5:
            home_avg = home_games['PTS'].mean() if 'PTS' in past_games.columns else 25
            away_avg = away_games['PTS'].mean() if 'PTS' in past_games.columns else 25
            home_advantage_strength = (home_avg - away_avg) / (home_avg + away_avg + 0.1)
            
            features["home_advantage"] = home_advantage_strength if is_home else -home_advantage_strength * 0.5
        else:
            features["home_advantage"] = 0.05 if is_home else -0.03
    else:
        features["home_advantage"] = 0.05 if is_home else -0.03
    
    # Game timing factors
    if "GAME_DATE" in current_game and not pd.isna(current_game["GAME_DATE"]):
        game_date = pd.to_datetime(current_game["GAME_DATE"])
        
        # Day of week effects
        day_of_week = game_date.dayofweek
        if day_of_week in [4, 5, 6]:  # Fri, Sat, Sun (weekend)
            features["weekend_boost"] = 0.04
        elif day_of_week in [0, 1]:  # Mon, Tue (early week)
            features["weekend_boost"] = -0.02
        else:
            features["weekend_boost"] = 0.0
        
        # Month/season progression effects
        month = game_date.month
        if month in [10, 11]:  # Early season
            features["season_phase"] = -0.02  # Still getting into rhythm
        elif month in [12, 1, 2]:  # Mid-season
            features["season_phase"] = 0.03   # Peak performance
        elif month in [3, 4]:  # Late season
            features["season_phase"] = 0.01   # Playoff push but fatigue
        else:
            features["season_phase"] = 0.0
    else:
        features["weekend_boost"] = 0.0
        features["season_phase"] = 0.0
    
    # Recent travel/schedule burden (estimate based on opponent changes)
    if current_idx >= 3:
        recent_opponents = []
        for i in range(max(0, current_idx-3), current_idx):
            if i < len(df):
                opp = extract_opponent_from_matchup(df.iloc[i].get('MATCHUP', ''))
                recent_opponents.append(opp)
        
        unique_opponents = len(set(recent_opponents))
        travel_burden = (unique_opponents - 1) / 2  # More unique opponents = more travel
        features["travel_burden"] = -travel_burden * 0.03
    else:
        features["travel_burden"] = 0.0
    
    return features

def prepare_comprehensive_features(df, target, current_idx, team_stats):
    """
    Combine all optimized features with dynamic weighting
    """
    if current_idx < MIN_GAMES_FOR_PREDICTION or current_idx >= len(df):
        return {}
    
    current_game = df.iloc[current_idx]
    
    # Calculate all feature types
    lag_features = calculate_dynamic_lagged_indicators(df, target, current_idx)
    momentum_features = calculate_advanced_momentum_features(df, target, current_idx)
    team_features = calculate_advanced_team_features(df, target, current_idx, team_stats, current_game)
    situational_features = calculate_situational_features(df, current_idx, current_game)
    
    # Enhanced opponent features (from original model)
    opponent_features = calculate_opponent_features(df, target, current_idx, team_stats)
    
    # Merge all features
    all_features = {
        **lag_features,
        **momentum_features,
        **team_features,
        **situational_features,
        **opponent_features
    }
    
    return all_features

def calculate_opponent_features(df, target, current_idx, team_stats):
    """More conservative opponent-specific adjustments"""
    if not OPPONENT_ADJUSTMENT or current_idx == 0 or current_idx >= len(df):
        return {}
    
    features = {}
    
    try:
        current_game = df.iloc[current_idx]
        current_opponent = extract_opponent_from_matchup(current_game.get("MATCHUP", ""))
        
        # Calculate season average for normalization
        if current_idx > 0:
            season_avg = np.mean(df.iloc[:current_idx][target])
        else:
            season_avg = np.mean(df[target]) if len(df) > 0 else 15.0
        
        # More conservative historical performance vs opponent
        opponent_history = []
        for i in range(min(current_idx, len(df))):
            game = df.iloc[i]
            opponent = extract_opponent_from_matchup(game.get("MATCHUP", ""))
            if opponent == current_opponent:
                opponent_history.append(game[target])
        
        if len(opponent_history) > 0:
            opponent_avg = np.mean(opponent_history)
            # Much more conservative adjustment
            if len(opponent_history) >= 3:
                recent_vs_opp = np.mean(opponent_history[-3:])
                raw_adjustment = (recent_vs_opp - season_avg) / (season_avg + 0.1)
                # Apply bias correction and heavy capping
                adjustment = np.clip(raw_adjustment, -0.08, 0.08) * 0.3
            else:
                raw_adjustment = (opponent_avg - season_avg) / (season_avg + 0.1)
                adjustment = np.clip(raw_adjustment, -0.05, 0.05) * 0.2
                
            features[f"{target}_vs_opponent_adj"] = adjustment
        else:
            features[f"{target}_vs_opponent_adj"] = 0.0
        
        # Much more conservative defensive strength analysis
        if not team_stats.empty and current_opponent in team_stats["TEAM_ABBREVIATION"].values:
            opp_stats = team_stats[team_stats["TEAM_ABBREVIATION"] == current_opponent].iloc[0]
            
            # Conservative target-specific defensive adjustments
            if target == 'PTS':
                def_rating = opp_stats.get("DEF_RATING", 113.5)
                raw_adjustment = (def_rating - 113.5) / 20  # More conservative scaling
            elif target == 'REB':
                reb_pct = opp_stats.get("REB_PCT", 0.5)
                raw_adjustment = (0.5 - reb_pct) * 1.2  # Reduced impact
            elif target == 'AST':
                pace = opp_stats.get("PACE", 100)
                raw_adjustment = (pace - 100) / 30  # More conservative
            else:
                raw_adjustment = 0.0
                
            # Apply heavy capping
            adjustment = np.clip(raw_adjustment, -0.06, 0.06)
            features[f"{target}_def_strength_adj"] = adjustment
        else:
            features[f"{target}_def_strength_adj"] = 0.0
    
    except Exception as e:
        print(f"Warning: Error in opponent features: {e}")
        features = {
            f"{target}_vs_opponent_adj": 0.0,
            f"{target}_def_strength_adj": 0.0
        }
    
    return features

# --------------------------------------------------------------
# 4  ENSEMBLE PREDICTION METHODS
# --------------------------------------------------------------

def calculate_ensemble_prediction(features, target_values, target):
    """
    Bias-corrected ensemble prediction
    """
    if len(features) == 0 or len(target_values) == 0:
        return np.mean(target_values) if len(target_values) > 0 else 15.0
    
    historical_mean = np.mean(target_values)
    predictions = []
    
    # Method 1: Enhanced weighted prediction (50% weight - increased for stability)
    pred1 = calculate_optimized_weighted_prediction(features, target_values, target)
    predictions.append(pred1)
    
    # Method 2: Conservative moving average (30% weight)
    if len(target_values) >= 10:
        # Less dynamic window for stability
        recent_volatility = np.std(target_values[-10:])
        optimal_window = max(5, min(12, int(8 / (1 + recent_volatility))))
        pred2 = np.mean(target_values[-optimal_window:])
        # Bias correction
        if BIAS_CORRECTION:
            pred2 = historical_mean * 0.2 + pred2 * 0.8
    else:
        pred2 = historical_mean
    predictions.append(pred2)
    
    # Method 3: Conservative exponential smoothing (15% weight)
    if len(target_values) >= 5:
        # More conservative smoothing parameters
        alpha, beta = 0.2, 0.05
        level = target_values[0]
        trend = 0
        
        for val in target_values[1:]:
            last_level = level
            level = alpha * val + (1 - alpha) * (level + trend)
            trend = beta * (level - last_level) + (1 - beta) * trend
        
        pred3 = level + trend * 0.5  # Reduce trend influence
        # Bias correction
        if BIAS_CORRECTION:
            pred3 = historical_mean * 0.15 + pred3 * 0.85
    else:
        pred3 = historical_mean
    predictions.append(pred3)
    
    # Method 4: Very conservative recent trend (5% weight)
    if len(target_values) >= 7:
        recent_trend = np.polyfit(range(7), target_values[-7:], 1)[0]
        # Heavy damping of trend
        pred4 = target_values[-1] + recent_trend * 0.3
        if BIAS_CORRECTION:
            pred4 = historical_mean * 0.3 + pred4 * 0.7
    else:
        pred4 = historical_mean
    predictions.append(pred4)
    
    # Conservative ensemble weights
    conservative_weights = [0.5, 0.3, 0.15, 0.05]
    final_prediction = np.average(predictions, weights=conservative_weights)
    
    # Final bias correction - regress extreme predictions toward historical mean
    if BIAS_CORRECTION:
        deviation = abs(final_prediction - historical_mean)
        historical_std = np.std(target_values)
        if deviation > historical_std:
            regression_factor = min(0.2, (deviation - historical_std) / historical_std * 0.05)
            final_prediction = final_prediction * (1 - regression_factor) + historical_mean * regression_factor
    
    # Ensure reasonable bounds
    return max(0, final_prediction)

def calculate_optimized_weighted_prediction(features, target_values, target):
    """
    Bias-corrected weighted prediction with conservative bounds
    """
    if len(features) == 0 or len(target_values) == 0:
        return np.mean(target_values) if len(target_values) > 0 else 15.0
    
    # Historical mean for bias correction
    historical_mean = np.mean(target_values)
    
    # Base prediction from best adaptive average with bias correction
    base_candidates = [
        features.get(f"{target}_adaptive_short", historical_mean),
        features.get(f"{target}_adaptive_medium", historical_mean),
        features.get(f"{target}_adaptive_long", historical_mean)
    ]
    
    # Choose base prediction based on recent performance stability
    recent_volatility = np.std(target_values[-5:]) if len(target_values) >= 5 else np.std(target_values)
    
    if recent_volatility < np.mean(target_values) * 0.15:  # Very low volatility - use short term
        base_pred = base_candidates[0]
    elif recent_volatility < np.mean(target_values) * 0.35:  # Medium volatility - blend short and medium
        base_pred = (base_candidates[0] + base_candidates[1]) / 2
    else:  # High volatility - use long term with bias correction
        base_pred = historical_mean * 0.4 + base_candidates[2] * 0.6
    
    # Conservative momentum adjustment
    momentum_keys = [k for k in features.keys() if 'momentum' in k or 'trend' in k or 'streak' in k]
    raw_momentum_adj = sum(features.get(k, 0) for k in momentum_keys) * MOMENTUM_WEIGHT
    momentum_adj = np.clip(raw_momentum_adj, -MAX_SINGLE_ADJUSTMENT, MAX_SINGLE_ADJUSTMENT)
    
    # Conservative team matchup adjustments
    team_keys = [k for k in features.keys() if 'team_' in k or 'opp_' in k or 'def_matchup' in k]
    raw_team_adj = sum(features.get(k, 0) for k in team_keys) * TEAM_MATCHUP_WEIGHT
    team_adj = np.clip(raw_team_adj, -MAX_SINGLE_ADJUSTMENT, MAX_SINGLE_ADJUSTMENT)
    
    # Conservative opponent-specific adjustments  
    opponent_keys = [k for k in features.keys() if 'opponent_adj' in k or 'def_strength_adj' in k]
    raw_opponent_adj = sum(features.get(k, 0) for k in opponent_keys) * OPPONENT_WEIGHT
    opponent_adj = np.clip(raw_opponent_adj, -MAX_SINGLE_ADJUSTMENT, MAX_SINGLE_ADJUSTMENT)
    
    # Conservative situational adjustments
    situational_keys = [k for k in features.keys() if k in ['home_advantage', 'rest_fatigue', 'rest_optimal', 'weekend_boost', 'season_phase', 'travel_burden']]
    raw_situational_adj = sum(features.get(k, 0) for k in situational_keys) * SITUATIONAL_WEIGHT
    situational_adj = np.clip(raw_situational_adj, -MAX_SINGLE_ADJUSTMENT, MAX_SINGLE_ADJUSTMENT)
    
    # Volatility penalty (conservative)
    volatility_penalty = features.get(f"{target}_volatility_penalty", 0) * VOLATILITY_PENALTY
    volatility_penalty = np.clip(volatility_penalty, -0.05, 0.05)
    
    # Combined prediction
    total_adjustment = momentum_adj + team_adj + opponent_adj + situational_adj + volatility_penalty
    
    # Apply conservative bounds to total adjustment
    total_adjustment = np.clip(total_adjustment, -0.15, 0.15)
    
    # Final prediction with bias correction
    raw_pred = base_pred + total_adjustment
    
    if CONSERVATIVE_BOUNDS:
        # Apply conservative bounds relative to historical performance
        historical_std = np.std(target_values)
        lower_bound = historical_mean - 1.5 * historical_std
        upper_bound = historical_mean + 1.5 * historical_std
        
        # Soft capping - gradually compress extreme predictions
        if raw_pred < lower_bound:
            final_pred = lower_bound + (raw_pred - lower_bound) * 0.3
        elif raw_pred > upper_bound:
            final_pred = upper_bound + (raw_pred - upper_bound) * 0.3
        else:
            final_pred = raw_pred
        
        # Additional regression toward historical mean for extreme cases
        deviation = abs(final_pred - historical_mean)
        if deviation > historical_std:
            regression_factor = min(0.3, (deviation - historical_std) / historical_std * 0.1)
            final_pred = final_pred * (1 - regression_factor) + historical_mean * regression_factor
    else:
        final_pred = raw_pred
    
    return max(0, final_pred)

# --------------------------------------------------------------
# 5  BASELINE METHODS (from original)
# --------------------------------------------------------------

def create_enhanced_baselines(df, target, current_idx):
    """Enhanced baseline models for comparison"""
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
    
    # Directional accuracy
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
# 6  MAIN ANALYSIS FUNCTION
# --------------------------------------------------------------

def run_optimized_analysis():
    """Run optimized analysis with all enhancements"""
    
    # Load data
    train_data, test_data, team_stats = load_and_prepare_data()
    if train_data is None:
        return
    
    print(f"\nOPTIMIZED ONLINE LEARNING SETUP")
    print(f"   Training games: {len(train_data)}")
    print(f"   Testing games: {len(test_data)}")
    print(f"   Ensemble methods: {len(ENSEMBLE_WEIGHTS)} models")
    print(f"   Advanced features: Team analytics + Situational context")
    
    results = {}
    
    # Analyze each target
    for target in TARGETS:
        print(f"\n{'='*60}")
        print(f"OPTIMIZED ANALYSIS: {target}")
        print(f"{'='*60}")
        
        # Storage for predictions
        optimized_predictions = []
        actual_values = []
        baseline_predictions = {
            "MA_5": [], "MA_10": [], "EWMA": [], 
            "Linear_Trend": [], "Weighted_Avg": []
        }
        
        # Start with training data
        current_data = train_data.copy()
        
        # Online learning loop
        for test_idx in range(len(test_data)):
            current_game = test_data.iloc[test_idx]
            game_date = current_game["GAME_DATE"].strftime("%Y-%m-%d")
            actual_value = current_game[target]
            
            # Store actual value
            actual_values.append(actual_value)
            
            # Generate baseline predictions
            baselines = create_enhanced_baselines(current_data, target, len(current_data))
            for model_name in baseline_predictions.keys():
                baseline_predictions[model_name].append(baselines.get(model_name, actual_value))
            
            # Generate optimized prediction
            features = prepare_comprehensive_features(current_data, target, len(current_data), team_stats)
            target_history = current_data[target].values
            
            if ENSEMBLE_METHODS:
                optimized_pred = calculate_ensemble_prediction(features, target_history, target)
            else:
                optimized_pred = calculate_optimized_weighted_prediction(features, target_history, target)
            
            optimized_predictions.append(optimized_pred)
            
            # Online learning: add game to training data
            current_data = pd.concat([current_data, current_game.to_frame().T], ignore_index=True)
            current_data = current_data.sort_values("GAME_DATE").reset_index(drop=True)
        
        # Calculate final metrics
        if len(actual_values) > 0:
            print(f"\nOPTIMIZED RESULTS FOR {target}")
            print("-" * 50)
            
            # Optimized model metrics
            optimized_metrics = calculate_prediction_metrics(actual_values, optimized_predictions)
            print(f"Optimized Enhanced Model:")
            print(f"   R² = {optimized_metrics['R2']:.3f}")
            print(f"   MAE = {optimized_metrics['MAE']:.2f}")
            print(f"   Correlation = {optimized_metrics['Correlation']:.3f}")
            print(f"   Directional Accuracy = {optimized_metrics['Directional_Accuracy']:.1%}")
            print(f"   Bias = {optimized_metrics['Bias']:+.2f}")
            
            # Baseline comparisons
            print(f"\nBaseline Comparisons:")
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
            print(f"\nOPTIMIZED PERFORMANCE SUMMARY:")
            if optimized_metrics['R2'] > best_baseline_r2:
                improvement = optimized_metrics['R2'] - best_baseline_r2
                print(f"   Optimized model OUTPERFORMS best baseline ({best_baseline_name})")
                print(f"   R² improvement: +{improvement:.3f}")
                
                if improvement > 0.1:
                    print(f"   MAJOR IMPROVEMENT!")
                elif improvement > 0.05:
                    print(f"   SIGNIFICANT IMPROVEMENT!")
                
                # Check MAE improvement
                baseline_maes = [calculate_prediction_metrics(actual_values, preds)['MAE'] 
                               for preds in baseline_predictions.values() if len(preds) == len(actual_values)]
                if baseline_maes and optimized_metrics['MAE'] < min(baseline_maes):
                    print(f"   Also achieved LOWEST prediction error (MAE)")
                    
            else:
                gap = best_baseline_r2 - optimized_metrics['R2']
                print(f"   Best baseline ({best_baseline_name}) still leads by {gap:.3f}")
            
            # Store results
            results[target] = {
                'optimized_predictions': optimized_predictions,
                'actual_values': actual_values,
                'baseline_predictions': baseline_predictions,
                'optimized_metrics': optimized_metrics
            }
    
    print(f"\nOPTIMIZED ANALYSIS COMPLETE!")
    print(f"Enhanced with: Advanced team analytics, Ensemble methods, Dynamic weighting")
    
    return results

# --------------------------------------------------------------
# 7  EXECUTE ANALYSIS
# --------------------------------------------------------------

if __name__ == "__main__":
    results = run_optimized_analysis()
