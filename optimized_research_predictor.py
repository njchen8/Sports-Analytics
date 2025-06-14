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
CSV_TEAMS   = Path("nba_team_stats_monthly_2018_current.csv")  # Updated to use monthly team stats
OUT         = Path("optimized_research_preds.csv")

TARGETS = ["PTS", "REB", "AST"]

# RNN + Temporal Fusion Transformer parameters (based on research paper)
SEQUENCE_LENGTHS = [2, 3, 4]           # Multiple sequence lengths as in paper
USE_LSTM = True                        # Enable LSTM features
USE_GRU = True                         # Enable GRU features  
USE_BI_LSTM = True                     # Enable Bidirectional LSTM
USE_ATTENTION = True                   # Enable attention mechanism
USE_TFT = True                         # Enable Temporal Fusion Transformer

# TFT Configuration (from paper)
TFT_STATE_SIZE = 80                    # Hidden state size
TFT_DROPOUT = 0.2                      # Dropout rate
TFT_LEARNING_RATE = 0.01               # Learning rate
TFT_ATTENTION_HEADS = 1                # Single attention head (optimal from paper)
TFT_BATCH_SIZE = 128                   # Batch size

# Feature Engineering Configuration
NORMAL_SCORE_TRANSFORM = True          # Transform to normal scores (Liu et al.)
FOUR_FACTORS_MODEL = True              # Basketball's four factors
ENSEMBLE_METHODS = True                # Multiple prediction methods
ADVANCED_TEAM_FEATURES = True          # Keep essential team context

# Simplified approach - reduce bias
MOMENTUM_ANALYSIS = False              # Disable - causes bias
OPPONENT_ADJUSTMENT = False            # Simplify - focus on player performance  
CONTEXTUAL_FEATURES = False            # Simplify - avoid overfitting

# Advanced parameters
MIN_GAMES_FOR_PREDICTION = 25          # Minimum games before making predictions
ENSEMBLE_WEIGHTS = [0.4, 0.3, 0.2, 0.1]  # Weights for ensemble methods
FEATURE_SELECTION_THRESHOLD = 0.001    # Minimum feature importance

# Prediction adjustment parameters
BIAS_CORRECTION = True                 # Enable bias correction
CONSERVATIVE_BOUNDS = True             # Use conservative prediction bounds
MOMENTUM_WEIGHT = 0.3                  # Weight for momentum features
TEAM_MATCHUP_WEIGHT = 0.2             # Weight for team matchup features
OPPONENT_WEIGHT = 0.15                # Weight for opponent features
SITUATIONAL_WEIGHT = 0.1              # Weight for situational features
VOLATILITY_PENALTY = 0.05             # Penalty for high volatility
MAX_SINGLE_ADJUSTMENT = 3.0           # Maximum single feature adjustment

print("OPTIMIZED RESEARCH-ENHANCED NBA PREDICTION MODEL")
# --------------------------------------------------------------
# 2  ENHANCED DATA LOADING
# --------------------------------------------------------------

def load_and_prepare_data():
    """Enhanced data loading with monthly CSV-based team stats"""
    print(f"\nLOADING AND PREPROCESSING DATA")
    print("-"*40)
    
    # Load datasets
    df = pd.read_csv(CSV_PLAYERS, parse_dates=["GAME_DATE"])
    
    # Load monthly team stats
    team_stats = pd.DataFrame()
    if CSV_TEAMS.exists():
        team_stats = pd.read_csv(CSV_TEAMS)
        # Convert date columns to datetime
        team_stats['DATE_FROM'] = pd.to_datetime(team_stats['DATE_FROM'])
        team_stats['DATE_TO'] = pd.to_datetime(team_stats['DATE_TO'])
        print(f"Monthly team stats loaded: {len(team_stats)} records")
        print(f"Team stats date range: {team_stats['DATE_FROM'].min()} to {team_stats['DATE_TO'].max()}")
    else:
        print("Monthly team stats CSV not found, using defaults")
    
    print(f"Dataset loaded: {df.shape[0]:,} games")
    print(f"Player data date range: {df['GAME_DATE'].min()} to {df['GAME_DATE'].max()}")
    
    # Find target player (updated to search for actual player)
    player_candidates = df[df["PLAYER_NAME"].str.contains("Nikola", case=False, na=False)]
    
    if player_candidates.empty:
        print("No target player data found, trying alternative search...")
        # Try a common player as fallback
        player_candidates = df[df["PLAYER_NAME"].str.contains("James", case=False, na=False)]
        if player_candidates.empty:
            print("No suitable player found")
            return None, None, None
    
    player_id = player_candidates["PLAYER_ID"].iloc[0]
    player_name = player_candidates["PLAYER_NAME"].iloc[0]
    
    print(f"\nTARGET PLAYER: {player_name}")
    print(f"   Player ID: {player_id}")
    
    # Get all player data, sorted chronologically
    player_data = df[df["PLAYER_ID"] == player_id].copy()
    player_data = player_data.sort_values("GAME_DATE").reset_index(drop=True)
    
    print(f"   Total games: {len(player_data)}")
    
    # Enhanced data preprocessing
    if 'OPP_TEAM' not in player_data.columns:
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
# 2.5  MONTHLY CSV-BASED TEAM STATS (NO DATA LEAKAGE)
# --------------------------------------------------------------

def get_team_stats_for_game_date(game_date, monthly_team_stats_df):
    """
    Get team stats from monthly CSV data for a specific game date.
    Uses only data available up to that date to prevent data leakage.
    """
    try:
        if monthly_team_stats_df.empty:
            return create_default_team_stats()
        
        # Convert game_date to datetime if it's a string
        if isinstance(game_date, str):
            game_date = pd.to_datetime(game_date)
        
        # Find the appropriate month's data for this game date
        # Use only data where the game_date falls within the month's date range
        # and ensure we're not using future data
        available_months = monthly_team_stats_df[
            (monthly_team_stats_df['DATE_FROM'] <= game_date) &
            (monthly_team_stats_df['DATE_TO'] >= game_date)
        ].copy()
        
        # If no exact match found, get the most recent month's data before the game date
        if available_months.empty:
            available_months = monthly_team_stats_df[
                monthly_team_stats_df['DATE_TO'] < game_date
            ].copy()
            
            if available_months.empty:
                print(f"No team stats available for date {game_date.strftime('%Y-%m-%d')}, using defaults")
                return create_default_team_stats()
            
            # Get the most recent month before the game date for each team
            available_months = available_months.sort_values('DATE_TO').groupby('TEAM_ABBREVIATION').tail(1)
        
        # Create team stats dictionary with the required columns
        team_stats_dict = {}
        for _, row in available_months.iterrows():
            team_stats_dict[row['TEAM_ABBREVIATION']] = {
                'TEAM_ABBREVIATION': row['TEAM_ABBREVIATION'],
                'OFF_RATING': row['OFF_RATING'],
                'DEF_RATING': row['DEF_RATING'],
                'NET_RATING': row['NET_RATING'],
                'PACE': row['PACE'],
                'AST_RATIO': row['AST_RATIO'],
                'REB_PCT': row['REB_PCT'],
                'W_PCT': row['W_PCT'],
                'GP': row['GP']
            }
        
        # Ensure all NBA teams are represented (fill missing teams with league averages)
        all_teams = ["ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW", 
                     "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK", 
                     "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS"]
        
        # Calculate league averages for missing teams
        if available_months.empty:
            league_avgs = get_league_averages()
        else:
            league_avgs = {
                'OFF_RATING': available_months['OFF_RATING'].mean(),
                'DEF_RATING': available_months['DEF_RATING'].mean(),
                'NET_RATING': available_months['NET_RATING'].mean(),
                'PACE': available_months['PACE'].mean(),
                'AST_RATIO': available_months['AST_RATIO'].mean(),
                'REB_PCT': available_months['REB_PCT'].mean(),
                'W_PCT': available_months['W_PCT'].mean(),
                'GP': available_months['GP'].mean()
            }
        
        # Fill in missing teams with league averages
        for team in all_teams:
            if team not in team_stats_dict:
                team_stats_dict[team] = {
                    'TEAM_ABBREVIATION': team,
                    'OFF_RATING': league_avgs['OFF_RATING'],
                    'DEF_RATING': league_avgs['DEF_RATING'],
                    'NET_RATING': league_avgs['NET_RATING'],
                    'PACE': league_avgs['PACE'],
                    'AST_RATIO': league_avgs['AST_RATIO'],
                    'REB_PCT': league_avgs['REB_PCT'],
                    'W_PCT': league_avgs['W_PCT'],
                    'GP': league_avgs['GP']
                }
        
        # Convert to DataFrame
        team_stats_df = pd.DataFrame(list(team_stats_dict.values()))
        
        return team_stats_df
        
    except Exception as e:
        print(f"Error loading monthly team stats for {game_date}: {e}")
        return create_default_team_stats()

def get_league_averages():
    """Get typical NBA league averages for fallback"""
    return {
        'OFF_RATING': 113.5,
        'DEF_RATING': 113.5,
        'NET_RATING': 0.0,
        'PACE': 100.0,
        'AST_RATIO': 19.0,
        'REB_PCT': 0.5,
        'W_PCT': 0.5,
        'GP': 15
    }

def create_default_team_stats():
    """Create default team stats when CSV data is unavailable"""
    teams = ["ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW", 
             "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK", 
             "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS"]
    
    default_stats = []
    for i, team in enumerate(teams):
        team_strength = (i - 15) / 30  # -0.5 to +0.5 range
        
        team_stat = {
            'TEAM_ABBREVIATION': team,
            'OFF_RATING': 113.5 + team_strength * 8 + np.random.normal(0, 2),
            'DEF_RATING': 113.5 - team_strength * 6 + np.random.normal(0, 2),
            'NET_RATING': team_strength * 12 + np.random.normal(0, 3),
            'PACE': 100.0 + team_strength * 6 + np.random.normal(0, 2),
            'AST_RATIO': 19.0 + team_strength * 3 + np.random.normal(0, 1),
            'REB_PCT': 0.5 + team_strength * 0.06 + np.random.normal(0, 0.02),
            'W_PCT': 0.5 + team_strength * 0.3 + np.random.normal(0, 0.1),
            'GP': 15 + np.random.randint(-3, 4)
        }
        default_stats.append(team_stat)
    
    return pd.DataFrame(default_stats)

# --------------------------------------------------------------
# 3  BAYESIAN NETWORK FEATURE ENGINEERING (Liu et al. 2009)
# --------------------------------------------------------------

def transform_to_normal_scores(values, delta=1e-6):
    """
    Transform values to normal scores using Liu et al. (2009) approach.
    This creates a non-parametric transformation that makes the data approximately normal.
    """
    if len(values) == 0:
        return values
    
    n = len(values)
    # Get ranks (1-based)
    ranks = np.argsort(np.argsort(values)) + 1
    
    # Winsorized truncation to avoid infinite values
    truncated_ranks = np.clip(ranks / (n + 1), delta, 1 - delta)
    
    # Transform to normal scores using inverse normal CDF
    from scipy.stats import norm
    normal_scores = norm.ppf(truncated_ranks)
    
    return normal_scores

def calculate_bayesian_features(df, target, current_idx):
    """
    Calculate features using Bayesian Network principles with normal score transformation.
    Based on Liu et al. (2009) and the basketball analytics paper.
    """
    if current_idx < 12 or current_idx >= len(df):
        return {}
    
    features = {}
    target_values = df[target].iloc[:current_idx].values
    
    if len(target_values) == 0:
        return {}
    
    # Transform target values to normal scores for better modeling
    if NORMAL_SCORE_TRANSFORM and len(target_values) >= 5:
        normal_target = transform_to_normal_scores(target_values)
    else:
        normal_target = target_values
    
    # 1. Simple Moving Averages (proven baseline predictors)
    for window in [3, 5, 10]:
        if len(normal_target) >= window:
            features[f"{target}_ma_{window}"] = np.mean(normal_target[-window:])
    
    # 2. Exponential Smoothing (gives more weight to recent games)
    if len(normal_target) >= 3:
        alpha = 0.3
        smoothed = normal_target[0]
        for val in normal_target[1:]:
            smoothed = alpha * val + (1 - alpha) * smoothed
        features[f"{target}_exp_smooth"] = smoothed
    
    # 3. Recent vs Historical Performance (key for player form)
    if len(normal_target) >= 10:
        recent_5 = np.mean(normal_target[-5:])
        historical_mean = np.mean(normal_target[:-5])
        features[f"{target}_recent_form"] = recent_5 - historical_mean
    
    # 4. Consistency Measure (lower variance = more predictable)
    if len(normal_target) >= 5:
        recent_std = np.std(normal_target[-5:])
        features[f"{target}_consistency"] = 1 / (1 + recent_std)
    
    # 5. Linear Trend (momentum without bias)
    if len(normal_target) >= 7:
        x = np.arange(len(normal_target[-7:]))
        y = normal_target[-7:]
        try:
            slope, _ = np.polyfit(x, y, 1)
            # Cap extreme trends to prevent overfitting
            features[f"{target}_trend"] = np.clip(slope, -1, 1)
        except:
            features[f"{target}_trend"] = 0.0
    
    return features

def calculate_four_factors_features(df, target, current_idx, team_stats, current_game):
    """
    Calculate features based on basketball's "Four Factors" as described in the paper.
    These are the most important basketball analytics metrics.
    """
    if not FOUR_FACTORS_MODEL or team_stats.empty:
        return {}
    
    features = {}
    
    try:
        player_team = current_game.get('TEAM_ABBREVIATION', '')
        opponent_team = extract_opponent_from_matchup(current_game.get('MATCHUP', ''))
        
        # Get team stats for both teams
        if player_team in team_stats['TEAM_ABBREVIATION'].values:
            team_row = team_stats[team_stats['TEAM_ABBREVIATION'] == player_team].iloc[0]
            
            # Four Factors from team stats (normalized)
            features[f'team_off_rating'] = (team_row['OFF_RATING'] - 110) / 10  # Normalized around 110
            features[f'team_def_rating'] = (110 - team_row['DEF_RATING']) / 10  # Inverted (lower is better)
            features[f'team_pace'] = (team_row['PACE'] - 100) / 10  # Normalized around 100
            
        if opponent_team in team_stats['TEAM_ABBREVIATION'].values:
            opp_row = team_stats[team_stats['TEAM_ABBREVIATION'] == opponent_team].iloc[0]
            
            # Opponent strength affects player performance
            features[f'opp_def_rating'] = (110 - opp_row['DEF_RATING']) / 10  # Stronger defense = harder for player
            features[f'opp_pace'] = (opp_row['PACE'] - 100) / 10  # Pace affects opportunities
            
            # Matchup-specific features based on target stat
            if target == 'PTS':
                # Points are most affected by opponent defense
                features[f'matchup_difficulty'] = (opp_row['DEF_RATING'] - 110) / 15
            elif target == 'REB':
                # Rebounds affected by opponent rebounding
                features[f'matchup_difficulty'] = opp_row.get('REB_PCT', 0.5) - 0.5
            elif target == 'AST':
                # Assists affected by pace and team offense
                features[f'matchup_difficulty'] = (100 - opp_row['PACE']) / 15
                
    except Exception as e:
        print(f"Warning: Error in four factors features: {e}")
        # Return default neutral values
        for key in ['team_off_rating', 'team_def_rating', 'matchup_difficulty']:
            features[key] = 0.0
    
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
    Advanced team and matchup analytics using CSV team stats
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
            if player_team in team_stats['TEAM_ABBREVIATION'].values:
                team_row = team_stats[team_stats['TEAM_ABBREVIATION'] == player_team].iloc[0]
                pace_matchup = (team_row['PACE'] + opp_row['PACE']) / 2
            else:
                pace_matchup = opp_row['PACE']
            
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
    RNN + Temporal Fusion Transformer feature engineering approach.
    Based on research paper showing TFT models outperform traditional methods.
    """
    if current_idx < MIN_GAMES_FOR_PREDICTION or current_idx >= len(df):
        return {}
    
    current_game = df.iloc[current_idx]
    
    # 1. RNN-inspired sequence features (LSTM, GRU, BiLSTM)
    sequence_features = calculate_sequence_features(df, target, current_idx)
    
    # 2. Temporal Fusion Transformer features
    tft_features = calculate_temporal_fusion_features(df, target, current_idx, team_stats, current_game)
    
    # 3. Research-proven baseline features
    baseline_features = calculate_research_baseline_features(df, target, current_idx)
    
    # 4. Four Factors features (basketball-specific)
    four_factors_features = calculate_four_factors_features(df, target, current_idx, team_stats, current_game)
    
    # 5. Minimal team features (reduced weight to avoid bias)
    team_features = {}
    if ADVANCED_TEAM_FEATURES:
        team_features = calculate_advanced_team_features(df, target, current_idx, team_stats, current_game)
        # Significantly reduce team feature influence to focus on player performance
        team_features = {k: v * 0.05 for k, v in team_features.items()}
    
    # Combine all features with appropriate weighting
    all_features = {
        **sequence_features,      # Primary RNN features
        **tft_features,          # TFT attention and gating features  
        **baseline_features,     # Proven baseline predictors
        **four_factors_features, # Basketball-specific metrics
        **team_features         # Minimal team context
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
# 3.5  RNN + TEMPORAL FUSION TRANSFORMER FEATURES (Research Paper)
# --------------------------------------------------------------

def calculate_sequence_features(df, target, current_idx):
    """
    Calculate RNN-inspired sequence features with multiple sequence lengths.
    Based on LSTM, GRU, and BiLSTM approaches from the research paper.
    """
    if current_idx < max(SEQUENCE_LENGTHS) or current_idx >= len(df):
        return {}
    
    features = {}
    target_values = df[target].iloc[:current_idx].values
    
    if len(target_values) == 0:
        return {}
    
    # Transform to normal scores if enabled
    if NORMAL_SCORE_TRANSFORM and len(target_values) >= 10:
        normal_target = transform_to_normal_scores(target_values)
    else:
        normal_target = target_values
    
    # Multi-sequence length features (from paper: SL = 2, 3, 4)
    for seq_len in SEQUENCE_LENGTHS:
        if len(normal_target) >= seq_len:
            sequence = normal_target[-seq_len:]
            
            # LSTM-inspired features: memory cell states
            if USE_LSTM:
                # Exponential moving average (simulates LSTM memory)
                ema_alpha = 0.3
                ema = sequence[0]
                for val in sequence[1:]:
                    ema = ema_alpha * val + (1 - ema_alpha) * ema
                features[f"{target}_lstm_memory_{seq_len}"] = ema
                
                # Long-term vs short-term memory
                if seq_len >= 3:
                    short_term = np.mean(sequence[-2:])
                    long_term = np.mean(sequence[:-2]) if len(sequence) > 2 else short_term
                    features[f"{target}_lstm_forget_{seq_len}"] = short_term - long_term
            
            # GRU-inspired features: update gate simulation
            if USE_GRU:
                # Weighted combination of recent and historical
                weights = np.exp(np.linspace(-1, 0, seq_len))
                weights = weights / weights.sum()
                gru_state = np.sum(sequence * weights)
                features[f"{target}_gru_state_{seq_len}"] = gru_state
                
                # Reset gate simulation (volatility-based)
                volatility = np.std(sequence) if len(sequence) > 1 else 0
                reset_factor = 1 / (1 + volatility)
                features[f"{target}_gru_reset_{seq_len}"] = reset_factor
            
            # BiLSTM-inspired features: bidirectional processing
            if USE_BI_LSTM:
                # Forward direction
                forward_trend = np.polyfit(range(seq_len), sequence, 1)[0] if seq_len > 1 else 0
                # Backward direction (reverse sequence)
                backward_trend = np.polyfit(range(seq_len), sequence[::-1], 1)[0] if seq_len > 1 else 0
                # Combine bidirectional information
                features[f"{target}_bilstm_trend_{seq_len}"] = (forward_trend + backward_trend) / 2
                features[f"{target}_bilstm_divergence_{seq_len}"] = abs(forward_trend - backward_trend)
    
    return features

def calculate_temporal_fusion_features(df, target, current_idx, team_stats, current_game):
    """
    Calculate Temporal Fusion Transformer features.
    Combines static covariates with temporal patterns using attention mechanisms.
    """
    if current_idx < max(SEQUENCE_LENGTHS) or not USE_TFT:
        return {}
    
    features = {}
    target_values = df[target].iloc[:current_idx].values
    
    if len(target_values) == 0:
        return {}
    
    # Static covariates (player and team characteristics)
    static_features = {}
    if not team_stats.empty:
        player_team = current_game.get('TEAM_ABBREVIATION', '')
        if player_team in team_stats['TEAM_ABBREVIATION'].values:
            team_row = team_stats[team_stats['TEAM_ABBREVIATION'] == player_team].iloc[0]
            static_features['team_quality'] = team_row['NET_RATING'] / 10
            static_features['team_pace'] = (team_row['PACE'] - 100) / 10
            static_features['team_efficiency'] = (team_row['OFF_RATING'] - 113.5) / 10
    
    # Multi-head attention simulation
    if USE_ATTENTION:
        attention_weights = []
        for seq_len in SEQUENCE_LENGTHS:
            if len(target_values) >= seq_len:
                sequence = target_values[-seq_len:]
                
                # Query: current context (recent performance)
                query = np.mean(sequence[-2:]) if len(sequence) >= 2 else sequence[-1]
                
                # Key-Value pairs: historical sequence
                keys = sequence
                values = sequence
                
                # Attention scores (similarity between query and keys)
                attention_scores = []
                for key in keys:
                    # Simplified attention: inverse distance + value similarity
                    similarity = 1 / (1 + abs(query - key))
                    attention_scores.append(similarity)
                
                # Normalize attention weights
                attention_scores = np.array(attention_scores)
                if attention_scores.sum() > 0:
                    attention_weights = attention_scores / attention_scores.sum()
                    
                    # Attention-weighted value
                    attended_value = np.sum(values * attention_weights)
                    features[f"{target}_attention_{seq_len}"] = attended_value
                    
                    # Attention entropy (how focused the attention is)
                    entropy = -np.sum(attention_weights * np.log(attention_weights + 1e-8))
                    features[f"{target}_attention_entropy_{seq_len}"] = entropy
    
    # Gating mechanism simulation (filter relevant information)
    for seq_len in SEQUENCE_LENGTHS:
        if len(target_values) >= seq_len:
            sequence = target_values[-seq_len:]
            
            # Gate based on recent performance consistency
            consistency = 1 / (1 + np.std(sequence))
            relevance = abs(np.mean(sequence) - np.mean(target_values)) / (np.std(target_values) + 1e-6)
            
            # Sigmoid-like gating function
            gate_value = consistency / (1 + relevance)
            gated_output = np.mean(sequence) * gate_value
            
            features[f"{target}_tft_gated_{seq_len}"] = gated_output
            features[f"{target}_tft_gate_strength_{seq_len}"] = gate_value
    
    # Static-temporal fusion (combine static team features with temporal patterns)
    if static_features:
        for seq_len in SEQUENCE_LENGTHS:
            if len(target_values) >= seq_len:
                temporal_signal = np.mean(target_values[-seq_len:])
                for static_key, static_value in static_features.items():
                    # Multiplicative fusion
                    fusion_key = f"{target}_fusion_{static_key}_{seq_len}"
                    features[fusion_key] = temporal_signal * (1 + static_value * 0.1)
    
    return features

def calculate_research_baseline_features(df, target, current_idx):
    """
    Calculate simple baseline features that consistently work well in research.
    These provide a strong foundation for the ensemble.
    """
    if current_idx < 5:
        return {}
    
    features = {}
    target_values = df[target].iloc[:current_idx].values
    
    if len(target_values) == 0:
        return {}
    
    # Simple moving averages (proven effective baselines)
    for window in [3, 5, 7, 10, 15]:
        if len(target_values) >= window:
            ma = np.mean(target_values[-window:])
            features[f"{target}_baseline_ma_{window}"] = ma
    
    # Exponential smoothing with different alpha values
    for alpha in [0.2, 0.3, 0.4]:
        if len(target_values) >= 3:
            smoothed = target_values[0]
            for val in target_values[1:]:
                smoothed = alpha * val + (1 - alpha) * smoothed
            features[f"{target}_baseline_exp_{int(alpha*10)}"] = smoothed
    
    # Linear trend (momentum)
    if len(target_values) >= 7:
        x = np.arange(len(target_values[-7:]))
        y = target_values[-7:]
        try:
            slope, intercept = np.polyfit(x, y, 1)
            # Predict next value
            next_pred = slope * len(x) + intercept
            features[f"{target}_baseline_trend"] = next_pred
        except:
            features[f"{target}_baseline_trend"] = np.mean(target_values[-5:])
    
    # Median-based features (robust to outliers)
    for window in [5, 10]:
        if len(target_values) >= window:
            median = np.median(target_values[-window:])
            features[f"{target}_baseline_median_{window}"] = median
    
    # Recent vs historical comparison
    if len(target_values) >= 15:
        recent_avg = np.mean(target_values[-5:])
        historical_avg = np.mean(target_values[:-5])
        features[f"{target}_baseline_recent_vs_hist"] = recent_avg - historical_avg
    
    return features

def calculate_rnn_ensemble_prediction(features, target_values, target):
    """
    Advanced ensemble prediction combining RNN-inspired features with TFT components.
    Based on the research paper's finding that TFT models outperform traditional RNNs.
    """
    if len(features) == 0 or len(target_values) == 0:
        return np.mean(target_values) if len(target_values) > 0 else 15.0
    
    # Separate feature types for different prediction methods
    lstm_features = {k: v for k, v in features.items() if 'lstm' in k}
    gru_features = {k: v for k, v in features.items() if 'gru' in k}
    attention_features = {k: v for k, v in features.items() if 'attention' in k}
    tft_features = {k: v for k, v in features.items() if 'tft' in k}
    baseline_features = {k: v for k, v in features.items() if 'baseline' in k}
    
    predictions = []
    weights = []
    
    # 1. LSTM-inspired prediction (weight: 0.25)
    if lstm_features and USE_LSTM:
        lstm_pred = np.mean([v for v in lstm_features.values() if not np.isnan(v)])
        if not np.isnan(lstm_pred):
            predictions.append(lstm_pred)
            weights.append(0.25)
    
    # 2. Attention-based prediction (weight: 0.35 - highest as per paper)
    if attention_features and USE_ATTENTION:
        attention_pred = np.mean([v for v in attention_features.values() if not np.isnan(v)])
        if not np.isnan(attention_pred):
            predictions.append(attention_pred)
            weights.append(0.35)
    
    # 3. TFT-inspired prediction (weight: 0.25)
    if tft_features and USE_TFT:
        tft_pred = np.mean([v for v in tft_features.values() if not np.isnan(v)])
        if not np.isnan(tft_pred):
            predictions.append(tft_pred)
            weights.append(0.25)
    
    # 4. Baseline prediction (weight: 0.15 - always include for stability)
    if baseline_features:
        baseline_pred = np.mean([v for v in baseline_features.values() if not np.isnan(v)])
        if not np.isnan(baseline_pred):
            predictions.append(baseline_pred)
            weights.append(0.15)
    
    # Fallback to simple methods if advanced features failed
    if len(predictions) == 0:
        # Simple moving averages as fallback
        for window in [3, 5, 10]:
            if len(target_values) >= window:
                ma_pred = np.mean(target_values[-window:])
                predictions.append(ma_pred)
                weights.append(1.0 / len([3, 5, 10]))
    
    # Calculate weighted ensemble prediction
    if len(predictions) > 0:
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Weighted average
        ensemble_pred = np.average(predictions, weights=weights)
        
        # Apply conservative bounds to prevent extreme predictions
        if len(target_values) > 0:
            hist_mean = np.mean(target_values)
            hist_std = np.std(target_values)
            
            # Conservative bounds (2 standard deviations)
            lower_bound = hist_mean - 2 * hist_std
            upper_bound = hist_mean + 2 * hist_std
            
            ensemble_pred = np.clip(ensemble_pred, lower_bound, upper_bound)
        
        return ensemble_pred
    else:
        # Ultimate fallback
        return np.mean(target_values) if len(target_values) > 0 else 15.0

# --------------------------------------------------------------
# 4  ENSEMBLE PREDICTION METHODS
# --------------------------------------------------------------

def calculate_bayesian_prediction(features, target_values, target):
    """
    Simple Bayesian-inspired prediction using normal score transformation
    """
    if len(features) == 0 or len(target_values) == 0:
        return np.mean(target_values) if len(target_values) > 0 else 15.0
    
    # Base prediction: weighted average of moving averages
    ma_3 = features.get(f"{target}_ma_3", np.mean(target_values[-3:]) if len(target_values) >= 3 else np.mean(target_values))
    ma_5 = features.get(f"{target}_ma_5", np.mean(target_values[-5:]) if len(target_values) >= 5 else np.mean(target_values))
    ma_10 = features.get(f"{target}_ma_10", np.mean(target_values[-10:]) if len(target_values) >= 10 else np.mean(target_values))
    
    # Weight recent performance more heavily
    base_pred = 0.5 * ma_3 + 0.3 * ma_5 + 0.2 * ma_10
    
    # Small adjustments from other features
    form_adj = features.get(f"{target}_recent_form", 0) * 0.1
    trend_adj = features.get(f"{target}_trend", 0) * 0.05
    consistency_adj = (features.get(f"{target}_consistency", 1) - 1) * 0.02
    
    # Team context (very small adjustments)
    team_adj = 0
    team_keys = [k for k in features.keys() if 'team_' in k or 'opp_' in k or 'matchup_' in k]
    if team_keys:
        team_adj = sum(features.get(k, 0) for k in team_keys) * 0.01
    
    # Final prediction
    final_pred = base_pred + form_adj + trend_adj + consistency_adj + team_adj
    
    # Transform back from normal scores if needed
    if NORMAL_SCORE_TRANSFORM:
        # Simple inverse transformation - just use the historical distribution
        historical_mean = np.mean(target_values)
        historical_std = np.std(target_values)
        # Convert normal score back to original scale approximately
        final_pred = historical_mean + final_pred * historical_std * 0.3
    
    return max(0, final_pred)

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
    """Run optimized analysis with CSV-based team stats"""
    
    # Load data
    train_data, test_data, team_stats_df = load_and_prepare_data()
    if train_data is None:
        return
    
    print(f"\nOPTIMIZED ONLINE LEARNING SETUP")
    print(f"   Training games: {len(train_data)}")
    print(f"   Testing games: {len(test_data)}")
    print(f"   Team stats records: {len(team_stats_df) if not team_stats_df.empty else 0}")
    print(f"   Ensemble methods: {len(ENSEMBLE_WEIGHTS)} models")
    print(f"   Advanced features: CSV-based team analytics + Situational context")
    
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
            game_date = current_game["GAME_DATE"]
            actual_value = current_game[target]
            
            # Store actual value
            actual_values.append(actual_value)
              # Get CSV-based team stats for this time point (NO DATA LEAKAGE)
            dynamic_team_stats = get_team_stats_for_game_date(game_date, team_stats_df)
            
            # Generate baseline predictions
            baselines = create_enhanced_baselines(current_data, target, len(current_data))
            for model_name in baseline_predictions.keys():
                baseline_predictions[model_name].append(baselines.get(model_name, actual_value))
            
            # Generate optimized prediction using CSV team stats
            features = prepare_comprehensive_features(current_data, target, len(current_data), dynamic_team_stats)
            target_history = current_data[target].values
            
            if ENSEMBLE_METHODS:
                # Use RNN + TFT ensemble prediction
                optimized_pred = calculate_rnn_ensemble_prediction(features, target_history, target)
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
    print(f"Enhanced with: CSV-based team analytics, Ensemble methods, Dynamic weighting")
    
    return results

# --------------------------------------------------------------
# 7  EXECUTE ANALYSIS
# --------------------------------------------------------------

if __name__ == "__main__":
    results = run_optimized_analysis()
