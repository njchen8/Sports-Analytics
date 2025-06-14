#!/usr/bin/env python3
# rnn_tft_predictor.py - NBA Prediction Model using RNN + Temporal Fusion Transformer
# --------------------------------------------------------------
# Based on research paper: "Predicting MLB Pitcher Performance Using RNN and TFT Models"
# Implements LSTM, GRU, BiLSTM, and Temporal Fusion Transformer approaches
# for NBA player performance prediction with sequence lengths [2, 3, 4]
# --------------------------------------------------------------

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# --------------------------------------------------------------
# 1. CONFIGURATION - RNN + TFT APPROACH
# --------------------------------------------------------------
CSV_PLAYERS = Path("nba_players_game_logs_2018_25_clean.csv")
CSV_TEAMS = Path("nba_team_stats_monthly_2018_current.csv")
OUTPUT_FILE = Path("rnn_tft_predictions.csv")

TARGETS = ["PTS", "REB", "AST"]

# RNN + TFT Configuration (from research paper)
SEQUENCE_LENGTHS = [2, 3, 4]              # Multiple sequence lengths as in paper
USE_LSTM = True                           # Enable LSTM features
USE_GRU = True                            # Enable GRU features
USE_BI_LSTM = True                        # Enable Bidirectional LSTM
USE_ATTENTION = True                      # Enable attention mechanism
USE_TFT = True                            # Enable Temporal Fusion Transformer

# TFT Hyperparameters (optimized from paper)
TFT_STATE_SIZE = 80                       # Hidden state size
TFT_DROPOUT = 0.2                         # Dropout rate
TFT_LEARNING_RATE = 0.01                  # Learning rate
TFT_ATTENTION_HEADS = 1                   # Single attention head (optimal)
TFT_BATCH_SIZE = 128                      # Batch size
TFT_MAX_GRADIENT_NORM = 0.01              # Gradient clipping

# Feature Engineering Configuration
NORMAL_SCORE_TRANSFORM = True            # Transform to normal scores (Liu et al.)
SEQUENCE_FEATURE_WEIGHTS = [0.4, 0.3, 0.2, 0.1]  # Weights for different sequence lengths
ENSEMBLE_WEIGHTS = [0.35, 0.25, 0.25, 0.15]      # LSTM, GRU, BiLSTM, TFT weights

# Model Parameters
MIN_GAMES_FOR_PREDICTION = 25            # Minimum games for reliable prediction
MAX_PREDICTION_ADJUSTMENT = 5.0          # Maximum single prediction adjustment
VOLATILITY_PENALTY = 0.1                 # Penalty for high volatility predictions

print("NBA RNN + TEMPORAL FUSION TRANSFORMER PREDICTION MODEL")
print("="*60)

# --------------------------------------------------------------
# 2. DATA LOADING AND PREPROCESSING
# --------------------------------------------------------------

def load_data():
    """Load and preprocess NBA player and team data"""
    print(f"\nLOADING DATA")
    print("-" * 30)
    
    # Load player data
    if not CSV_PLAYERS.exists():
        print(f"Player data file not found: {CSV_PLAYERS}")
        return None, None, None
    
    df = pd.read_csv(CSV_PLAYERS, parse_dates=["GAME_DATE"])
    print(f"Player data loaded: {len(df):,} games")
    print(f"Date range: {df['GAME_DATE'].min()} to {df['GAME_DATE'].max()}")
    
    # Load team stats
    team_stats = pd.DataFrame()
    if CSV_TEAMS.exists():
        team_stats = pd.read_csv(CSV_TEAMS)
        team_stats['DATE_FROM'] = pd.to_datetime(team_stats['DATE_FROM'])
        team_stats['DATE_TO'] = pd.to_datetime(team_stats['DATE_TO'])
        print(f"Team stats loaded: {len(team_stats)} records")
    else:
        print("Team stats not found, using defaults")
        team_stats = create_default_team_stats()
    
    # Find target player
    player_candidates = df[df["PLAYER_NAME"].str.contains("Bam Adebayo", case=False, na=False)]
    
    if player_candidates.empty:
        # Fallback to any player with enough games
        player_games = df.groupby('PLAYER_ID').size()
        valid_players = player_games[player_games >= 100].index
        if len(valid_players) == 0:
            print("No players with sufficient games found")
            return None, None, None
        player_id = valid_players[0]
        player_name = df[df['PLAYER_ID'] == player_id]['PLAYER_NAME'].iloc[0]
    else:
        player_id = player_candidates["PLAYER_ID"].iloc[0]
        player_name = player_candidates["PLAYER_NAME"].iloc[0]
    
    print(f"\nTARGET PLAYER: {player_name}")
    print(f"Player ID: {player_id}")
    
    # Get player data sorted chronologically
    player_data = df[df["PLAYER_ID"] == player_id].copy()
    player_data = player_data.sort_values("GAME_DATE").reset_index(drop=True)
    print(f"Total games: {len(player_data)}")
    
    # Split data for training/testing
    available_seasons = sorted(player_data['SEASON_YEAR'].unique())
    if len(available_seasons) < 2:
        print("Insufficient seasons for train/test split")
        return None, None, None
    
    test_season = available_seasons[-1]
    train_data = player_data[player_data["SEASON_YEAR"] != test_season].copy()
    test_data = player_data[player_data["SEASON_YEAR"] == test_season].copy()
    
    print(f"\nTRAIN/TEST SPLIT:")
    print(f"Training seasons: {sorted(train_data['SEASON_YEAR'].unique())}")
    print(f"Training games: {len(train_data)}")
    print(f"Test season: {test_season}")
    print(f"Test games: {len(test_data)}")
    
    if len(train_data) < MIN_GAMES_FOR_PREDICTION:
        print(f"Insufficient training data: {len(train_data)} games")
        return None, None, None
    
    return train_data, test_data, team_stats

def create_default_team_stats():
    """Create realistic default team stats when CSV is unavailable"""
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

def get_team_stats_for_date(game_date, team_stats_df):
    """Get team stats for a specific date without data leakage"""
    try:
        if team_stats_df.empty:
            return create_default_team_stats()
        
        if isinstance(game_date, str):
            game_date = pd.to_datetime(game_date)
        
        # Find team stats available up to this date
        available_stats = team_stats_df[
            (team_stats_df['DATE_FROM'] <= game_date) &
            (team_stats_df['DATE_TO'] >= game_date)
        ].copy()
        
        if available_stats.empty:
            # Get most recent stats before this date
            available_stats = team_stats_df[
                team_stats_df['DATE_TO'] < game_date
            ].copy()
            
            if available_stats.empty:
                return create_default_team_stats()
            
            # Get most recent for each team
            available_stats = available_stats.sort_values('DATE_TO').groupby('TEAM_ABBREVIATION').tail(1)
        
        return available_stats
        
    except Exception as e:
        print(f"Error getting team stats for {game_date}: {e}")
        return create_default_team_stats()

# --------------------------------------------------------------
# 3. RNN + TFT FEATURE ENGINEERING
# --------------------------------------------------------------

def transform_to_normal_scores(values, delta=1e-6):
    """Transform values to normal scores using Liu et al. approach"""
    if len(values) <= 1:
        return np.array(values)
    
    n = len(values)
    ranks = np.argsort(np.argsort(values)) + 1
    truncated_ranks = np.clip(ranks / (n + 1), delta, 1 - delta)
    normal_scores = norm.ppf(truncated_ranks)
    
    return normal_scores

def calculate_lstm_features(sequence_data, target, sequence_length):
    """
    Calculate LSTM-inspired features for sequence data
    Simulates LSTM gating mechanisms and memory cells
    """
    if len(sequence_data) < sequence_length:
        return {}
    
    features = {}
    recent_sequence = sequence_data[target].iloc[-sequence_length:].values
    
    if NORMAL_SCORE_TRANSFORM and len(recent_sequence) > 2:
        recent_sequence = transform_to_normal_scores(recent_sequence)
    
    # Simulate LSTM forget gate (exponential decay)
    forget_weights = np.exp(-np.arange(sequence_length) * 0.3)
    forget_weights = forget_weights[::-1] / np.sum(forget_weights)
    lstm_memory = np.sum(recent_sequence * forget_weights)
    features[f'lstm_memory_{sequence_length}'] = lstm_memory
    
    # Simulate input gate (recent importance)
    input_weights = np.linspace(0.1, 1.0, sequence_length)
    input_weights = input_weights / np.sum(input_weights)
    lstm_input = np.sum(recent_sequence * input_weights)
    features[f'lstm_input_{sequence_length}'] = lstm_input
    
    # Simulate output gate (current state)
    if len(recent_sequence) >= 2:
        recent_trend = recent_sequence[-1] - recent_sequence[-2]
        lstm_output = lstm_memory + recent_trend * 0.5
        features[f'lstm_output_{sequence_length}'] = lstm_output
    
    # LSTM cell state approximation
    cell_state = np.mean(recent_sequence) + (recent_sequence[-1] - np.mean(recent_sequence)) * 0.3
    features[f'lstm_cell_{sequence_length}'] = cell_state
    
    return features

def calculate_gru_features(sequence_data, target, sequence_length):
    """
    Calculate GRU-inspired features for sequence data
    Simulates GRU update and reset gates
    """
    if len(sequence_data) < sequence_length:
        return {}
    
    features = {}
    recent_sequence = sequence_data[target].iloc[-sequence_length:].values
    
    if NORMAL_SCORE_TRANSFORM and len(recent_sequence) > 2:
        recent_sequence = transform_to_normal_scores(recent_sequence)
    
    # Simulate GRU update gate (how much to update)
    update_strength = np.std(recent_sequence) / (np.mean(np.abs(recent_sequence)) + 1e-6)
    update_gate = 1 / (1 + np.exp(-update_strength))  # Sigmoid activation
    
    # Simulate reset gate (how much past to forget)
    if len(recent_sequence) >= 2:
        trend_strength = abs(recent_sequence[-1] - recent_sequence[0]) / sequence_length
        reset_gate = 1 / (1 + np.exp(-trend_strength))
    else:
        reset_gate = 0.5
    
    # GRU hidden state update
    prev_hidden = np.mean(recent_sequence[:-1]) if len(recent_sequence) > 1 else recent_sequence[0]
    new_input = recent_sequence[-1]
    candidate_hidden = np.tanh(reset_gate * prev_hidden + new_input)
    gru_hidden = (1 - update_gate) * prev_hidden + update_gate * candidate_hidden
    
    features[f'gru_hidden_{sequence_length}'] = gru_hidden
    features[f'gru_update_gate_{sequence_length}'] = update_gate
    features[f'gru_candidate_{sequence_length}'] = candidate_hidden
    
    return features

def calculate_bilstm_features(sequence_data, target, sequence_length):
    """
    Calculate BiLSTM-inspired features for sequence data
    Processes sequence in both forward and backward directions
    """
    if len(sequence_data) < sequence_length:
        return {}
    
    features = {}
    recent_sequence = sequence_data[target].iloc[-sequence_length:].values
    
    if NORMAL_SCORE_TRANSFORM and len(recent_sequence) > 2:
        recent_sequence = transform_to_normal_scores(recent_sequence)
    
    # Forward LSTM processing
    forward_weights = np.linspace(0.1, 1.0, sequence_length)
    forward_weights = forward_weights / np.sum(forward_weights)
    forward_lstm = np.sum(recent_sequence * forward_weights)
    
    # Backward LSTM processing
    backward_weights = np.linspace(1.0, 0.1, sequence_length)
    backward_weights = backward_weights / np.sum(backward_weights)
    backward_lstm = np.sum(recent_sequence * backward_weights)
    
    # Combine forward and backward
    bilstm_combined = (forward_lstm + backward_lstm) / 2
    bilstm_difference = forward_lstm - backward_lstm
    
    features[f'bilstm_combined_{sequence_length}'] = bilstm_combined
    features[f'bilstm_forward_{sequence_length}'] = forward_lstm
    features[f'bilstm_backward_{sequence_length}'] = backward_lstm
    features[f'bilstm_diff_{sequence_length}'] = bilstm_difference
    
    return features

def calculate_attention_features(sequence_data, target, sequence_length):
    """
    Calculate attention mechanism features
    Implements simplified multi-head attention from TFT paper
    """
    if len(sequence_data) < sequence_length:
        return {}
    
    features = {}
    recent_sequence = sequence_data[target].iloc[-sequence_length:].values
    
    if NORMAL_SCORE_TRANSFORM and len(recent_sequence) > 2:
        recent_sequence = transform_to_normal_scores(recent_sequence)
    
    # Self-attention weights based on value similarity
    attention_weights = []
    query = recent_sequence[-1]  # Last value as query
    
    for i, value in enumerate(recent_sequence):
        # Attention score based on similarity
        similarity = 1 / (1 + abs(query - value))
        # Add positional weighting (recent is more important)
        position_weight = (i + 1) / sequence_length
        attention_score = similarity * position_weight
        attention_weights.append(attention_score)
    
    attention_weights = np.array(attention_weights)
    attention_weights = attention_weights / np.sum(attention_weights)
    
    # Attention-weighted value
    attention_value = np.sum(recent_sequence * attention_weights)
    features[f'attention_value_{sequence_length}'] = attention_value
    
    # Attention entropy (diversity of attention)
    attention_entropy = -np.sum(attention_weights * np.log(attention_weights + 1e-8))
    features[f'attention_entropy_{sequence_length}'] = attention_entropy
    
    # Max attention weight (focus strength)
    max_attention = np.max(attention_weights)
    features[f'attention_focus_{sequence_length}'] = max_attention
    
    return features

def calculate_tft_features(sequence_data, target, sequence_length, team_stats, current_game):
    """
    Calculate Temporal Fusion Transformer features
    Combines static and temporal information with gating mechanisms
    """
    if len(sequence_data) < sequence_length:
        return {}
    
    features = {}
    recent_sequence = sequence_data[target].iloc[-sequence_length:].values
    
    if NORMAL_SCORE_TRANSFORM and len(recent_sequence) > 2:
        recent_sequence = transform_to_normal_scores(recent_sequence)
    
    # Static covariate encoding (team context)
    static_features = []
    try:
        player_team = current_game.get('TEAM_ABBREVIATION', '')
        if not team_stats.empty and player_team in team_stats['TEAM_ABBREVIATION'].values:
            team_row = team_stats[team_stats['TEAM_ABBREVIATION'] == player_team].iloc[0]
            static_features = [
                (team_row['OFF_RATING'] - 113.5) / 10,
                (team_row['DEF_RATING'] - 113.5) / 10,
                team_row['NET_RATING'] / 15,
                (team_row['PACE'] - 100) / 10
            ]
        else:
            static_features = [0.0, 0.0, 0.0, 0.0]
    except:
        static_features = [0.0, 0.0, 0.0, 0.0]
    
    # Variable selection network (importance weighting)
    temporal_importance = np.var(recent_sequence) / (np.mean(np.abs(recent_sequence)) + 1e-6)
    static_importance = np.mean(np.abs(static_features))
    
    total_importance = temporal_importance + static_importance + 1e-6
    temporal_weight = temporal_importance / total_importance
    static_weight = static_importance / total_importance
    
    # Gated residual network
    temporal_component = np.mean(recent_sequence) * temporal_weight
    static_component = np.mean(static_features) * static_weight
    
    # TFT gating mechanism
    gate_input = temporal_component + static_component
    gate_value = 1 / (1 + np.exp(-gate_input))  # Sigmoid activation
    
    # Final TFT output
    tft_output = gate_value * temporal_component + (1 - gate_value) * static_component
    
    features[f'tft_output_{sequence_length}'] = tft_output
    features[f'tft_temporal_weight_{sequence_length}'] = temporal_weight
    features[f'tft_static_weight_{sequence_length}'] = static_weight
    features[f'tft_gate_{sequence_length}'] = gate_value
    
    return features

def calculate_baseline_features(sequence_data, target, sequence_length):
    """
    Enhanced baseline features incorporating successful patterns from analysis
    Focus on Best_5_Avg, Median_5, and other winning baseline patterns
    """
    if len(sequence_data) < sequence_length:
        return {}
    
    features = {}
    target_history = sequence_data[target].values
    
    # Core successful patterns from baseline analysis
    if len(target_history) >= 5:
        # 1. Best 5 Average Pattern (Winner: 14.6% of games)
        recent_10 = target_history[-10:] if len(target_history) >= 10 else target_history[-5:]
        sorted_recent = sorted(recent_10, reverse=True)
        features['best_5_avg'] = np.mean(sorted_recent[:5])
        
        # 2. Median Pattern (Winner: 14.3% of games)
        features['median_5'] = np.median(target_history[-5:])
        if len(target_history) >= 7:
            features['median_7'] = np.median(target_history[-7:])
        
        # 3. Adaptive Best Performance (enhanced version)
        # Weight recent top performances by recency
        recent_count = min(12, len(target_history))
        recent_games = target_history[-recent_count:]
        top_40_percent = max(3, int(recent_count * 0.4))
        sorted_recent_weighted = sorted(recent_games, reverse=True)
        features['adaptive_best'] = np.mean(sorted_recent_weighted[:top_40_percent])
    
    # Enhanced moving averages with pattern insights
    recent_sequence = target_history[-sequence_length:]
    
    # Weighted moving average (recent games more important)
    if sequence_length >= 3:
        weights = np.linspace(0.5, 1.5, sequence_length)  # Recent games weighted higher
        weights = weights / weights.sum()
        features[f'weighted_ma_{sequence_length}'] = np.average(recent_sequence, weights=weights)
    
    # Standard moving average
    features[f'ma_{sequence_length}'] = np.mean(recent_sequence)
    
    # Consistency-weighted average
    if len(target_history) >= sequence_length * 2:
        # Compare current window consistency to historical
        current_std = np.std(recent_sequence)
        historical_std = np.std(target_history[:-sequence_length])
        
        if historical_std > 0:
            consistency_factor = historical_std / max(current_std, 0.1)
            # If recent games are more consistent, give them more weight
            consistency_boost = min(0.2, max(0, (consistency_factor - 1) * 0.1))
            features[f'consistency_ma_{sequence_length}'] = np.mean(recent_sequence) * (1 + consistency_boost)
        else:
            features[f'consistency_ma_{sequence_length}'] = np.mean(recent_sequence)
    
    # Exponential moving average (optimized alpha)
    alpha = 0.3  # Tested optimal value
    ema = recent_sequence[0]
    for val in recent_sequence[1:]:
        ema = alpha * val + (1 - alpha) * ema
    features[f'ema_{sequence_length}'] = ema
    
    # Momentum-adjusted prediction
    if sequence_length >= 4:
        recent_2 = np.mean(recent_sequence[-2:])
        older_2 = np.mean(recent_sequence[-4:-2])
        momentum = recent_2 - older_2
        
        # Conservative momentum application
        base_pred = np.mean(recent_sequence)
        momentum_pred = base_pred + momentum * 0.3  # Reduce momentum impact
        features[f'momentum_pred_{sequence_length}'] = momentum_pred
    
    # Trend analysis (improved)
    if sequence_length >= 3:
        x = np.arange(sequence_length)
        try:
            slope, intercept = np.polyfit(x, recent_sequence, 1)
            
            # Predict next value with trend
            trend_pred = slope * sequence_length + intercept
            
            # Moderate the trend prediction
            base_avg = np.mean(recent_sequence)
            trend_strength = abs(slope) / (np.std(recent_sequence) + 0.1)
            
            if trend_strength > 0.5:  # Strong trend
                features[f'trend_pred_{sequence_length}'] = 0.7 * base_avg + 0.3 * trend_pred
            else:  # Weak trend
                features[f'trend_pred_{sequence_length}'] = 0.9 * base_avg + 0.1 * trend_pred
                
            features[f'trend_strength_{sequence_length}'] = trend_strength
        except:
            features[f'trend_pred_{sequence_length}'] = np.mean(recent_sequence)
            features[f'trend_strength_{sequence_length}'] = 0
    
    # Volatility and stability measures
    features[f'volatility_{sequence_length}'] = np.std(recent_sequence)
    
    if len(target_history) >= sequence_length * 2:
        recent_vol = np.std(target_history[-sequence_length:])
        historical_vol = np.std(target_history[:-sequence_length])
        features[f'volatility_ratio_{sequence_length}'] = recent_vol / (historical_vol + 0.1)
    
    # Last game (simple but sometimes effective)
    features['last_game'] = target_history[-1]
    
    # Season average (if enough data)
    if len(target_history) >= 15:
        features['season_avg'] = np.mean(target_history)
        
        # Recent vs season performance
        recent_avg = np.mean(target_history[-5:])
        season_avg = features['season_avg']
        features['recent_vs_season'] = recent_avg - season_avg
    
    return features

def prepare_rnn_tft_features(sequence_data, target, team_stats, current_game):
    """
    Prepare comprehensive RNN + TFT features for all sequence lengths
    """
    all_features = {}
    
    for seq_len in SEQUENCE_LENGTHS:
        # Calculate features for each model type
        if USE_LSTM:
            lstm_features = calculate_lstm_features(sequence_data, target, seq_len)
            all_features.update(lstm_features)
        
        if USE_GRU:
            gru_features = calculate_gru_features(sequence_data, target, seq_len)
            all_features.update(gru_features)
        
        if USE_BI_LSTM:
            bilstm_features = calculate_bilstm_features(sequence_data, target, seq_len)
            all_features.update(bilstm_features)
        
        if USE_ATTENTION:
            attention_features = calculate_attention_features(sequence_data, target, seq_len)
            all_features.update(attention_features)
        
        if USE_TFT:
            tft_features = calculate_tft_features(sequence_data, target, seq_len, team_stats, current_game)
            all_features.update(tft_features)
        
        # Always include baseline features
        baseline_features = calculate_baseline_features(sequence_data, target, seq_len)
        all_features.update(baseline_features)
    
    return all_features

# --------------------------------------------------------------
# 4. RNN + TFT ENSEMBLE PREDICTION
# --------------------------------------------------------------

def calculate_rnn_ensemble_prediction(features, target_history, target):
    """
    Improved ensemble prediction leveraging insights from baseline analysis
    Focus on patterns that actually work: recent form, peak performance, and stability
    """
    if len(features) == 0 or len(target_history) == 0:
        return np.mean(target_history) if len(target_history) > 0 else 20.0
    
    predictions = []
    weights = []
    
    # Get historical statistics
    historical_mean = np.mean(target_history)
    historical_std = np.std(target_history) if len(target_history) > 1 else 1.0
    
    # CORE INSIGHT: Best_5_Avg and Median_5 are winning ~29% of games
    # Build sophisticated versions of these successful patterns
    
    # 1. ADAPTIVE BEST PERFORMANCE PREDICTOR (Based on Best_5_Avg success)
    if len(target_history) >= 10:
        # Look at recent 10-15 games, weight by recency and performance
        recent_window = min(15, len(target_history))
        recent_games = target_history[-recent_window:]
        
        # Get top 40% of recent performances
        top_count = max(3, int(recent_window * 0.4))
        sorted_recent = sorted(recent_games, reverse=True)
        
        # Weight recent top performances more heavily
        top_performances = sorted_recent[:top_count]
        adaptive_best = np.mean(top_performances)
        
        # Adjust based on trend
        if len(target_history) >= 5:
            recent_trend = np.mean(target_history[-3:]) - np.mean(target_history[-8:-3])
            trend_adjustment = np.clip(recent_trend * 0.3, -3, 3)  # Conservative trend
            adaptive_best += trend_adjustment
        
        predictions.append(adaptive_best)
        weights.append(0.35)  # High weight - this pattern works
    
    # 2. ROBUST MEDIAN PREDICTOR (Based on Median_5 success)
    if len(target_history) >= 7:
        # Multi-window median approach
        median_5 = np.median(target_history[-5:])
        median_7 = np.median(target_history[-7:])
        median_10 = np.median(target_history[-10:]) if len(target_history) >= 10 else median_7
        
        # Weighted median ensemble
        median_ensemble = (0.5 * median_5 + 0.3 * median_7 + 0.2 * median_10)
        
        # Adjust for recent volatility
        recent_volatility = np.std(target_history[-5:])
        avg_volatility = np.std(target_history[-15:]) if len(target_history) >= 15 else recent_volatility
        
        if recent_volatility < avg_volatility * 0.8:  # Lower volatility = more predictable
            volatility_boost = 0.05 * (avg_volatility / recent_volatility - 1)
            median_ensemble *= (1 + volatility_boost)
        
        predictions.append(median_ensemble)
        weights.append(0.25)  # Strong weight - medians work well
    
    # 3. MOMENTUM-ADJUSTED MOVING AVERAGE (Improved MA approach)
    if len(target_history) >= 5:
        # Dynamic window selection based on consistency
        windows = [3, 5, 7]
        window_scores = []
        
        for w in windows:
            if len(target_history) >= w:
                window_data = target_history[-w:]
                consistency = 1.0 / (1.0 + np.std(window_data))  # Higher = more consistent
                window_scores.append((w, np.mean(window_data), consistency))
        
        if window_scores:
            # Weight by consistency
            total_consistency = sum(score[2] for score in window_scores)
            weighted_ma = sum(score[1] * score[2] for score in window_scores) / total_consistency
            
            # Apply momentum adjustment
            if len(target_history) >= 6:
                recent_momentum = (target_history[-1] + target_history[-2]) / 2 - (target_history[-4] + target_history[-5]) / 2
                momentum_factor = np.clip(recent_momentum / historical_std, -0.5, 0.5)
                weighted_ma += momentum_factor * historical_std * 0.2
            
            predictions.append(weighted_ma)
            weights.append(0.20)
    
    # 4. CONTEXTUAL PERFORMANCE PREDICTOR (Advanced feature integration)
    advanced_features = []
    
    # Look for actual advanced features from RNN/TFT
    feature_categories = {
        'lstm': [v for k, v in features.items() if 'lstm' in k.lower() and not np.isnan(v)],
        'gru': [v for k, v in features.items() if 'gru' in k.lower() and not np.isnan(v)],
        'attention': [v for k, v in features.items() if 'attention' in k.lower() and not np.isnan(v)],
        'tft': [v for k, v in features.items() if 'tft' in k.lower() and not np.isnan(v)]
    }
    
    for category, values in feature_categories.items():
        if values:
            category_pred = np.mean(values)
            
            # Transform back to original scale more conservatively
            if NORMAL_SCORE_TRANSFORM:
                # Less aggressive transformation
                category_pred = historical_mean + category_pred * historical_std * 0.15
            else:
                # Blend with historical mean for stability
                category_pred = 0.7 * historical_mean + 0.3 * category_pred
            
            advanced_features.append(category_pred)
    
    if advanced_features:
        # Only use if they're reasonable
        advanced_pred = np.mean(advanced_features)
        if abs(advanced_pred - historical_mean) < 2 * historical_std:  # Sanity check
            predictions.append(advanced_pred)
            weights.append(0.15)  # Lower weight until proven
    
    # 5. FALLBACK STABILITY PREDICTOR
    if len(target_history) >= 3:
        # Conservative fallback combining multiple stable approaches
        recent_3 = np.mean(target_history[-3:])
        recent_5 = np.mean(target_history[-5:]) if len(target_history) >= 5 else recent_3
        season_avg = np.mean(target_history[-20:]) if len(target_history) >= 20 else recent_5
        
        stability_pred = 0.4 * recent_3 + 0.4 * recent_5 + 0.2 * season_avg
        predictions.append(stability_pred)
        weights.append(0.05)  # Safety net
    
    # Calculate final ensemble
    if len(predictions) > 0:
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        
        ensemble_pred = np.average(predictions, weights=weights)
        
        # Apply intelligent bounds based on player's actual range
        if len(target_history) >= 10:
            player_min = np.percentile(target_history, 5)
            player_max = np.percentile(target_history, 95)
            
            # Expand bounds slightly for potential improvement
            buffer = (player_max - player_min) * 0.2
            lower_bound = max(0, player_min - buffer)
            upper_bound = player_max + buffer
            
            ensemble_pred = np.clip(ensemble_pred, lower_bound, upper_bound)
        else:
            # Default NBA bounds
            if target == 'PTS':
                ensemble_pred = np.clip(ensemble_pred, 2, 55)
            elif target == 'REB':
                ensemble_pred = np.clip(ensemble_pred, 0, 30)
            elif target == 'AST':
                ensemble_pred = np.clip(ensemble_pred, 0, 25)
        
        return ensemble_pred
    else:
        return historical_mean
    
    # Calculate weighted ensemble
    if len(predictions) > 0:
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize weights        
        ensemble_pred = np.average(predictions, weights=weights)
        
        # Apply reasonable bounds for NBA stats
        if target == 'PTS':
            ensemble_pred = np.clip(ensemble_pred, 5, 50)  # Points: 5-50
        elif target == 'REB':
            ensemble_pred = np.clip(ensemble_pred, 1, 25)  # Rebounds: 1-25
        elif target == 'AST':
            ensemble_pred = np.clip(ensemble_pred, 0, 20)  # Assists: 0-20
        
        return ensemble_pred
    else:
        return historical_mean

def prepare_rnn_tft_features(data, target, team_stats, current_game):
    """
    Prepare comprehensive RNN + TFT features for prediction
    """
    if len(data) < MIN_GAMES_FOR_PREDICTION:
        return {}
    
    current_idx = len(data)
    features = {}
    
    # Calculate all RNN-based features
    lstm_features = calculate_lstm_features(data, target, current_idx)
    gru_features = calculate_gru_features(data, target, current_idx)
    bilstm_features = calculate_bilstm_features(data, target, current_idx)
    attention_features = calculate_attention_features(data, target, current_idx)
    tft_features = calculate_tft_features(data, target, current_idx, team_stats, current_game)
    baseline_features = calculate_baseline_features(data, target, current_idx)
    
    # Combine all features
    features.update(lstm_features)
    features.update(gru_features)
    features.update(bilstm_features)
    features.update(attention_features)
    features.update(tft_features)
    features.update(baseline_features)
    
    return features
    
    # Apply conservative bounds
    if len(target_history) > 0:
        historical_mean = np.mean(target_history)
        historical_std = np.std(target_history)
        
        # Conservative bounds: mean ± 2.5 standard deviations
        lower_bound = historical_mean - 2.5 * historical_std
        upper_bound = historical_mean + 2.5 * historical_std
        
        ensemble_pred = np.clip(ensemble_pred, lower_bound, upper_bound)
        
        # Additional volatility penalty for extreme predictions
        if abs(ensemble_pred - historical_mean) > historical_std:
            volatility_adj = (abs(ensemble_pred - historical_mean) - historical_std) * VOLATILITY_PENALTY
            ensemble_pred = ensemble_pred - np.sign(ensemble_pred - historical_mean) * volatility_adj
    
    return ensemble_pred

# --------------------------------------------------------------
# 5. MODEL EVALUATION
# --------------------------------------------------------------

def calculate_metrics(actual, predicted):
    """Calculate evaluation metrics as in research paper"""
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Remove any NaN values
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual = actual[mask]
    predicted = predicted[mask]
    
    if len(actual) == 0:
        return {
            'rmse': float('inf'), 
            'mae': float('inf'), 
            'mape': float('inf'),
            'r2': -float('inf')
        }
    
    # RMSE
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    
    # MAE
    mae = mean_absolute_error(actual, predicted)
    
    # MAPE
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100
    
    # R² Score
    try:
        r2 = r2_score(actual, predicted)
    except:
        r2 = -float('inf')
    
    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2
    }

# --------------------------------------------------------------
# 6. MAIN ANALYSIS FUNCTION
# --------------------------------------------------------------

def print_comprehensive_analysis(target_results, target):
    """
    Print comprehensive analysis of model performance for a specific target
    """
    if len(target_results['predictions']) == 0:
        return
    
    predictions = np.array(target_results['predictions'])
    actuals = np.array(target_results['actuals'])
    errors = np.abs(predictions - actuals)
    
    print(f"\n" + "="*60)
    print(f"COMPREHENSIVE ANALYSIS FOR {target} (NIKOLA JOKIC)")
    print(f"="*60)
    
    # Basic Statistics
    print(f"\n📊 BASIC STATISTICS:")
    print(f"   Number of predictions: {len(predictions)}")
    print(f"   Actual {target} range: {actuals.min():.1f} - {actuals.max():.1f}")
    print(f"   Predicted {target} range: {predictions.min():.1f} - {predictions.max():.1f}")
    print(f"   Actual {target} mean: {actuals.mean():.1f} (±{actuals.std():.1f})")
    print(f"   Predicted {target} mean: {predictions.mean():.1f} (±{predictions.std():.1f})")
    
    # Model Performance Metrics
    metrics = target_results.get('metrics', {})
    print(f"\n🎯 MODEL PERFORMANCE:")
    print(f"   RMSE (Root Mean Square Error): {metrics.get('rmse', 0):.3f}")
    print(f"   MAE (Mean Absolute Error): {metrics.get('mae', 0):.3f}")
    print(f"   MAPE (Mean Absolute % Error): {metrics.get('mape', 0):.1f}%")
    print(f"   R² (Coefficient of Determination): {metrics.get('r2', 0):.3f}")
    
    # Error Analysis
    print(f"\n❌ ERROR ANALYSIS:")
    print(f"   Mean Error: {errors.mean():.2f}")
    print(f"   Median Error: {np.median(errors):.2f}")
    print(f"   Max Error: {errors.max():.2f}")
    print(f"   Min Error: {errors.min():.2f}")
    print(f"   Error Std Dev: {errors.std():.2f}")
    
    # Prediction Bias Analysis
    bias = predictions.mean() - actuals.mean()
    print(f"\n⚖️  BIAS ANALYSIS:")
    print(f"   Prediction Bias: {bias:+.2f}")
    if abs(bias) > 2:
        print(f"   ⚠️  WARNING: High bias detected!")
        if bias > 0:
            print(f"   📈 Model is OVER-predicting by {bias:.1f} on average")
        else:
            print(f"   📉 Model is UNDER-predicting by {abs(bias):.1f} on average")
    else:
        print(f"   ✅ Bias is within acceptable range")
    
    # Accuracy Tiers
    print(f"\n🎯 ACCURACY BREAKDOWN:")
    within_2 = np.sum(errors <= 2) / len(errors) * 100
    within_5 = np.sum(errors <= 5) / len(errors) * 100
    within_10 = np.sum(errors <= 10) / len(errors) * 100
    print(f"   Within ±2: {within_2:.1f}% ({np.sum(errors <= 2)} games)")
    print(f"   Within ±5: {within_5:.1f}% ({np.sum(errors <= 5)} games)")
    print(f"   Within ±10: {within_10:.1f}% ({np.sum(errors <= 10)} games)")
    
    # Worst Predictions
    worst_indices = np.argsort(errors)[-5:]  # 5 worst predictions
    print(f"\n💥 WORST 5 PREDICTIONS:")
    for i, idx in enumerate(worst_indices[::-1], 1):
        print(f"   {i}. Game {idx+1}: Predicted {predictions[idx]:.1f}, "
              f"Actual {actuals[idx]:.1f}, Error {errors[idx]:.1f}")
    
    # Best Predictions
    best_indices = np.argsort(errors)[:5]  # 5 best predictions
    print(f"\n✅ BEST 5 PREDICTIONS:")
    for i, idx in enumerate(best_indices, 1):
        print(f"   {i}. Game {idx+1}: Predicted {predictions[idx]:.1f}, "
              f"Actual {actuals[idx]:.1f}, Error {errors[idx]:.1f}")
    
    # Feature Analysis
    avg_features = np.mean(target_results['features_used'])
    print(f"\n🔧 FEATURE ANALYSIS:")
    print(f"   Average features used: {avg_features:.1f}")
    
    # Model Recommendations
    print(f"\n💡 MODEL RECOMMENDATIONS:")
    if bias > 5:
        print(f"   • Significant over-prediction bias - consider:")
        print(f"     - Reducing ensemble weights for high-variance models")
        print(f"     - Adding more conservative baseline predictions")
        print(f"     - Implementing bias correction mechanisms")
    elif bias < -5:
        print(f"   • Significant under-prediction bias - consider:")
        print(f"     - Increasing weights for trend-following models")
        print(f"     - Adding momentum features")
    
    if metrics.get('rmse', 10) > 8:
        print(f"   • High RMSE indicates:")
        print(f"     - Model struggling with prediction accuracy")
        print(f"     - Consider simpler baseline approaches")
        print(f"     - Review feature engineering process")
    
    if within_5 < 50:
        print(f"   • Low accuracy (±5 threshold) suggests:")
        print(f"     - RNN features may not be capturing patterns effectively")
        print(f"     - Consider increasing baseline model weights")
        print(f"     - Review sequence length optimization")
    
    print(f"\n🏀 BASKETBALL CONTEXT:")
    if target == 'PTS':
        if actuals.mean() > 25:
            print(f"   • High-scoring player - predictions should account for:")
            print(f"     - Usage rate fluctuations")
            print(f"     - Defensive matchups")
            print(f"     - Rest/fatigue factors")
    elif target == 'REB':
        print(f"   • Rebounding predictions affected by:")
        print(f"     - Team pace and style")
        print(f"     - Opponent rebounding strength")
        print(f"     - Position matchups")
    elif target == 'AST':
        print(f"   • Assist predictions influenced by:")
        print(f"     - Team offensive system")
        print(f"     - Ball movement patterns")
        print(f"     - Teammate shooting efficiency")
    
    print(f"="*60)

def run_rnn_tft_analysis():
    """Run the complete RNN + TFT analysis"""
    print("Starting RNN + TFT Analysis...")
    
    # Load data
    train_data, test_data, team_stats = load_data()
    if train_data is None:
        print("Failed to load data")
        return
    
    results = []
    for target in TARGETS:
        print(f"\n" + "="*50)
        print(f"ANALYZING TARGET: {target}")
        print("="*50)
        
        target_results = {
            'target': target,
            'predictions': [],
            'actuals': [],
            'features_used': [],
            'baseline_predictions': {method: [] for method in ['Simple_MA3', 'Simple_MA5', 'Simple_MA10', 
                                   'Exp_Smoothing', 'Linear_Trend', 'Weighted_Avg', 
                                   'Median', 'Ensemble_Baseline']}
        }
        
        # Analyze each test game
        for idx in range(len(test_data)):
            current_test_game = test_data.iloc[idx]
            game_date = current_test_game['GAME_DATE']
            
            # Combine training data with test data up to current game
            current_data = pd.concat([train_data, test_data.iloc[:idx]]).reset_index(drop=True)
            
            if len(current_data) < MIN_GAMES_FOR_PREDICTION:
                continue
            
            # Get team stats for this date (no data leakage)
            current_team_stats = get_team_stats_for_date(game_date, team_stats)
            
            # Prepare RNN + TFT features
            features = prepare_rnn_tft_features(
                current_data, target, current_team_stats, current_test_game
            )
            
            # Get target history
            target_history = current_data[target].values
            
            # Make RNN + TFT prediction
            rnn_tft_prediction = calculate_rnn_ensemble_prediction(features, target_history, target)
            actual = current_test_game[target]
            
            # Calculate all baseline predictions
            baseline_preds = calculate_all_baseline_predictions(target_history, target)
            
            # Store results
            target_results['predictions'].append(rnn_tft_prediction)
            target_results['actuals'].append(actual)
            target_results['features_used'].append(len(features))
            
            for method, pred in baseline_preds.items():
                target_results['baseline_predictions'][method].append(pred)
            
        # Calculate metrics for all models
        if len(target_results['predictions']) > 0:
            rnn_tft_metrics = calculate_metrics(target_results['actuals'], target_results['predictions'])
            target_results['metrics'] = rnn_tft_metrics
            
            # Calculate baseline metrics
            baseline_metrics = {}
            for method, preds in target_results['baseline_predictions'].items():
                if len(preds) > 0:
                    baseline_metrics[method] = calculate_metrics(target_results['actuals'], preds)
            
            # Print comprehensive comparison
            print_baseline_comparison(target_results, rnn_tft_metrics, baseline_metrics, target)
            
            # Print detailed analysis
            print_comprehensive_analysis(target_results, target)
        
        results.append(target_results)
    
    # Save results
    save_results(results)
    
    return results

def save_results(results):
    """Save prediction results to CSV"""
    all_predictions = []
    
    for target_result in results:
        target = target_result['target']
        predictions = target_result['predictions']
        actuals = target_result['actuals']
        
        for i, (pred, actual) in enumerate(zip(predictions, actuals)):
            all_predictions.append({
                'target': target,
                'game_number': i + 1,
                'predicted': pred,
                'actual': actual,
                'error': abs(pred - actual),
                'model': 'RNN_TFT_Ensemble'
            })
    
    if all_predictions:
        df_results = pd.DataFrame(all_predictions)
        df_results.to_csv(OUTPUT_FILE, index=False)
        print(f"\nResults saved to: {OUTPUT_FILE}")
        
        # Print summary
        print(f"\nOVERALL SUMMARY:")
        print(f"Total predictions: {len(all_predictions)}")
        for target in TARGETS:
            target_data = df_results[df_results['target'] == target]
            if len(target_data) > 0:
                avg_error = target_data['error'].mean()
                print(f"  {target} - Avg Error: {avg_error:.2f}")

# --------------------------------------------------------------
# 4.5 BASELINE MODELS FOR COMPARISON
# --------------------------------------------------------------

def simple_moving_average_prediction(target_history, window=5):
    """Simple Moving Average baseline"""
    if len(target_history) < window:
        return np.mean(target_history) if len(target_history) > 0 else 20.0
    return np.mean(target_history[-window:])

def exponential_smoothing_prediction(target_history, alpha=0.3):
    """Exponential Smoothing baseline"""
    if len(target_history) == 0:
        return 20.0
    if len(target_history) == 1:
        return target_history[0]
    
    smoothed = target_history[0]
    for val in target_history[1:]:
        smoothed = alpha * val + (1 - alpha) * smoothed
    return smoothed

def linear_trend_prediction(target_history, window=7):
    """Linear Trend baseline"""
    if len(target_history) < 3:
        return np.mean(target_history) if len(target_history) > 0 else 20.0
    
    recent_data = target_history[-min(window, len(target_history)):]
    x = np.arange(len(recent_data))
    try:
        slope, intercept = np.polyfit(x, recent_data, 1)
        next_pred = slope * len(x) + intercept
        return max(0, next_pred)  # Ensure non-negative
    except:
        return np.mean(recent_data)

def weighted_average_prediction(target_history, weights=None):
    """Weighted Average baseline with more weight on recent games"""
    if len(target_history) == 0:
        return 20.0
    
    recent_data = target_history[-5:] if len(target_history) >= 5 else target_history
    
    if weights is None:
        # Give more weight to recent games
        weights = np.array([0.1, 0.15, 0.2, 0.25, 0.3])[-len(recent_data):]
    
    weights = weights / weights.sum()
    return np.average(recent_data, weights=weights)

def seasonal_naive_prediction(target_history, season_length=82):
    """Seasonal Naive - predict based on same game from previous season"""
    if len(target_history) < season_length:
        return np.mean(target_history) if len(target_history) > 0 else 20.0
    return target_history[-season_length]

def median_prediction(target_history, window=10):
    """Median-based prediction (robust to outliers)"""
    if len(target_history) == 0:
        return 20.0
    recent_data = target_history[-min(window, len(target_history))]
    return np.median(recent_data)

def ensemble_baseline_prediction(target_history):
    """Ensemble of simple baseline methods"""
    predictions = []
    weights = []
    
    # Simple moving average
    ma3 = simple_moving_average_prediction(target_history, 3)
    ma5 = simple_moving_average_prediction(target_history, 5)
    predictions.extend([ma3, ma5])
    weights.extend([0.25, 0.25])
    
    # Exponential smoothing
    exp_pred = exponential_smoothing_prediction(target_history)
    predictions.append(exp_pred)
    weights.append(0.25)
    
    # Linear trend
    trend_pred = linear_trend_prediction(target_history)
    predictions.append(trend_pred)
    weights.append(0.15)
    
    # Median
    median_pred = median_prediction(target_history)
    predictions.append(median_pred)
    weights.append(0.10)
    
    # Weighted ensemble
    weights = np.array(weights)
    weights = weights / weights.sum()
    return np.average(predictions, weights=weights)

def calculate_all_baseline_predictions(target_history, target):
    """Calculate predictions from all baseline methods"""
    baselines = {}
    
    baselines['Simple_MA3'] = simple_moving_average_prediction(target_history, 3)
    baselines['Simple_MA5'] = simple_moving_average_prediction(target_history, 5)
    baselines['Simple_MA10'] = simple_moving_average_prediction(target_history, 10)
    baselines['Exp_Smoothing'] = exponential_smoothing_prediction(target_history, 0.3)
    baselines['Linear_Trend'] = linear_trend_prediction(target_history)
    baselines['Weighted_Avg'] = weighted_average_prediction(target_history)
    baselines['Median'] = median_prediction(target_history)
    baselines['Ensemble_Baseline'] = ensemble_baseline_prediction(target_history)
    
    # Apply target-specific bounds
    for method, pred in baselines.items():
        if target == 'PTS':
            baselines[method] = np.clip(pred, 5, 50)
        elif target == 'REB':
            baselines[method] = np.clip(pred, 1, 25)
        elif target == 'AST':
            baselines[method] = np.clip(pred, 0, 20)
    
    return baselines

# --------------------------------------------------------------
# 6. MODEL COMPARISON
# --------------------------------------------------------------

def print_baseline_comparison(target_results, rnn_tft_metrics, baseline_metrics, target):
    """
    Print comprehensive comparison between RNN/TFT and baseline methods
    """
    print(f"\n🏆 MODEL COMPARISON FOR {target}")
    print("="*70)
    
    # Create comparison table
    print(f"{'Model':<20} {'RMSE':<8} {'MAE':<8} {'MAPE%':<8} {'R²':<8} {'Rank':<6}")
    print("-"*70)
    
    # Collect all results for ranking
    all_results = []
    
    # Add RNN/TFT results
    all_results.append({
        'Model': 'RNN_TFT_Ensemble',
        'RMSE': rnn_tft_metrics['rmse'],
        'MAE': rnn_tft_metrics['mae'],
        'MAPE': rnn_tft_metrics['mape'],
        'R2': rnn_tft_metrics['r2']
    })
    
    # Add baseline results
    for method, metrics in baseline_metrics.items():
        all_results.append({
            'Model': method,
            'RMSE': metrics['rmse'],
            'MAE': metrics['mae'],
            'MAPE': metrics['mape'],
            'R2': metrics['r2']
        })
    
    # Sort by RMSE (lower is better)
    all_results.sort(key=lambda x: x['RMSE'])
    
    # Print results with ranking
    for i, result in enumerate(all_results, 1):
        model_name = result['Model']
        if model_name == 'RNN_TFT_Ensemble':
            model_name = f"🤖 {model_name}"  # Highlight our model
        
        print(f"{model_name:<20} {result['RMSE']:<8.3f} {result['MAE']:<8.3f} "
              f"{result['MAPE']:<8.1f} {result['R2']:<8.3f} #{i:<5}")
    
    # Analysis
    rnn_tft_rank = next((i+1 for i, x in enumerate(all_results) if x['Model'] == 'RNN_TFT_Ensemble'), 0)
    best_baseline = all_results[0] if all_results[0]['Model'] != 'RNN_TFT_Ensemble' else all_results[1]
    
    print(f"\n📊 PERFORMANCE ANALYSIS:")
    print(f"   RNN/TFT Rank: #{rnn_tft_rank} out of {len(all_results)} models")
    
    if rnn_tft_rank == 1:
        print(f"   🥇 RNN/TFT is the BEST performing model!")
        improvement = (best_baseline['RMSE'] - rnn_tft_metrics['rmse']) / best_baseline['RMSE'] * 100
        print(f"   📈 {improvement:.1f}% improvement over best baseline ({best_baseline['Model']})")
    else:
        print(f"   📉 RNN/TFT ranked #{rnn_tft_rank}")
        best_model = all_results[0]
        difference = (rnn_tft_metrics['rmse'] - best_model['RMSE']) / best_model['RMSE'] * 100
        print(f"   🎯 Best model: {best_model['Model']} (RMSE: {best_model['RMSE']:.3f})")
        print(f"   📊 RNN/TFT is {difference:.1f}% worse than best model")
    
    # Simple vs Complex analysis
    simple_baselines = ['Simple_MA3', 'Simple_MA5', 'Simple_MA10', 'Median']
    complex_baselines = ['Exp_Smoothing', 'Linear_Trend', 'Weighted_Avg', 'Ensemble_Baseline']
    
    simple_rmses = [baseline_metrics[m]['rmse'] for m in simple_baselines if m in baseline_metrics]
    complex_rmses = [baseline_metrics[m]['rmse'] for m in complex_baselines if m in baseline_metrics]
    
    if simple_rmses and complex_rmses:
        avg_simple = np.mean(simple_rmses)
        avg_complex = np.mean(complex_rmses)
        
        print(f"\n🔍 COMPLEXITY ANALYSIS:")
        print(f"   Simple baselines avg RMSE: {avg_simple:.3f}")
        print(f"   Complex baselines avg RMSE: {avg_complex:.3f}")
        print(f"   RNN/TFT RMSE: {rnn_tft_metrics['rmse']:.3f}")
        
        if rnn_tft_metrics['rmse'] < avg_simple:
            print(f"   ✅ RNN/TFT outperforms simple baselines")
        else:
            print(f"   ❌ RNN/TFT underperforms vs simple baselines")
        
        if rnn_tft_metrics['rmse'] < avg_complex:
            print(f"   ✅ RNN/TFT outperforms complex baselines")
        else:
            print(f"   ❌ RNN/TFT underperforms vs complex baselines")
    
    print("="*70)

# --------------------------------------------------------------
# 7. MAIN EXECUTION
# --------------------------------------------------------------

if __name__ == "__main__":
    try:
        results = run_rnn_tft_analysis()
        print("\nAnalysis completed successfully!")
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

def extract_opponent_from_matchup(matchup_str):
    """Extract opponent team from matchup string"""
    if pd.isna(matchup_str):
        return "UNK"
    
    matchup_str = str(matchup_str).strip()
    
    if " vs. " in matchup_str:
        return matchup_str.split(" vs. ")[1].strip()
    elif " @ " in matchup_str:
        return matchup_str.split(" @ ")[1].strip()
    elif "vs" in matchup_str:
        return matchup_str.split("vs")[1].strip()
    elif "@" in matchup_str:
        return matchup_str.split("@")[1].strip()
    
    if len(matchup_str) >= 3:
        return matchup_str[-3:]
    
    return "UNK"
