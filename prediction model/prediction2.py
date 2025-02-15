import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from unidecode import unidecode  # Normalize names

print("Loading CSV files...")

# Load CSV files
underdog_df = pd.read_csv("underdog_home_team_props.csv")
nba_logs_df = pd.read_csv("nba_players_game_logs_2018_25.csv", low_memory=False)
nba_logs_df.columns = nba_logs_df.columns.str.strip().str.upper()
nba_logs_df = nba_logs_df.loc[:, ~nba_logs_df.columns.duplicated()]

print("Preprocessing data...")

# Normalize player names to remove special characters
underdog_df['Player'] = underdog_df['Player'].apply(lambda x: unidecode(str(x).strip().lower()))
nba_logs_df['PLAYER_NAME'] = nba_logs_df['PLAYER_NAME'].apply(lambda x: unidecode(str(x).strip().lower()))

# Helper function to parse numeric values
def parse_numeric(val):
    try:
        if isinstance(val, str):
            part = val.split()[0].replace("x", "")
            return float(part)
        return float(val)
    except:
        return np.nan

underdog_df['Prop Line'] = underdog_df['Prop Line'].apply(parse_numeric)
underdog_df['Higher Payout'] = underdog_df['Higher Payout'].apply(parse_numeric)
underdog_df['Lower Payout'] = underdog_df['Lower Payout'].apply(parse_numeric)

# Mapping props to stats
def map_prop_to_stats(prop_type):
    if "period_1" in prop_type:
        return []
    elif "pts_rebs_asts" in prop_type:
        return ["PTS", "REB", "AST"]
    elif "points" in prop_type:
        return ["PTS"]
    elif "rebounds" in prop_type:
        return ["REB"]
    elif "assists" in prop_type:
        return ["AST"]
    else:
        return []

# Function to calculate weighted mean and variance
# Function to calculate weighted mean and variance with log transformation
def weighted_mean_and_variance(series, dates, decay_rate, min_variance=5.0, apply_log=False, weights=None):
    """
    Calculate the weighted mean and variance using exponential decay weights.
    
    Parameters:
        series (pd.Series): The data series (e.g., player stats).
        dates (pd.Series): The corresponding dates or match indices for the series.
        decay_rate (float): The exponential decay rate.
        min_variance (float): Minimum variance to prevent overconfidence.
        apply_log (bool): Whether to apply a log transformation to the series.
    
    Returns:
        weighted_mean (float): The weighted mean of the series.
        weighted_variance (float): The weighted variance of the series.
        num_data_points (int): The number of data points used.
        date_range (str): The date range of the matches used.
        unweighted_mean (float): The unweighted mean of the series.
    """
    # Ensure we only use the last 25 matches
    series = series.tail(25)
    dates = dates.tail(25)
    
    # Apply log transformation if specified
    if apply_log:
        series = np.log(series + 1)  # Add 1 to handle zeros
    
    # Calculate the date range
    date_range = f"{dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}"
    
    # Calculate the unweighted mean
    unweighted_mean = series.mean()
    
    # Calculate the time difference (in days) from the most recent date
    time_diff = (dates.max() - dates).dt.days
    
    # Apply decay every 5 matches
    match_group = np.arange(len(series)) // 5  # Group matches into sets of 5
    decay_factor = np.exp(-decay_rate * match_group)  # Apply decay factor to each group
    
    # Calculate exponential decay weights
    weights = np.exp(-decay_rate * time_diff) * decay_factor
    weights /= weights.sum()  # Normalize weights to sum to 1
    
    # Calculate weighted mean
    weighted_mean = np.average(series, weights=weights)
    
    # Calculate weighted variance with a minimum threshold
    weighted_variance = np.average((series - weighted_mean)**2, weights=weights)
    weighted_variance = max(weighted_variance, min_variance)  # Apply minimum variance
    
    # Number of data points used
    num_data_points = len(series)
    
    return weighted_mean, weighted_variance, num_data_points, date_range, unweighted_mean

# Function to find the optimal decay rate
def find_optimal_decay_rate(series, dates, metric='mse', bounds=(0.001, 1.0)):
    """
    Find the optimal decay rate that minimizes a given metric (e.g., mean squared error).
    
    Parameters:
        series (pd.Series): The data series (e.g., player stats).
        dates (pd.Series): The corresponding dates or match indices for the series.
        metric (str): The metric to optimize ('mse' for mean squared error).
        bounds (tuple): The bounds for the decay rate search.
    
    Returns:
        optimal_decay_rate (float): The optimal decay rate.
    """
    def objective(decay_rate):
        # Calculate weighted mean and variance
        weighted_mean, _, _, _, _ = weighted_mean_and_variance(series, dates, decay_rate)  # Unpack all five values
        
        # Calculate the metric to optimize
        if metric == 'mse':
            # Mean squared error between weighted mean and actual values
            return np.mean((series - weighted_mean)**2)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    
    # Find the decay rate that minimizes the objective function
    result = minimize_scalar(objective, bounds=bounds, method='bounded')
    return result.x

# Function to print calculations for top parlays
def print_top_parlays(prop_results_df, top_n=10):
    """
    Print the calculations for the top parlays, including EV, variance, data points, date range, and unweighted average.
    
    Parameters:
        prop_results_df (pd.DataFrame): The DataFrame containing prop results.
        top_n (int): The number of top parlays to print.
    """
    # Sort by EV and select the top N parlays
    top_parlays = prop_results_df.sort_values("EV", ascending=False).head(top_n)
    
    print("Top Parlays Calculations:")
    print("=" * 50)
    for idx, row in top_parlays.iterrows():
        player = row['Player']
        stat_type = row['Stat Type']
        prop_line = row['Prop Line']
        ev = row['EV']
        prob_hit = row['Probability_Hit']
        variance = row['Variance']
        num_data_points = row['Num_Data_Points']
        date_range = row['Date Range']
        unweighted_avg = row['Unweighted Average']
        
        print(f"Player: {player}")
        print(f"Stat Type: {stat_type}")
        print(f"Prop Line: {prop_line}")
        print(f"Weighted Average: {row['Predicted']:.2f}")
        print(f"Expected Value (EV): {ev:.6f}")
        print(f"Probability Hit: {prob_hit:.6f}")
        print(f"Variance: {variance:.6f}")
        print(f"Number of Data Points: {num_data_points}")
        print(f"Date Range: {date_range}")
        print(f"Unweighted Average: {unweighted_avg:.2f}")
        print("-" * 50)

from scipy.optimize import minimize_scalar

def find_optimal_team_weight(player_logs, team_weight, current_team, opponent_team):
    """
    Find the optimal weight for games where the player is playing a specific team.
    
    Parameters:
        player_logs (pd.DataFrame): The player's game logs.
        team_weight (float): The weight to apply for games against the specified team.
        current_team (str): The player's current team.
        opponent_team (str): The opponent team.
    
    Returns:
        optimal_weight (float): The optimal weight for games against the specified team.
    """
    def objective(weight):
        # Ensure the required columns exist
        if 'TEAM_ABBREVIATION' not in player_logs.columns or 'MATCHUP' not in player_logs.columns or 'PTS' not in player_logs.columns:
            raise KeyError("Required columns ('TEAM_ABBREVIATION', 'MATCHUP', or 'PTS') are missing in player_logs.")
        
        # Determine which team the player is on and which team they are playing
        player_logs.loc[:, 'Is_Current_Team'] = player_logs['TEAM_ABBREVIATION'] == current_team
        player_logs.loc[:, 'Is_Opponent_Team'] = player_logs['MATCHUP'].str.contains(opponent_team)
        
        # Apply the weight to games against the specified team
        weights = np.where(
            (player_logs['Is_Current_Team'] & player_logs['Is_Opponent_Team']),
            float(weight),  # Apply the weight for games against the specified team
            1.0            # Default weight for other games
        )
        
        # Ensure weights is a NumPy array of floats
        weights = np.array(weights, dtype=float)
        
        # Ensure player_logs['PTS'] is a NumPy array of floats
        pts = np.array(player_logs['PTS'], dtype=float)
        
        # Calculate the weighted mean and variance
        weighted_mean = np.average(pts, weights=weights)
        weighted_variance = np.average((pts - weighted_mean)**2, weights=weights)
        
        # Return the negative expected value (to maximize EV)
        return -weighted_mean  # Negative because we are minimizing the objective function
    
    # Find the optimal weight
    result = minimize_scalar(objective, bounds=(0.1, 10.0), method='bounded')
    return result.x

# Recalculate individual prop predictions with team-specific weighting
prop_results = []
print("Recalculating individual prop EVs with team-specific weighting...")

for idx, row in underdog_df.iterrows():
    player = row['Player']
    current_team = row['Current Team']
    opponent_team = row['Opponent Team']
    stat_type = row['Stat Type']
    prop_line = row['Prop Line']
    payout_high = row['Higher Payout']
    payout_low = row['Lower Payout']
    stats = map_prop_to_stats(stat_type)
    if not stats:
        continue

    player_logs = nba_logs_df[nba_logs_df['PLAYER_NAME'] == player]
    if player_logs.empty:
        print(f"no values for {player}")
        continue

    # Convert the 'MIN' column to numeric
    player_logs.loc[:, 'MIN'] = pd.to_numeric(player_logs['MIN'], errors='coerce')
    
    # Drop rows with NaN values in the 'MIN' column (invalid or non-numeric values)
    player_logs = player_logs.dropna(subset=['MIN'])
    
    # Filter out games where the player had less than 5 minutes of game time
    player_logs = player_logs[player_logs['MIN'] >= 5]

    # Extract dates for the series
    dates = player_logs.loc[:, 'GAME_DATE']  # Ensure 'GAME_DATE' is the correct column name
    dates = pd.to_datetime(dates, errors='coerce')  # Convert to datetime
    
    # Drop rows with missing or invalid dates
    valid_dates = dates.notna()
    player_logs = player_logs[valid_dates]
    dates = dates[valid_dates]
    
    if player_logs.empty:
        continue
    
    # Determine the player's current team based on the most recent game
    most_recent_game = player_logs.iloc[-1]
    current_team = most_recent_game['TEAM_ABBREVIATION']
    
    # Recalculate means and standard deviations
    stat_means, stat_vars, num_data_points_list, date_ranges, unweighted_means = [], [], [], [], []
    for stat in stats:
        if stat not in player_logs.columns:
            continue
        
        # For pts_rebs_asts, sum up points, rebounds, and assists
        if stat_type == "pts_rebs_asts":
            # Ensure the required columns exist
            if not all(col in player_logs.columns for col in ["PTS", "REB", "AST"]):
                continue
            
            # Sum up points, rebounds, and assists for each game
            series = (
                pd.to_numeric(player_logs["PTS"], errors='coerce') +
                pd.to_numeric(player_logs["REB"], errors='coerce') +
                pd.to_numeric(player_logs["AST"], errors='coerce')
            ).dropna()
        else:
            # For other stats, use the stat directly
            series = pd.to_numeric(player_logs[stat], errors='coerce').dropna()
        
        if series.empty:
            continue
        
        # Find the optimal team weight
        try:
            optimal_team_weight = find_optimal_team_weight(player_logs, team_weight=1.0, current_team=current_team, opponent_team=opponent_team)
        except KeyError as e:
            print(f"Skipping {player}: {e}")
            continue
        
        # Apply the optimal team weight to the series
        player_logs.loc[:, 'Is_Current_Team'] = player_logs['TEAM_ABBREVIATION'] == current_team
        player_logs.loc[:, 'Is_Opponent_Team'] = player_logs['MATCHUP'].str.contains(opponent_team)
        weights = np.where(
            (player_logs['Is_Current_Team'] & player_logs['Is_Opponent_Team']),
            float(optimal_team_weight),  # Apply the optimal weight for games against the specified team
            1.0                        # Default weight for other games
        )
        
        # Find the optimal decay rate for this stat
        optimal_decay_rate = find_optimal_decay_rate(series, dates, metric='mse')
        
        # Calculate the weighted mean and variance using the optimal decay rate and team weight
        mean, var, num_data_points, date_range, unweighted_mean = weighted_mean_and_variance(
            series, dates, optimal_decay_rate, min_variance=5.0, apply_log=False, weights=weights
        )
        
        stat_means.append(mean)
        stat_vars.append(var)
        num_data_points_list.append(num_data_points)
        date_ranges.append(date_range)
        unweighted_means.append(unweighted_mean)
    
    if not stat_means:
        continue

    # Calculate the combined mean and standard deviation
    total_mean = np.mean(stat_means)
    total_var = np.sum(stat_vars)  # Variance is summed up for independent stats
    total_std = np.sqrt(total_var) if total_var > 0 else 0.0

    # Handle edge case where standard deviation is 0
    if total_std == 0:
        total_std = 1  # Prevent division by zero, can adjust this based on context

    # Recalculate probability over and under correctly
    prob_over = 1 - norm.cdf(prop_line, loc=total_mean, scale=total_std)
    prob_under = norm.cdf(prop_line, loc=total_mean, scale=total_std)

    # Clip probabilities to a reasonable range
    prob_over = min(max(prob_over, 0.01), 0.99)
    prob_under = min(max(prob_under, 0.01), 0.99)

    # Expected value calculation
    ev_over = (1 + payout_high) * prob_over - (1 * (1 - prob_over))
    ev_under = (1 + payout_low) * prob_under - (1 * (1 - prob_under))
    
    # Store the best EV (whether over or under)
    if ev_over > ev_under:
        best_bet = 'Over'
        best_ev = ev_over
        best_payout = payout_high
        best_prob = prob_over
    else:
        best_bet = 'Under'
        best_ev = ev_under
        best_payout = payout_low
        best_prob = prob_under
    
    # Store the results
    prop_results.append({
        'Player': player,
        'Stat Type': stat_type,
        'Prop Line': prop_line,
        'Predicted': total_mean,
        'Probability_Hit': best_prob,
        'Payout': best_payout,
        'EV': best_ev,
        'Variance': total_var,
        'Num_Data_Points': np.mean(num_data_points_list),  # Average number of data points
        'Date Range': date_ranges[0],  # Use the first date range (all stats share the same range)
        'Unweighted Average': np.mean(unweighted_means),  # Average of unweighted means
        'Optimal Team Weight': optimal_team_weight,  # Optimal weight for games against the specified team
        'Bet Direction': best_bet
    })

# Convert to DataFrame
prop_results_df = pd.DataFrame(prop_results)

# Print calculations for the top parlays
print_top_parlays(prop_results_df, top_n=10)

print("Prop calculations completed.")