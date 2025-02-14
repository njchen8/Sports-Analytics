import pandas as pd
import numpy as np
import csv
from itertools import combinations
from scipy.stats import norm
from unidecode import unidecode  # Normalize names

print("Loading CSV files...")

# Load CSV files
underdog_df = pd.read_csv("underdog.csv")
nba_logs_df = pd.read_csv("nba_players_game_logs_2018_25.csv")
nba_logs_df.columns = nba_logs_df.columns.str.strip().str.upper()
nba_logs_df = nba_logs_df.loc[:, ~nba_logs_df.columns.duplicated()]
correlations_df = pd.read_csv("top_teammate_correlations_filtered.csv")

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
def weighted_mean_and_variance(series, years):
    # Ensure the weights are based on the same number of data points as the series
    weights = np.array([1.0 / (2025 - year) for year in years])  # Weight based on year recency
    if len(weights) != len(series):
        # If weights don't match the size of series, raise an error or adjust logic
        raise ValueError(f"Series length {len(series)} does not match weights length {len(weights)}.")
    
    weighted_mean = np.average(series, weights=weights)
    weighted_variance = np.average((series - weighted_mean)**2, weights=weights)
    return weighted_mean, weighted_variance

# Recalculate individual prop predictions with proper mean and standard deviation calculation
prop_results = []
print("Recalculating individual prop EVs with adjusted means and std deviations...")

# Prepare list for prop data (EVs and probabilities)
prop_data = []

for idx, row in underdog_df.iterrows():
    player = row['Player']
    stat_type = row['Stat Type']
    prop_line = row['Prop Line']
    payout_high = row['Higher Payout']
    payout_low = row['Lower Payout']
    current_team = row['Current Team']
    opponent_team = row['Opponent Team']
    stats = map_prop_to_stats(stat_type)
    if not stats:
        continue

    player_logs = nba_logs_df[nba_logs_df['PLAYER_NAME'] == player]
    if player_logs.empty:
        print(f"no values for {player}")
        continue

    # Get the seasons
    seasons = player_logs['SEASON'].unique()
    # Convert to numeric for weighted calculations
    try:
        years = player_logs['SEASON'].astype(str).apply(lambda x: int(x.split('-')[0])).values
    except ValueError as e:
        print(f"Error processing season data for {player}: {e}")
        continue

    # Recalculate means and standard deviations
    stat_means, stat_vars = [], []
    for stat in stats:
        if stat not in player_logs.columns:
            continue
        series = pd.to_numeric(player_logs[stat], errors='coerce').dropna()
        if series.empty:
            continue
        # Calculate the weighted mean and variance for this stat
        mean, var = weighted_mean_and_variance(series, years)
        stat_means.append(mean)
        stat_vars.append(var)

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

    # Expected value calculation should account for losing bets too
    ev_over = (1 + payout_high) * prob_over - (1 * (1 - prob_over))  # Lose stake if bet fails
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

    # No debugging logs for negative EV, it's part of the process
    prop_results.append({
        'Player': player,
        'Stat Type': stat_type,
        'Prop Line': prop_line,
        'Predicted': total_mean,
        'Probability_Hit': best_prob,
        'Payout': best_payout,
        'EV': best_ev,
        'Bet Direction': best_bet
    })
    
    # Store the prop data (EV, probability, direction, and prop line) for all props
    prop_data.append({
        'Player': player,
        'Stat Type': stat_type,
        'Prop Line': prop_line,  # Add prop line here
        'EV': best_ev,
        'Probability_Hit': best_prob,
        'Bet Direction': best_bet  # Add bet direction here
    })

print("Individual prop predictions completed.")

# Convert to DataFrame and find best prop per player
prop_results_df = pd.DataFrame(prop_results)
print(f"Total individual props available: {len(prop_results_df)}")
print(prop_results_df[['Player', 'EV', 'Probability_Hit']].head(10))
print("Selecting best prop by EV for each player...")
best_props_df = prop_results_df.sort_values("EV", ascending=False).groupby("Player").first().reset_index()

# Store prop EVs, probabilities, and directions to CSV
prop_data_df = pd.DataFrame(prop_data)
prop_data_output_file = "prediction model/prop_data.csv"
prop_data_df.to_csv(prop_data_output_file, index=False)

print(f"Prop data saved to {prop_data_output_file}")


# Define parlay multipliers
parlay_multiplier = {2: 3, 3: 6, 4: 10, 5: 20, 6: 35, 7: 65, 8: 120}
players_best = best_props_df.to_dict('records')

# File paths
output_file = "prediction model/parlay_analysis_live.csv"
sorted_output_file = "prediction model/parlay_analysis_sorted.csv"

print("Calculating parlays for best player props...")

parlay_results = []
top_props = prop_results_df.sort_values("EV", ascending=False).head(15)  # Consider top 15 bets
print(f"Total props considered for parlays: {len(top_props)}")
print(top_props[['Player', 'EV', 'Probability_Hit']].head(15))

with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Players", "Stat Types", "Bet Direction", "Prop Lines", "Num Props", "Combined_Probability", "Multiplier", "Parlay_EV"])

    for group_size in range(2, 9):  # Limit parlays from 2 to 8 players
        for combo in combinations(top_props.to_dict('records'), group_size):
            players_in_parlay = [p['Player'] for p in combo]
            bet_directions = [p['Bet Direction'] for p in combo]  # Keep track of Over/Under
            prop_lines = [p['Prop Line'] for p in combo]  # Collect prop lines for each bet

            # Avoid stacking too many players from the same team
            teams = [
                nba_logs_df[nba_logs_df['PLAYER_NAME'] == p['Player']]['TEAM_ABBREVIATION'].iloc[0]
                for p in combo if p['Player'] in nba_logs_df['PLAYER_NAME'].values
            ]
            if len(set(teams)) < len(teams) - 1:  # More than 2 players from the same team
                continue  

            combined_prob = np.prod([p['Probability_Hit'] for p in combo])
            multiplier = parlay_multiplier.get(group_size, 1)
            parlay_ev = multiplier * combined_prob

            parlay_results.append({
                'Players': "; ".join(players_in_parlay),
                'Stat Types': "; ".join([p['Stat Type'] for p in combo]),
                'Bet Direction': "; ".join(bet_directions),
                'Prop Lines': "; ".join([str(p['Prop Line']) for p in combo]),  # Add prop lines to results
                'Num Props': group_size,
                'Combined Probability': combined_prob,
                'Multiplier': multiplier,
                'Parlay EV': parlay_ev
            })

            # Write to CSV
            writer.writerow([ 
                "; ".join(players_in_parlay),
                "; ".join([p['Stat Type'] for p in combo]),
                "; ".join(bet_directions),
                "; ".join([str(p['Prop Line']) for p in combo]),  # Include prop lines in the CSV
                group_size, 
                combined_prob, 
                multiplier,
                parlay_ev
            ])

# Sort and save
parlay_results_df = pd.DataFrame(parlay_results)
parlay_results_df_sorted = parlay_results_df.sort_values("Combined Probability", ascending=False)
parlay_results_df_sorted.to_csv(sorted_output_file, index=False)

print("Parlay analysis completed.")
