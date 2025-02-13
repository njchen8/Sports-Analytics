import pandas as pd
import numpy as np
from itertools import combinations, product
from scipy.stats import norm

# Load data
correlations = pd.read_csv('/mnt/data/top_teammate_correlations_filtered.csv')
props = pd.read_csv('/mnt/data/underdog_home_team_props.csv')
game_logs = pd.read_csv('/mnt/data/nba_players_game_logs_2018_25.csv')

# Check if 'prop_type' column exists before filtering
if 'prop_type' in props.columns:
    props = props[~props['prop_type'].str.contains('pts_rebs_asts', na=False)]

# Function to estimate probability of hitting a prop based on past data
def estimate_hit_probability(player, prop_type, line):
    if player not in game_logs['player'].values or prop_type not in game_logs.columns:
        return 0.5  # Default probability if data is missing

    player_data = game_logs[game_logs['player'] == player]
    stat_values = player_data[prop_type].dropna()
    if len(stat_values) == 0:
        return 0.5

    mean_val, std_val = stat_values.mean(), stat_values.std()
    if pd.isna(std_val) or std_val == 0:
        return 0.5
    return 1 - norm.cdf(line, loc=mean_val, scale=std_val)

# Calculate EV for each prop
props['probability'] = props.apply(lambda x: estimate_hit_probability(x['player'], x['prop_type'], x['line']), axis=1)
props['EV'] = props['probability'] * (props['odds'] - 1) - (1 - props['probability'])

# Find top correlated players and their props
high_corr_pairs = correlations[correlations['correlation'] > 0.5]

# Generate all parlay combinations
parlay_combinations = []
for _, row in high_corr_pairs.iterrows():
    player1_props = props[props['player'] == row['player1']]
    player2_props = props[props['player'] == row['player2']]
    for _, p1 in player1_props.iterrows():
        for _, p2 in player2_props.iterrows():
            parlay = [p1, p2]
            total_ev = sum(item['EV'] for item in parlay)
            parlay_combinations.append((parlay, total_ev))

# Sort parlays by highest EV
parlay_combinations.sort(key=lambda x: x[1], reverse=True)

# Output top parlays
top_parlays = parlay_combinations[:10]

for i, (parlay, ev) in enumerate(top_parlays, 1):
    print(f"Parlay {i}: EV = {ev:.2f}")
    for prop in parlay:
        print(f"  {prop['player']} - {prop['prop_type']} line: {prop['line']} odds: {prop['odds']} probability: {prop['probability']:.2f}")
    print()
