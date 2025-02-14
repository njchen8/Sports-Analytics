import pandas as pd
import numpy as np
import csv
from itertools import combinations
from scipy.stats import norm
from unidecode import unidecode  # Normalize names

import pandas as pd
import numpy as np
from scipy.stats import norm
from unidecode import unidecode

print("Loading CSV files...")
underdog_df = pd.read_csv(r"underdog_player_props_fixed.csv")
nba_logs_df = pd.read_csv(r"nba_players_game_logs_2018_25.csv")
nba_logs_df.columns = nba_logs_df.columns.str.strip().str.upper()
nba_logs_df = nba_logs_df.loc[:, ~nba_logs_df.columns.duplicated()]

print("Preprocessing data...")
underdog_df['Player'] = underdog_df['Player'].apply(lambda x: unidecode(str(x).strip().lower()))
nba_logs_df['PLAYER_NAME'] = nba_logs_df['PLAYER_NAME'].apply(lambda x: unidecode(str(x).strip().lower()))

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

# Additional code logic...

prop_results_df = pd.DataFrame(prop_results)
expected_columns = ['Player', 'Stat Type', 'Prop Line', 'Bet Direction', 'EV', 'Probability_Hit']

if not all(col in prop_results_df.columns for col in expected_columns):
    print("Error: Missing one or more expected columns in prop_results_df")
    print("Available columns:", prop_results_df.columns.tolist())
else:
    print(prop_results_df[expected_columns].head(10))

def weighted_mean_and_variance(series, years):
    weights = np.array([1.0 / (2025 - year) for year in years])
    if len(weights) != len(series):
        raise ValueError(f"Series length {len(series)} does not match weights length {len(weights)}.")
    weighted_mean = np.average(series, weights=weights)
    weighted_variance = np.average((series - weighted_mean)**2, weights=weights)
    return weighted_mean, weighted_variance

prop_results = []
print("Recalculating individual prop EVs with adjusted means and std deviations...")

prop_data = []

for idx, row in underdog_df.iterrows():
    player = row['Player']
    stat_type = row['Stat Type']
    prop_line = row['Prop Line']
    payout_high = row['Higher Payout']
    payout_low = row['Lower Payout']

    stats = map_prop_to_stats(stat_type)
    if not stats:
        continue

    player_logs = nba_logs_df[nba_logs_df['PLAYER_NAME'] == player]
    if player_logs.empty:
        continue

    try:
        years = player_logs['SEASON'].astype(str).apply(lambda x: int(x.split('-')[0])).values
    except ValueError as e:
        print(f"Error processing season data for {player}: {e}")
        continue

    stat_means, stat_vars = [], []
    for stat in stats:
        if stat not in player_logs.columns:
            continue
        series = pd.to_numeric(player_logs[stat], errors='coerce').dropna()
        if series.empty:
            continue
        mean, var = weighted_mean_and_variance(series, years)
        stat_means.append(mean)
        stat_vars.append(var)

    if not stat_means:
        continue

    total_mean = np.mean(stat_means)
    total_var = np.sum(stat_vars)
    total_std = np.sqrt(total_var) if total_var > 0 else 0.0

    if total_std == 0:
        total_std = 1

    prob_over = 1 - norm.cdf(prop_line, loc=total_mean, scale=total_std)
    prob_under = norm.cdf(prop_line, loc=total_mean, scale=total_std)

    ev_over = (1 + payout_high) * prob_over - (1 * (1 - prob_over))
    ev_under = (1 + payout_low) * prob_under - (1 * (1 - prob_under))

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

    print(f"{player.title()} - {stat_type}: {prop_line} ({best_bet})")

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

    prop_data.append({
        'Player': player,
        'Stat Type': stat_type,
        'Prop Line': prop_line,
        'EV': best_ev,
        'Probability_Hit': best_prob,
        'Bet Direction': best_bet
    })

print("Individual prop predictions completed.")

prop_results_df = pd.DataFrame(prop_results)
print(f"Total individual props available: {len(prop_results_df)}")
print(prop_results_df[['Player', 'Stat Type', 'Prop Line', 'Bet Direction', 'EV', 'Probability_Hit']].head(10))
print("Selecting best prop by EV for each player...")
best_props_df = prop_results_df.sort_values("EV", ascending=False).groupby("Player").first().reset_index()

prop_data_df = pd.DataFrame(prop_data)
prop_data_output_file = r"prop_data.csv"
prop_data_df.to_csv(prop_data_output_file, index=False)

print(f"Prop data saved to {prop_data_output_file}")

parlay_multiplier = {2: 3, 3: 6, 4: 10, 5: 20, 6: 35, 7: 65, 8: 120}
players_best = best_props_df.to_dict('records')

output_file = r"parlay_analysis_live.csv"
sorted_output_file = r"parlay_analysis_sorted.csv"

print("Calculating parlays for best player props...")

top_props = prop_results_df.sort_values("EV", ascending=False).head(15)
print(f"Total props considered for parlays: {len(top_props)}")
print(top_props[['Player', 'Stat Type', 'Prop Line', 'Bet Direction', 'EV', 'Probability_Hit']].head(15))

print("Parlay analysis completed.")
