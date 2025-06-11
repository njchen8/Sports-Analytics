#!/usr/bin/env python
import pandas as pd
import numpy as np
import time
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
from datetime import datetime

def get_latest_stat(player_name, stat_type, season='2024-25'):
    """
    Given a player's full name and a stat type, this function returns the most recent game's stat.
    """
    player_list = players.find_players_by_full_name(player_name)
    if not player_list:
        print(f"NBA API: No data found for {player_name}.")
        return None, None, None
    player_id = player_list[0]['id']
    
    try:
        # Pause to help avoid rate limiting
        time.sleep(0.6)
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season)
        log_df = gamelog.get_data_frames()[0]
    except Exception as e:
        print(f"NBA API: Error retrieving game log for {player_name}: {e}")
        return None, None, None
    
    if log_df.empty:
        print(f"NBA API: No game log data for {player_name}.")
        return None, None, None

    # Use the most recent game (assumes log is sorted descending by game date)
    latest_game = log_df.iloc[0]
    latest_game_date = latest_game['GAME_DATE']
    opponent = latest_game['MATCHUP'].split()[-1]  # Extract opponent team abbreviation
    
    if stat_type == 'points':
        return latest_game['PTS'], latest_game_date, opponent
    elif stat_type == 'rebounds':
        return latest_game['REB'], latest_game_date, opponent
    elif stat_type == 'assists':
        return latest_game['AST'], latest_game_date, opponent
    else:
        return None, None, None

def compute_hit(row):
    """
    Determines if the prop line prediction was a hit.
    """
    if pd.isna(row['Actual Value']):
        return np.nan

    if row['Over/Under'] == 'over':
        return 1 if row['Actual Value'] >= row['Prop Line'] else 0
    else:
        return 1 if row['Actual Value'] <= row['Prop Line'] else 0

def main():
    # Read predictions from CSV
    predictions_csv = "NBA_Predictions.csv"
    predictions_df = pd.read_csv(predictions_csv)
    
    # Fetch actual performance from the NBA API for each prediction row.
    actual_values = []
    game_dates = []
    opponents = []

    for index, row in predictions_df.iterrows():
        actual_value, game_date, opponent = get_latest_stat(row['Player'], row['Stat Type'])
        actual_values.append(actual_value)
        game_dates.append(game_date)
        opponents.append(opponent)

    predictions_df['Actual Value'] = actual_values
    predictions_df['Game Date'] = game_dates
    predictions_df['Actual Opponent'] = opponents

    # Filter out rows where the actual opponent does not match the predicted opponent
    predictions_df = predictions_df[predictions_df['Actual Opponent'] == predictions_df['Opponent Team']]

    # Compute whether each prediction was a hit
    predictions_df['Hit'] = predictions_df.apply(compute_hit, axis=1)
    
    # Categorize by R²: "Positive R²" if >= 0, otherwise "Negative R²"
    predictions_df['R^2 Group'] = np.where(predictions_df['R^2 Value'] >= 0, 'Positive R²', 'Negative R²')
    
    # Calculate overall hit rates for each R² category
    positive_df = predictions_df[predictions_df['R^2 Group'] == 'Positive R²']
    negative_df = predictions_df[predictions_df['R^2 Group'] == 'Negative R²']
    
    positive_hit_pct = positive_df['Hit'].mean() * 100 if not positive_df.empty else None
    negative_hit_pct = negative_df['Hit'].mean() * 100 if not negative_df.empty else None
    
    print(f"\nOverall hit percentage for props with Positive R²: {positive_hit_pct:.2f}% (Count: {len(positive_df)})")
    
    # Add bins for prop line ranges
    bins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    predictions_df['Prob_Bin'] = pd.cut(predictions_df['Probability'], bins)
    
    prop_line_bins = [0, 5, 10, 15, 20, 25, float('inf')]
    prop_line_labels = ['0-5', '5-10', '10-15', '15-20', '20-25', '25+']
    predictions_df['Prop_Line_Bin'] = pd.cut(predictions_df['Prop Line'], bins=prop_line_bins, labels=prop_line_labels)
    
    # Calculate hit rates by probability bins
    hit_rates_by_prob = predictions_df.groupby('Prob_Bin', observed=False)['Hit'].mean() * 100
    counts_by_prob = predictions_df.groupby('Prob_Bin', observed=False).size()
    print("\nHit percentages by Probability bins:")
    for bin_label, hit_rate in hit_rates_by_prob.items():
        count = counts_by_prob[bin_label]
        print(f"{bin_label}: {hit_rate:.2f}% (Count: {count})")
    
    # Calculate hit rates by prop line bins
    hit_rates_by_prop_line = predictions_df.groupby('Prop_Line_Bin', observed=False)['Hit'].mean() * 100
    counts_by_prop_line = predictions_df.groupby('Prop_Line_Bin', observed=False).size()
    print("\nHit percentages by Prop Line bins:")
    for bin_label, hit_rate in hit_rates_by_prop_line.items():
        count = counts_by_prop_line[bin_label]
        print(f"{bin_label}: {hit_rate:.2f}% (Count: {count})")
    
    # Calculate hit rates by stat type and over/under
    hit_rates_by_stat_over_under = predictions_df.groupby(['Stat Type', 'Over/Under'], observed=False)['Hit'].mean() * 100
    counts_by_stat_over_under = predictions_df.groupby(['Stat Type', 'Over/Under'], observed=False).size()
    print("\nHit percentages by Stat Type and Over/Under:")
    for (stat_label, over_under_label), hit_rate in hit_rates_by_stat_over_under.items():
        count = counts_by_stat_over_under[(stat_label, over_under_label)]
        print(f"{stat_label} {over_under_label}: {hit_rate:.2f}% (Count: {count})")
    
    # Identify best conditions: for example, high probability (>= 0.8) and positive R².
    best_conditions = predictions_df[
        (predictions_df['Probability'] >= 0.8) & (predictions_df['R^2 Group'] == 'Positive R²')
    ]
    if not best_conditions.empty:
        best_hit_pct = best_conditions['Hit'].mean() * 100
        print(f"\nFor predictions with Probability >= 0.8 and Positive R², the hit rate is: {best_hit_pct:.2f}% (Count: {len(best_conditions)})")
    else:
        print("\nNo predictions met the conditions (Probability >= 0.8 and Positive R²).")

    # Print line and actual result
    print("\nProp Line vs. Actual Result:")
    for index, row in predictions_df.iterrows():
        hit_status = "Hit" if row['Hit'] == 1 else "Miss" if row['Hit'] == 0 else "N/A"
        print(f"Player: {row['Player']}, Stat: {row['Stat Type']}, Prop Line: {row['Prop Line']}, Actual: {row['Actual Value']}, Result: {hit_status}")
    
if __name__ == '__main__':
    main()