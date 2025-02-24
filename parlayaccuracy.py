#!/usr/bin/env python
import pandas as pd
import numpy as np
import time
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
from scipy.stats import norm

def get_latest_stat(player_name, stat_type, season='2024-25'):
    """
    Given a player's full name and a stat type, this function returns the most recent game's stat.
    For 'pts_rebs_asts', it sums points, rebounds, and assists.
    """
    player_list = players.find_players_by_full_name(player_name)
    if not player_list:
        print(f"NBA API: No data found for {player_name}.")
        return None
    player_id = player_list[0]['id']
    
    try:
        # Pause to help avoid rate limiting
        time.sleep(0.6)
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season)
        log_df = gamelog.get_data_frames()[0]
    except Exception as e:
        print(f"NBA API: Error retrieving game log for {player_name}: {e}")
        return None
    
    if log_df.empty:
        print(f"NBA API: No game log data for {player_name}.")
        return None

    # Use the most recent game (assumes log is sorted descending by game date)
    latest_game = log_df.iloc[0]
    
    if stat_type == 'points':
        return latest_game['PTS']
    elif stat_type == 'rebounds':
        return latest_game['REB']
    elif stat_type == 'assists':
        return latest_game['AST']
    elif stat_type == 'pts_rebs_asts':
        return latest_game['PTS'] + latest_game['REB'] + latest_game['AST']
    else:
        return None

def compute_hit(row):
    """
    Determines if the prop line prediction was a hit.
    For 'over' bets, hit if actual stat >= prop line.
    For 'under' bets, hit if actual stat <= prop line.
    """
    if pd.isna(row['Actual Value']):
        return np.nan
    if row['Best Bet'] == 'over':
        return 1 if row['Actual Value'] >= row['Prop Line'] else 0
    else:
        return 1 if row['Actual Value'] <= row['Prop Line'] else 0

def main():
    # Read predictions from CSV
    predictions_csv = "predictions_sorted.csv"
    predictions_df = pd.read_csv(predictions_csv)
    
    # Fetch actual performance from the NBA API for each prediction row.
    predictions_df['Actual Value'] = predictions_df.apply(
        lambda row: get_latest_stat(row['Player'], row['Stat Type']), axis=1
    )
    
    # Compute whether each prediction was a hit
    predictions_df['Hit'] = predictions_df.apply(compute_hit, axis=1)
    
    # Print the prop line and actual value for each game
    print("\nIndividual Game Results:")
    for idx, row in predictions_df.iterrows():
        print(f"Player: {row['Player']} - Prop Line: {row['Prop Line']}, Actual Value: {row['Actual Value']}")
    
    # Categorize by R²: "Positive R²" if >= 0, otherwise "Negative R²"
    predictions_df['R^2 Group'] = np.where(predictions_df['R^2 Value'] >= 0, 'Positive R²', 'Negative R²')
    
    # Calculate overall hit rates for each R² category
    positive_df = predictions_df[predictions_df['R^2 Group'] == 'Positive R²']
    negative_df = predictions_df[predictions_df['R^2 Group'] == 'Negative R²']
    
    positive_hit_pct = positive_df['Hit'].mean() * 100 if not positive_df.empty else None
    negative_hit_pct = negative_df['Hit'].mean() * 100 if not negative_df.empty else None
    
    print(f"\nOverall hit percentage for props with Positive R²: {positive_hit_pct:.2f}%")
    print(f"Overall hit percentage for props with Negative R²: {negative_hit_pct:.2f}%")
    
    # Group predictions by probability buckets.
    bins = [0, 0.5, 0.6, 0.7, 0.8, 1.0]
    predictions_df['Prob_Bin'] = pd.cut(predictions_df['Probability Max'], bins)
    
    hit_rates_by_prob = predictions_df.groupby('Prob_Bin')['Hit'].mean() * 100
    print("\nHit percentages by Probability Max bins (overall):")
    print(hit_rates_by_prob)
    
    hit_rates_by_group = predictions_df.groupby(['R^2 Group', 'Prob_Bin'])['Hit'].mean() * 100
    print("\nHit percentages by R² Group and Probability Max bins:")
    print(hit_rates_by_group)
    
    # Identify best conditions: for example, high probability (>= 0.8) and positive R².
    best_conditions = predictions_df[
        (predictions_df['Probability Max'] >= 0.8) & (predictions_df['R^2 Group'] == 'Positive R²')
    ]
    if not best_conditions.empty:
        best_hit_pct = best_conditions['Hit'].mean() * 100
        print(f"\nFor predictions with Probability Max >= 0.8 and Positive R², the hit rate is: {best_hit_pct:.2f}%")
    else:
        print("\nNo predictions met the conditions (Probability Max >= 0.8 and Positive R²).")
    
if __name__ == '__main__':
    main()
