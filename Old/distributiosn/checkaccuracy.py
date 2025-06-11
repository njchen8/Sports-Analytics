import pandas as pd
import numpy as np
import time
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
from concurrent.futures import ThreadPoolExecutor, TimeoutError

def get_latest_stat(player_name, stat_type, season='2024-25'):
    """
    Given a player's full name and a stat type, this function returns the most recent game's stat.
    """
    print(f"Fetching latest stat for {player_name} ({stat_type})...")
    player_list = players.find_players_by_full_name(player_name)
    if not player_list:
        print(f"NBA API: No data found for {player_name}.")
        return None, None, None
    player_id = player_list[0]['id']
    
    try:
        # Pause to help avoid rate limiting
        time.sleep(1.0)  # Increased delay to 1 second
        with ThreadPoolExecutor() as executor:
            future = executor.submit(playergamelog.PlayerGameLog, player_id=player_id, season=season)
            gamelog = future.result(timeout=2)  # Increased timeout to 10 seconds
        log_df = gamelog.get_data_frames()[0]
    except TimeoutError:
        print(f"NBA API: Timeout retrieving game log for {player_name}.")
        return None, None, None
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
    elif stat_type == 'pts_rebs_asts':
        return latest_game['PTS'] + latest_game['REB'] + latest_game['AST'], latest_game_date, opponent
    else:
        return None, None, None

def compute_hit(row):
    """
    Determines if the prop line prediction was a hit.
    """
    if pd.isna(row['Actual Value']):
        return np.nan

    if row['Bet'] == 'over':
        return 1 if row['Actual Value'] >= row['Prop Line'] else 0
    else:
        return 1 if row['Actual Value'] <= row['Prop Line'] else 0

def main():
    print("Starting the script...")
    
    # Read predictions from CSV
    predictions_csv = 'distributiosn/prop_probabilities.csv'
    print(f"Reading predictions from {predictions_csv}...")
    predictions_df = pd.read_csv(predictions_csv)
    
    # Filter predictions with probability > 0.5
    print("Filtering predictions with probability > 0.5...")
    predictions_df = predictions_df[predictions_df['Probability'] > 0.5]
    
    # Fetch actual performance from the NBA API for each prediction row.
    actual_values = []
    game_dates = []
    opponents = []
    valid_indices = []

    for index, row in predictions_df.iterrows():
        print(f"Processing row {index + 1}/{len(predictions_df)}...")
        try:
            actual_value, game_date, opponent = get_latest_stat(row['Player'], row['Stat Type'])
            if actual_value is not None:
                actual_values.append(actual_value)
                game_dates.append(game_date)
                opponents.append(opponent)
                valid_indices.append(index)
            else:
                print(f"Skipping row {index + 1} due to missing player data.")
        except Exception as e:
            print(f"Skipping row {index + 1} due to an error: {e}")

    # Filter the DataFrame to keep only valid rows
    predictions_df = predictions_df.loc[valid_indices]
    predictions_df['Actual Value'] = actual_values
    predictions_df['Game Date'] = game_dates
    predictions_df['Actual Opponent'] = opponents

    # Filter out rows where the actual opponent does not match the predicted opponent
    print("Filtering rows where the actual opponent does not match the predicted opponent...")
    predictions_df = predictions_df[predictions_df['Actual Opponent'] == predictions_df['Opponent Team']]

    # Compute whether each prediction was a hit
    print("Computing hits and misses...")
    predictions_df['Hit'] = predictions_df.apply(compute_hit, axis=1)
    
    # Calculate total hits and misses
    total_hits = predictions_df['Hit'].sum()
    total_misses = len(predictions_df) - total_hits
    
    print(f"Total Hits: {total_hits}")
    print(f"Total Misses: {total_misses}")

    # Print line and actual result
    print("\nProp Line vs. Actual Result:")
    for index, row in predictions_df.iterrows():
        hit_status = "Hit" if row['Hit'] == 1 else "Miss" if row['Hit'] == 0 else "N/A"
        print(f"Player: {row['Player']}, Stat: {row['Stat Type']}, Prop Line: {row['Prop Line']}, Actual: {row['Actual Value']}, Result: {hit_status}")

    # Save results to CSV
    # Sort by probability in descending order
    predictions_df = predictions_df.sort_values(by='Probability', ascending=False)
    
    results_df = predictions_df[['Player', 'Probability', 'Hit']]
    results_df.columns = ['Name', 'Probability', 'Hit or Miss']
    results_df.to_csv('distributiosn/results.csv', index=False)
    print("Results saved to results.csv")

if __name__ == "__main__":
    main()
