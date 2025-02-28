from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
import pandas as pd
import matplotlib.pyplot as plt

def get_player_id(player_name):
    player_info = players.find_players_by_full_name(player_name)
    if not player_info:
        raise ValueError(f"Player '{player_name}' not found. Check spelling.")
    return player_info[0]['id']

def get_last_50_stats(player_id, stat_name):
    all_seasons = ['2024-25', '2023-24', '2022-23', '2021-22', '2020-21']
    stats_list = []
    
    for season in all_seasons:
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season)
        df = gamelog.get_data_frames()[0]
        if stat_name.upper() not in df.columns:
            raise ValueError(f"'{stat_name}' is not a valid stat. Available stats: {list(df.columns)}")
        
        stats_list.extend(df[stat_name.upper()].tolist())
        
        if len(stats_list) >= 100:
            break

    return stats_list[:100]

def plot_stat_distribution(player_name, stat_name):
    try:
        player_id = get_player_id(player_name)
        stats = get_last_50_stats(player_id, stat_name)
        
        plt.figure(figsize=(10, 5))
        plt.hist(stats, bins=10, edgecolor='black')
        plt.title(f"Distribution of Last 50 {stat_name} Data Points for {player_name}")
        plt.xlabel(stat_name)
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(str(e))

if __name__ == "__main__":
    player_name = "Austin Reaves"  # Change to desired player's name
    stat_name = "AST"  # Change to the desired stat (e.g., "REB", "AST", etc.)
    plot_stat_distribution(player_name, stat_name)
