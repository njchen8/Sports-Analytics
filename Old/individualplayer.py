import pandas as pd
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
from datetime import datetime

def get_recent_games(player_name, num_games=10):
    """Retrieve the most recent matches for a given player using NBA API up to today's date."""
    player_dict = players.find_players_by_full_name(player_name)
    if not player_dict:
        print("Player not found.")
        return None
    
    player_id = player_dict[0]['id']
    game_log = playergamelog.PlayerGameLog(player_id=player_id, season='2024-25', season_type_all_star='Regular Season')
    game_df = game_log.get_data_frames()[0]
    
    # Convert GAME_DATE to datetime and filter by today's date
    game_df['GAME_DATE'] = pd.to_datetime(game_df['GAME_DATE'])
    today = datetime.today()
    recent_games = game_df[game_df['GAME_DATE'] <= today].sort_values(by='GAME_DATE', ascending=False).head(num_games)
    
    return recent_games

# Example usage
player_input = input("Enter player name: ")
recent_games = get_recent_games(player_input)
print(recent_games)