import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.stattools import adfuller
from scipy.stats import norm

# Load datasets
file_path = "nba_players_game_logs_cleaned.csv"  # Update with the correct path
df = pd.read_csv(file_path)
team_stats_path = "team_stats.csv"
team_stats_df = pd.read_csv(team_stats_path)

# Function to predict and plot results for a given player, stat type, and prop line
def predict_and_plot(player_name, stat_type, prop_line):
    # Filter player data
    player_df = df[df['PLAYER_NAME'] == player_name]
    if player_df.empty:
        print(f"No data found for {player_name}.")
        return
    
    # Sort by date and select the last 50 games
    player_df = player_df.sort_values(by='GAME_DATE', ascending=False).head(50)
    train_df = player_df.copy()
    
    # Assign matchup-based time intervals
    train_df = train_df.reset_index(drop=True)
    train_df['MATCHUP_INDEX'] = range(1, len(train_df) + 1)
    
    # Identify home vs away games
    train_df['HOME_GAME'] = train_df['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
    
    # Identify back-to-back games
    train_df['GAME_DATE'] = pd.to_datetime(train_df['GAME_DATE'])
    train_df['BACK_TO_BACK'] = train_df['GAME_DATE'].diff().dt.days.eq(1).astype(int)
    
    # Set target variable
    stat_column_map = {'points': 'PTS', 'rebounds': 'REB', 'assists': 'AST', 'pts_rebs_asts': 'PTS_REB_AST'}
    target_variable = stat_column_map.get(stat_type)
    if target_variable not in train_df.columns:
        print(f"Stat type {stat_type} not found for {player_name}.")
        return
    
    # Include past game performance indicators
    if stat_type == 'points':
        train_df['PTS_PREV_1'] = train_df['PTS'].shift(1, fill_value=train_df['PTS'].mean())
    elif stat_type == 'rebounds':
        train_df['PTS_PREV_1'] = train_df['REB'].shift(1, fill_value=train_df['REB'].mean())
    elif stat_type == 'assists':
        train_df['PTS_PREV_1'] = train_df['AST'].shift(1, fill_value=train_df['AST'].mean())
    elif stat_type == 'pts_rebs_asts':
        train_df['PTS_PREV_1'] = (train_df['PTS'] + train_df['REB'] + train_df['AST']).shift(1, fill_value=(train_df['PTS'] + train_df['REB'] + train_df['AST']).mean())
    
    train_df['PTS_PREV_2'] = train_df[target_variable].shift(2, fill_value=train_df[target_variable].mean())
    train_df['PTS_PREV_3'] = train_df[target_variable].shift(3, fill_value=train_df[target_variable].mean())
    
    # Calculate variance of the stat in the last 10 games
    train_df['PROP_VARIANCE_10'] = train_df[target_variable].rolling(window=10).var().fillna(train_df[target_variable].var())
    
    # Extract opponent team from MATCHUP column
    train_df['TEAM_MATCHUP'] = train_df['MATCHUP'].apply(lambda x: x.split()[-1])
    last_matchup = train_df.iloc[0]['TEAM_MATCHUP']
    train_df['AVG_VS_TEAM_5'] = train_df[train_df['TEAM_MATCHUP'] == last_matchup][target_variable].rolling(window=5).mean().fillna(train_df[target_variable].mean())
    
    # Merge opponent team stats
    train_df = train_df.merge(team_stats_df, left_on='TEAM_MATCHUP', right_on='TEAM_ABBREVIATION', how='left')
    
    # Prepare training data
    features = ['MATCHUP_INDEX', 'HOME_GAME', 'BACK_TO_BACK', 'PTS_PREV_1', 'PTS_PREV_2', 'PTS_PREV_3',
                'OFF_RATING', 'DEF_RATING', 'NET_RATING', 'PACE', 'AST_RATIO', 'REB_PCT', 'PROP_VARIANCE_10', 'AVG_VS_TEAM_5']
    X_train = train_df[features].values
    y_train = train_df[target_variable].values
    
    # Fit a regression model using Random Forest Regressor
    model = RandomForestRegressor(n_estimators=20, max_depth=5, min_samples_split=5, random_state=10)
    model.fit(X_train, y_train)
    
    # Predict the next game
    test_features = train_df.iloc[-1][features].values.reshape(1, -1)
    predicted_stat = model.predict(test_features)[0]
    
    # Generate the graph
    plt.figure(figsize=(10, 5))
    plt.plot(train_df['MATCHUP_INDEX'], train_df[target_variable], marker='o', linestyle='-', label='Actual')
    plt.axhline(y=prop_line, color='r', linestyle='--', label='Prop Line')
    plt.scatter(train_df['MATCHUP_INDEX'].max() + 1, predicted_stat, color='g', label='Predicted', marker='x', s=100)
    plt.xlabel('Matchup Index')
    plt.ylabel(stat_type.upper())
    plt.title(f"Prediction for {player_name} - {stat_type.upper()}")
    plt.legend()
    plt.show()
    
    print(f"Predicted {stat_type} for {player_name}: {predicted_stat:.2f}")
    print(f"Prop Line: {prop_line}")

# Example usage
player_name = "Julian Champagnie"  # Change as needed
stat_type = "rebounds"  # Choose from 'points', 'rebounds', 'assists', 'pts_rebs_asts'
prop_line = 1.5  # Set prop line
predict_and_plot(player_name, stat_type, prop_line)
