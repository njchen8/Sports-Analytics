import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.stattools import adfuller

# Load datasets
file_path = "nba_players_game_logs_cleaned.csv"  # Update with the correct path
df = pd.read_csv(file_path)
team_stats_path = "team_stats.csv"
team_stats_df = pd.read_csv(team_stats_path)
prop_data_path = "fixed_data.csv"
prop_df = pd.read_csv(prop_data_path)

# Initialize dictionary to store predictions
player_predictions = []

def merge_team_stats(df):
    return df.merge(team_stats_df, left_on='OPPONENT', right_on='TEAM_ABBREVIATION', how='left')

# Iterate over each player in prop_df
for _, row in prop_df.iterrows():
    player_name = row['Player']
    opponent_team = row['Opponent Team']
    stat_type = row['Stat Type']
    if stat_type not in ['points', 'rebounds', 'assists', 'pts_rebs_asts']:
        continue  # Ignore non-relevant lines
    
    # Filter player data
    player_df = df[df['PLAYER_NAME'] == player_name]
    if player_df.empty:
        continue
    
    # Sort by date and select the last 50 games
    player_df = player_df.sort_values(by='GAME_DATE', ascending=False).head(50)
    train_df = player_df.copy()
    
    # Assign matchup-based time intervals
    train_df = train_df.reset_index(drop=True)
    train_df['MATCHUP_INDEX'] = range(1, len(train_df) + 1)
    
    # Identify home vs away games
    train_df['HOME_GAME'] = train_df['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
    
    # Identify back-to-back games assuming today as game day
    train_df['GAME_DATE'] = pd.to_datetime(train_df['GAME_DATE'])
    last_game_date = train_df.iloc[0]['GAME_DATE']
    train_df['BACK_TO_BACK'] = train_df['GAME_DATE'].diff().dt.days.eq(1).astype(int)
    train_df.at[0, 'BACK_TO_BACK'] = int((last_game_date - train_df.iloc[1]['GAME_DATE']).days == 1)
    
    # Set target variable
    stat_column_map = {'points': 'PTS', 'rebounds': 'REB', 'assists': 'AST', 'pts_rebs_asts': 'PTS_REB_AST'}
    target_variable = stat_column_map.get(stat_type)
    if target_variable not in train_df.columns:
        continue
    
    # Include past game performance indicators
    train_df['PTS_PREV_1'] = train_df[target_variable].shift(1, fill_value=train_df[target_variable].mean())
    train_df['PTS_PREV_2'] = train_df[target_variable].shift(2, fill_value=train_df[target_variable].mean())
    train_df['PTS_PREV_3'] = train_df[target_variable].shift(3, fill_value=train_df[target_variable].mean())
    
    # Calculate variance of the stat in the last 10 games
    train_df['PROP_VARIANCE_10'] = train_df[target_variable].rolling(window=10).var().fillna(train_df[target_variable].var())
    
    # Merge opponent team stats
    train_df['OPPONENT'] = opponent_team
    train_df = merge_team_stats(train_df)
    
    # Prepare training data
    features = ['MATCHUP_INDEX', 'HOME_GAME', 'BACK_TO_BACK', 'PTS_PREV_1', 'PTS_PREV_2', 'PTS_PREV_3',
                'OFF_RATING', 'DEF_RATING', 'NET_RATING', 'PACE', 'AST_RATIO', 'REB_PCT', 'PROP_VARIANCE_10']
    X_train = train_df[features].values
    y_train = train_df[target_variable].values
    
    # Fit a regression model using Random Forest Regressor
    model = RandomForestRegressor(n_estimators=20, max_depth=5, min_samples_split=5, random_state=10)
    model.fit(X_train, y_train)
    
    # Predict the next game
    test_features = train_df.iloc[-1][features].values.reshape(1, -1)
    predicted_stat = model.predict(test_features)[0]
    
    # Store prediction
    player_predictions.append({
        'Prop Line': row['Prop Line'],  # Include prop line from prop_df
        'Player': player_name,
        'Stat Type': stat_type,
        'Predicted Value': predicted_stat,
        'Variance (Last 10)': train_df['PROP_VARIANCE_10'].iloc[-1]
    })

# Convert predictions to DataFrame and display
predictions_df = pd.DataFrame(player_predictions)
predictions_df.to_csv("predictions.csv", index=False)
print(predictions_df)
