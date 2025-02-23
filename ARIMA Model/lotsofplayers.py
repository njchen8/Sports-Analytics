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
prop_data_path = "fixed_data.csv"
prop_df = pd.read_csv(prop_data_path)

# Convert GAME_DATE to datetime
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
latest_date = df['GAME_DATE'].max()

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
    
    # Ensure player has played within the last 7 days
    last_game_date = player_df['GAME_DATE'].max()
    if (latest_date - last_game_date).days > 7:
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
    
    # Calculate average stat for same team matchup in the last 5 matchups
    train_df['TEAM_MATCHUP'] = train_df['MATCHUP'].apply(lambda x: x.split()[-1])
    train_df['AVG_VS_TEAM_5'] = train_df[train_df['TEAM_MATCHUP'] == opponent_team][target_variable].rolling(window=5).mean().fillna(train_df[target_variable].mean())
    train_df['AVG_VS_TEAM_5'] = train_df[target_variable].mean()
    
    # Merge opponent team stats
    train_df['OPPONENT'] = opponent_team
    train_df = merge_team_stats(train_df)
    
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
    
    # Calculate probability of prop line hitting using empirical distribution
    mean_stat = train_df[target_variable].mean()
    std_stat = train_df[target_variable].std()
    
    if std_stat > 0:
        probability = 1 - norm.cdf(row['Prop Line'], loc=mean_stat, scale=std_stat)
    else:
        probability = 1.0 if predicted_stat >= row['Prop Line'] else 0.0
    max_probability = 1 - probability if probability < 0.5 else probability
    best_bet = 'over' if probability > 0.5 else 'under'
    # Store prediction
    player_predictions.append({
        'Player': player_name,
        'Stat Type': stat_type,
        'Predicted Value': predicted_stat,
        'Variance (Last 10)': train_df['PROP_VARIANCE_10'].iloc[-1],
        'Prop Line': row['Prop Line'],
        'Probability Max': max_probability,
        'best_bet' : best_bet
    })

# Convert predictions to DataFrame
predictions_df = pd.DataFrame(player_predictions)

# Sort by highest probability
predictions_df = predictions_df.sort_values(by='Probability Max', ascending=False)

# Save to CSV
predictions_df.to_csv("predictions_sorted.csv", index=False)

# Display the sorted predictions
print(predictions_df)
