import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from itertools import product

# Load datasets
file_path = "nba_players_game_logs_cleaned.csv"  # Update with your actual file path
team_stats_path = "team_stats.csv"  # Update with your actual file path

df = pd.read_csv(file_path)
team_stats_df = pd.read_csv(team_stats_path)

# Function to optimize hyperparameters for maximizing R^2
def optimize_hyperparameters(player_name, stat_type):
    # Filter player data
    player_df = df[df['PLAYER_NAME'] == player_name]
    if player_df.empty:
        print(f"No data found for {player_name}.")
        return
    
    # Sort by date and select the last 70 games
    player_df = player_df.sort_values(by='GAME_DATE', ascending=False).head(70)
    train_df = player_df.iloc[20:].copy()  # Use 50 games for training
    test_df = player_df.iloc[:10].copy()   # Optimize on the most recent 10 games

    # Assign matchup-based time intervals
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    train_df['MATCHUP_INDEX'] = range(1, len(train_df) + 1)
    test_df['MATCHUP_INDEX'] = range(len(train_df) + 1, len(train_df) + 1 + len(test_df))
    
    # Identify home vs away games and back-to-back games
    for df_set in [train_df, test_df]:
        df_set['HOME_GAME'] = df_set['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
        df_set['GAME_DATE'] = pd.to_datetime(df_set['GAME_DATE'])
        df_set['BACK_TO_BACK'] = df_set['GAME_DATE'].diff().dt.days.eq(1).astype(int)

    # Set target variable
    stat_column_map = {'points': 'PTS', 'rebounds': 'REB', 'assists': 'AST', 'pts_rebs_asts': 'PTS_REB_AST'}
    target_variable = stat_column_map.get(stat_type)
    if target_variable not in train_df.columns:
        print(f"Stat type {stat_type} not found for {player_name}.")
        return
    
    # Feature engineering
    for df_set in [train_df, test_df]:
        df_set['PTS_PREV_1'] = df_set[target_variable].shift(1, fill_value=df_set[target_variable].mean())
        df_set['PTS_PREV_2'] = df_set[target_variable].shift(2, fill_value=df_set[target_variable].mean())
        df_set['PTS_PREV_3'] = df_set[target_variable].shift(3, fill_value=df_set[target_variable].mean())
        df_set['PROP_VARIANCE_10'] = df_set[target_variable].rolling(window=10).var().fillna(df_set[target_variable].var())

        # Extract opponent team from MATCHUP column
        df_set['TEAM_MATCHUP'] = df_set['MATCHUP'].apply(lambda x: x.split()[-1])
        df_set['AVG_VS_TEAM_5'] = df_set.groupby('TEAM_MATCHUP')[target_variable].rolling(window=5).mean().reset_index(level=0, drop=True).fillna(df_set[target_variable].mean())

    # Merge opponent team stats
    train_df = train_df.merge(team_stats_df, left_on='TEAM_MATCHUP', right_on='TEAM_ABBREVIATION', how='left')
    test_df = test_df.merge(team_stats_df, left_on='TEAM_MATCHUP', right_on='TEAM_ABBREVIATION', how='left')

    # Prepare training and testing data
    features = ['MATCHUP_INDEX', 'HOME_GAME', 'BACK_TO_BACK', 'PTS_PREV_1', 'PTS_PREV_2', 'PTS_PREV_3',
                'OFF_RATING', 'DEF_RATING', 'NET_RATING', 'PACE', 'AST_RATIO', 'REB_PCT', 'PROP_VARIANCE_10', 'AVG_VS_TEAM_5']
    
    X_train, y_train = train_df[features].values, train_df[target_variable].values
    X_test, y_test = test_df[features].values, test_df[target_variable].values

    # Hyperparameter Grid Search
    param_grid = {
        "n_estimators": [5, 10, 20, 50, 60],
        "max_depth": [1, 3, 5, 7],
        "min_samples_split": [2, 5, 7, 10]
    }
    
    best_r2 = float('-inf')
    best_params = None
    best_preds = None

    for n, d, s in product(param_grid["n_estimators"], param_grid["max_depth"], param_grid["min_samples_split"]):
        model = RandomForestRegressor(n_estimators=n, max_depth=d, min_samples_split=s, random_state=10)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        if r2 > best_r2:
            best_r2 = r2
            best_params = {"n_estimators": n, "max_depth": d, "min_samples_split": s}
            best_preds = y_pred

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.scatter(test_df['MATCHUP_INDEX'], y_test, label='Actual', color='blue', alpha=0.6)
    plt.plot(test_df['MATCHUP_INDEX'], y_test, label='Actual Trend', color='blue', linestyle='-')
    plt.plot(test_df['MATCHUP_INDEX'], best_preds, label='Best Model Prediction', color='red', linestyle='--')
    plt.xlabel('Matchup Index')
    plt.ylabel(stat_type.upper())
    plt.title(f"Optimized Model for {player_name} - {stat_type.upper()} (R² = {best_r2:.2f})")
    plt.legend()
    plt.show()

    # Print best parameters and R² score
    print(f"Best Parameters: {best_params}")
    print(f"Optimized R² value: {best_r2:.2f}")

# Example usage
player_name = "LaMelo Ball"  # Change as needed
stat_type = "points"  # Choose from 'points', 'rebounds', 'assists', 'pts_rebs_asts'
optimize_hyperparameters(player_name, stat_type)
