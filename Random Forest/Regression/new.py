import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor  # Using XGBoost for better performance

# Load datasets
file_path = "nba_players_game_logs_cleaned.csv"  # Update with your actual file path
team_stats_path = "team_stats.csv"  # Update with your actual file path

df = pd.read_csv(file_path)
team_stats_df = pd.read_csv(team_stats_path)

# Function to optimize R^2 using all games from the current season
def optimize_with_season_data(player_name, stat_type):
    # Convert game dates and filter for the current season (2023-24)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    current_season = df['GAME_DATE'].dt.year.max()  # Get the most recent season year
    season_df = df[df['GAME_DATE'].dt.year == current_season]  # Filter for this season

    # Filter player data for this season
    player_df = season_df[season_df['PLAYER_NAME'] == player_name]
    if player_df.empty:
        print(f"No data found for {player_name} in the current season.")
        return

    # Sort by date
    player_df = player_df.sort_values(by='GAME_DATE', ascending=True)  # Earliest to latest

    # Split into train/test: Train on all but last 10 games, test on last 10 games
    train_df = player_df.iloc[:-10].copy()
    test_df = player_df.iloc[-10:].copy()

    # Assign matchup-based time intervals
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    train_df['MATCHUP_INDEX'] = range(1, len(train_df) + 1)
    test_df['MATCHUP_INDEX'] = range(len(train_df) + 1, len(train_df) + 1 + len(test_df))

    # Identify home vs away games and back-to-back games
    for df_set in [train_df, test_df]:
        df_set['HOME_GAME'] = df_set['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
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

    # Train and optimize using XGBoost
    model = XGBRegressor(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=10)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Calculate R^2
    r2 = r2_score(y_test, y_pred)

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.scatter(test_df['MATCHUP_INDEX'], y_test, label='Actual', color='blue', alpha=0.6)
    plt.plot(test_df['MATCHUP_INDEX'], y_test, label='Actual Trend', color='blue', linestyle='-')
    plt.plot(test_df['MATCHUP_INDEX'], y_pred, label='Predicted Trend (XGBoost)', color='red', linestyle='--')
    plt.xlabel('Matchup Index')
    plt.ylabel(stat_type.upper())
    plt.title(f"Optimized Model Using This Season's Data\n{player_name} - {stat_type.upper()} (R² = {r2:.2f})")
    plt.legend()
    plt.show()

    # Print results
    print(f"Final R² value: {r2:.2f}")

# Example usage
player_name = "Julian Champagnie"  # Change as needed
stat_type = "rebounds"  # Choose from 'points', 'rebounds', 'assists', 'pts_rebs_asts'
optimize_with_season_data(player_name, stat_type)
