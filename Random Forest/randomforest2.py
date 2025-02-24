import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from scipy.stats import norm
from itertools import product
from joblib import Parallel, delayed

# Load datasets with low memory to prevent dtype warnings
file_path = "nba_players_game_logs_cleaned.csv"
df = pd.read_csv(file_path, low_memory=False)

team_stats_path = "team_stats.csv"
team_stats_df = pd.read_csv(team_stats_path, low_memory=False)

prop_data_path = "fixed_data.csv"
prop_df = pd.read_csv(prop_data_path, low_memory=False)

# Convert GAME_DATE to datetime
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
latest_date = df['GAME_DATE'].max()

# Initialize dictionary to store predictions
player_predictions = []

def merge_team_stats(df):
    return df.merge(team_stats_df, left_on='OPPONENT', right_on='TEAM_ABBREVIATION', how='left')

def optimize_hyperparameters(train_df, target_variable, features):
    param_grid = {
        "n_estimators": [5, 10, 20, 50, 60],
        "max_depth": [1, 3, 5, 7],
        "min_samples_split": [2, 5, 7, 10]
    }
    
    best_r2 = float('-inf')
    best_model = None

    X_train, y_train = train_df[features].values, train_df[target_variable].values
    
    for n, d, s in product(param_grid["n_estimators"], param_grid["max_depth"], param_grid["min_samples_split"]):
        model = RandomForestRegressor(n_estimators=n, max_depth=d, min_samples_split=s, random_state=10)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_train)
        r2 = r2_score(y_train, y_pred)
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
    
    return best_model

def process_player(row):
    player_name = row['Player']
    opponent_team = row['Opponent Team']
    stat_type = row['Stat Type']
    if stat_type not in ['points', 'rebounds', 'assists', 'pts_rebs_asts']:
        return None
    
    print(f"\n🔍 Processing player: {player_name} | Stat Type: {stat_type}")

    # Filter player data
    player_df = df[df['PLAYER_NAME'] == player_name]
    if player_df.empty:
        print(f"⚠️ No data found for {player_name}. Skipping.")
        return None
    
    # Ensure player has played within the last 7 days
    last_game_date = player_df['GAME_DATE'].max()
    if (latest_date - last_game_date).days > 7:
        print(f"⚠️ {player_name} hasn't played in the last 7 days. Skipping.")
        return None
    
    # Sort by date and select past 75 games (training: 15-75, testing: most recent 15)
    player_df = player_df.sort_values(by='GAME_DATE', ascending=False).head(75)
    test_df = player_df.head(15)
    train_df = player_df.tail(60)

    print(f"📊 Using {len(train_df)} games for training, {len(test_df)} for testing.")

    # Feature Engineering
    train_df = train_df.reset_index(drop=True)
    train_df['HOME_GAME'] = train_df['MATCHUP'].apply(lambda x: 1 if ' vs. ' in x else 0)

    stat_column_map = {'points': 'PTS', 'rebounds': 'REB', 'assists': 'AST', 'pts_rebs_asts': 'PTS_REB_AST'}
    target_variable = stat_column_map.get(stat_type)

    # **Auto-Skip if Target Variable is Missing**
    if target_variable not in train_df.columns:
        print(f"⚠️ Target variable {target_variable} not found for {player_name}. Skipping immediately.")
        return None

    # Calculate rolling variance for the last 10 games
    train_df['PROP_VARIANCE_10'] = train_df[target_variable].rolling(window=10, min_periods=1).var()

    # Add lag features
    for lag in range(1, 4):
        train_df[f'PTS_PREV_{lag}'] = train_df[target_variable].shift(lag).fillna(train_df[target_variable].mean())

    train_df['OPPONENT'] = opponent_team
    train_df = merge_team_stats(train_df)

    features = ['HOME_GAME', 'PTS_PREV_1', 'PTS_PREV_2', 'PTS_PREV_3',
                'OFF_RATING', 'DEF_RATING', 'NET_RATING', 'PACE', 'AST_RATIO', 'REB_PCT', 'PROP_VARIANCE_10']

    # Accumulate test results for proper R² calculation
    all_actuals = []
    all_predictions = []

    print("📡 Beginning rolling test evaluation...")

    # Optimize hyperparameters once per player
    best_model = optimize_hyperparameters(train_df, target_variable, features)

    for i in range(len(test_df)):
        # Prepare the next test match
        current_test_game = test_df.iloc[[i]].copy()
        current_test_game['HOME_GAME'] = current_test_game['MATCHUP'].apply(lambda x: 1 if ' vs. ' in x else 0)
        for lag in range(1, 4):
            current_test_game[f'PTS_PREV_{lag}'] = train_df[target_variable].shift(lag).fillna(train_df[target_variable].mean())
        current_test_game['PROP_VARIANCE_10'] = train_df[target_variable].rolling(window=10, min_periods=1).var().iloc[-1]
        current_test_game['OPPONENT'] = opponent_team
        current_test_game = merge_team_stats(current_test_game)

        # Ensure enough features exist
        if len(current_test_game) == 0:
            continue

        # Test prediction
        X_test, y_test = current_test_game[features].values, current_test_game[target_variable].values
        y_pred_test = best_model.predict(X_test)

        # Store results
        all_actuals.extend(y_test)
        all_predictions.extend(y_pred_test)

        print(f"✅ Test game {i+1}/{len(test_df)} completed.")

        # Update training set
        train_df = pd.concat([train_df, current_test_game]).reset_index(drop=True)

    # Compute R² after multiple games
    final_r2 = r2_score(all_actuals, all_predictions) if len(all_actuals) > 1 else np.nan

    print(f"📈 Final R² Score: {final_r2:.3f}")

    predicted_stat = best_model.predict(train_df[features].values[-1].reshape(1, -1))[0]
    
    # Calculate probability of prop line hitting
    mean_stat = train_df[target_variable].mean()
    std_stat = train_df[target_variable].std()
    
    if std_stat > 0:
        probability = 1 - norm.cdf(row['Prop Line'], loc=mean_stat, scale=std_stat)
    else:
        probability = 1.0 if predicted_stat >= row['Prop Line'] else 0.0
    max_probability = 1 - probability if probability < 0.5 else probability
    best_bet = 'over' if probability > 0.5 else 'under'
    
    return {
        'Player': player_name,
        'Stat Type': stat_type,
        'Predicted Value': predicted_stat,
        'Variance (Last 10)': train_df['PROP_VARIANCE_10'].iloc[-1],
        'Prop Line': row['Prop Line'],
        'Probability Max': max_probability,
        'Best Bet': best_bet,
        'R^2 Value': final_r2
    }

# Use parallel processing to handle multiple players
player_predictions = Parallel(n_jobs=-1)(delayed(process_player)(row) for _, row in prop_df.iterrows())

# Filter out None values
player_predictions = [pred for pred in player_predictions if pred is not None]

# Convert predictions to DataFrame
predictions_df = pd.DataFrame(player_predictions)

# Sort by positive R² vs negative R², then by highest probability
predictions_df['R^2 Group'] = np.where(predictions_df['R^2 Value'] >= 0, 'Positive R²', 'Negative R²')
predictions_df = predictions_df.sort_values(by=['R^2 Group', 'Probability Max'], ascending=[True, False])

# Save to CSV
predictions_df.to_csv("predictions_sorted.csv", index=False)

# Display the sorted predictions
print("\n✅ Predictions successfully generated and saved!")
print(predictions_df)