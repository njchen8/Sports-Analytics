import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from scipy.stats import norm
from itertools import product
from joblib import Parallel, delayed

# Load datasets with low memory to prevent dtype warnings
file_path = "nba_players_game_logs_2018_25.csv"
df = pd.read_csv(file_path, low_memory=False)

team_stats_path = "team_stats.csv"
team_stats_df = pd.read_csv(team_stats_path, low_memory=False)

prop_data_path = "fixed_data.csv"
prop_df = pd.read_csv(prop_data_path, low_memory=False)

# Convert GAME_DATE to datetime
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
latest_date = df['GAME_DATE'].max()

# Merge team stats
def merge_team_stats(data):
    return data.merge(team_stats_df, left_on='OPPONENT', right_on='TEAM_ABBREVIATION', how='left')

# Create features: HOME_GAME and lag features (no rolling window here)
def create_features(data, target_variable):
    data = data.copy()
    data['HOME_GAME'] = data['MATCHUP'].apply(lambda x: 1 if ' vs. ' in x else 0)
    for lag in range(1, 4):
        data[f'PTS_PREV_{lag}'] = data[target_variable].shift(lag).fillna(data[target_variable].mean())
    return data

# Optimize hyperparameters using cross-validation on the training set
def optimize_hyperparameters(train_df, target_variable, features):
    param_grid = {
        "n_estimators": [10, 50, 100],
        "max_depth": [3, 5, 7, None],
        "min_samples_split": [2, 5, 10]
    }
    
    best_cv_score = float('-inf')
    best_params = None
    X_train = train_df[features].values
    y_train = train_df[target_variable].values
    
    kf = KFold(n_splits=3, shuffle=True, random_state=10)
    
    for n, d, s in product(param_grid["n_estimators"], param_grid["max_depth"], param_grid["min_samples_split"]):
        cv_scores = []
        for train_index, val_index in kf.split(X_train):
            X_tr, X_val = X_train[train_index], X_train[val_index]
            y_tr, y_val = y_train[train_index], y_train[val_index]
            model = RandomForestRegressor(n_estimators=n, max_depth=d, 
                                          min_samples_split=s, random_state=10,
                                          bootstrap=True, oob_score=True)
            model.fit(X_tr, y_tr)
            y_pred_val = model.predict(X_val)
            cv_scores.append(r2_score(y_val, y_pred_val))
        mean_cv_score = np.mean(cv_scores)
        
        if mean_cv_score > best_cv_score:
            best_cv_score = mean_cv_score
            best_params = (n, d, s)
    
    best_model = RandomForestRegressor(n_estimators=best_params[0], max_depth=best_params[1],
                                       min_samples_split=best_params[2], random_state=10,
                                       bootstrap=True, oob_score=True)
    best_model.fit(X_train, y_train)
    print(f"Optimized hyperparameters: n_estimators={best_params[0]}, max_depth={best_params[1]}, min_samples_split={best_params[2]} (CV R²: {best_cv_score:.3f})")
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
    
    # Sort by date and select past 75 games (training: games 16-75, testing: most recent 15)
    player_df = player_df.sort_values(by='GAME_DATE', ascending=False).head(75)
    test_df = player_df.head(15).reset_index(drop=True)
    train_df = player_df.tail(60).reset_index(drop=True)

    print(f"📊 Using {len(train_df)} games for training, {len(test_df)} for testing.")

    stat_column_map = {'points': 'PTS', 'rebounds': 'REB', 'assists': 'AST', 'pts_rebs_asts': 'PTS_REB_AST'}
    target_variable = stat_column_map.get(stat_type)

    if target_variable not in train_df.columns:
        print(f"⚠️ Target variable {target_variable} not found for {player_name}. Skipping immediately.")
        return None

    # Generate features on training set and merge team stats
    train_df = create_features(train_df, target_variable)
    train_df['OPPONENT'] = opponent_team
    train_df = merge_team_stats(train_df)
    
    # Compute the variance from the last 10 games (or all available if <10) of the training set
    variance_10 = train_df[target_variable].tail(10).var()
    train_df['PROP_VARIANCE_10'] = variance_10
    
    features = ['HOME_GAME', 'PTS_PREV_1', 'PTS_PREV_2', 'PTS_PREV_3',
                'OFF_RATING', 'DEF_RATING', 'NET_RATING', 'PACE', 'AST_RATIO', 'REB_PCT', 'PROP_VARIANCE_10']
    
    # Optimize hyperparameters on the training set
    best_model = optimize_hyperparameters(train_df, target_variable, features)
    
    # --- Fixed Test Evaluation on the Past 15 Recent Games ---
    # Prepare the test set without updating the training set
    test_df = create_features(test_df, target_variable)
    test_df['OPPONENT'] = opponent_team
    test_df = merge_team_stats(test_df)
    # Use the variance from the last 10 games of the training set as a fixed feature for all test games
    test_df['PROP_VARIANCE_10'] = variance_10
    
    X_test = test_df[features].values
    y_test = test_df[target_variable].values
    y_pred_test = best_model.predict(X_test)
    final_r2 = r2_score(y_test, y_pred_test) if len(y_test) > 1 else np.nan
    print(f"📈 Final R² Score (on 15 recent games): {final_r2:.3f}")
    
    predicted_stat = best_model.predict(train_df[features].values[-1].reshape(1, -1))[0]
    
    # For probability calculation, use the mean and std from the last 10 training games
    recent_stats = train_df[target_variable].tail(10)
    mean_stat = recent_stats.mean()
    std_stat = np.sqrt(recent_stats.var())
    
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
        'Variance (Last 10)': variance_10,
        'Prop Line': row['Prop Line'],
        'Probability Max': max_probability,
        'Best Bet': best_bet,
        'R^2 Value': final_r2
    }

# Process players in parallel
player_predictions = Parallel(n_jobs=-1)(
    delayed(process_player)(row) for _, row in prop_df.iterrows()
)

# Filter out None values
player_predictions = [pred for pred in player_predictions if pred is not None]

# Convert predictions to DataFrame and sort results
predictions_df = pd.DataFrame(player_predictions)
predictions_df['R^2 Group'] = np.where(predictions_df['R^2 Value'] >= 0, 'Positive R²', 'Negative R²')
predictions_df = predictions_df.sort_values(by=['R^2 Group', 'Probability Max'], ascending=[False, False])

# Save predictions to CSV
predictions_df.to_csv("predictions_sorted.csv", index=False)

print("\n✅ Predictions successfully generated and saved!")
print(predictions_df)
