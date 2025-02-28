import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV, KFold
from scipy.stats import randint

# Load datasets
file_path = "nba_players_game_logs_2018_25.csv"
team_stats_path = "team_stats.csv"

df = pd.read_csv(file_path)
team_stats_df = pd.read_csv(team_stats_path)

def predict_player_stat_incremental(player_name, opponent_team, home_game, stat_type):
    """Predicts a player's stat incrementally and calculates R^2."""

    player_df = df[df['PLAYER_NAME'] == player_name]
    if player_df.empty:
        return f"No data found for {player_name}."

    player_df = player_df.sort_values(by='GAME_DATE', ascending=False)

    stat_column_map = {'points': 'PTS', 'rebounds': 'REB', 'assists': 'AST', 'pts_rebs_asts': 'PTS_REB_AST'}
    target_variable = stat_column_map.get(stat_type)

    if target_variable not in player_df.columns:
        return f"Stat type {stat_type} not found for {player_name}."

    player_df['HOME_GAME'] = player_df['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
    player_df['GAME_DATE'] = pd.to_datetime(player_df['GAME_DATE'])
    player_df['BACK_TO_BACK'] = player_df['GAME_DATE'].diff().dt.days.eq(1).astype(int)

    player_df['PTS_PREV_1'] = player_df[target_variable].shift(1, fill_value=player_df[target_variable].mean())
    player_df['PTS_PREV_2'] = player_df[target_variable].shift(2, fill_value=player_df[target_variable].mean())
    player_df['PTS_PREV_3'] = player_df[target_variable].shift(3, fill_value=player_df[target_variable].mean())
    player_df['PROP_VARIANCE_10'] = player_df[target_variable].rolling(window=10, min_periods=1).var().fillna(player_df[target_variable].var()) #min_periods=1
    player_df['TEAM_MATCHUP'] = player_df['MATCHUP'].apply(lambda x: x.split()[-1])
    player_df['AVG_VS_TEAM_5'] = player_df.groupby('TEAM_MATCHUP')[target_variable].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True).fillna(player_df[target_variable].mean()) #min_periods=1

    player_df = player_df.merge(team_stats_df, left_on='TEAM_MATCHUP', right_on='TEAM_ABBREVIATION', how='left')

    features = ['HOME_GAME', 'BACK_TO_BACK', 'PTS_PREV_1', 'PTS_PREV_2', 'PTS_PREV_3',
                'OFF_RATING', 'DEF_RATING', 'NET_RATING', 'PACE', 'AST_RATIO', 'REB_PCT', 'PROP_VARIANCE_10', 'AVG_VS_TEAM_5']

    player_df = player_df.dropna(subset=features + [target_variable])

    test_df = player_df.iloc[:15].copy()
    train_df = player_df.iloc[15:].copy()

    X_train, y_train = train_df[features].values, train_df[target_variable].values

    param_dist = {
        "n_estimators": randint(50, 200),
        "max_depth": [3, 5, 7, 10, None],
        "min_samples_split": randint(2, 10),
        "min_samples_leaf": randint(1, 5)
    }

    model = RandomForestRegressor(random_state=10)
    kf = KFold(n_splits=5, shuffle=True, random_state=10) # increase folds.
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=50, cv=kf, scoring='r2', n_jobs=-1, random_state=10)
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_

    predictions = []
    actuals = []

    for i in range(len(test_df)):
        current_game = test_df.iloc[i]

        input_data = pd.DataFrame({
            'HOME_GAME': [current_game['HOME_GAME']],
            'BACK_TO_BACK': [current_game['BACK_TO_BACK']],
            'PTS_PREV_1': [current_game['PTS_PREV_1']],
            'PTS_PREV_2': [current_game['PTS_PREV_2']],
            'PTS_PREV_3': [current_game['PTS_PREV_3']],
            'PROP_VARIANCE_10': [current_game['PROP_VARIANCE_10']],
            'TEAM_MATCHUP': [current_game['TEAM_MATCHUP']],
            'AVG_VS_TEAM_5': [current_game['AVG_VS_TEAM_5']]
        })

        input_data = input_data.merge(team_stats_df, left_on='TEAM_MATCHUP', right_on='TEAM_ABBREVIATION', how='left')
        input_features = input_data[features].values

        prediction = best_model.predict(input_features)[0]
        actual = current_game[target_variable]

        predictions.append(prediction)
        actuals.append(actual)

        # Update the model with the new data
        X_train = np.vstack([X_train, current_game[features].values])
        y_train = np.append(y_train, current_game[target_variable])
        best_model.fit(X_train, y_train)

    r2 = r2_score(actuals, predictions)
    print(f"R^2 on most recent 15 (incremental update): {r2}")

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(actuals)), actuals, label='Actual', marker='o')
    plt.plot(range(len(predictions)), predictions, label='Predicted', marker='x')
    plt.xlabel('Game Index (Most Recent 15)')
    plt.ylabel(stat_type.upper())
    plt.title(f'{player_name} {stat_type.upper()} Incremental Prediction vs Actual (R^2: {r2:.2f})')
    plt.legend()
    plt.grid(True)
    plt.show()

    return predictions[-1]

# Example usage
player = "Ivica Zubac"
opponent = "BOS"
home_away = "home"
stat = "points"

predicted_value = predict_player_stat_incremental(player, opponent, home_away, stat)
print(f"Predicted {stat} for {player} vs {opponent} ({home_away}): {predicted_value}")