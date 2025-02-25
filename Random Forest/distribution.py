import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV, KFold
from scipy.stats import randint, norm
from datetime import datetime, timedelta

# Load datasets
file_path = "nba_players_game_logs_2018_25.csv"
team_stats_path = "team_stats.csv"
prop_data_path = "fixed_data.csv"  # Replace with your prop data file path

df = pd.read_csv(file_path)
team_stats_df = pd.read_csv(team_stats_path)
prop_df = pd.read_csv(prop_data_path)

def predict_player_stat_distribution(player_name, opponent_team, home_game, stat_type, prop_line, num_samples=1000):
    """Predicts a player's stat and returns a distribution of probabilities."""

    print(f"\nProcessing: {player_name}, {stat_type} vs {opponent_team} ({'home' if home_game == 'home' else 'away'})")

    player_df = df[df['PLAYER_NAME'] == player_name]
    if player_df.empty:
        print(f"  -> Player not found: {player_name}")
        return None  # Player not found

    if len(player_df) < 35:
        print(f"  -> Less than 35 matches: {player_name}")
        return None  # Less than 35 matches

    player_df = player_df.sort_values(by='GAME_DATE', ascending=False)
    latest_game_date = pd.to_datetime(player_df['GAME_DATE'].iloc[0])
    if (datetime.now() - latest_game_date) > timedelta(days=7):
        print(f"  -> Not played in the last week: {player_name}")
        return None  # Not played in the last week

    stat_column_map = {'points': 'PTS', 'rebounds': 'REB', 'assists': 'AST'}
    target_variable = stat_column_map.get(stat_type)

    if target_variable not in player_df.columns:
        print(f"  -> Stat type not found: {player_name}, {stat_type}")
        return None  # Stat type not found

    player_df['HOME_GAME'] = player_df['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
    player_df['GAME_DATE'] = pd.to_datetime(player_df['GAME_DATE'])
    player_df['BACK_TO_BACK'] = player_df['GAME_DATE'].diff().dt.days.eq(1).astype(int)

    player_df['PTS_PREV_1'] = player_df[target_variable].shift(1, fill_value=player_df[target_variable].mean())
    player_df['PTS_PREV_2'] = player_df[target_variable].shift(2, fill_value=player_df[target_variable].mean())
    player_df['PTS_PREV_3'] = player_df[target_variable].shift(3, fill_value=player_df[target_variable].mean())
    player_df['PROP_VARIANCE_10'] = player_df[target_variable].rolling(window=10, min_periods=1).var().fillna(player_df[target_variable].var())
    player_df['TEAM_MATCHUP'] = player_df['MATCHUP'].apply(lambda x: x.split()[-1])
    player_df['AVG_VS_TEAM_5'] = player_df.groupby('TEAM_MATCHUP')[target_variable].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True).fillna(player_df[target_variable].mean())
    player_df['AVG_LAST_5'] = player_df[target_variable].rolling(window=5, min_periods=1).mean().fillna(player_df[target_variable].mean())

    player_df = player_df.merge(team_stats_df, left_on='TEAM_MATCHUP', right_on='TEAM_ABBREVIATION', how='left')

    features = ['HOME_GAME', 'BACK_TO_BACK', 'PTS_PREV_1', 'PTS_PREV_2', 'PTS_PREV_3',
                'OFF_RATING', 'DEF_RATING', 'NET_RATING', 'PACE', 'AST_RATIO', 'REB_PCT', 'PROP_VARIANCE_10', 'AVG_VS_TEAM_5','AVG_LAST_5']

    player_df = player_df.dropna(subset=features + [target_variable])

    # Split data
    test_df = player_df.iloc[:15].copy()
    train_df = player_df.iloc[15:].copy()
    X_train, y_train = train_df[features].values, train_df[target_variable].values
    X_test, y_test = test_df[features].values, test_df[target_variable].values

    # Hyperparameter tuning
    param_dist = {
        "n_estimators": randint(50, 200),
        "max_depth": [3, 5, 7, 10, None],
        "min_samples_split": randint(2, 10),
        "min_samples_leaf": randint(1, 5)
    }
    model = RandomForestRegressor(random_state=10)
    kf = KFold(n_splits=5, shuffle=True, random_state=10)
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=50, cv=kf, scoring='r2', n_jobs=-1, random_state=10)
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_

    # Prediction
    input_data = pd.DataFrame({
        'HOME_GAME': [1 if home_game == 'home' else 0],
        'BACK_TO_BACK': [0],
        'PTS_PREV_1': [player_df[target_variable].iloc[0]],
        'PTS_PREV_2': [player_df[target_variable].iloc[1]],
        'PTS_PREV_3': [player_df[target_variable].iloc[2]],
        'PROP_VARIANCE_10': [player_df['PROP_VARIANCE_10'].iloc[0]],
        'TEAM_MATCHUP': [opponent_team],
        'AVG_VS_TEAM_5': [player_df[player_df['TEAM_MATCHUP'] == opponent_team][target_variable].rolling(window=5).mean().iloc[-1] if not player_df[player_df['TEAM_MATCHUP'] == opponent_team].empty else player_df[target_variable].mean()],
        'AVG_LAST_5': [player_df[target_variable].rolling(window=5).mean().iloc[-1]]
    })
    input_data = input_data.merge(team_stats_df, left_on='TEAM_MATCHUP', right_on='TEAM_ABBREVIATION', how='left')
    input_features = input_data[features].values

    # Generate distribution using multiple predictions from the forest
    predictions = []
    for tree in best_model.estimators_:
        predictions.append(tree.predict(input_features)[0])
    predictions = np.array(predictions)

    # Increase variance
    variance_factor = 1.5 # Increase variance by 50%
    predictions_adjusted = np.random.normal(np.mean(predictions), np.std(predictions) * variance_factor, len(predictions))

    # Calculate probabilities
    if prop_line > 8.0:  # Adjust this threshold based on your data and domain knowledge
        variance_factor = 1.5
        predictions_adjusted = np.random.normal(np.mean(predictions), np.std(predictions) * variance_factor, len(predictions))
        distribution_type = "normal"
    else:
        mean_prediction = np.mean(predictions)
        predictions_adjusted = np.random.poisson(mean_prediction, len(predictions))
        distribution_type = "poisson"

    # Calculate probabilities
    probs = {}
    if distribution_type == "poisson":
        max_value = int(max(predictions_adjusted) * 2) #expand the range a little for poisson
        for i in range(0, max_value + 1):
            probs[i] = np.mean(predictions_adjusted == i)
    else:
        for i in range(int(min(predictions_adjusted)), int(max(predictions_adjusted)) + 2):
            probs[i] = np.mean(predictions_adjusted >= i) - np.mean(predictions_adjusted >= i + 1)

    # Ensure probabilities are non-negative
    for key, value in probs.items():
        probs[key] = max(0, value)

    # Clean distribution to remove jumps
    cleaned_probs = {}
    last_non_zero = -1
    for k in sorted(probs.keys()):
        if probs[k] > 0:
            cleaned_probs[k] = probs[k]
            last_non_zero = k
        elif last_non_zero != -1:
            cleaned_probs[k] = 0.0
        else:
            cleaned_probs[k] = 0.0

    # Calculate R^2 for the 15 game backtest
    r2_backtest = r2_score(y_test, best_model.predict(X_test))
    print(f"  -> R^2 of 15 game backtest: {r2_backtest:.3f}")

    # Calculate variance of the most recent 10 games
    variance_10 = player_df[target_variable].tail(10).var()
    print(f"  -> Variance of last 10 games: {variance_10:.3f}")

    # Check if mean prediction is too far from prop line
    std_dev = player_df[target_variable].tail(10).std()
    mean_prediction = np.mean(predictions)
    if abs(mean_prediction - prop_line) > std_dev:
        print(f"  -> Prediction too far from prop line: {player_name}, {stat_type}")
        return None

    print(f"  -> Predicted {stat_type} Distribution:")
    for stat_val, prob in cleaned_probs.items():
        print(f"    {stat_val}: {prob:.4f}")

    if distribution_type == "poisson":
        over_probability = sum(probs[i] for i in probs if i > prop_line)
        under_probability = sum(probs[i] for i in probs if i <= prop_line)
    else:
        over_probability = sum(probs[i] for i in probs if i > prop_line)
        under_probability = sum(probs[i] for i in probs if i <= prop_line)

    print(f"  -> Probability over {prop_line}: {over_probability:.4f}")
    print(f"  -> Probability under {prop_line}: {under_probability:.4f}")

    return cleaned_probs, r2_backtest, variance_10, over_probability, under_probability

results = []
processed_combinations = set()

for index, row in prop_df.iterrows():
    player = row['Player']
    opponent = row['Opponent Team']
    home_away = 'home' if row['Home Team'] == row['Current Team'] else 'away'
    prop_line = row['Prop Line']
    stat_types = ['points', 'rebounds', 'assists']

    for stat in stat_types:
        combination = (player, stat)
        if combination not in processed_combinations:
            result = predict_player_stat_distribution(player, opponent, home_away, stat, prop_line)
            if result is not None:
                predicted_distribution, r2_backtest, variance_10, over_probability, under_probability = result #Modified line
                results.append({
                    'Player': player,
                    'Opponent Team': opponent,
                    'Home Team': row['Home Team'],
                    'Stat Type': stat,
                    'Prop Line': prop_line,
                    'R^2 Value': r2_backtest,
                    'Variance (Last 10)': variance_10,
                    'Over Probability': over_probability,
                    'Under Probability': under_probability,
                    'Predicted Distribution': predicted_distribution
                    
                })
            processed_combinations.add(combination)

results_df = pd.DataFrame(results)

# Save the results to a CSV file
results_df.to_csv("nba_predictions_distribution.csv", index=False)

print("Predictions saved to nba_predictions_distribution.csv")