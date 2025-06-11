import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV, KFold
from scipy.stats import randint, norm
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output

# Load datasets
file_path = "nba_players_game_logs_2018_25.csv"
df = pd.read_csv(file_path, low_memory=True)

team_stats_path = "team_stats.csv"
team_stats_df = pd.read_csv(team_stats_path, low_memory=True)

prop_data_path = "fixed_data.csv"
prop_df = pd.read_csv(prop_data_path, low_memory=True)

# Convert GAME_DATE to datetime
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
latest_date = df['GAME_DATE'].max()

# Merge team stats (optimized)
team_stats_dict = team_stats_df.set_index('TEAM_ABBREVIATION').to_dict('index')

def merge_team_stats(data):
    def get_team_stats(opponent):
        return team_stats_dict.get(opponent, {})

    team_stats = data['OPPONENT'].apply(get_team_stats).apply(pd.Series)
    return pd.concat([data.reset_index(drop=True), team_stats], axis=1)

# Create features: HOME_GAME and lag features (optimized)
def create_features(data, target_variable):
    data = data.copy()
    data['HOME_GAME'] = data['MATCHUP'].apply(lambda x: 1 if ' vs. ' in x else 0).astype(np.int8)
    for lag in range(1, 4):
        data[f'PTS_PREV_{lag}'] = data[target_variable].shift(lag).fillna(data[target_variable].mean()).astype(np.float32)
    return data

# Optimize hyperparameters using RandomizedSearchCV (optimized)
def optimize_hyperparameters(train_df, target_variable, features):
    param_dist = {
        "n_estimators": randint(200, 500),
        "max_depth": [7, 10, None],
        "min_samples_split": randint(2, 8),
        "min_samples_leaf": randint(1, 4)
    }

    X_train = train_df[features].values.astype(np.float32)
    y_train = train_df[target_variable].values.astype(np.float32)

    model = RandomForestRegressor(random_state=10, bootstrap=True, oob_score=True, n_jobs=-1)

    kf = KFold(n_splits=3, shuffle=True, random_state=10)

    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=30,  # Reduced iterations for speed
        cv=kf,
        scoring='r2',
        n_jobs=-1,
        random_state=10
    )

    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    best_cv_score = random_search.best_score_

    print(f"Optimized hyperparameters: {best_params} (CV R²: {best_cv_score:.3f})")
    return best_model

def process_player(row):
    player_name = row['Player']
    opponent_team = row['Opponent Team']
    stat_type = row['Stat Type']
    if stat_type not in ['points', 'rebounds', 'assists', 'pts_rebs_asts']:
        return None

    player_df = df[df['PLAYER_NAME'] == player_name]
    if player_df.empty or (latest_date - player_df['GAME_DATE'].max()).days > 7:
        return None

    player_df = player_df.sort_values(by='GAME_DATE', ascending=False).head(75)
    test_df = player_df.head(15).reset_index(drop=True)
    train_df = player_df.tail(60).reset_index(drop=True)

    stat_column_map = {'points': 'PTS', 'rebounds': 'REB', 'assists': 'AST', 'pts_rebs_asts': 'PTS_REB_AST'}
    target_variable = stat_column_map.get(stat_type)

    if target_variable not in train_df.columns:
        return None

    train_df = create_features(train_df, target_variable)
    train_df['OPPONENT'] = opponent_team
    train_df = merge_team_stats(train_df)

    variance_10 = train_df[target_variable].tail(10).var()
    train_df['PROP_VARIANCE_10'] = variance_10

    features = ['HOME_GAME', 'PTS_PREV_1', 'PTS_PREV_2', 'PTS_PREV_3',
                'OFF_RATING', 'DEF_RATING', 'NET_RATING', 'PACE', 'AST_RATIO', 'REB_PCT', 'PROP_VARIANCE_10']

    best_model = optimize_hyperparameters(train_df, target_variable, features)

    test_df = create_features(test_df, target_variable)
    test_df['OPPONENT'] = opponent_team
    test_df = merge_team_stats(test_df)
    test_df['PROP_VARIANCE_10'] = variance_10

    X_test = test_df[features].values.astype(np.float32)
    y_test = test_df[target_variable].values.astype(np.float32)

    y_pred_test = best_model.predict(X_test)
    final_r2 = r2_score(y_test, y_pred_test) if len(y_test) > 1 else np.nan

    predicted_stat = best_model.predict(train_df[features].values[-1].reshape(1, -1).astype(np.float32))[0]

    recent_stats = train_df[target_variable].tail(10)
    mean_stat = recent_stats.mean()
    std_stat = np.sqrt(recent_stats.var())

    probability = 1 - norm.cdf(row['Prop Line'], loc=mean_stat, scale=std_stat) if std_stat > 0 else 1.0 if predicted_stat >= row['Prop Line'] else 0.0
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

player_predictions = Parallel(n_jobs=-1)(
    delayed(process_player)(row) for _, row in prop_df.iterrows()
)

player_predictions = [pred for pred in player_predictions if pred is not None]

predictions_df = pd.DataFrame(player_predictions)
predictions_df['R^2 Group'] = np.where(predictions_df['R^2 Value'] >= 0, 'Positive R²', 'Negative R²')
predictions_df = predictions_df.sort_values(by=['R^2 Group', 'Probability Max'], ascending=[False, False])

predictions_df.to_csv("predictions_sorted.csv", index=False)

print("\n✅ Predictions successfully generated and saved!")
print(predictions_df)