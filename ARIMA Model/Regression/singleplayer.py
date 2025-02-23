import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.stattools import adfuller

# Load dataset
file_path = "nba_players_game_logs_cleaned.csv"  # Update with the correct path
df = pd.read_csv(file_path)
team_stats_path = "team_stats.csv"
team_stats_df = pd.read_csv(team_stats_path)

# Filter for LeBron James' games
lebron_df = df[df['PLAYER_NAME'] == 'LeBron James']

# Sort by date and select the last 70 games
lebron_df = lebron_df.sort_values(by='GAME_DATE', ascending=False).head(70)
train_df = lebron_df.iloc[20:].copy()
test_df = lebron_df.iloc[:20].copy()

# Assign matchup-based time intervals
train_df = train_df.reset_index(drop=True)
train_df['MATCHUP_INDEX'] = range(1, len(train_df) + 1)
test_df = test_df.reset_index(drop=True)
test_df['MATCHUP_INDEX'] = range(len(train_df) + 1, len(train_df) + len(test_df) + 1)

# Identify home vs away games and extract opponent team
train_df['HOME_GAME'] = train_df['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
test_df['HOME_GAME'] = test_df['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
train_df['OPPONENT'] = train_df['MATCHUP'].apply(lambda x: x.split()[-1])
test_df['OPPONENT'] = test_df['MATCHUP'].apply(lambda x: x.split()[-1])

# Merge with team stats
def merge_team_stats(df):
    return df.merge(team_stats_df, left_on='OPPONENT', right_on='TEAM_ABBREVIATION', how='left')

train_df = merge_team_stats(train_df)
test_df = merge_team_stats(test_df)

# Identify back-to-back games
train_df['GAME_DATE'] = pd.to_datetime(train_df['GAME_DATE'])
test_df['GAME_DATE'] = pd.to_datetime(test_df['GAME_DATE'])
train_df['BACK_TO_BACK'] = train_df['GAME_DATE'].diff().dt.days.eq(1).astype(int)
test_df['BACK_TO_BACK'] = test_df['GAME_DATE'].diff().dt.days.eq(1).astype(int)

# Select a performance metric (e.g., POINTS)
target_variable = 'PTS' if 'PTS' in train_df.columns else train_df.columns[-1]

# Include past game performance indicators
for df in [train_df, test_df]:
    df['PTS_PREV_1'] = df[target_variable].shift(1, fill_value=df[target_variable].mean())
    df['PTS_PREV_2'] = df[target_variable].shift(2, fill_value=df[target_variable].mean())
    df['PTS_PREV_3'] = df[target_variable].shift(3, fill_value=df[target_variable].mean())

# Prepare training data
features = ['MATCHUP_INDEX', 'HOME_GAME', 'BACK_TO_BACK', 'PTS_PREV_1', 'PTS_PREV_2', 'PTS_PREV_3',
            'OFF_RATING', 'DEF_RATING', 'NET_RATING', 'PACE', 'AST_RATIO', 'REB_PCT']
X_train = train_df[features].values
y_train = train_df[target_variable].values

# Fit a regression model using Random Forest Regressor
model = RandomForestRegressor(n_estimators=20, max_depth=5, min_samples_split=5, random_state=10)
model.fit(X_train, y_train)

# Iteratively predict one game ahead
predictions = []
current_test_df = test_df.copy()
for i in range(len(test_df)):
    X_test = current_test_df.iloc[[i]][features].values
    pred = model.predict(X_test)[0]
    predictions.append(pred)
    
    # Update future data with the new prediction
    if i < len(test_df) - 1:
        current_test_df.at[i + 1, 'PTS_PREV_1'] = pred
        current_test_df.at[i + 1, 'PTS_PREV_2'] = current_test_df.at[i, 'PTS_PREV_1']
        current_test_df.at[i + 1, 'PTS_PREV_3'] = current_test_df.at[i, 'PTS_PREV_2']

# Plot the actual vs. predicted values
plt.figure(figsize=(10, 5))
plt.plot(range(1, 21), test_df[target_variable], label='Actual', marker='o')
plt.plot(range(1, 21), predictions, label='Predicted', linestyle='dashed', marker='x')
plt.xlabel('Matchup Index (Last 20 Games)')
plt.ylabel(target_variable)
plt.title(f'One-Step Ahead Predictions for the Last 20 Games with Opponent Stats')
plt.legend()
plt.show()

# Display the last 20 rows with predictions
test_df['Predicted_PTS'] = predictions
print(test_df[['MATCHUP', 'HOME_GAME', 'BACK_TO_BACK', 'PTS_PREV_1', 'PTS_PREV_2', 'PTS_PREV_3',
               'OFF_RATING', 'DEF_RATING', 'NET_RATING', 'PACE', 'AST_RATIO', 'REB_PCT', target_variable, 'Predicted_PTS']])
