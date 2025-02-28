import pandas as pd
import numpy as np
from scipy.stats import gamma, norm, beta, expon
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# Load data
team_stats = pd.read_csv('team_stats.csv')
game_logs = pd.read_csv('nba_players_game_logs_2018_25.csv')
fixed_data = pd.read_csv('fixed_data.csv')

# Function to calculate weighted mean and standard deviation
def weighted_stats(values, weights):
    weighted_mean = np.sum(values * weights) / np.sum(weights)
    weighted_variance = np.sum(weights * (values - weighted_mean) ** 2) / np.sum(weights)
    weighted_std = np.sqrt(weighted_variance)
    return weighted_mean, weighted_std

# Function to fit distributions and choose the best one
def fit_best_distribution(data):
    print("Fitting distributions for data...")
    distributions = [gamma, norm, beta, expon]
    best_distribution = None
    best_mse = float('inf')
    best_params = None

    for distribution in distributions:
        print(f"Trying distribution: {distribution.name}")
        params = distribution.fit(data)
        fitted_data = distribution.pdf(np.arange(len(data)), *params)
        mse = mean_squared_error(data, fitted_data)
        print(f"Mean Squared Error for {distribution.name}: {mse}")
        if mse < best_mse:
            best_mse = mse
            best_distribution = distribution
            best_params = params

    print(f"Best distribution: {best_distribution.name} with MSE: {best_mse}")
    return best_distribution, best_params

# Calculate expected stats for each player
results = []

for player in fixed_data['Player'].unique():
    print(f"Processing player: {player}")
    player_logs = game_logs[game_logs['PLAYER_NAME'] == player].sort_values(by='GAME_DATE', ascending=False).head(50)
    
    if player_logs.empty:
        print(f"No logs found for player: {player}")
        continue
    
    print(f"Player logs for {player}:")
    print(player_logs)
    
    # Decaying weights
    weights = np.exp(-0.1 * np.arange(len(player_logs)))
    
    points_mean, points_std = weighted_stats(player_logs['PTS'], weights)
    rebounds_mean, rebounds_std = weighted_stats(player_logs['REB'], weights)
    assists_mean, assists_std = weighted_stats(player_logs['AST'], weights)
    
    points_distribution, points_params = fit_best_distribution(player_logs['PTS'])
    rebounds_distribution, rebounds_params = fit_best_distribution(player_logs['REB'])
    assists_distribution, assists_params = fit_best_distribution(player_logs['AST'])
    
    results.append({
        'Player': player,
        'Points Mean': points_mean,
        'Points Std Dev': points_std,
        'Points Distribution': points_distribution.name,
        'Points Params': [float(param) for param in points_params],
        'Rebounds Mean': rebounds_mean,
        'Rebounds Std Dev': rebounds_std,
        'Rebounds Distribution': rebounds_distribution.name,
        'Rebounds Params': [float(param) for param in rebounds_params],
        'Assists Mean': assists_mean,
        'Assists Std Dev': assists_std,
        'Assists Distribution': assists_distribution.name,
        'Assists Params': [float(param) for param in assists_params]
    })

# Convert results to DataFrame and save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('distributiosn/player_expected_stats.csv', index=False)