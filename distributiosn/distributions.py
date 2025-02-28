import pandas as pd
import numpy as np
from scipy.stats import kstest
import warnings
from scipy.stats import norm, beta, expon, chi2, t, f, pareto, rayleigh, cauchy, triang, laplace, uniform, logistic, gumbel_r, gumbel_l, gamma, weibull_min, invgauss, genextreme

warnings.filterwarnings("ignore")

# Load data
team_stats = pd.read_csv('team_stats.csv')
game_logs = pd.read_csv('nba_players_game_logs_2018_25.csv')
fixed_data = pd.read_csv('fixed_data.csv')

# Function to fit distributions and choose the best one
def fit_best_distribution(data):
    print("Fitting distributions for data...")
    distributions = [
        norm, beta, expon, chi2, t, f, pareto, rayleigh, cauchy, triang, laplace, uniform, logistic, gumbel_r, gumbel_l, gamma, weibull_min, invgauss, genextreme
    ]
    best_distribution = None
    best_mse = float('inf')
    best_params = None

    for distribution in distributions:
        print(f"Trying distribution: {distribution.name}")
        params = distribution.fit(data.replace([np.inf, -np.inf], np.nan).dropna())
        ks_stat, p_value = kstest(data, distribution.name, args=params)
        print(f"KS Statistic for {distribution.name}: {ks_stat}, p-value: {p_value}")
        if ks_stat < best_mse:
            best_mse = ks_stat
            best_distribution = distribution
            best_params = params

    print(f"Best distribution: {best_distribution.name} with MSE: {best_mse}")
    return best_distribution, best_params, best_mse

# Calculate expected stats for each player
results = []

for player in fixed_data['Player'].unique():
    print(f"Processing player: {player}")
    player_logs = game_logs[game_logs['PLAYER_NAME'] == player].sort_values(by='GAME_DATE', ascending=False).head(69)
    
    if player_logs.empty:
        print(f"No logs found for player: {player}")
        continue
    
    print(f"Player logs for {player}:")
    print(player_logs)
    
    points_distribution, points_params, points_mse = fit_best_distribution(player_logs['PTS'].replace([np.inf, -np.inf], np.nan).dropna())
    rebounds_distribution, rebounds_params, rebounds_mse = fit_best_distribution(player_logs['REB'].replace([np.inf, -np.inf], np.nan).dropna())
    assists_distribution, assists_params, assists_mse = fit_best_distribution(player_logs['AST'].replace([np.inf, -np.inf], np.nan).dropna())
    
    # Calculate total points + rebounds + assists
    total_stats = player_logs['PTS'] + player_logs['REB'] + player_logs['AST']
    total_distribution, total_params, total_mse = fit_best_distribution(total_stats.replace([np.inf, -np.inf], np.nan).dropna())
    
    results.append({
        'Player': player,
        'Points Mean': points_distribution.mean(*points_params),
        'Points Std Dev': points_distribution.std(*points_params),
        'Points Distribution': points_distribution.name,
        'Points MSE': points_mse,
        'Points Params': [float(param) for param in points_params],
        'Rebounds Mean': rebounds_distribution.mean(*rebounds_params),
        'Rebounds Std Dev': rebounds_distribution.std(*rebounds_params),
        'Rebounds Distribution': rebounds_distribution.name,
        'Rebounds MSE': rebounds_mse,
        'Rebounds Params': [float(param) for param in rebounds_params],
        'Assists Mean': assists_distribution.mean(*assists_params),
        'Assists Std Dev': assists_distribution.std(*assists_params),
        'Assists Distribution': assists_distribution.name,
        'Assists MSE': assists_mse,
        'Assists Params': [float(param) for param in assists_params],
        'Total Mean': total_distribution.mean(*total_params),
        'Total Std Dev': total_distribution.std(*total_params),
        'Total Distribution': total_distribution.name,
        'Total MSE': total_mse,
        'Total Params': [float(param) for param in total_params],
    })

# Convert results to DataFrame and save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('distributiosn/player_expected_stats.csv', index=False)
