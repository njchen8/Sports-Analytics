import pandas as pd
import scipy.stats as stats

# Load the data
print("Loading player stats data...")
player_stats = pd.read_csv('distributiosn/player_expected_stats.csv')
print("Player stats data loaded.")

print("Loading fixed data...")
fixed_data = pd.read_csv('fixed_data.csv')
print("Fixed data loaded.")

# Function to get the distribution object
def get_distribution(dist_name):
    distributions = {
        'norm': stats.norm,
        'beta': stats.beta,
        'expon': stats.expon,
        'chi2': stats.chi2,
        't': stats.t,
        'f': stats.f,
        'pareto': stats.pareto,
        'rayleigh': stats.rayleigh,
        'cauchy': stats.cauchy,
        'triang': stats.triang,
        'laplace': stats.laplace,
        'uniform': stats.uniform,
        'logistic': stats.logistic,
        'gumbel_r': stats.gumbel_r,
        'gumbel_l': stats.gumbel_l,
        'gamma': stats.gamma,
        'weibull_min': stats.weibull_min,
        'invgauss': stats.invgauss,
        'genextreme': stats.genextreme
    }
    return distributions.get(dist_name, stats.norm)

# Function to calculate probability based on distribution type
def calculate_probability(row, player_stats):
    player = row['Player']
    prop_line = row['Prop Line']
    stat_type = row['Stat Type']
    
    print(f"Calculating probability for player: {player}, stat type: {stat_type}, prop line: {prop_line}")
    
    player_stat = player_stats[player_stats['Player'] == player]
    if player_stat.empty:
        print(f"No stats found for player: {player}. Using default probability.")
        return 0.5, 'under'  # or some default probability
    player_stat = player_stat.iloc[0]
    
    if stat_type == 'points':
        mean = player_stat['Points Mean']
        std = player_stat['Points Std Dev']
        dist_name = player_stat['Points Distribution']
    elif stat_type == 'rebounds':
        mean = player_stat['Rebounds Mean']
        std = player_stat['Rebounds Std Dev']
        dist_name = player_stat['Rebounds Distribution']
    elif stat_type == 'assists':
        mean = player_stat['Assists Mean']
        std = player_stat['Assists Std Dev']
        dist_name = player_stat['Assists Distribution']
    elif stat_type == 'pts_rebs_asts':
        mean = player_stat['Total Mean']
        std = player_stat['Total Std Dev']
        dist_name = player_stat['Total Distribution']
    else:
        print(f"Unknown stat type: {stat_type}. Using default probability.")
        return 0.5, 'under'  # or some default probability
    
    print(f"Using distribution: {dist_name}, mean: {mean}, std: {std}")
    distribution = get_distribution(dist_name)
    
    if dist_name in ['beta', 'gamma', 'weibull_min', 'pareto', 'triang', 'f', 'genextreme']:
        # These distributions require shape parameters
        shape_params = player_stat.get('Shape Params', None)
        if shape_params is not None:
            prob = distribution.cdf(prop_line, *shape_params, loc=mean, scale=std)
        else:
            print(f"Shape parameters missing for distribution: {dist_name}. Using default probability.")
            return 0.5, 'under'  # or some default probability
    elif dist_name in ['chi2', 't', 'f']:
        df = player_stat.get('Degrees of Freedom', None)
        if df is not None:
            prob = distribution.cdf(prop_line, df, loc=mean, scale=std)
        else:
            print(f"Degrees of freedom missing for distribution: {dist_name}. Using default probability.")
            return 0.5, 'under'  # or some default probability
    else:
        prob = distribution.cdf(prop_line, mean, std)
    
    final_prob = max(prob, 1 - prob)
    bet = 'over' if 1 - prob > prob else 'under'
    print(f"Calculated probability: {final_prob}, Bet: {bet}")
    return final_prob, bet

# Calculate probabilities for all stat types
print("Calculating probabilities for all rows...")
fixed_data[['Probability', 'Bet']] = fixed_data.apply(lambda row: calculate_probability(row, player_stats), axis=1, result_type='expand')

# Sort by probability
print("Sorting data by probability...")
fixed_data = fixed_data.sort_values(by='Probability', ascending=False)

# Save to CSV
print("Saving probabilities to CSV...")
fixed_data.to_csv('distributiosn/prop_probabilities.csv', index=False)
print("Probabilities saved to CSV.")