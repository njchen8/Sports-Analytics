import pandas as pd
import numpy as np
from scipy.stats import norm

# Load data files
props = pd.read_csv('player_expected_values.csv')

# Calculate probabilities for over and under based on given lines and expected values
parlays = []
for _, row in props.iterrows():
    mean = row['Predicted Value']
    line = row['Prop Line']
    std_dev = np.sqrt(row['Variance'])

    if pd.isna(std_dev) or std_dev == 0:
        continue

    # Calculate z-score
    z_score = (line - mean) / std_dev

    # Probability of hitting over
    prob_over = 1 - norm.cdf(z_score)
    # Probability of hitting under
    prob_under = norm.cdf(z_score)

    best_choice = 'over' if prob_over > prob_under else 'under'
    best_prob = max(prob_over, prob_under)

    parlays.append({
        'player': row['Player'],
        'stat': row['Stat Type'],
        'line': line,
        'best_bet': best_choice,
        'probability': best_prob,
        'expected_value': row['Predicted Value']
    })

# Convert to DataFrame and sort by highest probability
parlays_df = pd.DataFrame(parlays)
parlays_df = parlays_df.sort_values(by='probability', ascending=False)

# Display top parlays
print(parlays_df.head(10))