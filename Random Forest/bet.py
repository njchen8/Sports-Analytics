import pandas as pd
import numpy as np
from scipy.stats import poisson, norm

# Load the predictions CSV
predictions_csv = "nba_predictions.csv"  # Replace with your predictions CSV file path
predictions_df = pd.read_csv(predictions_csv)

def calculate_probability(row):
    """Calculates the probability and determines over/under."""

    predicted_value = row['Predicted Value']
    prop_line = row['Prop Line']
    variance = row['Variance (Last 10)']
    stat_type = row['Stat Type']

    if pd.isna(predicted_value) or pd.isna(prop_line) or pd.isna(variance):
        return None, None  # Handle missing values

    if variance <= 0:
        return None, None  # cannot calculate distribution with no variance.

    # Determine if we should use normal distribution
    use_normal_distribution = False
    if stat_type in ['rebounds', 'assists'] and predicted_value >= 10:  # Example threshold: predicted value >= 10
        use_normal_distribution = True
    elif stat_type == 'points':
        use_normal_distribution = True

    if use_normal_distribution:
        std_dev = np.sqrt(variance)
        if std_dev == 0:
            return None, None  # cannot calculate distribution with no variance.
        prob_over = 1 - norm.cdf(prop_line, loc=predicted_value, scale=std_dev)
        prob_under = 1 - prob_over
    elif stat_type in ['rebounds', 'assists']:
        if predicted_value <= 0:
            return None, None  # cannot calculate poisson with negative or zero predicted value.
        prob_over = 1 - poisson.cdf(int(prop_line - 1), mu=predicted_value)
        prob_under = 1 - prob_over
    else:
        return None, None  # Handle unknown stat types

    if prob_over >= 0.5:
        return prob_over, 'over'
    else:
        return prob_under, 'under'

# Calculate probabilities and over/under
probabilities = []
over_unders = []

for index, row in predictions_df.iterrows():
    probability, over_under = calculate_probability(row)
    if probability is not None:
        probabilities.append(probability)
        over_unders.append(over_under)
    else:
        probabilities.append(np.nan)
        over_unders.append(np.nan)

predictions_df['Probability'] = probabilities
predictions_df['Over/Under'] = over_unders

# Filter to probabilities >= 0.5
predictions_df = predictions_df[predictions_df['Probability'] >= 0.5]

# Sort by probability (descending)
predictions_df = predictions_df.sort_values(by='Probability', ascending=False)

# Save the updated DataFrame to a new CSV file
output_csv = "nba_predictions_with_probabilities_over_under_sorted.csv"
predictions_df.to_csv(output_csv, index=False)

print(f"Probabilities and Over/Under calculated, filtered, sorted, and saved to {output_csv}")
print(predictions_df)