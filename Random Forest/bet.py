import pandas as pd
import numpy as np
from scipy.stats import poisson, norm

# Load the predictions CSV
predictions_csv = "nba_predictions.csv"  # Replace with your predictions CSV file path
predictions_df = pd.read_csv(predictions_csv)

def calculate_probability(row):
    """Calculates the probability of a prop hitting based on the predicted value, prop line, and variance."""

    predicted_value = row['Predicted Value']
    prop_line = row['Prop Line']
    variance = row['Variance (Last 10)']
    stat_type = row['Stat Type']

    if pd.isna(predicted_value) or pd.isna(prop_line) or pd.isna(variance):
        return None  # Handle missing values

    if variance <= 0:
        return None  # cannot calculate distribution with no variance.

    # Determine if we should use normal distribution
    use_normal_distribution = False
    if stat_type in ['rebounds', 'assists'] and predicted_value >= 10:  # Example threshold: predicted value >= 10
        use_normal_distribution = True
    elif stat_type == 'points':
        use_normal_distribution = True

    if use_normal_distribution:
        std_dev = np.sqrt(variance)
        if std_dev == 0:
            return None  # cannot calculate distribution with no variance.
        probability = 1 - norm.cdf(prop_line, loc=predicted_value, scale=std_dev)
    elif stat_type in ['rebounds', 'assists']:
        if predicted_value <= 0:
            return None  # cannot calculate poisson with negative or zero predicted value.
        probability = 1 - poisson.cdf(int(prop_line - 1), mu=predicted_value)
    else:
        return None  # Handle unknown stat types

    return probability

# Calculate probabilities
probabilities = []

for index, row in predictions_df.iterrows():
    probability = calculate_probability(row)
    if probability is not None:
        probabilities.append(probability)
    else:
        probabilities.append(np.nan)

predictions_df['Probability'] = probabilities

# Sort by probability (descending)
predictions_df = predictions_df.sort_values(by='Probability', ascending=False)

# Save the updated DataFrame to a new CSV file
output_csv = "nba_predictions_with_probabilities_sorted.csv"
predictions_df.to_csv(output_csv, index=False)

print(f"Probabilities calculated, sorted, and saved to {output_csv}")
print(predictions_df)