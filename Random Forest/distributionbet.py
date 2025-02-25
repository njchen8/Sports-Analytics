import pandas as pd
import numpy as np
from scipy.stats import poisson, norm

# Load the predictions CSV with distributions
predictions_csv = "nba_predictions_distribution.csv"  # Replace with your predictions CSV file path
predictions_df = pd.read_csv(predictions_csv)

def calculate_probability_from_distribution(row):
    """Calculates the probability and determines over/under from a distribution."""

    distribution_str = row['Predicted Distribution']
    prop_line = row['Prop Line']

    if pd.isna(distribution_str) or pd.isna(prop_line):
        return None, None  # Handle missing values

    # Convert the string representation of the dictionary back to a dictionary
    try:
        distribution = eval(distribution_str)
    except (SyntaxError, NameError, TypeError):
        return None, None #Handle invalid distribution strings

    if not isinstance(distribution, dict):
        return None, None

    prob_over = 0
    prob_under = 0

    for stat_val, prob in distribution.items():
        if stat_val >= prop_line:
            prob_over += prob
        else:
            prob_under += prob

    if prob_over + prob_under == 0:
        return None, None #Handle when distribution is empty.

    prob_over /= (prob_over + prob_under)
    prob_under /= (prob_over + prob_under)

    if prob_over >= 0.5:
        return prob_over, 'over'
    else:
        return prob_under, 'under'

# Calculate probabilities and over/under
probabilities = []
over_unders = []

for index, row in predictions_df.iterrows():
    probability, over_under = calculate_probability_from_distribution(row)
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
output_csv = "nba_predictions_distribution_with_probabilities_over_under_sorted.csv"
predictions_df.to_csv(output_csv, index=False)

print(f"Probabilities and Over/Under calculated, filtered, sorted, and saved to {output_csv}")
print(predictions_df)