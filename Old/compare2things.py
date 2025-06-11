import csv

def read_csv(filepath):
    with open(filepath, mode='r') as file:
        reader = csv.DictReader(file)
        return list(reader)

def join_prop_lines(prop_probabilities, nba_predictions):
    joined_data = []
    for prop in prop_probabilities:
        for prediction in nba_predictions:
            if (prop['Player'] == prediction['Player'] and
                prop['Opponent Team'] == prediction['Opponent Team'] and
                prop['Stat Type'] == prediction['Stat Type'] and
                prop['Prop Line'] == prediction['Prop Line'] and
                prop['Bet'].lower() == prediction['Over/Under'].lower()):
                joined_record = {
                    'Player': prop['Player'],
                    'Probability j': prop['Probability'],
                    'Probability n': max(float(prediction['Over Probability']), float(prediction['Under Probability'])),
                    'Over/Under A': prop['Bet'],
                    'Over/Under B': prediction['Over/Under'],
                    'Stat Type': prop['Stat Type'],
                }
                joined_data.append(joined_record)

    return joined_data

def calculate_matching_props(joined_data, prop_probabilities, nba_predictions):
    total_props = min(len(prop_probabilities), len(nba_predictions))
    matching_props = len(joined_data)
    return matching_props, total_props

def write_csv(filepath, data):
    if data:
        keys = data[0].keys()
        with open(filepath, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)

prop_probabilities = read_csv('distributiosn/prop_probabilities.csv')
nba_predictions = read_csv('nba_predictions.csv')
joined_data = join_prop_lines(prop_probabilities, nba_predictions)

# Sort the joined data by 'Probability n'
sorted_joined_data = sorted(joined_data, key=lambda x: x['Probability n'], reverse=True)

matching_props, total_props = calculate_matching_props(sorted_joined_data, prop_probabilities, nba_predictions)

# Write the sorted joined data to a CSV file
write_csv('joined_data.csv', sorted_joined_data)

# Print the number of matching props out of the total number of props
print(f'Matching props: {matching_props} out of {total_props}')