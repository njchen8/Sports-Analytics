import pandas as pd
import numpy as np
from itertools import combinations

# Load the data
data = pd.read_csv('underdog_player_props.csv')

# Calculate expected value for each leg
def calculate_ev(row):
    prob_hit = 1 / row['Prop Line']  # Simplified probability based on Prop Line
    ev = (prob_hit * row['Higher Payout']) + ((1 - prob_hit) * row['Lower Payout'])
    return ev

data['EV'] = data.apply(calculate_ev, axis=1)

# Generate all possible parlays with at least two legs and players from different teams
def generate_parlays(data):
    parlays = []
    teams = data['Current Team'].unique()
    
    for team in teams:
        team_players = data[data['Current Team'] == team]
        other_players = data[data['Current Team'] != team]
        
        for _, player1 in team_players.iterrows():
            for _, player2 in other_players.iterrows():
                parlays.append([player1, player2])
    
    return parlays

parlays = generate_parlays(data)

# Calculate expected value for each parlay
def calculate_parlay_ev(parlay):
    ev = 1
    for leg in parlay:
        ev *= leg['EV']
    return ev

parlay_ev = {tuple((leg['Player'], leg['Stat Type']) for leg in parlay): calculate_parlay_ev(parlay) for parlay in parlays}

# Find the top parlays
top_parlays = sorted(parlay_ev.items(), key=lambda x: x[1], reverse=True)[:10]

# Output the top parlays
for parlay, ev in top_parlays:
    print(f"Parlay: {parlay}, Expected Value: {ev}")