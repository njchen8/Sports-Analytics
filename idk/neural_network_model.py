import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import itertools

print("🚀 Loading datasets...")

# Load datasets
player_data = pd.read_csv("nba_players_game_logs_2018_25.csv")
prop_data = pd.read_csv("underdog_player_props_fixed.csv")

print("✅ Datasets loaded successfully.")

# Merge datasets on player name and season
print("🔄 Merging datasets...")
player_data.rename(columns={"PLAYER_NAME": "Player"}, inplace=True)
merged_data = prop_data.merge(player_data, on=["Player"], how="left")

# Drop rows with missing values
merged_data.dropna(inplace=True)
print(f"✅ Merging complete. {len(merged_data)} rows remaining after dropping NaNs.")

# Convert payouts to float (Fix TypeError)
print("🛠 Cleaning and converting payout columns to float...")
merged_data["Higher Payout"] = merged_data["Higher Payout"].astype(str).str.replace("x", "", regex=False)
merged_data["Lower Payout"] = merged_data["Lower Payout"].astype(str).str.replace("x", "", regex=False)

# Convert to float, setting invalid values to NaN
merged_data["Higher Payout"] = pd.to_numeric(merged_data["Higher Payout"], errors="coerce")
merged_data["Lower Payout"] = pd.to_numeric(merged_data["Lower Payout"], errors="coerce")

# Drop rows where payout conversion failed
merged_data.dropna(subset=["Higher Payout", "Lower Payout"], inplace=True)
print("✅ Payout column conversion complete. Remaining rows:", len(merged_data))

# Feature Engineering
print("🛠 Extracting home/away status and opponent team...")
merged_data['Home'] = merged_data['MATCHUP'].apply(lambda x: 1 if "@" not in x else 0)
merged_data['Opponent'] = merged_data['MATCHUP'].apply(lambda x: x.split(" ")[-1])
print("✅ Feature engineering complete.")

# Select features
features = [
    'MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'Home'
]
X = merged_data[features].values
y = (merged_data['Stat Type'] == 'Over').astype(int)  # Binary classification

print("🔄 Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
print(f"✅ Data split complete. Train size: {len(X_train)}, Test size: {len(X_test)}.")

# Normalize data
print("⚖️ Normalizing data...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("✅ Data normalization complete.")

# Build Neural Network with Dropout & Regularization
print("🛠 Building neural network model...")
model = keras.Sequential([
    layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01), input_shape=(len(features),)),
    layers.Dropout(0.3),  # Dropout for regularization
    layers.Dense(32, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01)),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")  # Probability output
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
print("✅ Model built successfully.")

# Early Stopping Callback
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

print("🏋️ Training model...")
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])
print("✅ Model training complete.")

# Predict probabilities
print("📊 Generating prop predictions...")
merged_data["P(over)"] = model.predict(scaler.transform(merged_data[features]))
merged_data["P(under)"] = 1 - merged_data["P(over)"]
print("✅ Predictions generated.")

# Compute expected values (EV) for both Over and Under
print("📊 Calculating expected values (EV)...")
merged_data["EV_over"] = (
    merged_data["P(over)"] * (merged_data["Higher Payout"] - 1) - 
    (1 - merged_data["P(over)"]) * 1  # Losing amount is the $1 bet
)
merged_data["EV_under"] = (
    merged_data["P(under)"] * (merged_data["Lower Payout"] - 1) - 
    (1 - merged_data["P(under)"]) * 1  # Losing amount is the $1 bet
)

print("✅ EV calculations complete.")

# Function to calculate parlay EV
def calculate_parlay_ev(props):
    probabilities = [prop['P(over)'] if prop['Stat Type'] == 'Over' else prop['P(under)'] for prop in props]
    payouts = [prop['Higher Payout'] if prop['Stat Type'] == 'Over' else prop['Lower Payout'] for prop in props]
    
    # Check if any payout is invalid (NaN or None)
    if any(pd.isna(payout) or payout is None for payout in payouts):
        return None  # Return None if there's an invalid payout

    # Parlay hit probability: product of individual prop hit probabilities
    parlay_hit_prob = np.prod(probabilities)
    
    # Parlay payout: product of individual prop payouts (payouts are already numeric)
    parlay_payout = np.prod([payout - 1 for payout in payouts])
    
    # EV calculation for parlay
    parlay_ev = (parlay_hit_prob * parlay_payout) - (1 - parlay_hit_prob)
    return parlay_ev

# Generating parlays for 2 to 6 props
parlay_results = []
for n in range(2, 7):  # For parlays of 2, 3, 4, 5, 6 props
    for combination in itertools.combinations(merged_data.to_dict(orient='records'), n):
        parlay_ev = calculate_parlay_ev(combination)
        parlay_results.append({
            'Combination': combination,
            'Parlay EV': parlay_ev
        })

# Convert to DataFrame for output
parlay_results_df = pd.DataFrame(parlay_results)
print("📋 Parlay EV calculations complete.")

# Save parlay results to CSV
parlay_results_df.to_csv("parlay_ev_results.csv", index=False)
print("✅ Parlay EV results saved as 'parlay_ev_results.csv'.")

# Sample of parlay EV results
print(parlay_results_df.head())
