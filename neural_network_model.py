import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("🚀 Loading datasets...")

# Load datasets
player_data = pd.read_csv("nba_players_game_logs_2018_25.csv")
prop_data = pd.read_csv("underdog_player_props.csv")

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
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])
print("✅ Model training complete.")

# Predict probabilities
print("📊 Generating prop predictions...")
merged_data["P(over)"] = model.predict(scaler.transform(merged_data[features]))
merged_data["P(under)"] = 1 - merged_data["P(over)"]
print("✅ Predictions generated.")

# Compute expected values (EV)
print("📊 Calculating expected values (EV)...")
merged_data["EV_over"] = (
    merged_data["P(over)"] * (merged_data["Higher Payout"] - 1) -
    (1 - merged_data["P(over)"])
)
merged_data["EV_under"] = (
    merged_data["P(under)"] * (merged_data["Lower Payout"] - 1) -
    (1 - merged_data["P(under)"])
)
print("✅ EV calculations complete.")

# Select the best prop bet for each player
print("📊 Selecting best prop bets...")
merged_data["Best Bet"] = np.where(merged_data["EV_over"] > merged_data["EV_under"], "Over", "Under")
merged_data["Best EV"] = merged_data[["EV_over", "EV_under"]].max(axis=1)

# Sort by "Best EV" column in descending order
merged_data = merged_data.sort_values(by="Best EV", ascending=False)

# Remove duplicate player rows based on 'Player' and 'Best Bet'
print("🔄 Removing duplicate player rows...")
final_props = merged_data.drop_duplicates(subset=["Player", "Best Bet"])

# Select only relevant columns for the final CSV
final_props = final_props[["Player", "Stat Type", "Prop Line", "Best Bet", "Best EV"]]

# Save to CSV
print("💾 Saving relevant props to CSV...")
final_props.to_csv("relevant_props_sorted.csv", index=False)
print("✅ Relevant props saved as 'relevant_props_sorted.csv'.")

# Print first few rows to confirm
print("📋 Sample of saved props:")
print(final_props.head())
