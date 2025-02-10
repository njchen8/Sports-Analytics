import requests
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ NBA API Endpoint for Player Game Logs
NBA_GAME_LOGS_URL = "https://stats.nba.com/stats/playergamelogs"

# ✅ Request Headers (NBA.com requires these)
HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nba.com/stats/players/traditional",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true"
}

# ✅ User Inputs
PLAYER_ID = "201939"  # Example: Stephen Curry
PLAYER_NAME = "Stephen Curry"
OPPONENT_TEAM = "vs LAL"  # Example: Los Angeles Lakers
PLAYER_PROP = 27.5  # Betting line input

# ✅ Query Parameters for Game Logs
PARAMS = {
    "PlayerID": PLAYER_ID,
    "LeagueID": "00",
    "Season": "2023-24",  # Change this for the latest season
    "SeasonType": "Regular Season"  # Options: 'Regular Season', 'Playoffs'
}

# ✅ Fetch Data from API
response = requests.get(NBA_GAME_LOGS_URL, headers=HEADERS, params=PARAMS)

# ✅ Check if API request was successful
if response.status_code == 200:
    data = response.json()

    # ✅ Extract Headers (Column Names)
    headers = data["resultSets"][0]["headers"]

    # ✅ Extract Player Game Logs
    rows = data["resultSets"][0]["rowSet"]

    # ✅ Convert to DataFrame
    df = pd.DataFrame(rows, columns=headers)

    # ✅ Extract Relevant Columns
    df = df[["MATCHUP", "PTS", "GAME_DATE"]]
    df["PTS"] = df["PTS"].astype(float)

    # ✅ Weighting Strategy: More Weight for Recent Games & Opponent-Specific Games
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE", ascending=False)  # Sort by most recent games

    # ✅ Monte Carlo Simulation: Finding the Best Weighting
    best_weighting = None
    best_variance_reduction = np.inf

    # Define different decay factors and opponent bonuses to test
    decay_factors = np.linspace(0.01, 0.2, 10)
    opponent_bonuses = np.linspace(1.0, 3.0, 10)

    results = []

    for decay in decay_factors:
        for bonus in opponent_bonuses:
            df["Weight"] = np.exp(-decay * (pd.to_datetime("today") - df["GAME_DATE"]).dt.days)
            df["Weight"] *= np.where(df["MATCHUP"] == OPPONENT_TEAM, bonus, 1.0)  # Apply opponent bonus
            df["Weight"] /= df["Weight"].sum()  # Normalize weights

            # Compute Weighted Mean & Variance
            weighted_mean = np.sum(df["PTS"] * df["Weight"])
            weighted_variance = np.sum(df["Weight"] * (df["PTS"] - weighted_mean) ** 2)

            results.append({"Decay": decay, "Bonus": bonus, "Variance": weighted_variance, "Mean": weighted_mean})

            if weighted_variance < best_variance_reduction:
                best_variance_reduction = weighted_variance
                best_weighting = (decay, bonus)

    # ✅ Best Weighting Found
    best_decay, best_bonus = best_weighting
    df["Optimized_Weight"] = np.exp(-best_decay * (pd.to_datetime("today") - df["GAME_DATE"]).dt.days)
    df["Optimized_Weight"] *= np.where(df["MATCHUP"] == OPPONENT_TEAM, best_bonus, 1.0)
    df["Optimized_Weight"] /= df["Optimized_Weight"].sum()

    # ✅ Monte Carlo Simulation using Optimized Weights
    weighted_simulated_games = np.random.normal(
        np.sum(df["PTS"] * df["Optimized_Weight"]),
        np.sqrt(np.sum(df["Optimized_Weight"] * (df["PTS"] - np.sum(df["PTS"] * df["Optimized_Weight"])) ** 2)),
        10000
    )

    # ✅ Monte Carlo Simulation without Weights (Unweighted: Simply using historical mean and std dev)
    unweighted_simulated_games = np.random.normal(df["PTS"].mean(), df["PTS"].std(), 10000)

    # ✅ Probability of Exceeding Player Prop
    prob_over_weighted = np.mean(weighted_simulated_games > PLAYER_PROP)
    prob_over_unweighted = np.mean(unweighted_simulated_games > PLAYER_PROP)

    expected_value_weighted = (prob_over_weighted * 2) - 1  # Assuming Even Odds (+100)
    expected_value_unweighted = (prob_over_unweighted * 2) - 1  # Assuming Even Odds (+100)

    # ✅ Print Results
    print(f"✅ Optimal Decay: {best_decay:.4f}, Opponent Bonus: {best_bonus:.2f}")
    print(f"📈 Weighted Probability Over {PLAYER_PROP} Points: {prob_over_weighted:.2%}")
    print(f"💰 Weighted Expected Value (EV): {expected_value_weighted:.2f}")
    print(f"📈 Unweighted Probability Over {PLAYER_PROP} Points: {prob_over_unweighted:.2%}")
    print(f"💰 Unweighted Expected Value (EV): {expected_value_unweighted:.2f}")

    # ✅ Plot Distribution of Weighted vs. Unweighted Simulations
    plt.figure(figsize=(10, 6))

    sns.histplot(unweighted_simulated_games, bins=30, kde=True, label="Unweighted Simulation", color="red", stat="density", alpha=0.6)
    sns.histplot(weighted_simulated_games, bins=30, kde=True, label="Weighted Simulation", color="blue", stat="density", alpha=0.6)

    plt.axvline(PLAYER_PROP, color="black", linestyle="dashed", label=f"Prop Line: {PLAYER_PROP}")
    plt.axvline(np.mean(weighted_simulated_games), color="blue", linestyle="dashed", label=f"Weighted Mean: {np.mean(weighted_simulated_games):.2f}")
    plt.axvline(np.mean(unweighted_simulated_games), color="red", linestyle="dashed", label=f"Unweighted Mean: {np.mean(unweighted_simulated_games):.2f}")

    plt.title(f"Weighted vs. Unweighted Player Points Distribution ({OPPONENT_TEAM})")
    plt.xlabel("Points Scored")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

else:
    print(f"❌ Failed to fetch data. HTTP Status Code: {response.status_code}")
