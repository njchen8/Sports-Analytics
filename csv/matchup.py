#!/usr/bin/env python3
# -------------------------------------------
# clean_nba_logs.py
# -------------------------------------------
import pandas as pd
from pathlib import Path

# --- 1. CONFIG -----------------------------------------------------------
DATA_PATH   = Path("nba_players_game_logs.csv")
OUT_PATH    = DATA_PATH.with_name("nba_players_game_logs_2018_25_clean.csv")

# Columns to remove (keep only those that actually exist in the file)
DROP_COLS = [
    "DD2", "TD3", "WNBA_FANTAST_PTS", "WNBA_FANTASY_PTS",                  # both spellings just in case
    "GP_RANK","W_RANK","L_RANK","W_PCT_RANK","MIN_RANK",
    "FGM_RANK","FGA_RANK","FG_PCT_RANK","FG3M_RANK","FG3A_RANK","FG3_PCT_RANK",
    "FTM_RANK","FTA_RANK","FT_PCT_RANK","OREB_RANK","DREB_RANK","REB_RANK",
    "AST_RANK","TOV_RANK","STL_RANK","BLK_RANK","BLKA_RANK","PF_RANK","PFD_RANK",
    "PTS_RANK","PLUS_MINUS_RANK","NBA_FANTASY_PTS_RANK",
    "DD2_RANK","TD3_RANK","WNBA_FANTASY_PTS_RANK","AVAILABLE_FLAG"
]

ROLLS = [5, 10]  # Rolling window sizes
TARGETS = ["PTS", "REB", "AST"]  # Stats to calculate rolling features for

# --- 2. HELPERS ----------------------------------------------------------
def matchup_to_home_away(s: str) -> str:
    """Return 'Home' if the player's team is listed before 'vs'/'vs.', else 'Away'."""
    s_lower = s.lower()
    return "Home" if "vs" in s_lower and "@" not in s_lower else "Away"

def matchup_to_opp_team(s: str) -> str:
    """Extract the three-letter opponent code (always the last token)."""
    return s.replace(".", "").split()[-1].upper()

def calculate_rolling_features(df, targets, rolls):
    """Calculate rolling features for the given dataframe."""
    for stat in targets:
        for w in rolls:
            df[f"{stat.lower()}_roll_{w}"] = (
                df.groupby("PLAYER_ID")[stat]
                  .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
            )
    return df

# --- 3. LOAD & TRANSFORM -------------------------------------------------
df = pd.read_csv(DATA_PATH)

# Add home/away and opponent columns
df["HOME_AWAY"] = df["MATCHUP"].apply(matchup_to_home_away)
df["OPP_TEAM"]  = df["MATCHUP"].apply(matchup_to_opp_team)

# Calculate rolling features
df = calculate_rolling_features(df, TARGETS, ROLLS)

# Add rest_days and is_b2b columns
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])  # Ensure GAME_DATE is datetime
df["rest_days"] = (
    df.groupby("PLAYER_ID")["GAME_DATE"]
      .transform(lambda s: (s - s.shift(1)).dt.days.clip(lower=0, upper=7))
)
df["is_b2b"] = (df["rest_days"] == 0).astype(int)

# Drop only the columns that are present to avoid KeyErrors
cols_to_drop = [c for c in DROP_COLS if c in df.columns]
df = df.drop(columns=cols_to_drop)

# --- 4. SAVE -------------------------------------------------------------
df.to_csv(OUT_PATH, index=False)
print(f"Cleaned file with rolling features and is_b2b column written to: {OUT_PATH}")
