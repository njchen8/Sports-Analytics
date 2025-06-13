#!/usr/bin/env python3
# data_check.py - Quick data exploration
# --------------------------------------------------------------

import pandas as pd
import numpy as np

print("🔍 CHECKING DATA AVAILABILITY")
print("="*40)

# Load data
df = pd.read_csv('nba_players_game_logs_2018_25_clean.csv', parse_dates=['GAME_DATE'])
print(f"📊 Dataset shape: {df.shape}")
print(f"📅 Available seasons: {sorted(df['SEASON_YEAR'].unique())}")

# Check for LeBron
print(f"\n🏀 SEARCHING FOR LEBRON JAMES")
print("-"*30)

lebron_candidates = df[df['PLAYER_NAME'].str.contains('LeBron', case=False, na=False)]
print(f"Found {len(lebron_candidates)} games with 'LeBron' in name")

if len(lebron_candidates) > 0:
    print("\nLeBron player info:")
    lebron_info = lebron_candidates[['PLAYER_ID', 'PLAYER_NAME', 'SEASON_YEAR']].drop_duplicates()
    print(lebron_info)
    
    # Check each season
    for season in sorted(lebron_candidates['SEASON_YEAR'].unique()):
        season_games = lebron_candidates[lebron_candidates['SEASON_YEAR'] == season]
        print(f"\n{season}: {len(season_games)} games")
        if len(season_games) > 0:
            print(f"  Date range: {season_games['GAME_DATE'].min()} to {season_games['GAME_DATE'].max()}")
            print(f"  Sample stats - PTS: {season_games['PTS'].mean():.1f}, REB: {season_games['REB'].mean():.1f}, AST: {season_games['AST'].mean():.1f}")

    # Check most recent season data
    latest_season = max(lebron_candidates['SEASON_YEAR'].unique())
    latest_data = lebron_candidates[lebron_candidates['SEASON_YEAR'] == latest_season]
    print(f"\n📈 LATEST SEASON ({latest_season}) ANALYSIS")
    print(f"Games available: {len(latest_data)}")
    
    if len(latest_data) >= 20:
        print("✅ Sufficient data for analysis!")
        
        # Show column availability
        important_cols = ['PTS', 'REB', 'AST', 'MATCHUP', 'rest_days', 'is_b2b']
        print(f"\n📋 IMPORTANT COLUMNS CHECK:")
        for col in important_cols:
            if col in latest_data.columns:
                non_null = latest_data[col].notna().sum()
                print(f"  ✅ {col}: {non_null}/{len(latest_data)} non-null values")
            else:
                print(f"  ❌ {col}: column missing")
        
        # Check rolling features
        rolling_cols = [col for col in latest_data.columns if 'roll' in col.lower()]
        print(f"\n📊 ROLLING FEATURES: {len(rolling_cols)} found")
        for col in rolling_cols[:5]:  # Show first 5
            non_null = latest_data[col].notna().sum()
            print(f"  {col}: {non_null}/{len(latest_data)} non-null")
            
    else:
        print(f"⚠️  Only {len(latest_data)} games available - need at least 20")

else:
    print("❌ No LeBron James data found!")
    print("\nSample of available players:")
    sample_players = df[['PLAYER_NAME', 'SEASON_YEAR']].drop_duplicates().head(10)
    print(sample_players)

print(f"\n🎯 ANALYSIS RECOMMENDATION")
print("-"*25)
if len(lebron_candidates) >= 20:
    print("✅ Proceed with LeBron James analysis")
    print(f"   Use PLAYER_ID: {lebron_candidates['PLAYER_ID'].iloc[0]}")
    print(f"   Latest season: {max(lebron_candidates['SEASON_YEAR'].unique())}")
else:
    print("⚠️  Consider using a different player or season")
