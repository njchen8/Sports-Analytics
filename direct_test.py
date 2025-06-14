#!/usr/bin/env python3
# direct_test.py - Direct test of dynamic team stats

import sys
import os

# Add current directory to path
sys.path.insert(0, os.getcwd())

print("Starting direct test of dynamic team stats...")

try:
    # Test basic season detection
    import pandas as pd
    from datetime import datetime
    
    # Import the module
    import dynamic_team_stats as dts
    
    print("Module imported successfully")
    
    # Test season detection
    test_date = pd.to_datetime('2024-01-15')
    season = dts.get_season_from_date(test_date)
    print(f"Season for 2024-01-15: {season}")
    
    # Test default team stats creation
    default_stats = dts.create_default_team_stats()
    print(f"Default stats created: {len(default_stats)} teams")
    print(f"Columns: {list(default_stats.columns)}")
    
    # Show sample team
    if not default_stats.empty:
        sample = default_stats.iloc[0]
        print(f"Sample team: {sample['TEAM_ABBREVIATION']} - OFF: {sample['OFF_RATING']}, DEF: {sample['DEF_RATING']}")
    
    # Test team stats for a specific date (will use defaults since API might not work)
    print(f"\nTesting team stats for date: {test_date}")
    team_stats = dts.get_team_stats_for_date(test_date)
    print(f"Retrieved {len(team_stats)} teams")
    
    if not team_stats.empty:
        print("Sample team from date-specific stats:")
        sample = team_stats.iloc[0]
        print(f"{sample['TEAM_ABBREVIATION']} - OFF: {sample['OFF_RATING']:.1f}, DEF: {sample['DEF_RATING']:.1f}")
    
    # Test validation
    is_valid = dts.validate_team_stats(team_stats)
    print(f"Stats validation result: {is_valid}")
    
    print("\nAll tests completed successfully!")
    
except Exception as e:
    print(f"Error during testing: {e}")
    import traceback
    traceback.print_exc()
