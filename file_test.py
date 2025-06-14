#!/usr/bin/env python3
# file_test.py - Test with file output

import sys

# Redirect output to file
with open('test_results.txt', 'w') as f:
    sys.stdout = f
    sys.stderr = f
    
    print("Starting file-based test...")
    
    try:
        import dynamic_team_stats as dts
        import pandas as pd
        
        print("Module imported successfully")
        
        # Test season detection
        test_date = pd.to_datetime('2024-01-15')
        season = dts.get_season_from_date(test_date)
        print(f"Season: {season}")
        
        # Test default stats
        default_stats = dts.create_default_team_stats()
        print(f"Default stats: {len(default_stats)} teams")
        
        # Test getting stats for date
        team_stats = dts.get_team_stats_for_date(test_date)
        print(f"Team stats for date: {len(team_stats)} teams")
        
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

# Reset stdout
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

print("Test completed - check test_results.txt")
