from nba_api.stats.endpoints import leaguedashteamstats
import pandas as pd
from datetime import datetime, timedelta
import time

def get_season_string(year):
    """Convert year to NBA season string format (e.g., 2018 -> '2018-19')"""
    return f"{year}-{str(year + 1)[-2:]}"

def get_month_date_range(year, month):
    """Get start and end dates for a given month and year"""
    if month == 12:  # December
        start_date = datetime(year, month, 1)
        end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        start_date = datetime(year if month >= 10 else year + 1, month, 1)
        if month == 2:  # February
            end_date = datetime(year if month >= 10 else year + 1, month + 1, 1) - timedelta(days=1)
        else:
            end_date = datetime(year if month >= 10 else year + 1, month + 1, 1) - timedelta(days=1)
    
    return start_date.strftime('%m/%d/%Y'), end_date.strftime('%m/%d/%Y')

# Initialize empty list to store all data
all_team_stats = []

# Define seasons and months
start_year = 2018
current_year = 2024
months = [10, 11, 12, 1, 2, 3, 4]  # NBA season months (Oct-Apr)
month_names = {10: 'October', 11: 'November', 12: 'December', 1: 'January', 2: 'February', 3: 'March', 4: 'April'}

# Mapping team names to abbreviations
team_name_to_abbreviation = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN", "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE", "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET", "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "LA Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM", "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS"
}

print("Fetching NBA team stats from 2018-current by month...")

# Loop through each season
for year in range(start_year, current_year + 1):
    season_str = get_season_string(year)
    print(f"Processing season: {season_str}")
    
    # Loop through each month in the season
    for month in months:
        try:
            date_from, date_to = get_month_date_range(year, month)
            month_name = month_names[month]
            
            print(f"  Fetching {month_name} {year if month >= 10 else year + 1} data...")
            
            # Fetch team stats for the specific month
            team_stats = leaguedashteamstats.LeagueDashTeamStats(
                season=season_str,
                season_type_all_star='Regular Season',
                per_mode_detailed='PerGame',
                measure_type_detailed_defense='Advanced',
                date_from_nullable=date_from,
                date_to_nullable=date_to
            )
            
            # Convert to DataFrame
            team_stats_df = team_stats.get_data_frames()[0]
            
            # Skip if no data for this month
            if team_stats_df.empty:
                print(f"    No data available for {month_name}")
                continue
            
            # Add season, month, and year columns
            team_stats_df['SEASON'] = season_str
            team_stats_df['MONTH'] = month_name
            team_stats_df['YEAR'] = year if month >= 10 else year + 1
            team_stats_df['DATE_FROM'] = date_from
            team_stats_df['DATE_TO'] = date_to
            
            # Add team abbreviation column
            team_stats_df['TEAM_ABBREVIATION'] = team_stats_df['TEAM_NAME'].map(team_name_to_abbreviation)
            
            # Select relevant columns
            columns_to_keep = ['SEASON', 'MONTH', 'YEAR', 'DATE_FROM', 'DATE_TO', 'TEAM_ABBREVIATION', 'TEAM_NAME', 
                             'GP', 'W', 'L', 'W_PCT', 'OFF_RATING', 'DEF_RATING', 'NET_RATING', 'PACE', 'AST_RATIO', 'REB_PCT']
            
            # Only keep columns that exist in the dataframe
            available_columns = [col for col in columns_to_keep if col in team_stats_df.columns]
            team_stats_df = team_stats_df[available_columns]
            
            # Append to master list
            all_team_stats.append(team_stats_df)
            
            print(f"    Successfully fetched data for {len(team_stats_df)} teams")
            
            # Add delay to avoid rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            print(f"    Error fetching data for {month_name}: {str(e)}")
            continue

# Combine all data
if all_team_stats:
    final_df = pd.concat(all_team_stats, ignore_index=True)
    
    # Sort by season, month, and team name
    month_order = {'October': 1, 'November': 2, 'December': 3, 'January': 4, 'February': 5, 'March': 6, 'April': 7}
    final_df['MONTH_ORDER'] = final_df['MONTH'].map(month_order)
    final_df = final_df.sort_values(['SEASON', 'MONTH_ORDER', 'TEAM_NAME']).drop('MONTH_ORDER', axis=1)
    
    # Save to CSV
    output_filename = 'nba_team_stats_monthly_2018_current.csv'
    final_df.to_csv(output_filename, index=False)
    
    print(f"\nData collection complete!")
    print(f"Total records: {len(final_df)}")
    print(f"Data saved to: {output_filename}")
    print(f"Columns: {list(final_df.columns)}")
    
    # Display sample data
    print("\nSample data:")
    print(final_df.head())
else:
    print("No data was collected.")
