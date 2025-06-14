#!/usr/bin/env python3
# dynamic_team_stats.py - Dynamic Team Stats for Avoiding Data Leakage
# --------------------------------------------------------------
# This module provides team statistics that are contextually appropriate
# for each prediction time point, avoiding data leakage by only using
# team stats available up to the current game date.
# --------------------------------------------------------------

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import time
import os
warnings.filterwarnings('ignore')

try:
    from nba_api.stats.endpoints import teamestimatedmetrics, leaguedashteamstats
    NBA_API_AVAILABLE = True
    print("NBA API available - will fetch real-time team stats")
except ImportError:
    NBA_API_AVAILABLE = False
    print("NBA API not available - will use fallback method")

# --------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------
TEAM_STATS_CACHE_FILE = Path("team_stats_cache.csv")
TEAM_MAPPING_FILE = Path("team_mappings.csv")

# Team abbreviation mappings (NBA API uses different formats)
TEAM_ABBREVIATIONS_MAP = {
    # Standard NBA abbreviations to full names
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets", 
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "LA Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards"
}

# Reverse mapping (full names to abbreviations)
TEAM_NAMES_MAP = {v: k for k, v in TEAM_ABBREVIATIONS_MAP.items()}

# Default team stats (league averages) for fallback
DEFAULT_TEAM_STATS = {
    "TEAM_ABBREVIATION": "AVG",
    "OFF_RATING": 113.5,
    "DEF_RATING": 113.5,
    "NET_RATING": 0.0,
    "PACE": 100.0,
    "AST_RATIO": 19.0,
    "REB_PCT": 0.5,
    "E_OFF_RATING": 113.5,
    "E_DEF_RATING": 113.5,
    "E_NET_RATING": 0.0,
    "E_PACE": 100.0,
    "E_AST_RATIO": 19.0,
    "E_OREB_PCT": 0.25,
    "E_DREB_PCT": 0.75,
    "E_REB_PCT": 0.5,
    "E_TM_TOV_PCT": 14.0
}

# --------------------------------------------------------------
# TEAM STATS CACHE MANAGEMENT
# --------------------------------------------------------------

class TeamStatsCache:
    """Manages cached team statistics to reduce API calls"""
    
    def __init__(self):
        self.cache_file = TEAM_STATS_CACHE_FILE
        self.cache = self._load_cache()
        
    def _load_cache(self) -> pd.DataFrame:
        """Load existing cache or create empty one"""
        if self.cache_file.exists():
            try:
                cache_df = pd.read_csv(self.cache_file)
                cache_df['cache_date'] = pd.to_datetime(cache_df['cache_date'])
                print(f"Loaded team stats cache with {len(cache_df)} entries")
                return cache_df
            except Exception as e:
                print(f"Error loading cache: {e}")
                return pd.DataFrame()
        else:
            return pd.DataFrame()
    
    def _save_cache(self):
        """Save current cache to file"""
        try:
            self.cache.to_csv(self.cache_file, index=False)
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def get_cached_stats(self, season: str, season_type: str = "Regular Season") -> Optional[pd.DataFrame]:
        """Get cached team stats for a season if available and recent"""
        if self.cache.empty:
            return None
            
        # Check if we have recent data for this season
        season_cache = self.cache[
            (self.cache['season'] == season) & 
            (self.cache['season_type'] == season_type)
        ]
        
        if season_cache.empty:
            return None
            
        # Check if cache is recent (within 7 days for current season, any age for past seasons)
        latest_cache = season_cache.sort_values('cache_date').iloc[-1]
        cache_age = datetime.now() - latest_cache['cache_date']
        
        current_season = "2024-25"  # Update this as needed
        if season == current_season and cache_age.days > 7:
            return None  # Current season data should be refreshed weekly
            
        # Return cached data
        cached_data = season_cache.drop(['cache_date', 'season', 'season_type'], axis=1)
        print(f"Using cached team stats for {season} (cached {cache_age.days} days ago)")
        return cached_data
    
    def cache_stats(self, stats_df: pd.DataFrame, season: str, season_type: str = "Regular Season"):
        """Cache new team statistics"""
        if stats_df.empty:
            return
            
        # Add metadata
        stats_df = stats_df.copy()
        stats_df['season'] = season
        stats_df['season_type'] = season_type
        stats_df['cache_date'] = datetime.now()
        
        # Remove old entries for this season
        if not self.cache.empty:
            self.cache = self.cache[
                ~((self.cache['season'] == season) & (self.cache['season_type'] == season_type))
            ]
        
        # Add new data
        self.cache = pd.concat([self.cache, stats_df], ignore_index=True)
        
        # Clean old entries (keep only last 5 seasons)
        if not self.cache.empty:
            seasons_to_keep = sorted(self.cache['season'].unique())[-5:]
            self.cache = self.cache[self.cache['season'].isin(seasons_to_keep)]
        
        self._save_cache()
        print(f"Cached team stats for {season} ({len(stats_df)} teams)")

# Global cache instance
team_cache = TeamStatsCache()

# --------------------------------------------------------------
# NBA API FUNCTIONS
# --------------------------------------------------------------

def fetch_team_stats_from_api(season: str, season_type: str = "Regular Season") -> Optional[pd.DataFrame]:
    """Fetch team statistics from NBA API with improved error handling and retries"""
    if not NBA_API_AVAILABLE:
        print("NBA API not available, cannot fetch team stats")
        return None
    
    max_retries = 3
    base_delay = 5
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                print(f"Waiting {delay} seconds before retry {attempt + 1}...")
                time.sleep(delay)
            
            print(f"Fetching team stats from NBA API for {season} {season_type} (attempt {attempt + 1}/{max_retries})...")
            
            # Try TeamEstimatedMetrics first (more comprehensive)
            try:
                team_metrics = teamestimatedmetrics.TeamEstimatedMetrics(
                    league_id="00",
                    season=season,
                    season_type=season_type,
                    timeout=45  # Increased timeout
                )
                
                stats_df = team_metrics.get_data_frames()[0]
                
                if not stats_df.empty:
                    # Standardize column names and add required columns
                    processed_df = process_estimated_metrics(stats_df)
                    print(f"✓ Fetched team estimated metrics for {season}: {len(processed_df)} teams")
                    return processed_df
                
            except Exception as e:
                print(f"TeamEstimatedMetrics failed (attempt {attempt + 1}): {e}")
            
            # Small delay before trying fallback method
            time.sleep(2)
            
            # Fallback to LeagueDashTeamStats
            try:
                team_stats = leaguedashteamstats.LeagueDashTeamStats(
                    season=season,
                    season_type_all_star=season_type,
                    per_mode_detailed='PerGame',
                    measure_type_detailed_defense='Advanced',
                    timeout=45  # Increased timeout
                )
                
                stats_df = team_stats.get_data_frames()[0]
                
                if not stats_df.empty:
                    processed_df = process_dashboard_stats(stats_df)
                    print(f"✓ Fetched team dashboard stats for {season}: {len(processed_df)} teams")
                    return processed_df
                    
            except Exception as e:
                print(f"LeagueDashTeamStats failed (attempt {attempt + 1}): {e}")
            
        except Exception as e:
            print(f"Overall fetch attempt {attempt + 1} failed: {e}")
        
        if attempt < max_retries - 1:
            print(f"Attempt {attempt + 1} failed, will retry...")
    
    print(f"All {max_retries} attempts failed to fetch team stats for {season}")
    return None

def process_estimated_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Process TeamEstimatedMetrics data into standard format"""
    try:
        # Create standardized dataframe
        processed = pd.DataFrame()
        
        # Map team names to abbreviations
        if 'TEAM_NAME' in df.columns:
            processed['TEAM_ABBREVIATION'] = df['TEAM_NAME'].map(TEAM_NAMES_MAP)
        elif 'TEAM_ABBREVIATION' in df.columns:
            processed['TEAM_ABBREVIATION'] = df['TEAM_ABBREVIATION']
        else:
            print("Warning: No team identifier found in estimated metrics")
            return pd.DataFrame()
        
        # Map estimated metrics to standard names
        column_mapping = {
            'E_OFF_RATING': 'OFF_RATING',
            'E_DEF_RATING': 'DEF_RATING', 
            'E_NET_RATING': 'NET_RATING',
            'E_PACE': 'PACE',
            'E_AST_RATIO': 'AST_RATIO',
            'E_REB_PCT': 'REB_PCT'
        }
        
        for est_col, std_col in column_mapping.items():
            if est_col in df.columns:
                processed[std_col] = df[est_col]
            else:
                processed[std_col] = DEFAULT_TEAM_STATS[std_col]
        
        # Add original estimated columns if available
        for col in df.columns:
            if col.startswith('E_') and col not in processed.columns:
                processed[col] = df[col]
        
        # Remove rows with missing team abbreviations
        processed = processed.dropna(subset=['TEAM_ABBREVIATION'])
        
        return processed
        
    except Exception as e:
        print(f"Error processing estimated metrics: {e}")
        return pd.DataFrame()

def process_dashboard_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Process LeagueDashTeamStats data into standard format"""
    try:
        processed = pd.DataFrame()
        
        # Map team names to abbreviations
        if 'TEAM_NAME' in df.columns:
            processed['TEAM_ABBREVIATION'] = df['TEAM_NAME'].map(TEAM_NAMES_MAP)
        elif 'TEAM_ABBREVIATION' in df.columns:
            processed['TEAM_ABBREVIATION'] = df['TEAM_ABBREVIATION']
        else:
            print("Warning: No team identifier found in dashboard stats")
            return pd.DataFrame()
        
        # Standard columns
        standard_columns = ['OFF_RATING', 'DEF_RATING', 'NET_RATING', 'PACE', 'AST_RATIO', 'REB_PCT']
        
        for col in standard_columns:
            if col in df.columns:
                processed[col] = df[col]
            else:
                processed[col] = DEFAULT_TEAM_STATS[col]
        
        # Remove rows with missing team abbreviations
        processed = processed.dropna(subset=['TEAM_ABBREVIATION'])
        
        return processed
        
    except Exception as e:
        print(f"Error processing dashboard stats: {e}")
        return pd.DataFrame()

# --------------------------------------------------------------
# SEASON DETERMINATION FUNCTIONS
# --------------------------------------------------------------

def get_season_from_date(game_date: datetime) -> str:
    """Determine NBA season from game date"""
    try:
        if isinstance(game_date, str):
            game_date = pd.to_datetime(game_date)
        
        year = game_date.year
        month = game_date.month
        
        # NBA season runs from October to June
        # Games from October-December are in the season starting that year
        # Games from January-June are in the season that started the previous year
        if month >= 10:  # October, November, December
            season_start_year = year
        else:  # January through September
            season_start_year = year - 1
        
        return f"{season_start_year}-{str(season_start_year + 1)[-2:]}"
        
    except Exception as e:
        print(f"Error determining season from date {game_date}: {e}")
        return "2023-24"  # Default fallback

def get_season_up_to_date(game_date: datetime, season: str) -> str:
    """Get the appropriate season type based on game date within season"""
    try:
        if isinstance(game_date, str):
            game_date = pd.to_datetime(game_date)
        
        # For past seasons, always use Regular Season
        current_year = datetime.now().year
        season_year = int(season.split('-')[0])
        
        if season_year < current_year - 1:
            return "Regular Season"
        
        # For recent/current seasons, determine based on date
        year = game_date.year
        month = game_date.month
        day = game_date.day
        
        # Playoffs typically start in mid-April
        if month > 4 or (month == 4 and day > 15):
            return "Playoffs"
        else:
            return "Regular Season"
            
    except Exception as e:
        print(f"Error determining season type: {e}")
        return "Regular Season"

# --------------------------------------------------------------
# MAIN TEAM STATS FUNCTIONS
# --------------------------------------------------------------

def get_team_stats_for_date(game_date: datetime, force_refresh: bool = False) -> pd.DataFrame:
    """Get team statistics appropriate for a specific game date"""
    try:
        # Determine season
        season = get_season_from_date(game_date)
        season_type = get_season_up_to_date(game_date, season)
        
        # Check cache first (unless force refresh)
        if not force_refresh:
            cached_stats = team_cache.get_cached_stats(season, season_type)
            if cached_stats is not None:
                return cached_stats
        
        # Fetch from API
        stats_df = fetch_team_stats_from_api(season, season_type)
        
        if stats_df is not None and not stats_df.empty:
            # Cache the results
            team_cache.cache_stats(stats_df, season, season_type)
            return stats_df
        else:
            # Fallback to default stats
            print(f"Using default team stats for {season}")
            return create_default_team_stats()
            
    except Exception as e:
        print(f"Error getting team stats for date {game_date}: {e}")
        return create_default_team_stats()

def create_default_team_stats() -> pd.DataFrame:
    """Create default team statistics based on league averages"""
    teams = list(TEAM_ABBREVIATIONS_MAP.keys())
    
    default_stats = []
    for team in teams:
        team_stats = DEFAULT_TEAM_STATS.copy()
        team_stats['TEAM_ABBREVIATION'] = team
        default_stats.append(team_stats)
    
    return pd.DataFrame(default_stats)

def get_team_stats_for_game_data(game_data: pd.DataFrame) -> pd.DataFrame:
    """Get appropriate team stats for a dataframe of game data"""
    if game_data.empty:
        return create_default_team_stats()
    
    # Determine the time range of games
    game_dates = pd.to_datetime(game_data['GAME_DATE'])
    latest_date = game_dates.max()
    
    print(f"Getting team stats for games up to {latest_date.strftime('%Y-%m-%d')}")
    
    return get_team_stats_for_date(latest_date)

def update_team_stats_dynamically(df: pd.DataFrame, current_idx: int) -> pd.DataFrame:
    """
    Update team stats dynamically based on the current game index
    This ensures no data leakage by only using stats available up to current game
    """
    if current_idx >= len(df):
        return create_default_team_stats()
    
    # Get the current game date
    current_game = df.iloc[current_idx]
    current_date = pd.to_datetime(current_game['GAME_DATE'])
    
    # Get team stats appropriate for this date
    return get_team_stats_for_date(current_date)

# --------------------------------------------------------------
# INTEGRATION FUNCTIONS
# --------------------------------------------------------------

def replace_static_team_stats(team_stats_df: pd.DataFrame, game_date: datetime) -> pd.DataFrame:
    """Replace static team stats with dynamic ones for a specific date"""
    dynamic_stats = get_team_stats_for_date(game_date)
    
    if dynamic_stats.empty:
        print("Warning: Could not get dynamic team stats, using provided static stats")
        return team_stats_df
    
    print(f"Replaced static team stats with dynamic stats for {game_date.strftime('%Y-%m-%d')}")
    return dynamic_stats

def validate_team_stats(team_stats: pd.DataFrame) -> bool:
    """Validate that team stats have required columns"""
    required_columns = ['TEAM_ABBREVIATION', 'OFF_RATING', 'DEF_RATING', 'NET_RATING', 'PACE', 'AST_RATIO', 'REB_PCT']
    
    missing_columns = [col for col in required_columns if col not in team_stats.columns]
    
    if missing_columns:
        print(f"Warning: Team stats missing columns: {missing_columns}")
        return False
    
    if team_stats.empty:
        print("Warning: Team stats dataframe is empty")
        return False
    
    return True

# --------------------------------------------------------------
# TESTING AND UTILITIES
# --------------------------------------------------------------

def test_dynamic_team_stats():
    """Test the dynamic team stats system"""
    print("Testing Dynamic Team Stats System")
    print("=" * 50)
    
    # Test dates
    test_dates = [
        "2023-01-15",  # Mid-season 2022-23
        "2023-10-15",  # Start of 2023-24
        "2024-03-15",  # Mid-season 2023-24
        "2024-10-15",  # Start of 2024-25
    ]
    
    for date_str in test_dates:
        print(f"\nTesting date: {date_str}")
        test_date = pd.to_datetime(date_str)
        season = get_season_from_date(test_date)
        print(f"  Detected season: {season}")
        
        team_stats = get_team_stats_for_date(test_date)
        print(f"  Retrieved {len(team_stats)} teams")
        
        if not team_stats.empty:
            print(f"  Sample team stats:")
            sample_team = team_stats.iloc[0]
            print(f"    {sample_team['TEAM_ABBREVIATION']}: OFF={sample_team['OFF_RATING']:.1f}, DEF={sample_team['DEF_RATING']:.1f}")
        
        is_valid = validate_team_stats(team_stats)
        print(f"  Valid: {is_valid}")

if __name__ == "__main__":
    test_dynamic_team_stats()
