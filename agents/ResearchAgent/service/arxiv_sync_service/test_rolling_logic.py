#!/usr/bin/env python3
"""
Test script for rolling 7-day logic

This script tests the new rolling 7-day logic to ensure it works correctly.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from agents.ResearchAgent.service.arxiv_sync_service.arxiv_sync_service import ArxivSyncService


def test_rolling_week_dates():
    """Test the rolling week dates logic."""
    print("=== Testing Rolling 7-Day Logic ===")
    
    # Create a service instance to test the method
    service = ArxivSyncService()
    
    # Test the rolling week dates
    rolling_dates = service._get_rolling_week_dates()
    
    print(f"Today: {datetime.now().strftime('%Y-%m-%d (%A)')}")
    print(f"Rolling 7-day dates:")
    
    for i, date in enumerate(rolling_dates):
        day_name = date.strftime('%A')
        date_str = date.strftime('%Y-%m-%d')
        print(f"  {i+1}. {date_str} ({day_name})")
    
    # Verify the logic
    today = datetime.now()
    expected_dates = []
    for i in range(7):
        expected_date = today - timedelta(days=i)
        expected_dates.append(expected_date)
    
    # Check if the dates match
    dates_match = True
    for i, (actual, expected) in enumerate(zip(rolling_dates, expected_dates)):
        if actual.date() != expected.date():
            dates_match = False
            print(f"❌ Date mismatch at position {i}: {actual.date()} != {expected.date()}")
    
    if dates_match:
        print("✅ Rolling 7-day logic is working correctly!")
    else:
        print("❌ Rolling 7-day logic has issues!")
    
    # Show the date range
    oldest_date = min(rolling_dates)
    newest_date = max(rolling_dates)
    print(f"\nDate range: {oldest_date.strftime('%Y-%m-%d')} to {newest_date.strftime('%Y-%m-%d')}")
    print(f"Total days: {len(rolling_dates)}")
    
    return dates_match


def main():
    """Main test function."""
    success = test_rolling_week_dates()
    
    if success:
        print("\n🎉 Rolling 7-day logic test passed!")
        sys.exit(0)
    else:
        print("\n💥 Rolling 7-day logic test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
