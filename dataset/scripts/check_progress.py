"""
Test to verify failed track counting works correctly
"""
import os
import pandas as pd

# Paths
OUTPUT_FILE = '../songs_enhanced_full.csv'
FAILED_FILE = '../failed_tracks.csv'

def count_tracks():
    successful_count = 0
    failed_count = 0
    
    # Count successful
    if os.path.exists(OUTPUT_FILE):
        df = pd.read_csv(OUTPUT_FILE)
        successful_count = len(df)
        print(f"✓ Successful tracks: {successful_count}")
    else:
        print("⚠ No successful tracks file found")
    
    # Count failed
    if os.path.exists(FAILED_FILE):
        df = pd.read_csv(FAILED_FILE)
        failed_count = len(df)
        print(f"✓ Failed tracks: {failed_count}")
    else:
        print("⚠ No failed tracks file found")
    
    total = successful_count + failed_count
    print(f"\n📊 TOTAL PROCESSED: {total}")
    print(f"   - Success: {successful_count} ({100*successful_count/total:.1f}%)")
    print(f"   - Failed: {failed_count} ({100*failed_count/total:.1f}%)")
    
    return total

print("=" * 60)
print("Current Progress Check")
print("=" * 60)
total = count_tracks()

print("\n" + "=" * 60)
print(f"Next scrape will start from track #{total + 1}")
print("=" * 60)
