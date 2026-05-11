#!/usr/bin/env python3
"""
Diagnostic Script - Analyze Download Log Failures

Analyzes download_log_pilot.csv to categorize and count failure reasons.
Helps identify:
- Most common error types
- Success rate breakdown
- Confidence score vs success correlation
- Error patterns for retry strategy

Usage:
    python diagnose_failures.py
    python diagnose_failures.py --row-range 0 47899  # Analyze specific range
"""
import csv
import sys
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
LOG_FILE = PROJECT_ROOT / "data" / "logs" / "download_log_pilot.csv"
AUDIO_DIR = PROJECT_ROOT / "data" / "audio" / "pilot"


def parse_bool(value: str) -> bool:
    """Parse boolean string from CSV."""
    return value.strip().lower() == 'true'


def get_downloaded_audio_rows() -> Tuple[set, set]:
    """Return complete and partial audio row IDs from data/audio/pilot."""
    complete_rows = set()
    partial_rows = set()

    if not AUDIO_DIR.exists():
        return complete_rows, partial_rows

    row_id_pattern = re.compile(r'^(\d+)_')

    for file_path in AUDIO_DIR.iterdir():
        if not file_path.is_file():
            continue

        match = row_id_pattern.match(file_path.name)
        if not match:
            continue

        row_idx = int(match.group(1))
        if file_path.name.endswith('.part'):
            partial_rows.add(row_idx)
            continue

        complete_rows.add(row_idx)

    return complete_rows, partial_rows


def analyze_log(start_row: int = 0, end_row: int = None) -> Dict:
    """Analyze download log and categorize failures."""
    
    if not LOG_FILE.exists():
        print(f"ERROR: Log file not found: {LOG_FILE}")
        sys.exit(1)
    
    # Statistics containers
    stats = {
        'total': 0,
        'successful': 0,
        'failed': 0,
        'skipped_low_confidence': 0,
        'attempted': 0,  # Actually tried to download
        'download_failed': 0,  # Download attempted but failed
        'missing_on_disk': 0,
        'not_downloaded_for_some_reason_error': 0,
        'complete_files_detected': 0,
        'partial_files_ignored': 0,
    }
    
    # Error categorization
    error_categories = Counter()
    confidence_distribution = defaultdict(int)
    confidence_vs_success = defaultdict(lambda: {'total': 0, 'success': 0})
    failed_rows = []  # For all failures
    retryable_rows = []  # For failures worth retrying
    processed_rows = set()
    low_confidence_rows = set()
    
    print(f"Analyzing log file: {LOG_FILE}")
    print(f"Row range: {start_row} to {end_row if end_row else 'end'}")
    print("=" * 80)
    
    with open(LOG_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            row_idx = int(row['row_idx'])
            
            # Filter by row range
            if row_idx < start_row:
                continue
            if end_row and row_idx >= end_row:
                break

            processed_rows.add(row_idx)
            
            stats['total'] += 1
            
            # Parse fields (handle empty values)
            try:
                confidence_score = float(row['confidence_score']) if row['confidence_score'] else 0.0
            except ValueError:
                confidence_score = 0.0
            
            confidence_level = row['confidence_level'].strip() if row['confidence_level'] else 'unknown'
            download_success = parse_bool(row['download_success']) if row['download_success'] else False
            error_msg = row['error_msg'].strip() if row['error_msg'] else ''
            
            # Confidence distribution
            confidence_bucket = int(confidence_score // 10) * 10
            confidence_distribution[confidence_bucket] += 1
            
            # Track confidence vs success
            conf_bucket = int(confidence_score // 10) * 10
            confidence_vs_success[conf_bucket]['total'] += 1
            if download_success:
                confidence_vs_success[conf_bucket]['success'] += 1
            
            # Categorize outcome
            if download_success:
                stats['successful'] += 1
            else:
                stats['failed'] += 1
                failed_rows.append(row_idx)
                
                is_retryable = True
                
                # Categorize error
                if error_msg:
                    if "Low confidence" in error_msg:
                        stats['skipped_low_confidence'] += 1
                        error_categories['Low confidence (skipped)'] += 1
                        low_confidence_rows.add(row_idx)
                        is_retryable = False
                    elif "age" in error_msg.lower() or "restricted" in error_msg.lower():
                        error_categories['Age restricted'] += 1
                        stats['download_failed'] += 1
                        stats['attempted'] += 1
                    elif "available" in error_msg.lower() or "removed" in error_msg.lower():
                        error_categories['Video unavailable/removed'] += 1
                        stats['download_failed'] += 1
                        stats['attempted'] += 1
                    elif "private" in error_msg.lower():
                        error_categories['Private video'] += 1
                        stats['download_failed'] += 1
                        stats['attempted'] += 1
                    elif "copyright" in error_msg.lower() or "blocked" in error_msg.lower():
                        error_categories['Copyright/blocked'] += 1
                        stats['download_failed'] += 1
                        stats['attempted'] += 1
                    elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                        error_categories['Timeout'] += 1
                        stats['download_failed'] += 1
                        stats['attempted'] += 1
                    elif "network" in error_msg.lower() or "connection" in error_msg.lower():
                        error_categories['Network error'] += 1
                        stats['download_failed'] += 1
                        stats['attempted'] += 1
                    elif "429" in error_msg or "rate" in error_msg.lower():
                        error_categories['Rate limited'] += 1
                        stats['download_failed'] += 1
                        stats['attempted'] += 1
                    else:
                        error_categories['Other error'] += 1
                        stats['download_failed'] += 1
                        stats['attempted'] += 1
                else:
                    # No error message - probably low confidence
                    if confidence_score < 60:
                        stats['skipped_low_confidence'] += 1
                        error_categories['Low confidence (skipped)'] += 1
                        low_confidence_rows.add(row_idx)
                        is_retryable = False
                    else:
                        error_categories['Unknown failure'] += 1
                        stats['download_failed'] += 1
                
                if is_retryable:
                    retryable_rows.append(row_idx)

    # Validate against actual downloaded files on disk
    complete_rows, partial_rows = get_downloaded_audio_rows()
    stats['complete_files_detected'] = len(complete_rows)
    stats['partial_files_ignored'] = len(partial_rows)

    missing_on_disk_rows = sorted(processed_rows - complete_rows)
    not_downloaded_error_rows = sorted(
        row_idx for row_idx in missing_on_disk_rows if row_idx not in low_confidence_rows
    )

    stats['missing_on_disk'] = len(missing_on_disk_rows)
    stats['not_downloaded_for_some_reason_error'] = len(not_downloaded_error_rows)

    # Ensure retry candidates include all non-low-confidence rows that are missing on disk.
    retryable_rows = sorted(set(retryable_rows).union(not_downloaded_error_rows))
    
    # Calculate percentages
    if stats['total'] > 0:
        stats['success_rate'] = (stats['successful'] / stats['total']) * 100
        stats['skip_rate'] = (stats['skipped_low_confidence'] / stats['total']) * 100
    
    if stats['attempted'] > 0:
        stats['download_failure_rate'] = (stats['download_failed'] / stats['attempted']) * 100
    
    return {
        'stats': stats,
        'error_categories': error_categories,
        'confidence_distribution': confidence_distribution,
        'confidence_vs_success': confidence_vs_success,
        'failed_rows': failed_rows,
        'retryable_rows': retryable_rows,
        'missing_on_disk_rows': missing_on_disk_rows,
        'not_downloaded_error_rows': not_downloaded_error_rows,
    }


def print_report(results: Dict):
    """Print formatted diagnostic report."""
    
    stats = results['stats']
    error_categories = results['error_categories']
    confidence_distribution = results['confidence_distribution']
    confidence_vs_success = results['confidence_vs_success']
    failed_rows = results['failed_rows']
    retryable_rows = results['retryable_rows']
    missing_on_disk_rows = results['missing_on_disk_rows']
    not_downloaded_error_rows = results['not_downloaded_error_rows']
    
    # Overall Statistics
    print("\n📊 OVERALL STATISTICS")
    print("=" * 80)
    print(f"Total rows processed:     {stats['total']:,}")
    print(f"Successfully downloaded:  {stats['successful']:,} ({stats.get('success_rate', 0):.1f}%)")
    print(f"Failed:                   {stats['failed']:,}")
    print(f"  - Skipped (low conf):   {stats['skipped_low_confidence']:,} ({stats.get('skip_rate', 0):.1f}%)")
    print(f"  - Download failed:      {stats['download_failed']:,}")
    
    if stats['attempted'] > 0:
        print(f"\nDownload attempts:        {stats['attempted']:,}")
        print(f"Download failure rate:    {stats.get('download_failure_rate', 0):.1f}%")
    
    # Error Categories
    print("\n🔍 ERROR BREAKDOWN")
    print("=" * 80)
    if error_categories:
        for error_type, count in error_categories.most_common():
            percentage = (count / stats['failed']) * 100 if stats['failed'] > 0 else 0
            print(f"{error_type:<30} {count:>6,} ({percentage:>5.1f}%)")
    else:
        print("No errors found!")
    
    # Confidence Distribution
    print("\n📈 CONFIDENCE SCORE DISTRIBUTION")
    print("=" * 80)
    print(f"{'Score Range':<15} {'Count':<10} {'Success':<10} {'Success Rate'}")
    print("-" * 80)
    for bucket in sorted(confidence_distribution.keys()):
        count = confidence_distribution[bucket]
        cv_data = confidence_vs_success[bucket]
        success_count = cv_data['success']
        success_rate = (success_count / cv_data['total'] * 100) if cv_data['total'] > 0 else 0
        print(f"{bucket}-{bucket+9:<15} {count:<10,} {success_count:<10,} {success_rate:>5.1f}%")
    
    # Retry candidates
    print("\n🔄 RETRY CANDIDATES")
    print("=" * 80)
    
    print(f"Total failed rows:        {len(failed_rows):,}")
    print(f"Low confidence (skip):    {error_categories.get('Low confidence (skipped)', 0):,}")
    print(f"Missing rows on disk:     {len(missing_on_disk_rows):,}")
    print(f"Not downloaded for some reason error: {len(not_downloaded_error_rows):,}")
    print(f"Retryable failures:       {len(retryable_rows):,}")
    print(f"Complete files detected:  {stats.get('complete_files_detected', 0):,}")
    print(f"Partial files ignored:    {stats.get('partial_files_ignored', 0):,}")
    
    # Key insights for retry strategy
    print("\n💡 KEY INSIGHTS FOR RETRY")
    print("=" * 80)
    
    age_restricted = error_categories.get('Age restricted', 0)
    if age_restricted > 0:
        print(f"⚠️  Age restricted: {age_restricted:,} songs")
        print(f"   → Use --cookies-from-browser firefox flag")
    
    network_errors = (error_categories.get('Timeout', 0) + 
                     error_categories.get('Network error', 0) +
                     error_categories.get('Rate limited', 0))
    if network_errors > 0:
        print(f"⚠️  Network/timeout: {network_errors:,} songs")
        print(f"   → Retry may succeed with better connection")
    
    unavailable = error_categories.get('Video unavailable/removed', 0)
    if unavailable > 0:
        print(f"⚠️  Unavailable: {unavailable:,} songs")
        print(f"   → Retry unlikely to help (videos removed)")

    not_downloaded_error_count = stats.get('not_downloaded_for_some_reason_error', 0)
    if not_downloaded_error_count > 0:
        print(f"⚠️  Not downloaded for some reason error: {not_downloaded_error_count:,} songs")
        print(f"   → Detected by checking missing complete files on disk (excluding low confidence)")
    
    if stats['skipped_low_confidence'] > stats['successful']:
        print(f"⚠️  More songs skipped ({stats['skipped_low_confidence']:,}) than downloaded ({stats['successful']:,})")
        print(f"   → Consider lowering confidence threshold to 50-55")
    
    # Export failed rows for retry
    failed_file = PROJECT_ROOT / "data" / "logs" / "failed_rows.txt"
    with open(failed_file, 'w') as f:
        for row_idx in failed_rows:
            f.write(f"{row_idx}\n")
            
    retry_file = PROJECT_ROOT / "data" / "logs" / "retryable_rows.txt"
    with open(retry_file, 'w') as f:
        for row_idx in retryable_rows:
            f.write(f"{row_idx}\n")

    missing_file = PROJECT_ROOT / "data" / "logs" / "missing_on_disk_rows.txt"
    with open(missing_file, 'w') as f:
        for row_idx in missing_on_disk_rows:
            f.write(f"{row_idx}\n")

    not_downloaded_error_file = PROJECT_ROOT / "data" / "logs" / "not_downloaded_error_rows.txt"
    with open(not_downloaded_error_file, 'w') as f:
        for row_idx in not_downloaded_error_rows:
            f.write(f"{row_idx}\n")
    
    print(f"\n✅ All failed row indices saved to: {failed_file}")
    print(f"✅ Retryable row indices saved to: {retry_file}")
    print(f"✅ Missing-on-disk row indices saved to: {missing_file}")
    print(f"✅ Not-downloaded-error row indices saved to: {not_downloaded_error_file}")
    print(f"   Total retryable: {len(retryable_rows):,} rows")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze download log failures')
    parser.add_argument('--start-row', type=int, default=0, help='Start row index')
    parser.add_argument('--end-row', type=int, default=None, help='End row index (exclusive)')
    
    args = parser.parse_args()
    
    # Analyze log
    results = analyze_log(start_row=args.start_row, end_row=args.end_row)
    
    # Print report
    print_report(results)
    
    print("\n" + "=" * 80)
    print("📋 NEXT STEPS")
    print("=" * 80)
    print("1. Review error breakdown above")
    print("2. Check if age-restricted videos are significant (use cookies)")
    print("3. Consider retry strategy for network errors")
    print("4. Evaluate if confidence threshold should be adjusted")
    print("5. Export failed_rows.txt for targeted retry")


if __name__ == '__main__':
    main()
