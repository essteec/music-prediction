# Script execution order

Use 50k rows if you have ~175GB of free disk space. Adjust it acordingly.

- python scripts/audio-acquisition/01_pilot_download.py --start-row 0 --limit 50000  # increase start-row each cycle
- python scripts/audio-acquisition/deduplicate_log.py
- python scripts/audio-acquisition/diagnose_failures.py --start-row 0 --end-row 50000  # increase start-row and end-row each cycle
- python scripts/audio-acquisition/01_pilot_download.py --retry-file data/logs/retryable_rows.txt
- python scripts/audio-acquisition/deduplicate_log.py
- python scripts/audio-acquisition/diagnose_failures.py --start-row 0 --end-row 50000  # increase start-row and end-row each cycle

- python scripts/audio-embedding-extraction/run_all_extractors.py --resume
- python scripts/audio-embedding-extraction/analyze_outcomes.py

- rm data/audio/pilot/*.webm

Re-run all the above again with next 50k cycle.
