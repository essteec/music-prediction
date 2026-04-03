# Audio Download Pipeline - Technical Summary

## Overview

Concurrent YouTube audio download pipeline for acquiring 550K+ songs. Optimized from 73.7 days → 11.5 days (6.4x speedup).

**Status**: Pilot phase - 850/15,000 songs completed (5.7%)  
**Current Speed**: 1.8s per song (target: ≤30s)  
**Success Rate**: ~78% (target: ≥80%)

---

## Architecture

### Producer-Consumer Pattern

**Phase 1: Search (8 workers)**
- Query: `"{track} {artist1} {artist2} {artist3} official audio"`
- Tool: yt-dlp Python API (32% faster than subprocess)
- Output: Top 3 YouTube results with metadata

**Phase 2: Validate (sequential)**
- Confidence scoring: Title + Duration + Artist = 0-100
- Threshold: ≥60 required to proceed
- Filters: Skip videos >8 minutes (mixes/podcasts)

**Phase 3: Download (4 workers)**
- Format: 251 (Opus/WebM, ~3-4MB)
- Bandwidth-limited: 4 parallel downloads
- Retry: Not implemented (move on if fails)

**Phase 4: Checkpoint (every 50 songs)**
- Atomic writes to prevent corruption
- Resume from last successful batch
- Logs: CSV with 19 validation columns

---

## Confidence Scoring (0-100)

```python
score = title_similarity(40) + duration_match(30) + artist_verify(30)

# Title: Fuzzy token sort ratio
title_sim = fuzz.token_sort_ratio(csv_title, yt_title) * 0.4

# Duration: Tolerance bands
if abs(diff) <= 5s:  duration_pts = 30
elif abs(diff) <= 15s: duration_pts = 20  
elif abs(diff) <= 30s: duration_pts = 10
else: duration_pts = 0

# Artist: Presence check in title/uploader
artist_pts = (matches / total_artists) * 30

# Confidence level
if score >= 80: "high"
elif score >= 60: "medium"
else: "low" (skip)
```

---

## Performance Benchmarks

All benchmarks run on same system (April 3, 2026):

| Test | Baseline | Optimized | Speedup |
|------|----------|-----------|---------|
| **Search (subprocess)** | 1.39s | 1.05s (API) | 1.32x |
| **Search (4 workers)** | 1.09s | 0.35s | 3.1x |
| **Download (sequential)** | 4.24s | 2.69s (4 workers) | 1.6x |
| **Full pipeline** | 11.57s | 2.27s (theory) | 5.1x |
| **Real-world (50 songs)** | 11.57s | **1.8s** | **6.4x** ✅ |

### Bottleneck Analysis

**Sequential version**:
- 2s rate limit per song = 12.7 days of waiting
- Search blocks download (stop-and-wait)
- Subprocess overhead (~0.3s per call)

**Optimized version**:
- 0.5s delay per 50-song batch = 5.5 hours total
- Search and download overlap (producer-consumer)
- Native Python API (no subprocess)

---

## File Structure

```
scripts/audio-acquisition/
├── 01_pilot_download.py        # Main pipeline (530 lines, concurrent)
├── 01_pilot_download_old.py    # Backup (522 lines, sequential)
├── validation.py                # Confidence scoring (140 lines)
└── utils.py                     # Helpers (114 lines)

data/audio/pilot/
└── *.webm                       # Downloaded audio (693 files so far)

data/logs/
├── download_log_pilot.csv       # Results: 19 columns × 850 rows
└── checkpoint_pilot.json        # Resume: {"last_row": 849, ...}
```

---

## Key Features

### Robustness
- ✅ Checkpoint corruption handling (JSON decode errors)
- ✅ Atomic checkpoint writes (temp file → rename)
- ✅ Thread-safe logging (mutex lock)
- ✅ CSV formula injection protection (escape =, +, -, @)
- ✅ Duration validation (None/invalid checks)
- ✅ Rate limiting (0.5s between batches)

### Validation Safeguards
- ✅ Multi-candidate scoring (best of 3)
- ✅ Duration range check (skip >8 minutes)
- ✅ Numeric duration validation (prevent None crashes)
- ✅ Artist presence verification (fuzzy match)
- ✅ Confidence threshold enforcement (≥60)

### Error Handling
- ✅ File I/O errors (FileNotFoundError, PermissionError)
- ✅ CSV encoding errors (UnicodeDecodeError)
- ✅ Empty results (no YouTube matches)
- ✅ Network failures (yt-dlp exceptions)
- ✅ Zero-duration edge cases

---

## Usage

### Run Pilot (15K songs)
```bash
cd /home/esstee/projects/music-prediction
source .venv/bin/activate
cd scripts/audio-acquisition

# Start or resume
python 01_pilot_download.py --limit 15000

# In tmux for persistence
tmux new -s pilot
python 01_pilot_download.py --limit 15000
# Ctrl+B, D to detach
```

### Adjust Concurrency
```bash
# More aggressive (if network allows)
python 01_pilot_download.py --workers 16 --dl-workers 8

# Conservative (slower connection)
python 01_pilot_download.py --workers 4 --dl-workers 2
```

### Resume After Crash
```bash
# Auto-resumes from checkpoint
python 01_pilot_download.py --limit 15000

# Manual resume
python 01_pilot_download.py --start-row 5000 --limit 15000
```

### Fresh Start
```bash
# Ignore checkpoint, start from row 0
python 01_pilot_download.py --limit 15000 --no-resume
```

---

## Monitoring

### Check Progress
```bash
# Checkpoint state
cat data/logs/checkpoint_pilot.json

# Downloaded files
ls -1 data/audio/pilot/*.webm | wc -l

# Success rate
tail -100 data/logs/download_log_pilot.csv | \
  awk -F',' '{if($16=="True") s++; t++} END {print s/t*100"%"}'
```

### Storage Usage
```bash
# Current size
du -sh data/audio/pilot/

# Projection
python -c "
files = $(ls -1 data/audio/pilot/*.webm | wc -l)
total_mb = $(du -sm data/audio/pilot/ | cut -f1)
avg_mb = total_mb / files
print(f'Avg: {avg_mb:.2f}MB')
print(f'15K: {avg_mb * 15000 / 1024:.1f}GB')
print(f'550K: {avg_mb * 550000 / 1024:.1f}GB')
"
```

---

## Decision Criteria

After 15K pilot completes, analyze:

### Success Rate
```bash
# Overall success
tail -n +2 data/logs/download_log_pilot.csv | \
  awk -F',' '{if($16=="True") s++; t++} END {print "Success:", s, "/", t, "=", s/t*100"%"}'

# By confidence level
tail -n +2 data/logs/download_log_pilot.csv | \
  awk -F',' '{
    conf[$11]++; 
    if($16=="True") succ[$11]++
  } 
  END {
    for(c in conf) print c":", succ[c]"/"conf[c], "=", succ[c]/conf[c]*100"%"
  }'
```

### Speed Verification
```bash
# Average time per song
tail -n +2 data/logs/download_log_pilot.csv | \
  awk -F',' '{sum+=$18; n++} END {print "Avg time:", sum/n, "seconds"}'
```

### Storage Verification
```bash
# Average file size
tail -n +2 data/logs/download_log_pilot.csv | \
  awk -F',' '{if($17!="") {sum+=$17; n++}} END {print "Avg size:", sum/n, "MB"}'
```

### PROCEED if:
- Success rate ≥80%
- Avg time ≤30s per song
- Storage projection manageable (<2.5TB)

### MODIFY if:
- Success rate 70-80% → Add manual review for medium confidence
- Time 30-60s → Reduce to test+val only (175K songs)
- Storage >2.5TB → Increase compression or reduce quality

### ABORT if:
- Success rate <70% → Too many false matches
- Time >60s → Not feasible timeline
- Storage >3TB → Can't store full dataset

---

## Known Issues

### Not Implemented
- ❌ Retry logic on download failure (just logs and moves on)
- ❌ Duplicate detection (same YouTube ID for different songs)
- ❌ Quality verification (assumes yt-dlp downloads correctly)
- ❌ Audio fingerprinting (trust confidence score only)

### Workarounds
- **Duplicates**: Low impact - different perspectives of same performance
- **Quality**: Post-process validation after download completes
- **Fingerprinting**: Could add in Phase 3 if needed

### Accepted Limitations
- Some songs will fail (target: <20%)
- Some matches may be covers/remixes (confidence score minimizes)
- Storage intensive (2.2TB for full dataset)

---

## Next Phase Integration

After acquisition completes:

### Phase 3: Audio Embeddings
```python
# Extract MERT embeddings (768-d)
from transformers import AutoModel
model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M")

# Process all audio files
for audio_file in glob("data/audio/pilot/*.webm"):
    embedding = extract_mert(audio_file)
    # Save to data/embeddings/mert_audio_768d.npy
```

### Feature Composition
- Phase 0: 414 features (23 audio + 5 text + 2 sentiment + 384 MiniLM)
- Phase 1B: 798 features (23 audio + 5 text + 2 sentiment + 768 MPNet)
- **Phase 3**: 1566 features (23 audio + 5 text + 2 sentiment + 768 MPNet + 768 MERT)

### Expected Improvement
- Energy: 0.75 → 0.80+ (MERT captures energy patterns)
- Danceability: 0.50 → 0.56+ (rhythm from raw audio)
- May enable multimodal fusion architectures

---

## Optimization Insights

### What Worked
1. **yt-dlp Python API**: Eliminates subprocess overhead
2. **Producer-consumer**: Search doesn't wait for download
3. **Batch checkpoints**: Balance safety vs I/O overhead
4. **Reduced rate limiting**: 0.5s per batch vs 2s per song
5. **Thread-safe logging**: Single lock, append-only

### What Didn't Help
- Increasing workers beyond 8 (search) / 4 (download): Bandwidth saturated
- Smaller batch sizes: More checkpoint overhead
- Aggressive retry logic: Slows down pipeline

### Lessons Learned
- I/O-bound tasks benefit most from concurrency
- Rate limiting is often the real bottleneck
- Validation at search time prevents wasted downloads
- Checkpoint granularity matters (too frequent = slow, too rare = data loss)

---

## References

- **yt-dlp**: https://github.com/yt-dlp/yt-dlp
- **sentence-transformers**: https://www.sbert.net/
- **MERT**: https://huggingface.co/m-a-p/MERT-v1-330M
- **ThreadPoolExecutor**: https://docs.python.org/3/library/concurrent.futures.html

---

**Last Updated**: April 3, 2026  
**Pipeline Version**: 2.0 (Optimized Concurrent)  
**Status**: Pilot in progress (5.7% complete)
