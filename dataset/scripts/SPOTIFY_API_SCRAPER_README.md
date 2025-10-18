# Spotify API Direct Scraper

Updated scraper that uses **Spotify Web API** directly instead of scraping Chosic.com.

## Key Improvements

✅ **More reliable** - Direct API access, no web scraping issues  
✅ **Efficient batching** - 50 tracks per request (vs 1 track per request before)  
✅ **Rate limit handling** - Automatic retry with exponential backoff  
✅ **Status code monitoring** - Proper handling of 401, 429, 5xx errors  
✅ **Same checkpoint system** - Resume from where you left off  

## Setup

### 1. Get Spotify Access Token

Visit: https://developer.spotify.com/console/get-track/

1. Click **"Get Token"** button
2. Select required scopes (or use default)
3. Copy the token

### 2. Set Environment Variable (Optional but Recommended)

```bash
export SPOTIFY_ACCESS_TOKEN='your_token_here'
```

Or just run the script and it will prompt you for the token.

## Usage

```bash
cd dataset/scripts
python chosic_scraper_spotify.py
```

### Options

1. **Process full dataset** - Process all tracks (or resume if interrupted)
2. **Process a sample batch** - Process N tracks (default: 100)
3. **Exit**

## API Endpoints Used

### 1. Get Tracks (Batch)
```bash
GET https://api.spotify.com/v1/tracks?ids=id1,id2,...,id50
```
Returns: popularity, release year, explicit flag, artist IDs

### 2. Get Artists (Batch)
```bash
GET https://api.spotify.com/v1/artists?ids=id1,id2,...,id50
```
Returns: artist genres

## Rate Limiting

- **Spotify limit**: ~180 requests/minute
- **Our delay**: 100ms between requests (safe buffer)
- **Batch size**: 50 tracks per request (maximum allowed)
- **Auto-retry**: Handles 429 (rate limit) with Retry-After header

## Efficiency

**Before (Chosic scraping):**
- 1 track = 3 HTTP requests
- ~1-2 tracks/second

**Now (Spotify API):**
- 50 tracks = 2-3 HTTP requests (1 for tracks, 1-2 for artists)
- **~10-20x faster**
- Example: 50 tracks in ~0.5 seconds

## Error Handling

| Status Code | Behavior |
|-------------|----------|
| 200 | Success ✓ |
| 401 | Token expired - script exits with error message |
| 429 | Rate limited - waits for Retry-After seconds |
| 4xx | Client error - logs and continues |
| 5xx | Server error - retries with exponential backoff |
| Timeout | Retries up to 3 times |

## Output Files

- `songs_enhanced_full.csv` - Successfully processed tracks
- `failed_tracks.csv` - Failed tracks (for later retry)

## Token Expiration

Spotify access tokens expire after **1 hour**.

If you get a 401 error:
1. Get a new token from the console
2. Export it: `export SPOTIFY_ACCESS_TOKEN='new_token'`
3. Re-run the script (it will resume from checkpoint)

## Example Run

```
Processing batch: tracks 1-50 (50 tracks)
============================================================
[1/50] 7ouMYWpwJ422jRcDASZB7P: ✓ Year: 2018, Pop: 75, Genre: pop
[2/50] 4VqPOruhp5EdPBeR92t6lQ: ✓ Year: 2019, Pop: 82, Genre: rock
...
[50/50] 2takcwOaAZWiXQijPHIx7B: ✓ Year: 2020, Pop: 68, Genre: hip-hop

============================================================
Progress: 50/1000 (5.0%)
Success rate: 50/50 (100.0%)
Failed: 0/50
Speed: 25.5 tracks/sec
API requests made: 3
Rate limit remaining: 177
============================================================
```

## Troubleshooting

### "Invalid access token"
- Get a new token from the Spotify console
- Tokens expire after 1 hour

### "Rate limited"
- Script automatically handles this
- Waits for the time specified in Retry-After header

### "Failed to fetch tracks batch"
- Check internet connection
- Verify token is valid
- Check if Spotify API is down: https://status.developer.spotify.com/

## Architecture

```
SpotifyAPIScraper
│
├── get_tracks_batch(track_ids[])      # Get 50 tracks in 1 request
├── get_artists_batch(artist_ids[])    # Get 50 artists in 1 request
├── process_tracks_batch(track_ids[])  # Process 50 tracks efficiently
│   ├── Fetch all tracks (1 request)
│   ├── Extract artist IDs
│   ├── Fetch all artists (1-2 requests)
│   └── Map genres to tracks
│
└── process_csv()                      # Main loop with checkpointing
    └── Process in batches of 50
```

## Notes

- Preserves the original modular design
- Compatible with existing checkpoint files
- Works with `genre_mapper_http` for genre normalization
- No changes needed to `genre_mapper_http.py`
