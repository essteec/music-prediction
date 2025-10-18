# 🚀 Quick Start Commands

## Setup (First Time Only)

```bash
# Set your credentials (replace with your actual values)
export SPOTIFY_CLIENT_ID='your_client_id'
export SPOTIFY_CLIENT_SECRET='your_client_secret'

# Make permanent (optional)
echo "export SPOTIFY_CLIENT_ID='your_client_id'" >> ~/.zshrc
echo "export SPOTIFY_CLIENT_SECRET='your_client_secret'" >> ~/.zshrc
```

## Run the Scraper

```bash
# Navigate to scripts directory
cd /home/esstee/documents/bitirme/dataset/scripts

# Run the scraper
python chosic_scraper_spotify.py

# Choose option:
# 1 - Process all tracks (or resume)
# 2 - Process a batch (e.g., 100 tracks)
# 3 - Exit
```

## That's It! 🎉

The script will:
- ✅ Automatically get access token
- ✅ Auto-refresh every hour
- ✅ Process tracks in efficient batches
- ✅ Save progress continuously
- ✅ Resume if interrupted (Ctrl+C)

---

## Optional: Check Progress

```bash
# Count processed tracks
wc -l ../songs_enhanced_full.csv

# View last few processed tracks
tail ../songs_enhanced_full.csv

# Check failed tracks
wc -l ../failed_tracks.csv
```

---

## If You Get Errors

### "Failed to obtain access token"
→ Double-check your Client ID and Secret (no typos!)

### "Invalid access token"
→ Should auto-refresh, but if not, restart the script

### Script stopped?
→ Just run it again - it resumes from checkpoint!

---

## Full Command (Copy-Paste Ready)

```bash
export SPOTIFY_CLIENT_ID='YOUR_CLIENT_ID_HERE'
export SPOTIFY_CLIENT_SECRET='YOUR_CLIENT_SECRET_HERE'
cd /home/esstee/documents/bitirme/dataset/scripts
python chosic_scraper_spotify.py
# Press 1 and Enter to start processing
```
