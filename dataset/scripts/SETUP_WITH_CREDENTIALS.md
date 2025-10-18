# Quick Setup Guide with Client ID and Secret

## ✅ You have Client ID and Client Secret!

Perfect! This is the **best way** - your token will **auto-refresh every hour**. No need to manually get tokens!

## 🚀 Setup (2 minutes)

### Step 1: Set Environment Variables

```bash
export SPOTIFY_CLIENT_ID='your_client_id_here'
export SPOTIFY_CLIENT_SECRET='your_client_secret_here'
```

**Tip:** To make this permanent (so you don't need to type it every time), add to your `~/.zshrc`:

```bash
echo "export SPOTIFY_CLIENT_ID='your_client_id_here'" >> ~/.zshrc
echo "export SPOTIFY_CLIENT_SECRET='your_client_secret_here'" >> ~/.zshrc
source ~/.zshrc
```

### Step 2: Run the Script

```bash
cd /home/esstee/documents/bitirme/dataset/scripts
python chosic_scraper_spotify.py
```

That's it! ✅

## 🎯 What Happens Now?

1. **Script starts** → Automatically gets access token
2. **Token expires after 1 hour** → Script automatically refreshes it
3. **No interruption** → You can process thousands of tracks continuously!

## 📊 Example Output

```
============================================================
Spotify API Direct Scraper
============================================================

✓ Found credentials in environment variables
  → Token will auto-refresh every hour
  
🔄 Requesting access token from Spotify...
✓ Access token obtained! Valid for 60 minutes

Reading ../songs_with_attributes_and_lyrics.csv...
Found existing output file with 1250 processed tracks
Total tracks in dataset: 10000
Already processed: 1250
Remaining to process: 8750

Options:
1. Process full dataset (or resume if interrupted)
2. Process a sample batch (e.g., 100 tracks)
3. Exit

Enter your choice (1/2/3): 
```

## ⏰ Auto-Refresh in Action

After ~55 minutes:
```
⚠️  Access token expired, refreshing...
🔄 Requesting access token from Spotify...
✓ Access token obtained! Valid for 60 minutes

[Continues processing without interruption]
```

## 🔧 Alternative: Run Without Environment Variables

If you prefer not to use environment variables, the script will prompt you:

```bash
python chosic_scraper_spotify.py

# It will ask:
# Do you have Client ID and Secret? (y/n): y
# Enter Client ID: [paste here]
# Enter Client Secret: [paste here]

# Then it continues automatically!
```

## 📋 Your Credentials

Keep these safe (don't share publicly):

- **Client ID**: Found in your Spotify Dashboard → Your App → Settings
- **Client Secret**: Click "View client secret" in Settings

## ✅ Benefits of This Method

| Feature | With Client ID/Secret | Without |
|---------|---------------------|---------|
| Token refresh | ✅ Automatic | ❌ Manual every hour |
| Long sessions | ✅ Unlimited | ❌ Max 1 hour |
| Convenience | ✅ Set once | ❌ Paste token each time |
| Interruptions | ✅ None | ❌ Every hour |

## 🎉 You're All Set!

Just run:
```bash
export SPOTIFY_CLIENT_ID='your_id'
export SPOTIFY_CLIENT_SECRET='your_secret'
python chosic_scraper_spotify.py
```

Choose option 1 to process all tracks, and let it run! 🚀

The script will:
- ✅ Auto-refresh tokens
- ✅ Save progress continuously
- ✅ Resume if interrupted
- ✅ Handle rate limits
- ✅ Process ~25+ tracks/second
