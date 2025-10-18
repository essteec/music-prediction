# How to Get Spotify Access Token

## 🚀 Quick Method (5 minutes)

### For Testing and Development

**This is the EASIEST way and requires NO app registration!**

1. **Open Spotify Web Console:**
   ```
   https://developer.spotify.com/console/get-track/
   ```

2. **Click the green "Get Token" button** (on the right side)

3. **In the popup:**
   - You can leave all checkboxes as they are (or uncheck all)
   - Click "Request Token"

4. **Copy the token** - It appears in a text box at the top:
   ```
   BQDxK7j8P9mH3vFB2gYC8r... (very long string)
   ```

5. **Use it immediately:**
   ```bash
   export SPOTIFY_ACCESS_TOKEN='paste_your_token_here'
   python chosic_scraper_spotify.py
   ```

**⏰ Important:** Token expires after **1 hour**. Just get a new one when needed!

---

## 🔧 Automated Method (Production)

### If you want to auto-generate tokens

#### Step 1: Create Spotify App (One-time Setup)

1. Go to: **https://developer.spotify.com/dashboard**

2. **Log in** with your Spotify account (free account works!)

3. **Click "Create app"** button

4. **Fill in the form:**
   - **App name**: "My Music Scraper" (or anything)
   - **App description**: "Data collection for research"
   - **Website**: (can leave blank)
   - **Redirect URIs**: `http://localhost:8888/callback`
   - **Which API/SDKs are you planning to use?**: Check "Web API"

5. **Click "Save"**

6. **Click on your app** to open it

7. **Go to "Settings"**

8. **Copy these:**
   - **Client ID**: Something like `a3b4c5d6e7f8...`
   - **Client Secret**: Click "View client secret" and copy

#### Step 2: Use the Helper Script

```bash
cd dataset/scripts
python get_spotify_token.py
```

It will prompt you for:
- Client ID (paste from step 1)
- Client Secret (paste from step 1)

Then it automatically generates a fresh token!

#### Step 3: Save Credentials (Optional)

To avoid entering credentials every time:

```bash
# In your ~/.bashrc or ~/.zshrc
export SPOTIFY_CLIENT_ID='your_client_id'
export SPOTIFY_CLIENT_SECRET='your_client_secret'
```

Then you can run:
```bash
python get_spotify_token.py  # Auto-uses env vars
```

---

## 📝 Three Ways to Use Token

### Method 1: Environment Variable (Recommended)
```bash
export SPOTIFY_ACCESS_TOKEN='BQDxK7j8...'
python chosic_scraper_spotify.py
```

### Method 2: Script Prompts You
```bash
python chosic_scraper_spotify.py
# When prompted, paste the token
```

### Method 3: In Python Code
```python
from chosic_scraper_spotify import SpotifyAPIScraper

scraper = SpotifyAPIScraper(access_token='BQDxK7j8...')
```

---

## ⚠️ Token Expiration

**All tokens expire after 1 hour!**

When you see:
```
❌ Access token expired or invalid
   Get a new token from: https://developer.spotify.com/console/get-track/
```

Just:
1. Get a new token (using quick method above)
2. Export it: `export SPOTIFY_ACCESS_TOKEN='new_token'`
3. Re-run the script - **it will resume from checkpoint!**

---

## 🎯 Recommended Workflow

### For Short Sessions (< 1 hour)
Use the **Quick Method** - fastest and easiest!

### For Long Sessions (> 1 hour)
1. Use the **Quick Method** to get first token
2. Start processing
3. When it expires, get a new token (takes 30 seconds)
4. Continue - no data lost!

### For Automation
Set up the **Automated Method** once, then:
```bash
# Auto-refresh token when needed
export SPOTIFY_ACCESS_TOKEN=$(python get_spotify_token.py --quiet)
python chosic_scraper_spotify.py
```

---

## 🐛 Troubleshooting

### "Invalid Client ID or Secret"
- Double-check you copied them correctly from Spotify Dashboard
- Make sure there are no extra spaces

### "Token not working"
- Make sure you copied the FULL token (they're very long)
- Check if 1 hour has passed (token expired)
- Get a fresh token from the console

### "Rate limit exceeded"
- Script automatically handles this
- Just wait, it will resume automatically

---

## 🔒 Security Notes

- **Don't commit tokens to git!** (They're in `.gitignore`)
- **Don't share your Client Secret** publicly
- Tokens expire, so leaked tokens become useless after 1 hour
- Free tier: 180 requests/minute (more than enough!)

---

## ✅ Quick Start Checklist

- [ ] Visit https://developer.spotify.com/console/get-track/
- [ ] Click "Get Token" button
- [ ] Copy the token
- [ ] Run: `export SPOTIFY_ACCESS_TOKEN='your_token'`
- [ ] Run: `python chosic_scraper_spotify.py`
- [ ] Done! 🎉

---

## 📚 Additional Resources

- **Spotify Console**: https://developer.spotify.com/console/get-track/
- **Spotify Dashboard**: https://developer.spotify.com/dashboard
- **API Documentation**: https://developer.spotify.com/documentation/web-api/
- **API Status**: https://status.developer.spotify.com/
