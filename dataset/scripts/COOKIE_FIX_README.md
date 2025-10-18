# Fixing 403 Errors - Cookie-Based Authentication

## Problem
The Chosic scraper is getting **403 Forbidden** errors because Cloudflare has strengthened its bot protection. Automated tools can no longer bypass it reliably.

## Solution
Use valid cookies from your browser session. This makes the scraper appear as a legitimate browser request.

---

## Quick Start

### 1. Get Your Cookies (Required)

1. **Open Firefox or Chrome** and go to: https://www.chosic.com/music-genre-finder/
2. **Open Developer Tools**: Press `F12`
3. **Go to Network tab**
4. **Refresh the page** (F5)
5. **Click on any request** to `chosic.com` in the network list
6. **Find "Cookie" header** in Request Headers section
7. **Copy the entire cookie value** - it should look like:
   ```
   pll_language=en; cf_clearance=uAIsICQ7FF3q6_M5R_0ttJ...; r_c1062550=1760360183...
   ```

### 2. Save Your Cookies

Create a file called `cookies.txt` in the `dataset/scripts/` directory and paste your cookie string:

```bash
cd /home/esstee/documents/bitirme/dataset/scripts/
nano cookies.txt
# Paste your cookies, save (Ctrl+O, Enter, Ctrl+X)
```

### 3. Test Your Cookies

Run the test script to verify your cookies work:

```bash
python test_cookies.py
```

You should see:
```
✓ ALL TESTS PASSED!
You can now run chosic_scraper_http.py
```

### 4. Run the Scraper

```bash
python chosic_scraper_http.py
```

The scraper will automatically load cookies from `cookies.txt` and start working.

---

## How Long Do Cookies Last?

- Cookies typically last **a few hours to a few days**
- The `cf_clearance` cookie (Cloudflare bypass) expires after ~2 hours to 24 hours
- When you get 403 errors again, just get fresh cookies and update `cookies.txt`

---

## Troubleshooting

### Still getting 403 errors?
1. **Get fresh cookies** - they may have expired
2. **Make sure you copied the ENTIRE cookie string** including all parts
3. **Check the cookie includes `cf_clearance`** - this is critical for Cloudflare bypass

### "No cf_clearance cookie" warning?
- The `cf_clearance` cookie is essential
- Make sure you waited for the page to fully load before copying cookies
- Try refreshing the page a couple times to trigger Cloudflare check

### Test script fails?
- Don't run the main scraper yet
- Get fresh cookies
- Run `python test_cookies.py` until it passes

---

## Example Cookie Format

Your cookies should look like this:
```
pll_language=en; cf_clearance=uAIsICQ7FF3q6_M5R_0ttJTKoMRHIw6SEdZD5Y7VTcg-1760531285-1.2.1.1-uYtL.kJ2mi8CI9vOl0NWX_Zq9FFx3mwOuwgHnvlfX1BnDmSaHcTJ8gUcLnwwpwGkgYg8oMkYgxuLeoqaYFta2EkfArlsJKDJvDB145gJDakxwnMsyRzJLy0YvkKOOfuBG5pjftW2CfQV4KFFXh0CccJhIIGBdh7AmXLw4HcaBtKKNPGM2eJU7T9sJO9wxdSXvn92DH4Fynra.PW_Ji.PLmeA_zQpew7WvUchAFeWHMQ; r_c1062550=1760360183%7Cf3eb6cc232dacf2d%7Cca5b390238c2da6cb14b2088050409faaee4384c263077d99835ecea12e64c5d
```

Key parts:
- `pll_language=en` - Language preference
- `cf_clearance=...` - **CRITICAL** - Cloudflare bypass token
- `r_c1062550=...` - Session tracking

---

## What Changed in the Script?

1. **Removed automatic Cloudflare bypass** - wasn't working reliably
2. **Added cookie loading** - from `cookies.txt` or manual input
3. **Added cookie test** - verifies connection before starting
4. **Updated headers** - matches your working curl commands exactly
5. **Added Referer headers** - makes requests look more legitimate

---

## Tips for Success

1. **Keep browser tab open** while scraping (optional but helps)
2. **Use lower delay** - with valid cookies, rate limiting is less strict
3. **Get fresh cookies when needed** - takes 30 seconds, saves hours of debugging
4. **Save cookies.txt** - reuse until they expire

---

## Alternative: Manual Cookie Input

If you don't want to create `cookies.txt`, the scraper will prompt you:

```bash
python chosic_scraper_http.py

# It will ask:
# "Do you want to enter cookies now? (y/n):"
# Type: y
# Paste your cookies
# Choose whether to save to file
```

---

## Contact

If you still have issues after following these steps, check:
1. Your cookies are fresh (< 2 hours old)
2. You copied the ENTIRE cookie string
3. The test script passes before running the main scraper
