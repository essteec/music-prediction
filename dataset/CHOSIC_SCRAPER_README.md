# Chosic.com Track Metadata Scraper

This script uses Selenium and BeautifulSoup to scrape track metadata from [Chosic.com](https://www.chosic.com/). It reads a CSV file containing Spotify track IDs, scrapes additional information for each track, and saves the enriched data to a new CSV file.

## Overview

The primary goal of this script is to augment an existing dataset of songs with the following metadata:
-   **Year** of release
-   **Genre** information
-   **Explicit** content flag
-   **Popularity** score

It achieves this by automating a Chrome browser to visit the "Music Genre Finder" tool on Chosic.com for each track ID.

## Features

-   **Automated Scraping**: Uses `Selenium` to control a Chrome browser, allowing it to process dynamically loaded content.
-   **CSV Processing**: Leverages the `pandas` library to read an input CSV, process the tracks, and write to an output CSV.
-   **Headless Mode**: Can run Chrome in the background without a visible UI, making it suitable for servers.
-   **Rate Limiting**: Includes a configurable delay between requests to avoid overwhelming the server.
-   **Interactive Execution**: When run directly, it first processes a small sample and then prompts the user before proceeding with the full dataset, preventing accidental long-running tasks.

## Dependencies

-   **Python Libraries**:
    -   `pandas`
    -   `beautifulsoup4`
    -   `selenium`
-   **External Requirements**:
    -   **Google Chrome**: The script is configured to use the Chrome browser.
    -   **ChromeDriver**: `Selenium` requires the corresponding ChromeDriver to be installed and accessible in the system's PATH.

## How to Run

1.  Ensure all Python dependencies are installed:
    ```sh
    pip install pandas beautifulsoup4 selenium
    ```
2.  Make sure the input file (`songs_with_attributes_and_lyrics.csv`) is in the same directory.
3.  Execute the script from your terminal:
    ```sh
    python chosic_scraper.py
    ```
4.  The script will first process a sample of 5 tracks and save them to `songs_enhanced_sample.csv`.
5.  It will then ask for confirmation before processing the entire dataset. Enter `y` to continue.

## Code Structure

### `ChosicScraper` Class

This class encapsulates all the scraping logic.

-   `__init__(self, headless=True)`: Initializes the Selenium WebDriver and sets various Chrome options to avoid bot detection. It also prepares a `WebDriverWait` object for explicit waits.

-   `scrape_track_metadata(self, track_id)`: This is the core scraping function. It navigates to the Chosic URL for a given track ID, waits for the page to load, and attempts to parse the HTML to extract metadata.

-   `process_csv(self, input_file, output_file, sample_size=None)`: This method orchestrates the entire workflow. It reads the source CSV, iterates through each track, calls the scraping method, and saves the results.

### `main()` Function

This function serves as the entry point when the script is executed directly. It demonstrates how to use the `ChosicScraper` class and provides the user-friendly interactive prompt.

## Known Issues and Limitations

-   **Brittle Scraping Logic**: The current version relies heavily on regular expressions and broad text searches (`soup.get_text()`) to find the required data. This is very fragile and likely to break if Chosic.com changes its page layout. The `TODO` comment in the code explicitly notes that more precise and stable CSS selectors are needed.
-   **Fixed Sleep Interval**: The `scrape_track_metadata` function uses `time.sleep(3)` to wait for the page's JavaScript to execute. This is unreliable. A more robust solution would be to use the `WebDriverWait` instance to wait for a specific element to appear on the page, which would confirm that the data has loaded.
