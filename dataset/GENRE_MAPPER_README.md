# Genre Mapping System (`genre_mapper.py`)

This script provides a robust system for mapping music sub-genres to a predefined list of main genres. It uses a local CSV file as a cache for known mappings and falls back to web scraping from [Chosic.com](https://www.chosic.com/) to discover new relationships.

## Features

- **Local Caching**: Stores successful mappings in `genre_mappings.csv` to minimize redundant web scraping.
- **Web Scraping Fallback**: If a sub-genre is not found in the local cache, the script scrapes the corresponding genre page on Chosic.com to find its parent genre.
- **Recursive Mapping**: It can trace a chain of parent genres until it finds a known main genre.
- **Validation**: Ensures that only valid, predefined main genres are saved to the mapping file.
- **"Unknown" Handling**: Gracefully handles cases where a main genre cannot be determined by assigning an "Unknown" status without permanently saving it, allowing for future re-attempts.

## How It Works

The core logic resides in the `get_main_genre(subgenre)` function:

1.  **Check Cache**: The script first loads all existing mappings from `genre_mappings.csv`.
2.  **Direct Match**: It checks if the given `subgenre` is already in the mappings or if it is, by itself, one of the `MAIN_GENRES`.
3.  **Scrape Parent Genre**: If no match is found, the script converts the sub-genre into a URL slug (e.g., "Post-Grunge" becomes `post-grunge`) and attempts to scrape its parent genre from `https://www.chosic.com/genre-chart/{genre_slug}/`.
4.  **Map Parent Genre**: The script then attempts to find a main genre for the *scraped parent genre*.
5.  **Save New Mapping**: If a valid main genre is found through the parent, the script saves the mapping for the *original sub-genre* to the CSV file.
6.  **Handle Failures**: If the scraper fails or if the parent genre cannot be mapped to a main genre, the original sub-genre is temporarily classified as "Unknown".

## Dependencies

- **Python Libraries**:
  - `selenium`: For web scraping.
- **External**:
  - **Google Chrome**: The script is configured to use the Chrome browser.
  - **ChromeDriver**: The `selenium` library requires the corresponding ChromeDriver to be installed and accessible in the system's PATH.

## Configuration

- **`MAIN_GENRES`**: A constant list at the top of the file that defines the valid parent genres. All sub-genres will ultimately be mapped to one of these categories.
- **`GENRE_MAP_FILE`**: The name of the CSV file used for caching mappings. Defaults to `"genre_mappings.csv"`.

## Core Functions

- **`get_main_genre(subgenre)`**: The main entry point. Takes a single sub-genre string and returns its corresponding main genre.
- **`get_main_genre_for_list(subgenres)`**: A convenience function that takes a list of sub-genres and returns the first valid main genre it can find among them.
- **`load_genre_mappings()`**: Reads the `genre_mappings.csv` file into a dictionary.
- **`save_genre_mapping(subgenre, main_genre)`**: Appends a new mapping to the CSV file, but only if the `main_genre` is in the `MAIN_GENRES` list.
- **`scrape_parent_genre(genre_slug)`**: Launches a `selenium` browser instance to fetch and parse the parent genre from a Chosic genre page.

## Usage Example

To use the mapper in another script, you can import the `get_main_genre` function.

```python
from genre_mapper import get_main_genre, get_main_genre_for_list

# Get the main genre for a single sub-genre
main_genre = get_main_genre("post-grunge")
print(f"The main genre for 'post-grunge' is: {main_genre}")
# Expected Output (after initial scrape): The main genre for 'post-grunge' is: Rock

# Get the first available main genre from a list of sub-genres
genres_list = ["bubblegum pop", "art rock", "country rap"]
first_main_genre = get_main_genre_for_list(genres_list)
print(f"The first main genre found is: {first_main_genre}")
# Expected Output: The first main genre found is: Pop
```
