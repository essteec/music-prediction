"""
Utility functions for audio download pipeline.
"""
import json
import re
from typing import List, Dict, Any


def safe_eval_artists(artists_str: str) -> List[str]:
    """
    Safely parse artists JSON string from CSV.
    
    Args:
        artists_str: JSON string like '["Artist1", "Artist2"]'
        
    Returns:
        List of artist names, empty list if parsing fails
        
    Example:
        >>> safe_eval_artists('["The Beatles", "John Lennon"]')
        ['The Beatles', 'John Lennon']
    """
    try:
        artists = json.loads(artists_str)
        if isinstance(artists, list):
            return [str(a) for a in artists]
        return []
    except (json.JSONDecodeError, TypeError):
        # Fallback: try literal_eval for Python-style lists
        try:
            import ast
            artists = ast.literal_eval(artists_str)
            if isinstance(artists, list):
                return [str(a) for a in artists]
        except (ValueError, SyntaxError):
            pass
        return []


def format_query(track_name: str, artists: List[str], max_artists: int = 3) -> str:
    """
    Format YouTube search query from track and artist names.
    
    Args:
        track_name: Song title
        artists: List of artist names
        max_artists: Maximum number of artists to include in query
        
    Returns:
        Formatted query string optimized for YouTube search
        
    Example:
        >>> format_query("Bohemian Rhapsody", ["Queen"], 3)
        'Bohemian Rhapsody Queen official audio'
    """
    # Take first N artists
    artist_part = " ".join(artists[:max_artists])
    
    # Combine with track name and add search hints
    query = f"{track_name} {artist_part} official audio"
    
    # Clean up: remove multiple spaces, special chars that break search
    query = re.sub(r'\s+', ' ', query).strip()
    query = re.sub(r'[^\w\s\-\']', '', query)  # Keep only alphanumeric, space, dash, apostrophe
    
    return query


def sanitize_filename(text: str, max_length: int = 100) -> str:
    """
    Create safe filename from arbitrary text.
    
    Args:
        text: Input text (e.g., song name, artist)
        max_length: Maximum filename length
        
    Returns:
        Filesystem-safe filename
        
    Example:
        >>> sanitize_filename("Song: Name/Path\\Test")
        'song_name_path_test'
    """
    # Convert to lowercase
    text = text.lower()
    
    # Replace problematic characters with underscores
    text = re.sub(r'[^\w\s\-]', '_', text)
    
    # Replace whitespace with underscores
    text = re.sub(r'\s+', '_', text)
    
    # Remove consecutive underscores
    text = re.sub(r'_+', '_', text)
    
    # Trim underscores from ends
    text = text.strip('_')
    
    # Limit length
    if len(text) > max_length:
        text = text[:max_length].rstrip('_')
    
    return text


def seconds_to_readable(seconds: float) -> str:
    """
    Convert seconds to human-readable duration.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string like "1h 23m 45s" or "23m 45s" or "45s"
        
    Example:
        >>> seconds_to_readable(3665)
        '1h 1m 5s'
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def estimate_remaining_time(completed: int, total: int, elapsed_seconds: float) -> str:
    """
    Estimate remaining time based on current progress.
    
    Args:
        completed: Number of items completed
        total: Total number of items
        elapsed_seconds: Time elapsed so far
        
    Returns:
        Estimated time remaining as readable string
        
    Example:
        >>> estimate_remaining_time(100, 1000, 3600)
        '9h 0m 0s'
    """
    if completed == 0:
        return "calculating..."
    
    avg_per_item = elapsed_seconds / completed
    remaining_items = total - completed
    remaining_seconds = avg_per_item * remaining_items
    
    return seconds_to_readable(remaining_seconds)
