"""
Validation functions for YouTube audio matching.
Uses fuzzy string matching and duration verification.
"""
from typing import Dict, List, Any
from fuzzywuzzy import fuzz
import re


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison (lowercase, remove special chars).
    
    Args:
        text: Input text
        
    Returns:
        Normalized text for matching
    """
    text = text.lower()
    # Remove special characters but keep spaces
    text = re.sub(r'[^\w\s]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def fuzzy_title_match(csv_title: str, youtube_title: str) -> float:
    """
    Calculate fuzzy similarity between track title and YouTube video title.
    
    Uses token_sort_ratio which handles word order differences well.
    
    Args:
        csv_title: Track name from CSV dataset
        youtube_title: Video title from YouTube
        
    Returns:
        Similarity score 0-100 (100 = perfect match)
        
    Example:
        >>> fuzzy_title_match("Bohemian Rhapsody", "Queen - Bohemian Rhapsody (Official Video)")
        95
    """
    csv_norm = normalize_text(csv_title)
    yt_norm = normalize_text(youtube_title)
    
    # Use token_sort_ratio: handles word order differences
    # "Queen Bohemian Rhapsody" matches "Bohemian Rhapsody Queen"
    similarity = fuzz.token_sort_ratio(csv_norm, yt_norm)
    
    return float(similarity)


def duration_check(csv_duration_ms: int, youtube_duration_sec: float, tolerance_sec: int = 5) -> Dict[str, Any]:
    """
    Check if durations match within tolerance.
    
    Args:
        csv_duration_ms: Track duration in milliseconds from CSV
        youtube_duration_sec: Video duration in seconds from YouTube
        tolerance_sec: Allowed difference in seconds (default: 5)
        
    Returns:
        Dict with 'match' (bool), 'diff_seconds' (float), 'points' (int 0-30 or -1 for rejection)
        Returns -1 points for extreme mismatches to force rejection.
        
    Example:
        >>> duration_check(245000, 245.5, tolerance_sec=5)
        {'match': True, 'diff_seconds': 0.5, 'points': 30}
    """
    csv_duration_sec = csv_duration_ms / 1000.0
    diff = abs(csv_duration_sec - youtube_duration_sec)
    
    # Hard reject if difference > 60 seconds AND > 30% of expected duration
    max_allowed_diff = max(60, csv_duration_sec * 0.30)
    if diff > max_allowed_diff:
        return {
            'match': False,
            'diff_seconds': round(diff, 2),
            'points': -1  # Indicates rejection
        }
    
    # Score based on difference
    if diff <= tolerance_sec:
        points = 30
        match = True
    elif diff <= 15:
        points = 20
        match = False
    elif diff <= 30:
        points = 10
        match = False
    else:
        points = 0
        match = False
    
    return {
        'match': match,
        'diff_seconds': round(diff, 2),
        'points': points
    }


def artist_verification(csv_artists: List[str], youtube_title: str, youtube_uploader: str = "") -> Dict[str, Any]:
    """
    Verify artists appear in YouTube video metadata.
    
    Checks both video title and uploader/channel name.
    
    Args:
        csv_artists: List of artist names from CSV
        youtube_title: Video title from YouTube
        youtube_uploader: Channel/uploader name from YouTube (optional)
        
    Returns:
        Dict with 'matches' (int), 'total' (int), 'points' (int 0-30)
        
    Example:
        >>> artist_verification(["Queen", "David Bowie"], "Queen - Bohemian Rhapsody", "Queen Official")
        {'matches': 1, 'total': 2, 'points': 15}
    """
    if not csv_artists:
        return {'matches': 0, 'total': 0, 'points': 0}
    
    # Normalize text for matching
    yt_title_norm = normalize_text(youtube_title)
    yt_uploader_norm = normalize_text(youtube_uploader)
    
    matches = 0
    for artist in csv_artists:
        artist_norm = normalize_text(artist)
        
        # Check if artist appears in title or uploader
        if artist_norm in yt_title_norm or artist_norm in yt_uploader_norm:
            matches += 1
    
    # Calculate points: 30 if all artists match, proportional otherwise
    total = len(csv_artists)
    points = int((matches / total) * 30)
    
    return {
        'matches': matches,
        'total': total,
        'points': points
    }


def calculate_confidence_score(
    csv_row: Dict[str, Any],
    youtube_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculate overall confidence score for YouTube match.
    
    Multi-factor scoring:
    - Title similarity: 40 points (fuzzy match)
    - Duration match: 30 points (within tolerance)
    - Artist verification: 30 points (appears in metadata)
    Total: 0-100 scale
    
    Thresholds:
    - ≥80: High confidence (auto-download)
    - 60-79: Medium confidence (flag for review)
    - <60: Low confidence (skip)
    
    Args:
        csv_row: Song data from CSV with keys: 'name', 'artists', 'duration_ms'
        youtube_result: YouTube video data with keys: 'title', 'duration', 'uploader'
        
    Returns:
        Dict with detailed breakdown and overall score
        
    Example:
        >>> csv_row = {'name': 'Bohemian Rhapsody', 'artists': ['Queen'], 'duration_ms': 354000}
        >>> yt_result = {'title': 'Queen - Bohemian Rhapsody', 'duration': 354, 'uploader': 'Queen Official'}
        >>> result = calculate_confidence_score(csv_row, yt_result)
        >>> result['total_score']
        95
    """
    # Null checks to prevent NoneType errors
    csv_name = csv_row.get('name', '')
    yt_title = youtube_result.get('title', '')
    
    if not csv_name or not isinstance(csv_name, str):
        return {
            'total_score': 0.0,
            'confidence': 'low',
            'title_similarity': 0,
            'duration_match': False,
            'duration_diff': 0,
            'artist_matches': 0,
            'title_points': 0,
            'duration_points': 0,
            'artist_points': 0
        }
    
    if not yt_title or not isinstance(yt_title, str):
        return {
            'total_score': 0.0,
            'confidence': 'low',
            'title_similarity': 0,
            'duration_match': False,
            'duration_diff': 0,
            'artist_matches': 0,
            'title_points': 0,
            'duration_points': 0,
            'artist_points': 0
        }
    
    # 1. Title similarity (0-40 points)
    title_similarity = fuzzy_title_match(csv_name, yt_title)
    title_points = (title_similarity / 100.0) * 40
    
    # 2. Duration check (0-30 points, or -1 for rejection)
    duration_result = duration_check(
        csv_row['duration_ms'],
        youtube_result['duration']
    )
    duration_points = duration_result['points']
    
    # Hard rejection if duration is way off (-1 means reject)
    if duration_points == -1:
        return {
            'total_score': 0.0,
            'confidence': 'low',
            'title_similarity': title_similarity,
            'duration_match': False,
            'duration_diff': duration_result['diff_seconds'],
            'artist_matches': 0,
            'title_points': title_points,
            'duration_points': 0,
            'artist_points': 0,
            'rejected': 'duration_mismatch'
        }
    
    # 3. Artist verification (0-30 points)
    artist_result = artist_verification(
        csv_row['artists'],
        youtube_result['title'],
        youtube_result.get('uploader', '')
    )
    artist_points = artist_result['points']
    
    # Total score
    total_score = title_points + duration_points + artist_points
    
    # Confidence level
    if total_score >= 80:
        confidence = 'high'
    elif total_score >= 60:
        confidence = 'medium'
    else:
        confidence = 'low'
    
    return {
        'total_score': round(total_score, 2),
        'confidence': confidence,
        'title_similarity': round(title_similarity, 2),
        'title_points': round(title_points, 2),
        'duration_match': duration_result['match'],
        'duration_diff': duration_result['diff_seconds'],
        'duration_points': duration_points,
        'artist_matches': artist_result['matches'],
        'artist_total': artist_result['total'],
        'artist_points': artist_points
    }
