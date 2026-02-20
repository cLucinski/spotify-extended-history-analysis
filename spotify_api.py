# spotify_api.py
import spotipy
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials
import time
import pandas as pd
from typing import List, Dict, Optional, Tuple
import streamlit as st
from tenacity import retry, stop_after_attempt, wait_exponential
import requests
from io import BytesIO
from PIL import Image
import base64
import webbrowser
from urllib.parse import urlparse, parse_qs
import socket
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import secrets
import hashlib
import os
import zipfile
from pathlib import Path
import re

# ============================================================================
# SPOTIFY API AUTHENTICATION
# ============================================================================

class SpotifyAuthHandler:
    """Handles Spotify OAuth authentication with proper redirect URI"""
    
    def __init__(self):
        self.redirect_uri = "http://127.0.0.1:8888/callback"
        self.auth_code = None
        self.server = None
        
    def get_auth_url(self, client_id: str, scope: str) -> str:
        """Generate the authorization URL"""
        # Generate a random state for security
        state = secrets.token_urlsafe(16)
        
        # Store state in session for validation
        st.session_state.spotify_auth_state = state
        
        auth_url = (
            f"https://accounts.spotify.com/authorize"
            f"?client_id={client_id}"
            f"&response_type=code"
            f"&redirect_uri={self.redirect_uri}"
            f"&scope={scope}"
            f"&state={state}"
        )
        return auth_url

@st.cache_resource(ttl=3600)
def get_spotify_client(use_oauth: bool = False):
    """
    Initialize and return Spotify client with credentials from secrets.
    
    Args:
        use_oauth: If True, uses OAuth for user-specific data.
                   If False, uses Client Credentials for public data.
    """
    try:
        client_id = st.secrets["SPOTIFY_CLIENT_ID"]
        client_secret = st.secrets["SPOTIFY_CLIENT_SECRET"]
        
        if use_oauth:
            # For user-specific data (like playlists, saved albums)
            # Using loopback IP as required by new Spotify rules
            redirect_uri = "http://127.0.0.1:8888/callback"
            
            # Check if there is a token in session
            token_info = st.session_state.get('spotify_token_info', None)
            
            # Define scope - only need to read album data
            scope = "user-library-read"
            
            sp = spotipy.Spotify(
                auth_manager=SpotifyOAuth(
                    client_id=client_id,
                    client_secret=client_secret,
                    redirect_uri=redirect_uri,
                    scope=scope,
                    cache_handler=spotipy.cache_handler.CacheFileHandler(
                        cache_path=".spotify_cache"
                    ),
                    open_browser=False,  # We'll handle browser opening ourselves
                ),
                auth=token_info.get('access_token') if token_info else None
            )
            
        else:
            # For public data (album searches) - simpler authentication
            client_credentials_manager = SpotifyClientCredentials(
                client_id=client_id, 
                client_secret=client_secret
            )
            sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
        
        return sp
        
    except Exception as e:
        st.error(f"Failed to initialize Spotify client: {str(e)}")
        return None

def show_auth_instructions():
    """Display instructions for setting up Spotify API access"""
    with st.expander("ℹ️ How to set up Spotify API access (Updated for 2025)", expanded=True):
        st.markdown("""
        ### 🔑 Setup Instructions
        
        1. **Create a Spotify App**:
           - Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
           - Click "Create App"
           - App name: "Spotify Album Art Viewer" (or any name)
           - App description: "View album covers from listening history"
           - **Important**: Set Redirect URI to: `http://127.0.0.1:8888/callback`
           - Check the box for "Web API" (required)
           - Click "Save"
        
        2. **Get Your Credentials**:
           - From your app dashboard, copy:
             - **Client ID** (shown on the app page)
             - **Client Secret** (click "Show" or "View")
        
        3. **Create secrets file**:
           Create a file at `.streamlit/secrets.toml` with:
           ```toml
           SPOTIFY_CLIENT_ID = "your_client_id_here"
           SPOTIFY_CLIENT_SECRET = "your_client_secret_here"
           ```
        
        4. **Note about Redirect URIs**:
           - The URI **must** match exactly: `http://127.0.0.1:8888/callback`
           - `localhost` is **not** allowed as per Spotify's 2025 requirements
           - The port (8888) can be any available port, but must match exactly
        """)

# ============================================================================
# SEARCH FUNCTIONS (Updated to use track URIs)
# ============================================================================

def extract_track_id_from_uri(track_uri: str) -> Optional[str]:
    """Extract the track ID from a Spotify track URI"""
    if not track_uri or not isinstance(track_uri, str):
        return None
    
    # Handle both formats: "spotify:track:6qj02zSeEJGWZ4c0dn5QzJ" and URLs
    if track_uri.startswith('spotify:track:'):
        return track_uri.split(':')[-1]
    elif 'track' in track_uri and 'spotify.com' in track_uri:
        # Handle potential URL format
        parts = track_uri.split('/')
        return parts[-1].split('?')[0]
    else:
        return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def search_album_cover(sp, artist_name: str, album_name: str, track_uri: str = None) -> Optional[Dict]:
    """
    Search for an album on Spotify and return cover image URL if found.
    Uses track_uri first if available, then falls back to text search.
    
    Args:
        sp: Spotify client
        artist_name: Name of the artist
        album_name: Name of the album
        track_uri: Spotify track URI (e.g., "spotify:track:6qj02zSeEJGWZ4c0dn5QzJ")
    
    Returns:
        Dictionary with cover image info or None if not found
    """
    if not sp:
        return None
    
    try:
        # METHOD 1: Use track URI if available (most accurate)
        if track_uri:
            track_id = extract_track_id_from_uri(track_uri)
            if track_id:
                try:
                    # Get track details from Spotify
                    track = sp.track(track_id)
                    
                    if track and 'album' in track:
                        album = track['album']
                        images = album.get('images', [])
                        
                        if images:
                            return {
                                'url': images[0]['url'],
                                'width': images[0].get('width', 640),
                                'height': images[0].get('height', 640),
                                'spotify_id': album['id'],
                                'spotify_url': album['external_urls']['spotify'],
                                'source': 'track_uri'
                            }
                except Exception as e:
                    # Log the error but continue to fallback methods
                    print(f"Track URI lookup failed for {track_uri}: {str(e)}")
        
        # METHOD 2: Fallback to text search (original method)
        # Clean up search query
        clean_album = album_name.split('(')[0].split('[')[0].strip() if album_name else ""
        clean_artist = artist_name.split('(')[0].split('[')[0].strip() if artist_name else ""
        
        # If have no artist or album name, can't proceed with fallback
        if not clean_album and not clean_artist:
            return None
        
        # First try: Exact album search with artist filter
        if clean_album and clean_artist:
            query = f"album:{clean_album} artist:{clean_artist}"
            results = sp.search(q=query, type='album', limit=10)
            
            # Check each result for exact artist match
            for album in results['albums']['items']:
                # Get all artists for this album
                album_artists = [a['name'].lower() for a in album['artists']]
                
                # Check if artist is the primary artist or listed as an artist
                if clean_artist.lower() in album_artists:
                    # Additional check: make sure it's not a various artists compilation
                    # unless artist is specifically the main artist
                    if len(album['artists']) == 1 or album_artists[0] == clean_artist.lower():
                        images = album.get('images', [])
                        if images:
                            return {
                                'url': images[0]['url'],
                                'width': images[0].get('width', 640),
                                'height': images[0].get('height', 640),
                                'spotify_id': album['id'],
                                'spotify_url': album['external_urls']['spotify'],
                                'source': 'text_search'
                            }
        
        # Second try: More specific query with quotes for exact matching
        if clean_album and clean_artist:
            query = f'album:"{clean_album}" artist:"{clean_artist}"'
            results = sp.search(q=query, type='album', limit=10)
            
            for album in results['albums']['items']:
                album_artists = [a['name'].lower() for a in album['artists']]
                if clean_artist.lower() in album_artists:
                    # Check if it's a tribute/cover album (usually has multiple artists or "tribute" in name)
                    album_name_lower = album['name'].lower()
                    if 'tribute' not in album_name_lower and 'cover' not in album_name_lower:
                        images = album.get('images', [])
                        if images:
                            return {
                                'url': images[0]['url'],
                                'width': images[0].get('width', 640),
                                'height': images[0].get('height', 640),
                                'spotify_id': album['id'],
                                'spotify_url': album['external_urls']['spotify'],
                                'source': 'text_search_quoted'
                            }
        
        # Third try: Search for the album and filter manually
        if clean_artist or clean_album:
            query = f"{clean_artist} {clean_album}".strip()
            results = sp.search(q=query, type='album', limit=20)
            
            # Score each result based on relevance
            best_match = None
            best_score = 0
            
            for album in results['albums']['items']:
                album_artists = [a['name'].lower() for a in album['artists']]
                album_name_lower = album['name'].lower()
                
                score = 0
                
                # Artist match (highest weight)
                if clean_artist and clean_artist.lower() in album_artists:
                    score += 100
                    # Extra points if it's the primary artist
                    if album_artists[0] == clean_artist.lower():
                        score += 50
                
                # Album name match
                if clean_album and clean_album.lower() in album_name_lower:
                    score += 30
                    # Exact match bonus
                    if clean_album.lower() == album_name_lower:
                        score += 20
                
                # Penalize tribute/cover albums
                if 'tribute' in album_name_lower or 'cover' in album_name_lower:
                    score -= 100
                
                # Penalize compilations with many artists (unless it's a known compilation)
                if len(album['artists']) > 3:
                    score -= 50
                
                # Check if this is likely a karaoke or instrumental version
                if 'karaoke' in album_name_lower or 'instrumental' in album_name_lower:
                    score -= 75
                
                if score > best_score:
                    best_score = score
                    best_match = album
            
            # Only return if decent match (score above threshold)
            if best_match and best_score > 100:
                images = best_match.get('images', [])
                if images:
                    return {
                        'url': images[0]['url'],
                        'width': images[0].get('width', 640),
                        'height': images[0].get('height', 640),
                        'spotify_id': best_match['id'],
                        'spotify_url': best_match['external_urls']['spotify'],
                        'source': 'text_search_scored'
                    }
        
        # No match found
        return None
        
    except Exception as e:
        print(f"Error searching for {artist_name} - {album_name}: {str(e)}")
        return None

def batch_search_album_covers(sp, albums_df: pd.DataFrame, 
                              artist_col: str = 'artist',
                              album_col: str = 'album',
                              track_uri_col: Optional[str] = 'track_uri',
                              batch_size: int = 5) -> pd.DataFrame:
    """
    Batch search for multiple album covers with rate limiting.
    Uses track URIs if available for more accurate results.
    
    Args:
        sp: Spotify client
        albums_df: DataFrame with artist and album columns
        artist_col: Name of column containing artist names
        album_col: Name of column containing album names
        track_uri_col: Name of column containing Spotify track URIs (optional)
        batch_size: Number of requests between pauses (rate limiting)
    
    Returns:
        DataFrame with added columns for cover URL and Spotify metadata
    """
    if sp is None:
        st.error("Spotify client not initialized")
        return albums_df
    
    results_df = albums_df.copy()
    results_df['cover_url'] = None
    results_df['spotify_album_id'] = None
    results_df['spotify_album_url'] = None
    results_df['cover_width'] = None
    results_df['cover_height'] = None
    results_df['cover_source'] = None  # Track whether used URI or text search
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total = len(results_df)
    successful = 0
    uri_success = 0
    
    for idx, row in results_df.iterrows():
        if idx % batch_size == 0 and idx > 0:
            # Rate limiting pause
            time.sleep(1)
        
        status_text.text(f"Searching {idx+1}/{total}: {row[artist_col]} - {row[album_col]}")
        
        # Get track URI if available
        track_uri = row.get(track_uri_col) if track_uri_col and track_uri_col in row else None
        
        result = search_album_cover(sp, row[artist_col], row[album_col], track_uri)
        
        if result:
            results_df.at[idx, 'cover_url'] = result['url']
            results_df.at[idx, 'spotify_album_id'] = result['spotify_id']
            results_df.at[idx, 'spotify_album_url'] = result['spotify_url']
            results_df.at[idx, 'cover_width'] = result['width']
            results_df.at[idx, 'cover_height'] = result['height']
            results_df.at[idx, 'cover_source'] = result.get('source', 'unknown')
            successful += 1
            
            if result.get('source') == 'track_uri':
                uri_success += 1
        
        progress_bar.progress((idx + 1) / total)
    
    progress_bar.empty()
    status_text.empty()
    
    # if successful > 0:
    #     st.success(f"Found covers for {successful}/{total} albums ({successful/total*100:.1f}%)")
    #     if uri_success > 0:
    #         st.info(f"📌 {uri_success} albums found using track URIs (most accurate)")
    
    return results_df

# ============================================================================
# IMAGE DOWNLOAD AND PROCESSING
# ============================================================================

@st.cache_data(ttl=86400)  # Cache for 24 hours
def download_image(url: str, max_size: Tuple[int, int] = (640, 640)) -> Optional[Image.Image]:
    """Download image from URL and return PIL Image object"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        
        # Resize if needed while maintaining aspect ratio
        if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        return img
    except Exception as e:
        print(f"Error downloading image from {url}: {str(e)}")
        return None

def get_image_base64(img: Image.Image, format: str = 'JPEG') -> str:
    """Convert PIL Image to base64 string for HTML display"""
    buffered = BytesIO()
    img.save(buffered, format=format, quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def get_default_image_base64():
    """Load the default music note PNG image and return as base64"""
    default_image_path = "assets/music-note-icon-grey.png"
    
    try:
        # Check if the file exists
        if os.path.exists(default_image_path):
            with open(default_image_path, "rb") as img_file:
                img_bytes = img_file.read()
                return base64.b64encode(img_bytes).decode()
        else:
            return None
    except Exception as e:
        print(f"Error loading default image: {e}")
        # Return a simple data URI as last resort
        return ""

# ============================================================================
# STREAMLIT DISPLAY FUNCTIONS
# ============================================================================

def display_album_grid(albums_df: pd.DataFrame, 
                       cover_col: str = 'cover_url',
                       title_col: str = 'album',
                       subtitle_col: str = 'artist',
                       plays_col: Optional[str] = None,
                       url_col: Optional[str] = 'spotify_album_url',
                       cols: int = 4):
    """
    Display albums in a grid with their cover images - shows full album and artist names
    """
    if len(albums_df) == 0:
        st.warning("No albums to display")
        return
    
    display_df = albums_df.copy()
    default_img_base64 = get_default_image_base64()
    
    # Determine the unit label based on the column name
    unit_label = "plays" if plays_col == 'plays' else "hours" if plays_col == 'hours' else ""
    
    # Create grid
    rows = len(display_df) // cols + (1 if len(display_df) % cols > 0 else 0)
    
    for row in range(rows):
        row_cols = st.columns(cols)
        for col in range(cols):
            idx = row * cols + col
            if idx < len(display_df):
                album = display_df.iloc[idx]
                with row_cols[col]:
                    # Display cover image (or default if none)
                    if album.get(cover_col) and pd.notna(album[cover_col]):
                        try:
                            # If there is a Spotify URL, make the image clickable
                            if url_col and url_col in album and album[url_col] and pd.notna(album[url_col]):
                                st.markdown(
                                    f"<a href='{album[url_col]}' target='_blank'>"
                                    f"<img src='{album[cover_col]}' style='width:100%; border-radius:8px;'>"
                                    f"</a>",
                                    unsafe_allow_html=True
                                )
                            else:
                                st.markdown(
                                    f"<img src='{album[cover_col]}' style='width:100%; border-radius:8px;'>",
                                    unsafe_allow_html=True
                                )
                        except Exception as e:
                            # Show default image on error
                            st.markdown(
                                f"<img src='data:image/png;base64,{default_img_base64}' style='width:100%; border-radius:8px;'>",
                                unsafe_allow_html=True
                            )
                    else:
                        # Show default image
                        st.markdown(
                            f"<img src='data:image/png;base64,{default_img_base64}' style='width:100%; border-radius:8px;'>",
                            unsafe_allow_html=True
                        )
                    
                    # Display album info with full names
                    album_title = str(album[title_col])
                    artist_name = str(album[subtitle_col])
                    
                    st.markdown(
                        f"<div style='word-wrap: break-word;'><strong>{album_title}</strong></div>",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f"<div style='word-wrap: break-word; color: #666; font-size: 0.9em;'>{artist_name}</div>",
                        unsafe_allow_html=True
                    )
                    
                    if plays_col and plays_col in album:
                        plays_value = album[plays_col]
                        if pd.notna(plays_value):
                            # Add the unit label after the number
                            st.caption(f"🎧 {plays_value} {unit_label}")

def display_album_carousel(albums_df: pd.DataFrame, 
                          cover_col: str = 'cover_url',
                          title_col: str = 'album',
                          subtitle_col: str = 'artist',
                          plays_col: Optional[str] = None,
                          url_col: Optional[str] = 'spotify_album_url',
                          height: int = 200):
    """
    Create a horizontal scrollable carousel of albums
    """
    # Show all albums, using default image for missing covers
    display_df = albums_df.head(50).copy()
    
    if len(display_df) == 0:
        st.warning("No albums to display")
        return
    
    # Determine the unit label based on the column name
    unit_label = "plays" if plays_col == 'plays' else "hours" if plays_col == 'hours' else ""
    
    # Create HTML for carousel
    carousel_items = []
    default_img_base64 = get_default_image_base64()
    
    for _, album in display_df.iterrows():
        # Get image
        img_html = ""
        if album.get(cover_col) and pd.notna(album[cover_col]):
            try:
                # Use URL directly in img tag
                img_html = f"<img src='{album[cover_col]}' style='width:150px; height:150px; object-fit:cover; border-radius:8px;'>"
            except Exception:
                # Fall back to default
                img_html = f"<img src='data:image/png;base64,{default_img_base64}' style='width:150px; height:150px; object-fit:cover; border-radius:8px;'>"
        else:
            # Default image
            img_html = f"<img src='data:image/png;base64,{default_img_base64}' style='width:150px; height:150px; object-fit:cover; border-radius:8px;'>"
        
        # Build the album info text
        title_text = str(album[title_col])
        subtitle_text = str(album[subtitle_col])
        
        plays_text = ""
        if plays_col and plays_col in album and pd.notna(album[plays_col]):
            # Add the unit label after the number
            plays_text = f"🎧 {album[plays_col]} {unit_label}"
        
        # If there is a Spotify URL, wrap the image in a link
        if url_col and url_col in album and album[url_col] and pd.notna(album[url_col]):
            item_html = f'''
            <div style="display: inline-block; margin-right: 15px; text-align: center; width: 150px; vertical-align: top;">
                <a href="{album[url_col]}" target="_blank">
                    {img_html}
                </a>
                <div style="font-size: 12px; margin-top: 5px; width: 150px;">
                    <div style="word-wrap: break-word; white-space: normal;"><strong>{title_text}</strong></div>
                    <div style="word-wrap: break-word; white-space: normal; color: #666;">{subtitle_text}</div>
                    <div style="color: #999; font-size: 11px; white-space: normal;">{plays_text}</div>
                </div>
            </div>
            '''
        else:
            item_html = f'''
            <div style="display: inline-block; margin-right: 15px; text-align: center; width: 150px; vertical-align: top;">
                {img_html}
                <div style="font-size: 12px; margin-top: 5px; width: 150px;">
                    <div style="word-wrap: break-word; white-space: normal;"><strong>{title_text}</strong></div>
                    <div style="word-wrap: break-word; white-space: normal; color: #666;">{subtitle_text}</div>
                    <div style="color: #999; font-size: 11px; white-space: normal;">{plays_text}</div>
                </div>
            </div>
            '''
        carousel_items.append(item_html)
    
    if carousel_items:
        # Join all items and wrap in a scrollable container
        carousel_html = f'''
        <div style="overflow-x: auto; overflow-y: hidden; white-space: nowrap; padding: 10px 0; width: 100%;">
            {''.join(carousel_items)}
        </div>
        '''
        
        st.html(carousel_html)

# ============================================================================
# INTEGRATION WITH MAIN APP
# ============================================================================

def get_albums_for_cover_search(aggregates, top_n: int = 100, ranking_method: str = 'Number of Plays') -> pd.DataFrame:
    """
    Extract top albums from aggregates for cover searching.
    Now includes track URIs from the original listening data and supports different ranking methods.
    
    Args:
        aggregates: The precomputed aggregates dictionary
        top_n: Number of top albums to return
        ranking_method: Either 'Number of Plays' or 'Total Playtime'
    
    Returns:
        DataFrame with album information
    """
    # Get top albums based on ranking method
    if ranking_method == 'Total Playtime':
        top_albums = aggregates['top_albums_time'].head(top_n)
        value_name = 'hours'
    else:  # Default to play count
        top_albums = aggregates['top_albums_count'].head(top_n)
        value_name = 'plays'
    
    # Create DataFrame with artist, album, and track URI info
    albums_list = []
    for album_name, value in top_albums.items():
        # Get all listens for this album to extract track URIs
        album_data = aggregates['filtered_df'][
            aggregates['filtered_df']['master_metadata_album_album_name'] == album_name
        ]
        
        if len(album_data) > 0:
            artist = album_data['master_metadata_album_artist_name'].iloc[0]
            
            # Get the most common track URI for this album (or the first one)
            track_uris = album_data['spotify_track_uri'].dropna()
            
            if len(track_uris) > 0:
                # Get the most frequent track URI
                track_uri = track_uris.mode()[0] if len(track_uris) > 0 else track_uris.iloc[0]
            else:
                track_uri = None
            
            # Store the value with appropriate column name
            album_entry = {
                'album': album_name,
                'artist': artist,
                'track_uri': track_uri,
                value_name: round(value, 1) if value_name == 'hours' else int(value)  # Round hours to 1 decimal
            }
            albums_list.append(album_entry)
    
    return pd.DataFrame(albums_list)

# Function for creating zip file with album images
def create_album_covers_zip(albums_df: pd.DataFrame, 
                           cover_col: str = 'cover_url',
                           title_col: str = 'album',
                           artist_col: str = 'artist') -> Optional[bytes]:
    """
    Download all album covers and create a zip file in memory.
    
    Args:
        albums_df: DataFrame with album information and cover URLs
        cover_col: Column name containing cover image URLs
        title_col: Column name containing album titles
        artist_col: Column name containing artist names
    
    Returns:
        BytesIO object containing the zip file, or None if no images to download
    """
    # Filter to albums with covers
    covers_df = albums_df[albums_df[cover_col].notna()]
    
    if len(covers_df) == 0:
        return None
    
    # Create a BytesIO object to store the zip file
    zip_buffer = BytesIO()
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        successful = 0
        total = len(covers_df)
        
        for idx, (_, album) in enumerate(covers_df.iterrows()):
            try:
                # Get album info
                album_title = str(album[title_col])
                artist_name = str(album[artist_col])
                cover_url = album[cover_col]
                
                status_text.text(f"Downloading {idx+1}/{total}: {album_title}")
                
                # Download the image
                response = requests.get(cover_url, timeout=10)
                response.raise_for_status()
                
                # Create a safe filename
                # Remove characters that are problematic in filenames
                safe_album = re.sub(r'[<>:"/\\|?*]', '', album_title)
                safe_artist = re.sub(r'[<>:"/\\|?*]', '', artist_name)
                
                # Truncate if too long (max 100 chars for filename)
                if len(safe_album) > 80:
                    safe_album = safe_album[:80]
                if len(safe_artist) > 20:
                    safe_artist = safe_artist[:20]
                
                # Create filename: "Artist - Album.jpg"
                filename = f"{safe_artist} - {safe_album}.jpg"
                
                # Write to zip
                zip_file.writestr(filename, response.content)
                successful += 1
                
            except Exception as e:
                print(f"Error downloading {album_title}: {str(e)}")
                continue
            
            # Update progress
            progress_bar.progress((idx + 1) / total)
    
    progress_bar.empty()
    status_text.empty()
    
    if successful == 0:
        return None
    
    st.success(f"Successfully downloaded {successful} album covers!")
    
    # Return the zip file as bytes
    zip_buffer.seek(0)
    return zip_buffer.getvalue()
