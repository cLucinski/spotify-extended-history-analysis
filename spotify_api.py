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
            
            # Check if we have a token in session
            token_info = st.session_state.get('spotify_token_info', None)
            
            # Define scope - we only need to read album data
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
# SEARCH FUNCTIONS (Same as before, using Client Credentials flow)
# ============================================================================

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def search_album_cover(sp, artist_name: str, album_name: str) -> Optional[Dict]:
    """
    Search for an album on Spotify and return cover image URL if found.
    Uses retry logic for handling rate limits.
    """
    if not sp:
        return None
    
    try:
        # Clean up search query
        # Remove common artifacts like "(Remastered)", "[Explicit]", etc.
        clean_album = album_name.split('(')[0].split('[')[0].strip()
        clean_artist = artist_name.split('(')[0].split('[')[0].strip()
        
        # First try: Exact album search with artist filter
        query = f"album:{clean_album} artist:{clean_artist}"
        results = sp.search(q=query, type='album', limit=10)
        
        # Check each result for exact artist match
        for album in results['albums']['items']:
            # Get all artists for this album
            album_artists = [a['name'].lower() for a in album['artists']]
            
            # Check if our artist is the primary artist or listed as an artist
            if clean_artist.lower() in album_artists:
                # Additional check: make sure it's not a various artists compilation
                # unless our artist is specifically the main artist
                if len(album['artists']) == 1 or album_artists[0] == clean_artist.lower():
                    images = album.get('images', [])
                    if images:
                        return {
                            'url': images[0]['url'],
                            'width': images[0].get('width', 640),
                            'height': images[0].get('height', 640),
                            'spotify_id': album['id'],
                            'spotify_url': album['external_urls']['spotify']
                        }
        
        # Second try: More specific query with quotes for exact matching
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
                            'spotify_url': album['external_urls']['spotify']
                        }
        
        # Third try: Search for the album and filter manually
        query = f"{clean_artist} {clean_album}"
        results = sp.search(q=query, type='album', limit=20)
        
        # Score each result based on relevance
        best_match = None
        best_score = 0
        
        for album in results['albums']['items']:
            album_artists = [a['name'].lower() for a in album['artists']]
            album_name_lower = album['name'].lower()
            
            score = 0
            
            # Artist match (highest weight)
            if clean_artist.lower() in album_artists:
                score += 100
                # Extra points if it's the primary artist
                if album_artists[0] == clean_artist.lower():
                    score += 50
            
            # Album name match
            if clean_album.lower() in album_name_lower:
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
        
        # Only return if we have a decent match (score above threshold)
        if best_match and best_score > 100:
            images = best_match.get('images', [])
            if images:
                return {
                    'url': images[0]['url'],
                    'width': images[0].get('width', 640),
                    'height': images[0].get('height', 640),
                    'spotify_id': best_match['id'],
                    'spotify_url': best_match['external_urls']['spotify']
                }
        
        return None
        
    except Exception as e:
        print(f"Error searching for {artist_name} - {album_name}: {str(e)}")
        return None

def batch_search_album_covers(sp, albums_df: pd.DataFrame, 
                              artist_col: str = 'artist',
                              album_col: str = 'album',
                              batch_size: int = 5) -> pd.DataFrame:
    """
    Batch search for multiple album covers with rate limiting.
    
    Args:
        sp: Spotify client
        albums_df: DataFrame with artist and album columns
        artist_col: Name of column containing artist names
        album_col: Name of column containing album names
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
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total = len(results_df)
    successful = 0
    
    for idx, row in results_df.iterrows():
        if idx % batch_size == 0 and idx > 0:
            # Rate limiting pause
            time.sleep(1)
        
        status_text.text(f"Searching {idx+1}/{total}: {row[artist_col]} - {row[album_col]}")
        
        result = search_album_cover(sp, row[artist_col], row[album_col])
        
        if result:
            results_df.at[idx, 'cover_url'] = result['url']
            results_df.at[idx, 'spotify_album_id'] = result['spotify_id']
            results_df.at[idx, 'spotify_album_url'] = result['spotify_url']
            results_df.at[idx, 'cover_width'] = result['width']
            results_df.at[idx, 'cover_height'] = result['height']
            successful += 1
        
        progress_bar.progress((idx + 1) / total)
    
    progress_bar.empty()
    status_text.empty()
    
    if successful > 0:
        st.success(f"Found covers for {successful}/{total} albums ({successful/total*100:.1f}%)")
    
    return results_df

# ============================================================================
# IMAGE DOWNLOAD AND PROCESSING (Same as before)
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

# ============================================================================
# STREAMLIT DISPLAY FUNCTIONS (Enhanced with clickable links)
# ============================================================================

def display_album_grid(albums_df: pd.DataFrame, 
                       cover_col: str = 'cover_url',
                       title_col: str = 'album',
                       subtitle_col: str = 'artist',
                       plays_col: Optional[str] = None,
                       url_col: Optional[str] = 'spotify_album_url',
                       cols: int = 4):
    """
    Display albums in a grid with their cover images
    """
    if cover_col not in albums_df.columns:
        st.warning("No cover images available")
        return
    
    # Use all albums, show default image for missing covers
    display_df = albums_df.copy()

    if len(display_df) == 0:
        st.warning("No albums to display")
        return
    
    # Create grid
    rows = len(display_df) // cols + (1 if len(display_df) % cols > 0 else 0)
    
    for row in range(rows):
        row_cols = st.columns(cols)
        for col in range(cols):
            idx = row * cols + col
            if idx < len(display_df):
                album = display_df.iloc[idx]
                with row_cols[col]:
                    # Display cover image
                    if album[cover_col]:
                        try:
                            img = download_image(album[cover_col])
                            if img:
                                # If we have a Spotify URL, make the image clickable
                                if url_col and url_col in album and album[url_col]:
                                    st.markdown(
                                        f"<a href='{album[url_col]}' target='_blank'>"
                                        f"<img src='{album[cover_col]}' style='width:100%; border-radius:8px;'>"
                                        f"</a>",
                                        unsafe_allow_html=True
                                    )
                                else:
                                    st.image(img, width='stretch')
                            else:
                                # Show default image
                                st.image("assets/music-note-icon-grey.png", width='stretch')
                        except Exception as e:
                            print(f"Error displaying image: {e}")
                            # Show default image
                            st.image("assets/music-note-icon-grey.png", width='stretch')
                    else:
                        # Show default image
                        st.image("assets/music-note-icon-grey.png", width='stretch')
                    
                    # Display album info
                    st.markdown(f"**{album[title_col][:30]}**")
                    st.caption(f"{album[subtitle_col][:20]}")
                    
                    if plays_col and plays_col in album:
                        st.caption(f"🎧 {album[plays_col]:.0f} plays")

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
    # Filter to albums with covers
    display_df = albums_df[albums_df[cover_col].notna()].head(50)
    
    if len(display_df) == 0:
        st.warning("No album covers found")
        return
    
    # Create HTML for carousel
    carousel_items = []
    
    for _, album in display_df.iterrows():
        if album[cover_col]:
            try:
                img = download_image(album[cover_col], max_size=(200, 200))
                if img:
                    img_base64 = get_image_base64(img)
                    
                    # Build the album info text
                    title_text = album[title_col][:25]
                    subtitle_text = album[subtitle_col][:20]
                    plays_text = f"🎧 {album[plays_col]:.0f} plays" if plays_col and plays_col in album else ""
                    
                    # If we have a Spotify URL, wrap in link
                    if url_col and url_col in album and album[url_col]:
                        item_html = f'''
                        <div style="display: inline-block; margin-right: 15px; text-align: center; width: 150px; vertical-align: top;">
                            <a href="{album[url_col]}" target="_blank">
                                <img src="data:image/jpeg;base64,{img_base64}" 
                                     style="width: 150px; height: 150px; object-fit: cover; border-radius: 8px; cursor: pointer;">
                            </a>
                            <div style="font-size: 12px; margin-top: 5px; white-space: normal; word-wrap: break-word;">
                                <strong>{title_text}</strong><br>
                                <span style="color: #666;">{subtitle_text}</span><br>
                                <span style="color: #999; font-size: 11px;">{plays_text}</span>
                            </div>
                        </div>
                        '''
                    else:
                        item_html = f'''
                        <div style="display: inline-block; margin-right: 15px; text-align: center; width: 150px; vertical-align: top;">
                            <img src="data:image/jpeg;base64,{img_base64}" 
                                 style="width: 150px; height: 150px; object-fit: cover; border-radius: 8px;">
                            <div style="font-size: 12px; margin-top: 5px; white-space: normal; word-wrap: break-word;">
                                <strong>{title_text}</strong><br>
                                <span style="color: #666;">{subtitle_text}</span><br>
                                <span style="color: #999; font-size: 11px;">{plays_text}</span>
                            </div>
                        </div>
                        '''
                    carousel_items.append(item_html)
            except Exception as e:
                print(f"Error processing image: {e}")
    
    if carousel_items:
        # Join all items and wrap in a scrollable container
        carousel_html = f'''
        <div style="overflow-x: auto; overflow-y: hidden; white-space: nowrap; padding: 10px 0; width: 100%;">
            {''.join(carousel_items)}
        </div>
        '''
        
        st.html(carousel_html)
    else:
        st.warning("No images could be processed for the carousel")

# ============================================================================
# INTEGRATION WITH MAIN APP
# ============================================================================

def get_albums_for_cover_search(aggregates, top_n: int = 100) -> pd.DataFrame:
    """
    Extract top albums from aggregates for cover searching
    """
    # Get top albums by play count
    top_albums = aggregates['top_albums_count'].head(top_n)
    
    # Create DataFrame with artist and album info
    albums_list = []
    for album_name, plays in top_albums.items():
        # Get a sample artist for this album (first occurrence in filtered data)
        album_data = aggregates['filtered_df'][
            aggregates['filtered_df']['master_metadata_album_album_name'] == album_name
        ]
        
        if len(album_data) > 0:
            artist = album_data['master_metadata_album_artist_name'].iloc[0]
            
            albums_list.append({
                'album': album_name,
                'artist': artist,
                'plays': plays
            })
    
    return pd.DataFrame(albums_list)
