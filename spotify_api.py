# spotify_api.py
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import time
import pandas as pd
from typing import List, Dict, Optional, Tuple
import streamlit as st
from tenacity import retry, stop_after_attempt, wait_exponential
import requests
from io import BytesIO
from PIL import Image
import base64

# ============================================================================
# SPOTIFY API AUTHENTICATION
# ============================================================================

@st.cache_resource(ttl=3600)
def get_spotify_client():
    """Initialize and return Spotify client with credentials from secrets"""
    try:
        client_id = st.secrets["SPOTIFY_CLIENT_ID"]
        client_secret = st.secrets["SPOTIFY_CLIENT_SECRET"]
        
        client_credentials_manager = SpotifyClientCredentials(
            client_id=client_id, 
            client_secret=client_secret
        )
        sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
        return sp
    except Exception as e:
        st.error(f"Failed to initialize Spotify client: {str(e)}")
        return None

# ============================================================================
# SEARCH FUNCTIONS
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
        
        # Search for album
        query = f"album:{clean_album} artist:{clean_artist}"
        results = sp.search(q=query, type='album', limit=1)
        
        if results['albums']['items']:
            album = results['albums']['items'][0]
            # Get the largest image available (usually 640x640)
            images = album.get('images', [])
            if images:
                # Images are typically ordered by size (largest first)
                return {
                    'url': images[0]['url'],
                    'width': images[0].get('width', 640),
                    'height': images[0].get('height', 640),
                    'spotify_id': album['id'],
                    'spotify_url': album['external_urls']['spotify']
                }
        
        # Try a more general search if specific search fails
        query = f"{clean_artist} {clean_album}"
        results = sp.search(q=query, type='album', limit=3)
        
        for album in results['albums']['items']:
            # Check if artist matches approximately
            album_artists = [a['name'].lower() for a in album['artists']]
            if clean_artist.lower() in ' '.join(album_artists):
                images = album.get('images', [])
                if images:
                    return {
                        'url': images[0]['url'],
                        'width': images[0].get('width', 640),
                        'height': images[0].get('height', 640),
                        'spotify_id': album['id'],
                        'spotify_url': album['external_urls']['spotify']
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
    st.success(f"Found covers for {successful}/{total} albums ({successful/total*100:.1f}%)")
    
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

# ============================================================================
# STREAMLIT DISPLAY FUNCTIONS
# ============================================================================

def display_album_grid(albums_df: pd.DataFrame, 
                       cover_col: str = 'cover_url',
                       title_col: str = 'album',
                       subtitle_col: str = 'artist',
                       plays_col: Optional[str] = None,
                       cols: int = 4):
    """
    Display albums in a grid with their cover images
    """
    if cover_col not in albums_df.columns:
        st.warning("No cover images available")
        return
    
    # Filter to albums with covers
    display_df = albums_df[albums_df[cover_col].notna()].copy()
    
    if len(display_df) == 0:
        st.warning("No album covers found")
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
                                st.image(img, use_container_width=True)
                            else:
                                st.markdown("🎵")
                        except:
                            st.markdown("🎵")
                    else:
                        st.markdown("🎵")
                    
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
    carousel_html = """
    <div style="overflow-x: auto; white-space: nowrap; padding: 10px 0;">
    """
    
    for _, album in display_df.iterrows():
        if album[cover_col]:
            try:
                img = download_image(album[cover_col], max_size=(200, 200))
                if img:
                    img_base64 = get_image_base64(img)
                    carousel_html += f"""
                    <div style="display: inline-block; margin-right: 15px; text-align: center; width: 150px;">
                        <img src="data:image/jpeg;base64,{img_base64}" 
                             style="width: 150px; height: 150px; object-fit: cover; border-radius: 8px;">
                        <div style="font-size: 12px; white-space: normal; word-wrap: break-word;">
                            <b>{album[title_col][:25]}</b><br>
                            {album[subtitle_col][:20]}
                        </div>
                    </div>
                    """
            except Exception as e:
                print(f"Error processing image: {e}")
    
    carousel_html += "</div>"
    
    st.markdown(carousel_html, unsafe_allow_html=True)

# ============================================================================
# INTEGRATION WITH MAIN APP
# ============================================================================

def get_albums_for_cover_search(aggregates, top_n: int = 100) -> pd.DataFrame:
    """
    Extract top albums from aggregates for cover searching
    """
    # Get top albums by play count (or time)
    top_albums = aggregates['top_albums_count'].head(top_n)
    
    # Create DataFrame with artist and album info
    albums_list = []
    for album_name, plays in top_albums.items():
        # Get a sample artist for this album (first occurrence in filtered data)
        artist = aggregates['filtered_df'][
            aggregates['filtered_df']['master_metadata_album_album_name'] == album_name
        ]['master_metadata_album_artist_name'].iloc[0]
        
        albums_list.append({
            'album': album_name,
            'artist': artist,
            'plays': plays
        })
    
    return pd.DataFrame(albums_list)
