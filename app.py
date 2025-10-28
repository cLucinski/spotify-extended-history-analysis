# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import io
import gc

# Configure Streamlit
st.set_page_config(page_title="Spotify Analysis", layout="wide")

# ============================================================================
# CHART FUNCTIONS
# ============================================================================

def create_timeline_chart(aggregates, frequency, analysis_type):
    """Create timeline chart based on selected frequency and analysis type"""
    if frequency == 'Daily':
        data = aggregates['daily_listens'].copy()
        data['date_dt'] = pd.to_datetime(data['date'])
        
        if analysis_type == 'Total Playtime':
            y_col = 'total_hours'
            title = 'Daily Listening Time'
            y_label = 'Hours Played'
        else:
            y_col = 'count'
            title = 'Daily Listening Activity'
            y_label = 'Number of Plays'
        
        fig = px.line(data, x='date_dt', y=y_col, 
                     title=title,
                     labels={'date_dt': 'Date', y_col: y_label})
        
    else:  # Monthly
        data = aggregates['monthly_listens'].copy()
        
        if analysis_type == 'Total Playtime':
            y_col = 'total_hours'
            title = 'Monthly Listening Time'
            y_label = 'Hours Played'
        else:
            y_col = 'count'
            title = 'Monthly Listening Activity'
            y_label = 'Number of Plays'
            
        fig = px.bar(data, x='month', y=y_col,
                    title=title,
                    labels={'month': 'Month', y_col: y_label})
    
    fig.update_layout(
        hovermode='x unified',
        showlegend=False,
        height=400
    )
    return fig

def create_top_artists_chart(aggregates, top_n, analysis_type):
    """Create horizontal bar chart for top artists"""
    if analysis_type == 'Total Playtime':
        top_artists = aggregates['top_artists_time'].head(top_n)
        x_values = top_artists.values
        title = f'Top {top_n} Artists by Playtime'
        x_label = 'Hours Played'
        color_scale = 'viridis'
    else:
        top_artists = aggregates['top_artists_count'].head(top_n)
        x_values = top_artists.values
        title = f'Top {top_n} Artists by Plays'
        x_label = 'Number of Plays'
        color_scale = 'plasma'
    
    fig = px.bar(
        x=x_values, 
        y=top_artists.index,
        orientation='h',
        title=title,
        labels={'x': x_label, 'y': 'Artist'},
        color=x_values,
        color_continuous_scale=color_scale
    )
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False,
        height=500
    )
    return fig

def create_top_songs_chart(aggregates, top_n, analysis_type):
    """Create horizontal bar chart for top songs"""
    if analysis_type == 'Total Playtime':
        top_songs = aggregates['top_songs_time'].head(top_n).copy()
        x_col = 'total_hours'
        title = f'Top {top_n} Songs by Playtime'
        x_label = 'Hours Played'
        color_scale = 'viridis'
    else:
        top_songs = aggregates['top_songs_count'].head(top_n).copy()
        x_col = 'count'
        title = f'Top {top_n} Songs by Plays'
        x_label = 'Number of Plays'
        color_scale = 'plasma'
    
    top_songs['song_artist'] = top_songs['master_metadata_track_name'] + ' - ' + top_songs['master_metadata_album_artist_name']
    
    fig = px.bar(
        top_songs, 
        x=x_col, 
        y='song_artist',
        orientation='h',
        title=title,
        labels={x_col: x_label, 'song_artist': 'Song'},
        color=x_col,
        color_continuous_scale=color_scale
    )
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False,
        height=600
    )
    return fig

def create_artist_timeline_chart(df, top_artists, top_n, frequency):
    """Show listening timeline for top artists"""
    top_artist_names = top_artists.head(top_n).index
    
    if frequency == 'Daily':
        artist_timeline = df[df['master_metadata_album_artist_name'].isin(top_artist_names)].groupby(
            ['date_dt', 'master_metadata_album_artist_name']
        ).size().reset_index(name='count')
        x_col = 'date_dt'
    else:
        artist_timeline = df[df['master_metadata_album_artist_name'].isin(top_artist_names)].groupby(
            ['month', 'master_metadata_album_artist_name']
        ).size().reset_index(name='count')
        artist_timeline['month'] = artist_timeline['month'].astype(str)
        x_col = 'month'
    
    fig = px.line(
        artist_timeline, 
        x=x_col, 
        y='count',
        color='master_metadata_album_artist_name',
        title=f'Listening Timeline for Top {top_n} Artists'
    )
    fig.update_layout(height=500, showlegend=True)
    return fig

# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

@st.cache_data(show_spinner=False, ttl=3600)
def load_and_process_chunk(uploaded_file, min_seconds=30):
    """Load and process a single file chunk with minimum play time filter"""
    try:
        content = uploaded_file.getvalue()
        df = pd.read_json(io.BytesIO(content))
        
        df['ts'] = pd.to_datetime(df['ts'])
        df['date'] = df['ts'].dt.date
        df['date_dt'] = pd.to_datetime(df['date'])
        df['month'] = df['ts'].dt.to_period('M')
        
        # Convert ms_played to seconds and apply filter
        df['seconds_played'] = df['ms_played'] / 1000
        initial_count = len(df)
        df = df[df['seconds_played'] >= min_seconds]
        filtered_count = initial_count - len(df)
        
        # Filter out podcasts
        music_df = df[df['master_metadata_track_name'].notna()].copy()
        
        # Store filtering stats
        music_df.attrs['filtered_short_plays'] = filtered_count
        return music_df
    except Exception as e:
        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=3600)
@st.cache_data(show_spinner=False, ttl=3600)
def load_data_optimized(uploaded_files, min_seconds=30):
    """Optimized data loading with minimum play time filter"""
    if not uploaded_files:
        return pd.DataFrame()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_dfs = []
    total_files = len(uploaded_files)
    total_filtered = 0
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {uploaded_file.name} ({i+1}/{total_files})")
        df_chunk = load_and_process_chunk(uploaded_file, min_seconds)
        if not df_chunk.empty:
            all_dfs.append(df_chunk)
            total_filtered += df_chunk.attrs.get('filtered_short_plays', 0)
        progress_bar.progress((i + 1) / total_files)
    
    status_text.text("Combining data...")
    
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df = combined_df.sort_values('ts').reset_index(drop=True)
        
        # Show filtering statistics
        if total_filtered > 0:
            st.sidebar.info(f"Filtered out {total_filtered:,} tracks under {min_seconds}s")
        
        status_text.text("Data loaded successfully!")
        progress_bar.empty()
        status_text.empty()
        
        return combined_df
    else:
        progress_bar.empty()
        status_text.empty()
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def precompute_aggregates(_df, date_range=None, artist_filter=None, min_seconds=30):
    """Precompute aggregates for both play count and playtime"""
    filtered_df = _df.copy()
    
    # Apply minimum seconds filter
    filtered_df = filtered_df[filtered_df['ms_played'] >= (min_seconds * 1000)]
    
    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['date_dt'] >= pd.to_datetime(start_date)) & 
            (filtered_df['date_dt'] <= pd.to_datetime(end_date))
        ]
    
    if artist_filter:
        filtered_df = filtered_df[filtered_df['master_metadata_album_artist_name'].isin(artist_filter)]
    
    # Calculate playtime in hours for easier reading
    filtered_df['hours_played'] = filtered_df['ms_played'] / (1000 * 60 * 60)
    filtered_df['minutes_played'] = filtered_df['ms_played'] / (1000 * 60)
    
    # Daily aggregates (both count and playtime)
    daily_plays = filtered_df.groupby('date_dt').size().reset_index(name='count')
    daily_playtime = filtered_df.groupby('date_dt')['hours_played'].sum().reset_index(name='total_hours')
    daily_listens = daily_plays.merge(daily_playtime, on='date_dt')
    daily_listens = daily_listens.rename(columns={'date_dt': 'date'})
    
    # Monthly aggregates (both count and playtime)
    monthly_plays = filtered_df.groupby('month').size().reset_index(name='count')
    monthly_playtime = filtered_df.groupby('month')['hours_played'].sum().reset_index(name='total_hours')
    monthly_listens = monthly_plays.merge(monthly_playtime, on='month')
    monthly_listens['month'] = monthly_listens['month'].astype(str)
    
    # Top artists (both count and playtime)
    top_artists_count = filtered_df['master_metadata_album_artist_name'].value_counts()
    top_artists_time = filtered_df.groupby('master_metadata_album_artist_name')['hours_played'].sum().sort_values(ascending=False)
    
    # Top songs (both count and playtime)
    top_songs_count = (filtered_df.groupby(['master_metadata_track_name', 'master_metadata_album_artist_name'])
                      .size()
                      .reset_index(name='count')
                      .sort_values('count', ascending=False))
    
    top_songs_time = (filtered_df.groupby(['master_metadata_track_name', 'master_metadata_album_artist_name'])
                     ['hours_played'].sum()
                     .reset_index(name='total_hours')
                     .sort_values('total_hours', ascending=False))
    
    return {
        'daily_listens': daily_listens,
        'monthly_listens': monthly_listens,
        'top_artists_count': top_artists_count,
        'top_artists_time': top_artists_time,
        'top_songs_count': top_songs_count,
        'top_songs_time': top_songs_time,
        'filtered_df': filtered_df
    }

def clear_cache_and_reload():
    """Clear cache and force reload"""
    st.cache_data.clear()
    gc.collect()
    st.rerun()

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.title("🎵 Spotify Listening History Analyzer")
    
    # File upload section
    st.sidebar.header("1. Upload Your Data")
    
    uploaded_files = st.sidebar.file_uploader(
        "Select Spotify JSON files",
        type=['json'],
        accept_multiple_files=True,
        help="Upload all your Streaming_History_Audio_*.json files"
    )
    
    # Add minimum seconds filter in sidebar
    st.sidebar.header("2. Data Filters")
    min_seconds = st.sidebar.slider(
        "Minimum play time (seconds)",
        min_value=0,
        max_value=120,
        value=30,
        help="Filter out tracks played for less than this many seconds"
    )
    
    if uploaded_files:
        total_size = sum(file.size for file in uploaded_files) / (1024 * 1024)
        st.sidebar.info(f"Selected {len(uploaded_files)} files ({total_size:.1f} MB)")
        
        if st.sidebar.button("Load and Process Data", type="primary"):
            with st.spinner("Loading and processing your Spotify data..."):
                # Pass min_seconds to the loading function
                df = load_data_optimized(uploaded_files, min_seconds)
            
            if df.empty:
                st.error("No data could be loaded. Please check your files.")
                return
            
            st.session_state.df = df
            st.session_state.data_loaded = True
            st.session_state.min_seconds = min_seconds
            
            # Show filtering info
            st.success(f"✅ Successfully loaded {len(df):,} listening records (≥{min_seconds}s plays)!")
            
            # Quick summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Date Range", f"{df['date'].min()} to {df['date'].max()}")
            with col2:
                st.metric("Total Plays", f"{len(df):,}")
            with col3:
                st.metric("Unique Artists", f"{df['master_metadata_album_artist_name'].nunique():,}")
    
    # Check if data is loaded
    if not st.session_state.get('data_loaded', False):
        st.info("""
        ## 📁 How to use this analyzer:
        1. **Export your Spotify data** from [Spotify's Privacy Settings](https://www.spotify.com/us/account/privacy/)
        2. **Wait for email** (usually takes a few days)
        3. **Upload all your** `Streaming_History_Audio_*.json` files using the file browser on the left
        4. **Click "Load and Process Data"** to begin analysis
        """)
        return
    
    # Get data from session state
    df = st.session_state.df
    
    # Filters sidebar
    st.sidebar.header("2. Analysis Filters")
    
    min_date = df['date'].min()
    max_date = df['date'].max()
    
    date_range = st.sidebar.date_input(
        "Date Range",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    all_artists = df['master_metadata_album_artist_name'].unique()
    selected_artists = st.sidebar.multiselect(
        "Filter by Artists",
        options=sorted(all_artists),
        help="Select specific artists or leave empty for all"
    )
    
    # Analysis parameters
    st.sidebar.header("3. Chart Settings")
    top_n = st.sidebar.slider("Number of Top Items", 5, 50, 15)
    timeline_freq = st.sidebar.radio("Timeline Frequency", ['Daily', 'Monthly'])

    analysis_type = st.sidebar.radio(
        "Analysis Type", 
        ['Number of Plays', 'Total Playtime'],
        help="Show charts based on play count or total time played"
    )
    
    # Clear cache button
    st.sidebar.header("4. System")
    if st.sidebar.button("Clear Cache & Reload"):
        clear_cache_and_reload()
    
    # Precompute aggregates
    aggregates = precompute_aggregates(df, date_range, selected_artists if selected_artists else None)
    
    # ============================================================================
    # CHART DISPLAY SECTION
    # ============================================================================
    
    # Display charts in tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Timeline", "🎤 Top Artists", "🎵 Top Songs", "👨‍🎤 Artist Timeline", "📊 Summary"
    ])
    
    with tab1:
        st.subheader("Listening Timeline")
        timeline_fig = create_timeline_chart(aggregates, timeline_freq, analysis_type)
        st.plotly_chart(timeline_fig, use_container_width=True)
        
        # Update stats based on analysis type
        col1, col2, col3 = st.columns(3)
        with col1:
            if analysis_type == 'Total Playtime':
                total_value = aggregates['filtered_df']['hours_played'].sum()
                st.metric("Total Playtime", f"{total_value:.1f} hours")
            else:
                total_value = len(aggregates['filtered_df'])
                st.metric("Total Plays", f"{total_value:,}")
        
        with col2:
            if analysis_type == 'Total Playtime':
                avg_value = aggregates['daily_listens']['total_hours'].mean()
                st.metric("Average Per Day", f"{avg_value:.1f} hours")
            else:
                avg_value = aggregates['daily_listens']['count'].mean()
                st.metric("Average Plays Per Day", f"{avg_value:.1f}")
        
        with col3:
            if analysis_type == 'Total Playtime':
                peak_data = aggregates['daily_listens'].loc[aggregates['daily_listens']['total_hours'].idxmax()]
                st.metric("Peak Listening Day", f"{peak_data['total_hours']:.1f} hours")
            else:
                peak_data = aggregates['daily_listens'].loc[aggregates['daily_listens']['count'].idxmax()]
                st.metric("Peak Listening Day", f"{peak_data['count']} plays")

    with tab2:
        st.subheader(f"Top {top_n} Artists")
        artists_fig = create_top_artists_chart(aggregates, top_n, analysis_type)
        st.plotly_chart(artists_fig, use_container_width=True)
        
        # Artist statistics
        col1, col2 = st.columns(2)
        with col1:
            if analysis_type == 'Total Playtime':
                top_artist = aggregates['top_artists_time'].index[0]
                top_artist_value = aggregates['top_artists_time'].iloc[0]
                st.metric("Top Artist", f"{top_artist} ({top_artist_value:.1f} hours)")
            else:
                top_artist = aggregates['top_artists_count'].index[0]
                top_artist_value = aggregates['top_artists_count'].iloc[0]
                st.metric("Top Artist", f"{top_artist} ({top_artist_value:,} plays)")
        
        with col2:
            unique_artists = aggregates['filtered_df']['master_metadata_album_artist_name'].nunique()
            st.metric("Unique Artists", f"{unique_artists:,}")

    with tab3:
        st.subheader(f"Top {top_n} Songs")
        songs_fig = create_top_songs_chart(aggregates, top_n, analysis_type)
        st.plotly_chart(songs_fig, use_container_width=True)
    
    with tab4:
        st.subheader(f"Artist Listening Timeline")
        artist_timeline_fig = create_artist_timeline_chart(
            aggregates['filtered_df'], 
            aggregates['top_artists_count'], 
            min(top_n, 10),  # Limit to top 10 for clarity
            timeline_freq
        )
        st.plotly_chart(artist_timeline_fig, use_container_width=True)
    
    with tab5:
        st.subheader("Data Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Filter Summary**")
            st.metric("Total Plays in Filter", f"{len(aggregates['filtered_df']):,}")
            
            # ADD THIS LINE for Total Playtime:
            total_hours = aggregates['filtered_df']['hours_played'].sum()
            st.metric("Total Playtime in Filter", f"{total_hours:.1f} hours")
            
            st.metric("Date Range", f"{aggregates['filtered_df']['date'].min()} to {aggregates['filtered_df']['date'].max()}")
            st.metric("Filtered Artists", f"{aggregates['filtered_df']['master_metadata_album_artist_name'].nunique():,}")
        
        with col2:
            st.write("**Overall Statistics**")
            st.metric("Total Listening Days", f"{aggregates['daily_listens']['date'].nunique():,}")
            st.metric("Most Active Month", f"{aggregates['monthly_listens'].loc[aggregates['monthly_listens']['count'].idxmax(), 'month']}")
            st.metric("Total Unique Songs", f"{aggregates['filtered_df']['master_metadata_track_name'].nunique():,}")

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

if __name__ == "__main__":
    main()
