# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import io
import gc
from spotify_api import (
    get_spotify_client, batch_search_album_covers, 
    display_album_grid, display_album_carousel, get_albums_for_cover_search
)

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
        
        fig = px.bar(data, x='date_dt', y=y_col, 
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
    """Create horizontal bar chart for top artists with rank-based coloring"""
    if analysis_type == 'Total Playtime':
        top_artists = aggregates['top_artists_time'].head(top_n)
        title = f'Top {top_n} Artists by Playtime'
        x_label = 'Hours Played'
    else:
        top_artists = aggregates['top_artists_count'].head(top_n)
        title = f'Top {top_n} Artists by Plays'
        x_label = 'Number of Plays'
    
    # Convert Series to DataFrame for ranking
    chart_df = pd.DataFrame({
        'artist': top_artists.index,
        'value': top_artists.values
    })
    
    # Create ranking column with "min" method to handle ties
    chart_df['rank'] = chart_df['value'].rank(method="min", ascending=False).astype(int)
    
    fig = px.bar(
        chart_df,
        x='value', 
        y='artist',
        orientation='h',
        title=title,
        labels={'value': x_label, 'artist': 'Artist'},
        color='rank',
        color_continuous_scale='plasma_r',
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False,
        coloraxis_showscale=False,
        height=600,
        # Add grid lines for better readability
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        )
    )
    
    # # Customize color scale to make ranking more visible
    # fig.update_coloraxes(
    #     # colorbar_title="Rank",
    #     # colorscale="plasma_r",  # Dark colors for lower ranks, bright for top ranks
    #     showscale=False
    # )
    
    return fig

def create_top_songs_chart(aggregates, top_n, analysis_type):
    """Create horizontal bar chart for top songs with rank-based coloring"""
    if analysis_type == 'Total Playtime':
        top_songs = aggregates['top_songs_time'].head(top_n).copy()
        x_col = 'total_hours'
        title = f'Top {top_n} Songs by Playtime'
        x_label = 'Hours Played'
    else:
        top_songs = aggregates['top_songs_count'].head(top_n).copy()
        x_col = 'count'
        title = f'Top {top_n} Songs by Plays'
        x_label = 'Number of Plays'
    
    top_songs['song_artist'] = top_songs['master_metadata_track_name'] + ' - ' + top_songs['master_metadata_album_artist_name']
    
    # Create ranking column with "min" method to handle ties
    top_songs['rank'] = top_songs[x_col].rank(method="min", ascending=False).astype(int)
    
    fig = px.bar(
        top_songs, 
        x=x_col, 
        y='song_artist',
        orientation='h',
        title=title,
        labels={x_col: x_label, 'song_artist': 'Song'},
        color='rank',
        color_continuous_scale='plasma_r'
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False,
        coloraxis_showscale=False,
        height=600,
        # Add grid lines for better readability
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        )
    )
    
    # Customize color scale
    # fig.update_coloraxes(
    #     # colorbar_title="Rank",
    #     # colorscale="plasma_r",
    #     showscale=False
    # )
    
    return fig


def create_top_albums_chart(aggregates, top_n, analysis_type):
    """Create horizontal bar chart for top albums with rank-based coloring"""
    if analysis_type == 'Total Playtime':
        top_albums = aggregates['top_albums_time'].head(top_n)
        title = f'Top {top_n} Albums by Playtime'
        x_label = 'Hours Played'
    else:
        top_albums = aggregates['top_albums_count'].head(top_n)
        title = f'Top {top_n} Albums by Plays'
        x_label = 'Number of Plays'
    
    # Convert Series to DataFrame for ranking
    chart_df = pd.DataFrame({
        'album': top_albums.index,
        'value': top_albums.values
    })
    
    # Create ranking column with "min" method to handle ties
    chart_df['rank'] = chart_df['value'].rank(method="min", ascending=False).astype(int)
    
    fig = px.bar(
        chart_df,
        x='value', 
        y='album',
        orientation='h',
        title=title,
        labels={'value': x_label, 'album': 'Album'},
        color='rank',
        color_continuous_scale='plasma_r',
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False,
        coloraxis_showscale=False,
        height=600,
        # Add grid lines for better readability
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        )
    )
    
    return fig

def create_top_podcasts_chart(top_shows: pd.Series, top_n: int = 20):
    if top_shows.empty:
        return None

    df = (
        top_shows
        .head(top_n)
        .reset_index()
        .rename(columns={
            'episode_show_name': 'Podcast',
            0: 'hours_played'
        })
    )

    df['rank'] = df['hours_played'].rank(method='min', ascending=False).astype(int)

    fig = px.bar(
        df,
        x='hours_played',
        y='Podcast',
        color='rank',
        color_continuous_scale='plasma_r',
        orientation='h',
        title=f"Top {min(top_n, len(top_shows))} Podcasts by Listening Time",
        labels={
            'hours_played': 'Hours Played'
        }
    )

    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False,
        coloraxis_showscale=False,
        height=600,
        # Add grid lines for better readability
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        )
    )

    # Remove colour scale
    # fig.update_coloraxes(showscale=False)

    return fig

# TODO: Fix hover labels
# TODO: Improve hover info
# TODO: Create an individual song/artist/album analysis page

def create_cumulative_timeline_chart(aggregates, frequency, analysis_type):
    """Create cumulative timeline chart based on selected frequency and analysis type"""
    if frequency == 'Daily':
        data = aggregates['daily_listens'].copy()
        data['date_dt'] = pd.to_datetime(data['date'])
        
        if analysis_type == 'Total Playtime':
            y_col = 'total_hours'
            title = 'Cumulative Listening Time'
            y_label = 'Cumulative Hours Played'
        else:
            y_col = 'count'
            title = 'Cumulative Listening Activity'
            y_label = 'Cumulative Number of Plays'
        
        # Calculate cumulative sum
        data['cumulative'] = data[y_col].cumsum()
        
        fig = px.line(data, x='date_dt', y='cumulative', 
                     title=title,
                     labels={'date_dt': 'Date', 'cumulative': y_label})
        
    else:  # Monthly
        data = aggregates['monthly_listens'].copy()
        
        if analysis_type == 'Total Playtime':
            y_col = 'total_hours'
            title = 'Cumulative Monthly Listening Time'
            y_label = 'Cumulative Hours Played'
        else:
            y_col = 'count'
            title = 'Cumulative Monthly Listening Activity'
            y_label = 'Cumulative Number of Plays'
            
        # Calculate cumulative sum
        data['cumulative'] = data[y_col].cumsum()
        
        fig = px.line(data, x='month', y='cumulative',
                    title=title,
                    labels={'month': 'Month', 'cumulative': y_label})
    
    fig.update_layout(
        hovermode='x unified',
        showlegend=False,
        height=600
    )
    return fig

def create_cumulative_artist_chart(df, top_artists, top_n, frequency, analysis_type):
    """Create cumulative timeline for top artists, ordered by total"""
    top_artist_names = top_artists.head(top_n).index
    
    # Get the filtered data for top artists
    artist_df = df[df['master_metadata_album_artist_name'].isin(top_artist_names)].copy()
    
    if frequency == 'Daily':
        if analysis_type == 'Total Playtime':
            # Group by date and artist, sum playtime
            daily_data = artist_df.groupby(['date_dt', 'master_metadata_album_artist_name'])['hours_played'].sum().reset_index()
        else:
            # Group by date and artist, count plays
            daily_data = artist_df.groupby(['date_dt', 'master_metadata_album_artist_name']).size().reset_index(name='count')
        
        # Create cumulative sum for each artist
        cumulative_data = []
        for artist in top_artist_names:
            artist_data = daily_data[daily_data['master_metadata_album_artist_name'] == artist].sort_values('date_dt')
            if analysis_type == 'Total Playtime':
                artist_data['cumulative'] = artist_data['hours_played'].cumsum()
            else:
                artist_data['cumulative'] = artist_data['count'].cumsum()
            cumulative_data.append(artist_data)
        
        cumulative_df = pd.concat(cumulative_data, ignore_index=True)
        x_col = 'date_dt'
        
    else:  # Monthly
        if analysis_type == 'Total Playtime':
            # Group by month and artist, sum playtime
            monthly_data = artist_df.groupby(['month', 'master_metadata_album_artist_name'])['hours_played'].sum().reset_index()
        else:
            # Group by month and artist, count plays
            monthly_data = artist_df.groupby(['month', 'master_metadata_album_artist_name']).size().reset_index(name='count')
        
        monthly_data['month'] = monthly_data['month'].astype(str)
        
        # Create cumulative sum for each artist
        cumulative_data = []
        for artist in top_artist_names:
            artist_data = monthly_data[monthly_data['master_metadata_album_artist_name'] == artist].sort_values('month')
            if analysis_type == 'Total Playtime':
                artist_data['cumulative'] = artist_data['hours_played'].cumsum()
            else:
                artist_data['cumulative'] = artist_data['count'].cumsum()
            cumulative_data.append(artist_data)
        
        cumulative_df = pd.concat(cumulative_data, ignore_index=True)
        x_col = 'month'
    
    # Calculate final totals for legend ordering
    if analysis_type == 'Total Playtime':
        final_totals = cumulative_df.groupby('master_metadata_album_artist_name')['cumulative'].max().sort_values(ascending=False)
    else:
        # For play count, we need to get the actual total counts from the original data
        if analysis_type == 'Total Playtime':
            final_totals = artist_df.groupby('master_metadata_album_artist_name')['hours_played'].sum().sort_values(ascending=False)
        else:
            final_totals = artist_df['master_metadata_album_artist_name'].value_counts()
    
    # Create ordered category for legend
    legend_order = final_totals.index.tolist()
    
    fig = px.line(
        cumulative_df, 
        x=x_col, 
        y='cumulative',
        color='master_metadata_album_artist_name',
        title=f'Cumulative Listening for Top {top_n} Artists',
        labels={'date_dt': "Date", 
                'cumulative': 'Cumulative Hours Played' if analysis_type == 'Total Playtime' else 'Cumulative Number of Plays',
                'master_metadata_album_artist_name': 'Artist'},
        category_orders={"master_metadata_album_artist_name": legend_order}
    )
    
    fig.update_layout(
        height=600, 
        showlegend=True,
        legend=dict(
            traceorder='normal'  # This ensures the order is respected
        )
    )
    return fig

def create_cumulative_song_chart(df, top_songs_data, top_n, frequency, analysis_type):
    """Create cumulative timeline for top songs, ordered by total"""
    # Get top songs and create combined identifier
    top_songs = top_songs_data.head(top_n).copy()
    top_songs['song_artist'] = top_songs['master_metadata_track_name'] + ' - ' + top_songs['master_metadata_album_artist_name']
    top_song_identifiers = top_songs['song_artist'].tolist()
    
    # Create the same identifier in the main dataframe
    df['song_artist'] = df['master_metadata_track_name'] + ' - ' + df['master_metadata_album_artist_name']
    song_df = df[df['song_artist'].isin(top_song_identifiers)].copy()
    
    if frequency == 'Daily':
        if analysis_type == 'Total Playtime':
            daily_data = song_df.groupby(['date_dt', 'song_artist'])['hours_played'].sum().reset_index()
        else:
            daily_data = song_df.groupby(['date_dt', 'song_artist']).size().reset_index(name='count')
        
        # Create cumulative sum for each song
        cumulative_data = []
        for song in top_song_identifiers:
            song_data = daily_data[daily_data['song_artist'] == song].sort_values('date_dt')
            if analysis_type == 'Total Playtime':
                song_data['cumulative'] = song_data['hours_played'].cumsum()
            else:
                song_data['cumulative'] = song_data['count'].cumsum()
            cumulative_data.append(song_data)
        
        cumulative_df = pd.concat(cumulative_data, ignore_index=True)
        x_col = 'date_dt'
        
    else:  # Monthly
        if analysis_type == 'Total Playtime':
            monthly_data = song_df.groupby(['month', 'song_artist'])['hours_played'].sum().reset_index()
        else:
            monthly_data = song_df.groupby(['month', 'song_artist']).size().reset_index(name='count')
        
        monthly_data['month'] = monthly_data['month'].astype(str)
        
        # Create cumulative sum for each song
        cumulative_data = []
        for song in top_song_identifiers:
            song_data = monthly_data[monthly_data['song_artist'] == song].sort_values('month')
            if analysis_type == 'Total Playtime':
                song_data['cumulative'] = song_data['hours_played'].cumsum()
            else:
                song_data['cumulative'] = song_data['count'].cumsum()
            cumulative_data.append(song_data)
        
        cumulative_df = pd.concat(cumulative_data, ignore_index=True)
        x_col = 'month'
    
    # Calculate final totals for legend ordering
    if analysis_type == 'Total Playtime':
        final_totals = cumulative_df.groupby('song_artist')['cumulative'].max().sort_values(ascending=False)
    else:
        final_totals = song_df['song_artist'].value_counts()
    
    legend_order = final_totals.index.tolist()
    
    fig = px.line(
        cumulative_df, 
        x=x_col, 
        y='cumulative',
        color='song_artist',
        title=f'Cumulative Listening for Top {top_n} Songs',
        labels={'date_dt': "Date", 
                'cumulative': 'Cumulative Hours Played' if analysis_type == 'Total Playtime' else 'Cumulative Number of Plays',
                'song_artist': 'Song - Artist'},
        category_orders={"song_artist": legend_order}
    )
    
    fig.update_layout(
        height=600, 
        showlegend=True,
        legend=dict(
            traceorder='normal'
        )
    )
    return fig

def create_cumulative_album_chart(df, top_albums, top_n, frequency, analysis_type):
    """Create cumulative timeline for top albums, ordered by total"""
    top_album_names = top_albums.head(top_n).index
    
    album_df = df[df['master_metadata_album_album_name'].isin(top_album_names)].copy()
    
    if frequency == 'Daily':
        if analysis_type == 'Total Playtime':
            daily_data = album_df.groupby(['date_dt', 'master_metadata_album_album_name'])['hours_played'].sum().reset_index()
        else:
            daily_data = album_df.groupby(['date_dt', 'master_metadata_album_album_name']).size().reset_index(name='count')
        
        # Create cumulative sum for each album
        cumulative_data = []
        for album in top_album_names:
            album_data = daily_data[daily_data['master_metadata_album_album_name'] == album].sort_values('date_dt')
            if analysis_type == 'Total Playtime':
                album_data['cumulative'] = album_data['hours_played'].cumsum()
            else:
                album_data['cumulative'] = album_data['count'].cumsum()
            cumulative_data.append(album_data)
        
        cumulative_df = pd.concat(cumulative_data, ignore_index=True)
        x_col = 'date_dt'
        
    else:  # Monthly
        if analysis_type == 'Total Playtime':
            monthly_data = album_df.groupby(['month', 'master_metadata_album_album_name'])['hours_played'].sum().reset_index()
        else:
            monthly_data = album_df.groupby(['month', 'master_metadata_album_album_name']).size().reset_index(name='count')
        
        monthly_data['month'] = monthly_data['month'].astype(str)
        
        # Create cumulative sum for each album
        cumulative_data = []
        for album in top_album_names:
            album_data = monthly_data[monthly_data['master_metadata_album_album_name'] == album].sort_values('month')
            if analysis_type == 'Total Playtime':
                album_data['cumulative'] = album_data['hours_played'].cumsum()
            else:
                album_data['cumulative'] = album_data['count'].cumsum()
            cumulative_data.append(album_data)
        
        cumulative_df = pd.concat(cumulative_data, ignore_index=True)
        x_col = 'month'
    
    # Calculate final totals for legend ordering
    if analysis_type == 'Total Playtime':
        final_totals = cumulative_df.groupby('master_metadata_album_album_name')['cumulative'].max().sort_values(ascending=False)
    else:
        final_totals = album_df['master_metadata_album_album_name'].value_counts()
    
    legend_order = final_totals.index.tolist()
    
    fig = px.line(
        cumulative_df, 
        x=x_col, 
        y='cumulative',
        color='master_metadata_album_album_name',
        title=f'Cumulative Listening for Top {top_n} Albums',
        labels={'date_dt': "Date", 
                'cumulative': 'Cumulative Hours Played' if analysis_type == 'Total Playtime' else 'Cumulative Number of Plays',
                'master_metadata_album_album_name': 'Album'},
        category_orders={"master_metadata_album_album_name": legend_order}
    )
    
    fig.update_layout(
        height=600, 
        showlegend=True,
        legend=dict(
            traceorder='normal'
        )
    )
    return fig

def create_cumulative_podcasts_chart(
    podcast_df: pd.DataFrame,
    top_n: int,
    timeline_freq: str
):
    if podcast_df.empty:
        return None

    df = podcast_df.copy()

    # Compute total listening per podcast (descending)
    podcast_totals = (
        df.groupby('episode_show_name')['hours_played']
        .sum()
        .sort_values(ascending=False)
    )

    top_podcasts = podcast_totals.head(top_n).index.tolist()

    df = df[df['episode_show_name'].isin(top_podcasts)]

    # Time bucketing
    if timeline_freq == "Daily":
        df['period'] = df['date_dt'].dt.date
    elif timeline_freq == "Weekly":
        df['period'] = df['date_dt'].dt.to_period('W').dt.start_time
    elif timeline_freq == "Monthly":
        df['period'] = df['date_dt'].dt.to_period('M').dt.start_time

    # Aggregate per podcast per period
    grouped = (
        df.groupby(['episode_show_name', 'period'])['hours_played']
        .sum()
        .reset_index()
        .sort_values('period')
    )

    # Compute cumulative sum per podcast
    grouped['cumulative_hours'] = (
        grouped
        .groupby('episode_show_name')['hours_played']
        .cumsum()
    )

    fig = px.line(
        grouped,
        x='period',
        y='cumulative_hours',
        color='episode_show_name',
        title=f"Cumulative Podcast Listening — Top {top_n} Podcasts",
        labels={
            'period': 'Date',
            'cumulative_hours': 'Cumulative Hours',
            'episode_show_name': 'Podcast'
        },
        category_orders={
            'episode_show_name': top_podcasts
        }
    )

    fig.update_layout(
        height=600
    )

    return fig


def create_cumulative_total_podcasts_chart(podcast_df: pd.DataFrame, timeline_freq: str):
    if podcast_df.empty:
        return None

    df = podcast_df.copy()

    if timeline_freq == "Daily":
        df['period'] = df['date_dt'].dt.date
    elif timeline_freq == "Monthly":
        df['period'] = df['date_dt'].dt.to_period('M').dt.start_time

    grouped = (
        df.groupby('period')['hours_played']
        .sum()
        .reset_index()
        .sort_values('period')
    )

    grouped['cumulative_hours'] = grouped['hours_played'].cumsum()

    fig = px.line(
        grouped,
        x='period',
        y='cumulative_hours',
        title="Cumulative Podcast Listening Over Time",
        labels={
            'period': 'Date',
            'cumulative_hours': 'Cumulative Hours'
        }
    )

    fig.update_traces(mode='lines')

    fig.update_layout(
        height=400,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig

def create_yearly_cumulative_comparison(
    df,
    timeline_freq,
    analysis_type,
    normalize=False
):
    """Compare cumulative listening across years on a normalized timeline"""

    data = df.copy()
    data['year'] = data['date_dt'].dt.year

    if timeline_freq == 'Daily':
        # Day-of-year alignment
        data['time_index'] = data['date_dt'].dt.dayofyear
        data = data[data['time_index'] <= 365]

        x_label = 'Day of Year'
    else:
        # Month-of-year alignment
        data['time_index'] = data['date_dt'].dt.month
        x_label = 'Month'

    if analysis_type == 'Total Playtime':
        grouped = (
            data.groupby(['year', 'time_index'])['hours_played']
            .sum()
            .reset_index(name='value')
        )
        base_y_label = 'Cumulative Hours Played'
        title = 'Year-over-Year Cumulative Listening (Hours)'
    else:
        grouped = (
            data.groupby(['year', 'time_index'])
            .size()
            .reset_index(name='value')
        )
        base_y_label = 'Cumulative Plays'
        title = 'Year-over-Year Cumulative Listening (Plays)'

    # Sort and cumulative sum per year
    grouped = grouped.sort_values(['year', 'time_index'])
    grouped['cumulative'] = grouped.groupby('year')['value'].cumsum()

    if normalize:
        # Normalize to percent of final year total
        grouped['final_total'] = grouped.groupby('year')['cumulative'].transform('max')
        grouped['cumulative'] = (
            grouped['cumulative'] / grouped['final_total'] * 100
        )
        y_label = '% of Year Total'
        title += ' — Normalized'
    else:
        y_label = base_y_label

    fig = px.line(
        grouped,
        x='time_index',
        y='cumulative',
        color='year',
        title=title,
        labels={
            'time_index': x_label,
            'cumulative': y_label,
            'year': 'Year'
        }
    )

    if timeline_freq == 'Monthly':
        fig.update_xaxes(
            tickmode='array',
            tickvals=list(range(1, 13)),
            ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        )

    fig.update_layout(
        height=600,
        hovermode='x unified'
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
    
    fig = px.bar(
        artist_timeline, 
        x=x_col, 
        y='count',
        color='master_metadata_album_artist_name',
        title=f'Listening Timeline for Top {top_n} Artists'
    )
    fig.update_layout(height=600, showlegend=True)
    return fig

# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

# TODO: Figure out timezone stuff

@st.cache_data(show_spinner=False, ttl=3600)
def load_and_process_chunk(uploaded_file, min_seconds=30):
    """Load and process a single file chunk"""
    try:
        content = uploaded_file.getvalue()
        df = pd.read_json(io.BytesIO(content))
        
        # Convert to datetime and remove timezone info
        df['ts'] = pd.to_datetime(df['ts']).dt.tz_localize(None)
        
        df['date'] = df['ts'].dt.date
        df['date_dt'] = pd.to_datetime(df['date'])
        df['month'] = df['ts'].dt.to_period('M')
        
        # Convert ms_played to seconds and filter
        df['seconds_played'] = df['ms_played'] / 1000
        df = df[df['seconds_played'] >= min_seconds]
        
        # Identify content type
        df['content_type'] = np.where(
            df['master_metadata_track_name'].notna(),
            'Music',
            'Podcast'
        )
        return df

    except Exception as e:
        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        return pd.DataFrame()

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

    # Keep music only for existing analysis
    filtered_df = filtered_df[filtered_df['content_type'] == 'Music']
    
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
    
    # Top albums (both count and playtime)
    top_albums_count = filtered_df['master_metadata_album_album_name'].value_counts()
    top_albums_time = filtered_df.groupby('master_metadata_album_album_name')['hours_played'].sum().sort_values(ascending=False)
    
    return {
        'daily_listens': daily_listens,
        'monthly_listens': monthly_listens,
        'top_artists_count': top_artists_count,
        'top_artists_time': top_artists_time,
        'top_songs_count': top_songs_count,
        'top_songs_time': top_songs_time,
        'top_albums_count': top_albums_count,
        'top_albums_time': top_albums_time,
        'filtered_df': filtered_df
    }

@st.cache_data(show_spinner=False)
def compute_podcast_aggregates(df, date_range=None, min_seconds=30):
    """Aggregate podcast listening data"""

    podcast_df = df[df['content_type'] == 'Podcast'].copy()

    if podcast_df.empty:
        return None

    podcast_df = podcast_df[podcast_df['ms_played'] >= (min_seconds * 1000)]
    podcast_df['hours_played'] = podcast_df['ms_played'] / (1000 * 60 * 60)

    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        podcast_df = podcast_df[
            (podcast_df['date_dt'] >= pd.to_datetime(start_date)) &
            (podcast_df['date_dt'] <= pd.to_datetime(end_date))
        ]

    # Daily aggregates
    daily = (
        podcast_df.groupby('date_dt')
        .agg(
            count=('ms_played', 'size'),
            total_hours=('hours_played', 'sum')
        )
        .reset_index()
    )

    # Top podcast shows (by playtime)
    top_shows = (
        podcast_df.groupby('episode_show_name')['hours_played']
        .sum()
        .sort_values(ascending=False)
    )

    return {
        'df': podcast_df,
        'daily': daily,
        'top_shows': top_shows,
    }

def clear_cache_and_reload():
    """Clear cache, session state, and force full reload"""
    # Clear all cached data
    st.cache_data.clear()
    
    # Clear ALL session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    # Force garbage collection
    gc.collect()
    
    # Rerun the app from scratch
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

    current_files_signature = (
        tuple((f.name, f.size) for f in uploaded_files)
        if uploaded_files else None
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

    # Mark data stale if inputs changed
    if (
        st.session_state.last_uploaded_files != current_files_signature
        or st.session_state.last_min_seconds != min_seconds
    ):
        st.session_state.data_stale = True



    if uploaded_files:
        total_size = sum(file.size for file in uploaded_files) / (1024 * 1024)
        st.sidebar.info(f"Selected {len(uploaded_files)} files ({total_size:.1f} MB)")

        load_button = st.sidebar.button(
            "Load and Process Data",
            type="primary" if st.session_state.data_stale else "secondary"
        )

        if load_button:
            with st.spinner("Loading and processing your Spotify data..."):
                df = load_data_optimized(uploaded_files, min_seconds)

            if df.empty:
                st.error("No data could be loaded. Please check your files.")
                return

            st.session_state.df = df
            st.session_state.data_loaded = True
            st.session_state.min_seconds = min_seconds

            # Persist comparison values
            st.session_state.last_uploaded_files = current_files_signature
            st.session_state.last_min_seconds = min_seconds
            st.session_state.data_stale = False

            st.rerun()

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

    # Detect changes in uploaded files
    current_files_signature = (
        tuple((f.name, f.size) for f in uploaded_files) if uploaded_files else None
    )

    if (st.session_state.last_uploaded_files != current_files_signature
        or st.session_state.last_min_seconds != min_seconds
    ):
        st.session_state.data_stale = True

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

    all_artists = (
        df.loc[
            (df['content_type'] == 'Music') &
            (df['master_metadata_album_artist_name'].notna()),
            'master_metadata_album_artist_name'
        ]
        .unique()
    )
    if len(all_artists) == 0:
        st.sidebar.warning("No music artists found in current filter.")

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

    # Get min_seconds from session state
    current_min_seconds = st.session_state.get('min_seconds', 30)

    # Precompute aggregates
    aggregates = precompute_aggregates(
        df, 
        date_range, 
        selected_artists if selected_artists else None,
        current_min_seconds)

    # ============================================================================
    # CHART DISPLAY SECTION
    # ============================================================================

    # Display charts in tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Timeline", "🎤 Top Artists", "🎵 Top Songs",
        "💽 Top Albums", "👨‍🎤 Artist Timeline",
        "🎙 Podcasts", "📋 Summary", "🎨 Album Art"
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

        # Cumulative overall timeline
        cumulative_fig = create_cumulative_timeline_chart(aggregates, timeline_freq, analysis_type)
        st.plotly_chart(cumulative_fig, use_container_width=True)

        # Year-over-Year comparison
        st.subheader("Year-over-Year Cumulative Comparison")

        normalize_yoy = st.checkbox(
            "Normalize Year-over-Year to % of Year Total",
            help="Compare how quickly listening accumulated within each year"
        )

        yoy_fig = create_yearly_cumulative_comparison(
            aggregates['filtered_df'],
            timeline_freq,
            analysis_type,
            normalize_yoy
        )
        st.plotly_chart(yoy_fig, use_container_width=True)

    with tab2:
        st.subheader(f"Top {top_n} Artists")
        artists_fig = create_top_artists_chart(aggregates, top_n, analysis_type)
        st.plotly_chart(artists_fig, use_container_width=True)

        # Cumulative artists chart
        # st.subheader(f"Top {min(top_n, 15)} Artists")
        artist_cumulative_fig = create_cumulative_artist_chart(
            aggregates['filtered_df'],
            aggregates['top_artists_count' if analysis_type == 'Number of Plays' else 'top_artists_time'],
            min(top_n, 15),  # Limit for clarity
            timeline_freq,
            analysis_type
        )
        st.plotly_chart(artist_cumulative_fig, use_container_width=True)

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

        # Cumulative songs chart
        # st.subheader(f"Top {min(top_n, 15)} Songs")
        song_cumulative_fig = create_cumulative_song_chart(
            aggregates['filtered_df'],
            aggregates['top_songs_count' if analysis_type == 'Number of Plays' else 'top_songs_time'],
            min(top_n, 15),  # Limit for clarity
            timeline_freq,
            analysis_type
        )
        st.plotly_chart(song_cumulative_fig, use_container_width=True)

    with tab4:
        st.subheader(f"Top {top_n} Albums")
        albums_fig = create_top_albums_chart(aggregates, top_n, analysis_type)
        st.plotly_chart(albums_fig, use_container_width=True)

        # Cumulative albums chart
        # st.subheader(f"Top {min(top_n, 15)} Albums")
        album_cumulative_fig = create_cumulative_album_chart(
            aggregates['filtered_df'],
            aggregates['top_albums_count' if analysis_type == 'Number of Plays' else 'top_albums_time'],
            min(top_n, 15),  # Limit for clarity
            timeline_freq,
            analysis_type
        )
        st.plotly_chart(album_cumulative_fig, use_container_width=True)

    with tab5:
        st.subheader(f"Artist Listening Timeline")
        artist_timeline_fig = create_artist_timeline_chart(
            aggregates['filtered_df'], 
            aggregates['top_artists_count'], 
            min(top_n, 10),  # Limit to top 10 for clarity
            timeline_freq
        )
        st.plotly_chart(artist_timeline_fig, use_container_width=True)

    with tab6:
        st.subheader("🎙 Podcast Listening")

        podcast_data = compute_podcast_aggregates(
            st.session_state.df,
            date_range,
            current_min_seconds
        )

        if not podcast_data:
            st.info("No podcast listening data found.")
        else:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Total Podcast Plays",
                    f"{len(podcast_data['df']):,}"
                )

            with col2:
                st.metric(
                    "Total Listening Time",
                    f"{podcast_data['df']['hours_played'].sum():.1f} hrs"
                )

            with col3:
                st.metric(
                    "Unique Podcasts",
                    f"{podcast_data['df']['episode_show_name'].nunique():,}"
                )

            st.subheader("Cumulative Podcast Listening")
            cum_fig = create_cumulative_total_podcasts_chart(
                podcast_data['df'],
                timeline_freq
            )
            if cum_fig:
                st.plotly_chart(cum_fig, use_container_width=True)

            st.subheader("Top Podcasts")
            top_fig = create_top_podcasts_chart(
                podcast_data['top_shows'],
                top_n=top_n
            )
            if top_fig:
                st.plotly_chart(top_fig, use_container_width=True)

            cum_podcast_fig = create_cumulative_podcasts_chart(
                podcast_data['df'],
                top_n=top_n,
                timeline_freq=timeline_freq
            )
            if cum_podcast_fig:
                st.plotly_chart(cum_podcast_fig, use_container_width=True)

    with tab7:
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
    with tab8:
        st.subheader("🎨 Album Cover Gallery")
        
        # Instructions for Spotify API setup
        with st.expander("ℹ️ How to set up Spotify API access"):
            st.markdown("""
            1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
            2. Click "Create App"
            3. Set any name and description
            4. Add `http://localhost:8501` to Redirect URIs
            5. Copy Client ID and Client Secret
            6. Add them to `.streamlit/secrets.toml` file:
            ```toml
            SPOTIFY_CLIENT_ID = "your_client_id"
            SPOTIFY_CLIENT_SECRET = "your_client_secret"
            ```
            """)
        
        # Initialize Spotify client
        sp = get_spotify_client()
        
        if sp is None:
            st.error("""
            ⚠️ Spotify API credentials not configured.
            
            Please set up your credentials in `.streamlit/secrets.toml` to enable album art.
            See the instructions above.
            """)
        else:
            st.success("✅ Spotify API connected!")
            
            # Get top albums for cover search
            cover_top_n = st.slider(
                "Number of top albums to search for",
                min_value=10,
                max_value=200,
                value=50,
                step=10
            )
            
            # Get albums data
            albums_df = get_albums_for_cover_search(aggregates, top_n=cover_top_n)
            
            col1, col2 = st.columns(2)
            
            with col1:
                search_button = st.button("🔍 Search for Album Covers", type="primary")
            
            with col2:
                display_mode = st.radio("Display Mode", ["Grid", "Carousel"])
            
            if search_button:
                with st.spinner("Searching Spotify for album covers..."):
                    albums_with_covers = batch_search_album_covers(
                        sp, 
                        albums_df,
                        artist_col='artist',
                        album_col='album',
                        batch_size=5
                    )
                    st.session_state.albums_with_covers = albums_with_covers
            
            # Display results if we have them
            if 'albums_with_covers' in st.session_state:
                if display_mode == "Grid":
                    display_album_grid(
                        st.session_state.albums_with_covers,
                        cover_col='cover_url',
                        title_col='album',
                        subtitle_col='artist',
                        plays_col='plays',
                        cols=4
                    )
                else:
                    display_album_carousel(
                        st.session_state.albums_with_covers,
                        cover_col='cover_url',
                        title_col='album',
                        subtitle_col='artist',
                        plays_col='plays',
                        height=200
                    )
                
                # Show statistics
                found_covers = st.session_state.albums_with_covers['cover_url'].notna().sum()
                st.info(f"Found {found_covers} album covers out of {len(st.session_state.albums_with_covers)} searched")

# Initialize session state variables if they don't exist
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None

if 'data_stale' not in st.session_state:
    st.session_state.data_stale = True
if 'last_min_seconds' not in st.session_state:
    st.session_state.last_min_seconds = None
if 'last_uploaded_files' not in st.session_state:
    st.session_state.last_uploaded_files = None


if __name__ == "__main__":
    main()
