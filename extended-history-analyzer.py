import argparse
import glob
import json
import logging
import os
import numpy as np
import pandas as pd
import plotly.express as px
from typing import List, Dict, Any, Set, Tuple

# Global configuration dictionary
global_config = {
    "aggregate_by_album": False,  # Controls if playtime is aggregated by album for 'master_metadata_album_artist_name'. Set to 'True' to enable album-level aggregation.
    "dark_mode": False  # Controls if the chart will have a dark or light background.
}

parser = argparse.ArgumentParser(description='Analyze Spotify streaming history data.')
parser.add_argument('-u', '--user',
                    help='The user whose data to analyze',
                    default='chris',
                    required=False)
parser.add_argument('-v', '--verbose',
                    help='Increase output verbosity',
                    action='store_true',
                    default=False)
parser.add_argument('--dark-mode',
                    help='Enable dark mode for the chart',
                    action='store_true')  # Adds a boolean flag for dark mode

def load_files(file_pattern: str) -> List[Dict[str, Any]]:
    """Loads and combines JSON data from files matching a pattern."""
    combined_data = []
    file_list = sorted(glob.glob(file_pattern))
    for file_path in file_list:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            combined_data.extend(data)
    return combined_data

def convert_to_dataframe(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Converts raw data into a pandas DataFrame."""
    if not data:
        raise ValueError("Input data is empty.")
    return pd.DataFrame(data)

def extract_insights(data: pd.DataFrame) -> Tuple[Set[str], Set[str], float]:
    """Extract unique tracks, artists, and total playback time from a DataFrame."""
    if "master_metadata_track_name" not in data.columns or "master_metadata_album_artist_name" not in data.columns or "ms_played" not in data.columns:
        raise ValueError("Required columns are missing in the data.")
    
    unique_tracks = set(data["master_metadata_track_name"].dropna().unique())
    unique_artists = set(data["master_metadata_album_artist_name"].dropna().unique())
    total_playback_time_sec = data["ms_played"].sum() / 1000  # Convert milliseconds to seconds

    return unique_tracks, unique_artists, total_playback_time_sec

def write_results(output_file: str, data: pd.DataFrame, unique_tracks: Set[str], unique_artists: Set[str], total_playback_time_sec: float):
    """Writes analysis results to a file."""
    output_lines = [
        f"Total track entries read: {len(data)}",
        f"Total unique tracks: {len(unique_tracks)}",
        f"Total unique artists: {len(unique_artists)}",
        f"Total playback time (seconds): {total_playback_time_sec:.2f}",
        f"Total playback time (minutes): {(total_playback_time_sec / 60):.2f}",
        f"Total playback time (hours): {(total_playback_time_sec / (60 * 60)):.2f}",
        f"Total playback time (days): {(total_playback_time_sec / (60 * 60 * 24)):.2f}",
        "\nUnique Tracks:"
    ] + [f"- {track}" for track in sorted(unique_tracks)] + [
        "\nUnique Artists:"
    ] + [f"- {artist}" for artist in sorted(unique_artists)]

    with open(output_file, "w", encoding="utf-8") as file:
        file.write("\n".join(output_lines))

    logging.info(f"Analysis results written to {output_file}")

def filter_by_date_range(df: pd.DataFrame, date_range: Tuple[str, str]) -> pd.DataFrame:
    """
    Filters a DataFrame to include only rows within a specified date range.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        date_range (Tuple[str, str]): A tuple of start and end dates in 'YYYY-MM-DD' format.
        
    Returns:
        pd.DataFrame: A DataFrame containing only rows within the specified date range.
    
    Raises:
        ValueError: If required columns are missing or if the filtered DataFrame is empty.
    """
    if "ts" not in df.columns:
        raise ValueError("The DataFrame must contain a 'ts' column with timestamps.")

    start_date, end_date = date_range
    
    # Create a copy to avoid modifying the original DataFrame
    copy_df = df.copy()
    
    try:
        # Ensure the 'ts' column is in datetime format
        copy_df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize(None)
        # Strip ["ts"] coloumn of time because we don't need that here
        copy_df["ts"] = copy_df["ts"].dt.date

        # Convert strings to dates for proper comparison with copy_df["ts"]
        start_date = pd.to_datetime(start_date).date()
        end_date = pd.to_datetime(end_date).date()

        # Filter by the date range
        filtered_df = copy_df[(copy_df["ts"] >= start_date) & (copy_df["ts"] <= end_date)]

        if filtered_df.empty:
            raise ValueError(f"No data found in the specified date range: {start_date} to {end_date}.")
        
        return filtered_df
    except Exception as e:
        raise ValueError(f"Failed to filter data by date range: {e}")

def filter_by_playback_time(df: pd.DataFrame, min_played_seconds: int) -> pd.DataFrame:
    """Filters the DataFrame to include only rows where playback time meets the minimum threshold."""
    if "ms_played" not in df.columns:
        raise ValueError("Required column 'ms_played' is missing in the data.")
    filtered_df = df[df["ms_played"] / 1000 >= min_played_seconds]
    
    if filtered_df.empty:
        raise ValueError(f"No data remaining after filtering by playback duration >= {min_played_seconds} seconds.")

    return filtered_df


def filter_by_group(df: pd.DataFrame, search_category: str, values: List[str]) -> pd.DataFrame:
    """
    Filters the DataFrame based on the specified grouping column (e.g., artist, album, or song).
    
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        search_category (str): The column to filter by (e.g., 'master_metadata_album_artist_name', 
                        'master_metadata_album_album_name', or 'master_metadata_track_name').
        values (List[str]): The values to filter by (e.g., a list of artist names, album names, or song names).
        
    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    if search_category not in df.columns:
        raise ValueError(f"Required column '{search_category}' is missing in the data.")

    filtered_df = df[df[search_category].isin(values)]

    if filtered_df.empty:
        raise ValueError(f"No data found for the specified values in column '{search_category}'.")
    
    return filtered_df

def prepare_histogram_data_for_listens(df: pd.DataFrame, search_category: str) -> pd.DataFrame:
    """
    Prepares data for a histogram by grouping listens per month by the specified column.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        search_category (str): The column to group by (e.g., artist, album, or song).
        
    Returns:
        pd.DataFrame: Aggregated data for the histogram.
    """
    if "ts" not in df.columns:
        raise ValueError("Required column 'ts' is missing in the data.")
    
    # Convert timestamps and group by month
    df["timestamp"] = pd.to_datetime(df["ts"]).dt.tz_localize(None)
    df["month"] = df["timestamp"].dt.to_period("M").dt.to_timestamp()

    # Adjust grouping for album-level aggregation if search_category is "master_metadata_album_artist_name"
    if search_category == "master_metadata_album_artist_name" and global_config.get("aggregate_by_album", False):
        # If searching for artist, include album data
        listens_per_month = df.groupby(["month", "master_metadata_album_album_name"]).size().reset_index(name="num_listens")
        listens_per_month.rename(columns={"master_metadata_album_album_name": "group"}, inplace=True)  # Rename column to be easily labelled by bar chart
    else:
        # Searching for album, just group by month
        listens_per_month = df.groupby(["month", search_category]).size().reset_index(name="num_listens")   
        listens_per_month.rename(columns={search_category: "group"}, inplace=True)  # Rename column to be easily labelled by bar chart
    return listens_per_month

def build_histogram_by_listens(
        listens_per_month: pd.DataFrame, 
        search_category: str, 
        values: List[str], 
        min_played_seconds: int, 
        date_range: Tuple[str, str] = None
    ):
    """
    Generates and displays an interactive histogram for the specified grouping (artist, album, or song).
    
    Args:
        listens_per_month (pd.DataFrame): Aggregated data for the histogram.
        search_category (str): The column to group by (e.g., artist, album, or song).
        values (List[str]): The values to display (e.g., artist names, album names, or song names).
        min_played_seconds (int): Minimum playback time (in seconds) to filter by.
        date_range (Tuple[str, str]): Optional date range for the x-axis.
    """
    title = f"Monthly Number of Listens ({min_played_seconds} seconds or more) for {', '.join(values)} "
    if date_range:
        title = title + f"from {date_range[0]} to {date_range[1]}"
    else:
        title = title + "of All Time"
    
    # If searching for Artist, add colours to group by Albums
    fig = px.bar(
        listens_per_month,
        x="month",
        y="num_listens",
        color="group",
        template="plotly_dark" if global_config.get("dark_mode", False) else "plotly",
        title=title,
        labels={
            "month": "Month", 
            "num_listens": "Number of Listens", 
            "group": "Album" if search_category == "master_metadata_album_artist_name" and global_config.get("aggregate_by_album", False) else search_category.split("_")[-2].capitalize()
        }
    )
    
    # Format x-axis
    # fig.update_layout(bargap=0.90)  # In case bars are too few and too wide.
    fig.update_xaxes(dtick="M1", tickformat="%b %Y")
    fig.show()

def create_histogram_by_listens(
        df: pd.DataFrame, 
        search_category: str, 
        values: List[str], 
        min_played_seconds: int = 0, 
        date_range: Tuple[str, str] = None
    ):
    """
    Creates a histogram for listening data grouped by artist, album, or song.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        search_category (str): The column to group by (e.g., 'master_metadata_album_artist_name', 
                        'master_metadata_album_album_name', or 'master_metadata_track_name').
        values (List[str]): The values to filter by (e.g., a list of artist names, album names, or song names).
        min_played_seconds (int): Minimum playback time (in seconds) to filter by.
        date_range (Tuple[str, str]): Optional date range for filtering data.
    """
    try:
        # Filter data by group (artist, album, or song)
        filtered_data = filter_by_group(df, search_category, values)

        # Filter data by playtime
        filtered_data = filter_by_playback_time(filtered_data, min_played_seconds)
        
        # Filter for date range, if provided
        if date_range:
            filtered_data = filter_by_date_range(filtered_data, date_range)

        # Prepare histogram data
        histogram_data = prepare_histogram_data_for_listens(filtered_data, search_category)

        # Generate histogram
        build_histogram_by_listens(histogram_data, search_category, values, min_played_seconds, date_range)
    except ValueError as e:
        logging.error(e)


def prepare_histogram_data_with_time_units(filtered_data: pd.DataFrame, search_category: str) -> pd.DataFrame:
    """
    Prepares data for a histogram by grouping playtime per month and converting playtime to hours and minutes.

    Args:
        filtered_data (pd.DataFrame): Filtered DataFrame with relevant data.
        search_category (str): The column to group by (e.g., artist, album, or song).

    Returns:
        pd.DataFrame: Aggregated DataFrame with monthly playtime in hours and minutes.
    """
    # Convert timestamps and group by month
    filtered_data["timestamp"] = pd.to_datetime(filtered_data["ts"]).dt.tz_localize(None)
    filtered_data["month"] = filtered_data["timestamp"].dt.to_period("M").dt.to_timestamp()

    # Aggregate playtime per month for the specified category
    # Adjust grouping for album-level aggregation if search_category is "master_metadata_album_artist_name"
    if search_category == "master_metadata_album_artist_name" and global_config.get("aggregate_by_album", False):
        histogram_data = (
            filtered_data.groupby(["month", "master_metadata_album_album_name"])["ms_played"].sum().reset_index()
        )
        histogram_data.rename(columns={"master_metadata_album_album_name": "group"}, inplace=True)  # Rename column to be easily labelled by bar chart
    else:
        histogram_data = (
            filtered_data.groupby(["month", search_category])["ms_played"].sum().reset_index()
        )
        histogram_data.rename(columns={search_category: "group"}, inplace=True)  # Rename column to be easily labelled by bar chart

    # Convert playtime to hours and minutes for better readability in the chart
    histogram_data["playtime_minutes"] = histogram_data["ms_played"] / (1000 * 60)
    histogram_data["playtime_hours"] = histogram_data["playtime_minutes"] / 60

    return histogram_data

def build_histogram_by_playtime(
        histogram_data: pd.DataFrame, 
        search_category: str, 
        values: List[str], 
        min_played_seconds: int,
        date_range: Tuple[str, str] = None
    ):
    """
    Builds and displays a histogram for playtime data grouped by month.

    Args:
        histogram_data (pd.DataFrame): Data prepared for the histogram.
        search_category (str): The column to group by (e.g., artist, album, or song).
        values (List[str]): The values to filter by (e.g., a list of artist names, album names, or song names).
        min_played_seconds (int): Minimum playback time (in seconds) to filter by.
        date_range (Tuple[str, str]): Optional date range for the x-axis.
    """
    title = f"Monthly Total Playtime (minutes) for {', '.join(values)} "
    if date_range:
        title += f"from {date_range[0]} to {date_range[1]}"
    else:
        title = title + "of All Time"
    subtitle = f"(Only considers tracks played for at least {min_played_seconds} seconds.)"

    fig = px.bar(
        histogram_data,
        x="month",
        y="playtime_minutes",
        color="group",
        template="plotly_dark" if global_config.get("dark_mode", False) else "plotly",
        title=title,
        labels={
            "month": "Month",
            "playtime_minutes": "Total Playtime (minutes)",
            "group": "Album" if (search_category == "master_metadata_album_artist_name" and global_config.get("aggregate_by_album", False)) else search_category.split("_")[-2].capitalize()
        },
        hover_data={
            "playtime_hours": ":.2f",
            "playtime_minutes": True,
            "group": True
        }
    )
    # Add subtitle to chart
    fig.update_layout(
        title=dict(
            subtitle=dict(
                text=subtitle, 
                font=dict(color="gray", size=13))
        )
    )

    # Format x-axis
    fig.update_xaxes(dtick="M1", tickformat="%b %Y")
    fig.show()


def create_histogram_by_playtime(
        df: pd.DataFrame, 
        search_category: str, 
        values: List[str], 
        min_played_seconds: int = 0, 
        date_range: Tuple[str, str] = None
    ):
    """
    Creates a histogram for listening data grouped by artist, album, or song,
    based on total playtime (ms_played).

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        search_category (str): The column to group by (e.g., 'master_metadata_album_artist_name', 
                        'master_metadata_album_album_name', or 'master_metadata_track_name').
        values (List[str]): The values to filter by (e.g., a list of artist names, album names, or song names).
        min_played_seconds (int): Minimum playback time (in seconds) to filter by.
        date_range (Tuple[str, str]): Optional date range for filtering data.
    """
    try:
        # Filter data by group (artist, album, or song)
        filtered_data = filter_by_group(df, search_category, values)

        # Filter data by playtime
        filtered_data = filter_by_playback_time(filtered_data, min_played_seconds)
        
        # Filter for date range, if provided
        if date_range:
            filtered_data = filter_by_date_range(filtered_data, date_range)

        # Prepare histogram data
        histogram_data = prepare_histogram_data_with_time_units(filtered_data, search_category)

        # Build histogram
        build_histogram_by_playtime(histogram_data, search_category, values, min_played_seconds, date_range)

    except ValueError as e:
        logging.error(e)


def filter_data_for_stacked_area_chart(df: pd.DataFrame, artist_names: List[str], min_played_seconds: int) -> pd.DataFrame:
    """
    Filters the DataFrame for the specified artists and minimum playback time.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        artist_names (List[str]): List of artist names to filter by.
        min_played_seconds (int): Minimum playback time (in seconds) to include an entry.
    
    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    # Filter data for the specified artists
    artist_data = df[df["master_metadata_album_artist_name"].isin(artist_names)]
    
    if artist_data.empty:
        raise ValueError(f"No data found for artists: {', '.join(artist_names)}")
    
    # Filter out entries with playback time less than the specified minimum
    artist_data = artist_data[artist_data["ms_played"] / 1000 >= min_played_seconds]
    
    if artist_data.empty:
        raise ValueError(f"No data remaining after filtering by playback duration for artist(s): {', '.join(artist_names)}")
    
    return artist_data

def prepare_stacked_area_chart_data(df: pd.DataFrame, date_range: Tuple[str, str]) -> pd.DataFrame:
    """
    Prepares data for the stacked area chart by grouping listens per month and album.
    
    Args:
        df (pd.DataFrame): The filtered DataFrame containing the data.
        date_range (Tuple[str, str]): A tuple of start and end dates in 'YYYY-MM-DD' format.
    
    Returns:
        pd.DataFrame: Aggregated data for the stacked area chart.
    """
    # Convert timestamps and group by month
    df["timestamp"] = pd.to_datetime(df["ts"]).dt.tz_localize(None)
    df["month"] = df["timestamp"].dt.to_period("M").dt.to_timestamp()

    # Aggregate listens per month, divided by album
    listens_per_month_album_agg = df.groupby(["month", "master_metadata_album_album_name"]).size().reset_index(name="num_listens")

    # Ensure the range includes all months between start and end, filling missing months
    if date_range:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        full_range = pd.date_range(start=start_date, end=end_date, freq="MS")
        all_albums = listens_per_month_album_agg["master_metadata_album_album_name"].unique()
        listens_per_month_album_agg = (
            listens_per_month_album_agg
            .set_index(["month", "master_metadata_album_album_name"])
            .reindex(pd.MultiIndex.from_product([full_range, all_albums], names=["month", "master_metadata_album_album_name"]), fill_value=0)
            .reset_index()
        )

    # Calculate cumulative listens over time per album
    listens_per_month_album_agg["cumulative_listens"] = listens_per_month_album_agg.groupby("master_metadata_album_album_name")["num_listens"].cumsum()

    return listens_per_month_album_agg

def build_stacked_area_chart(listens_per_month_album_agg: pd.DataFrame, artist_names: List[str], min_played_seconds: int, date_range: Tuple[str, str]):
    """
    Builds and displays a stacked area chart for the specified artists.
    The thickest bars (most listens) are at the bottom, and the legend is reordered to match the stack order.
    
    Args:
        listens_per_month_album_agg (pd.DataFrame): Aggregated data for the stacked area chart.
        artist_names (List[str]): List of artist names to filter by.
        min_played_seconds (int): Minimum playback time (in seconds) to include an entry.
        date_range (Tuple[str, str]): A tuple of start and end dates in 'YYYY-MM-DD' format.
    """
    # Sort the data so that the albums with the most cumulative listens are at the bottom
    # Calculate the total cumulative listens per album
    total_listens_per_album = listens_per_month_album_agg.groupby("master_metadata_album_album_name")["cumulative_listens"].max().reset_index()
    total_listens_per_album = total_listens_per_album.sort_values(by="cumulative_listens", ascending=False)  # Sort descending so largest is at the bottom
    sorted_albums = total_listens_per_album["master_metadata_album_album_name"].tolist()

    # Reorder the DataFrame based on the sorted albums
    listens_per_month_album_agg["master_metadata_album_album_name"] = pd.Categorical(
        listens_per_month_album_agg["master_metadata_album_album_name"],
        categories=sorted_albums,
        ordered=True
    )
    listens_per_month_album_agg = listens_per_month_album_agg.sort_values(["master_metadata_album_album_name", "month"])

    # Build the chart
    title = f"Monthly Cumulative Listens for {', '.join(artist_names)} (Filtered by {min_played_seconds} seconds)"
    
    fig = px.area(
        listens_per_month_album_agg,
        x="month",
        y="cumulative_listens",
        color="master_metadata_album_album_name",
        template="plotly_dark" if global_config.get("dark_mode", False) else "plotly",
        pattern_shape="master_metadata_album_album_name",
        title=title,
        labels={"month": "Month", "cumulative_listens": "Cumulative Listens", "master_metadata_album_album_name": "Album"},
        category_orders={"master_metadata_album_album_name": sorted_albums}  # Ensure the legend matches the stack order
    )

    # Force the x-axis range if date_range is specified
    if date_range:
        fig.update_xaxes(range=[date_range[0], date_range[1]])
    
    # Format the x-axis for better readability
    fig.update_xaxes(dtick="M1", tickformat="%b %Y")  # Format x-axis as 'Month Year'

    # Ensure the legend is displayed in the same order as the stack
    fig.update_layout(legend_traceorder="reversed")  # Keep legend in the same order as the stack

    fig.show()


def generate_stacked_area_chart(
        df: pd.DataFrame, 
        artist_names: List[str],
        date_range: Tuple[str, str], 
        min_played_seconds: int = 0
    ):
    """
    Creates an interactive stacked area chart for specific artists' listening data.
    The x-axis represents dates grouped by month, and the y-axis represents the cumulative number of listens per album.
    Entries with playback duration shorter than `min_played_seconds` are excluded.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        artist_names (List[str]): List of artist names to filter by.
        min_played_seconds (int): Minimum playback time (in seconds) to include an entry.
        date_range (Tuple[str, str]): A tuple of start and end dates in 'YYYY-MM-DD' format to force the x-axis range.
    """
    try:
        # Filter data for the specified artists and minimum playback time
        filtered_data = filter_data_for_stacked_area_chart(df, artist_names, min_played_seconds)

        # Prepare data for the stacked area chart
        listens_per_month_album_agg = prepare_stacked_area_chart_data(filtered_data, date_range)

        # Build and display the stacked area chart
        build_stacked_area_chart(listens_per_month_album_agg, artist_names, min_played_seconds, date_range)

    except ValueError as e:
        logging.error(e)


def prepare_ranking_data(df: pd.DataFrame, search_category: str, top_n: int) -> pd.DataFrame:
    """
    Prepares data for a ranking chart by grouping and ranking the number of listens.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        search_category (str): The column to group by (e.g., artist, album, or song).
        top_n (int): Number of top entries to display.
    
    Returns:
        pd.DataFrame: Aggregated and ranked data.
    """
    # Count the number of listens by the search category
    grouped_df = df.groupby(search_category).size().reset_index(name="num_listens")
    # Sort by the number of listens in descending order
    grouped_df = grouped_df.sort_values(by="num_listens", ascending=False)
    # Add ranking column with "min" method to handle ties
    grouped_df["rank"] = grouped_df["num_listens"].rank(method="min", ascending=False).astype(int)
    # Return the top n entries
    return grouped_df[grouped_df["rank"] <= top_n]


def get_category_labels(search_category: str):
    """
    The dictionary maps the `search_category` to a tuple of labels:
    - First element: Plural form for the chart title (e.g., "Artists", "Albums", "Songs").
    - Second element: Singular form for the y-axis label (e.g., "Artist", "Album", "Song").
    If the `search_category` does not match any known key, it defaults to
    ("[undefined]", "[undefined]") to handle unexpected or missing categories.

    Args:
        search_category (str): The column representing the ranking category.
    
    Returns: Tuple[str, str]: Tuple with labels for the bar chart.

    """
    category_labels = {
            "master_metadata_album_artist_name": ("Artists", "Artist"),
            "master_metadata_album_album_name": ("Albums", "Album"),
            "master_metadata_track_name": ("Songs", "Song")
        }.get(search_category, ("[undefined]", "[undefined]"))
    
    return category_labels


def get_top_n_chart_title(
        top_n: int, 
        category_label: str,
        min_played_seconds: int, 
        date_range: Tuple[str, str] = None):
    """
    Generates the title for the top n ranking chart

    Args:
        top_n (int): Number of top entries displayed.
        category_label (str): Search category type.
        min_played_seconds (int): Minimum playback time (in seconds) to filter by.
        date_range (Tuple[str, str]): Optional date range for the chart title.
    """
    
    title = f"Top {top_n} {category_label} by Number of Listens ({min_played_seconds} seconds or more) "
    if date_range:
        title = title + f"from {date_range[0]} to {date_range[1]}"
    else:
        title = title + "of All Time"
    return title


def build_top_n_chart_by_listens(
        ranked_data: pd.DataFrame,
        search_category: str,
        title: str,
        category_label: str,
    ):
    """
    Generates and displays a bar chart for ranked data.
    
    Args:
        ranked_data (pd.DataFrame): The ranked data to plot.
        search_category (str): The column representing the ranking category.
        title (str): The title for the bar chart.
        category_label (str): Label for the y-axis.
    """
    
    fig = px.bar(
        ranked_data,
        y=search_category,
        x="num_listens",
        title=title,
        color="rank",
        color_continuous_scale=px.colors.sequential.Plasma_r if global_config.get("dark_mode", False) else px.colors.sequential.Plasma,
        template="plotly_dark" if global_config.get("dark_mode", False) else "plotly",
        labels={
            search_category: category_label,
            "num_listens": "Number of Listens"
        }
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))  # Reverse bar order for better readability
    fig.update_coloraxes(showscale=False)
    fig.show()


def create_top_n_chart_by_listens(
        df: pd.DataFrame,
        top_n: int,
        search_category: str,
        min_played_seconds: int = 0,
        date_range: Tuple[str, str] = None
    ):
    """
    Creates a ranking bar chart for top artists, albums, or songs by number of listens.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        top_n (int): Number of top entries to display.
        search_category (str): The column to group by.
        min_played_seconds (int): Minimum playback time (in seconds) to filter by.
        date_range (Tuple[str, str]): Optional date range for filtering data.
    """
    try:
        # Filter data by playback time and date range
        filtered_data = filter_by_playback_time(df, min_played_seconds)
        if date_range:
            filtered_data = filter_by_date_range(filtered_data, date_range)

        # Prepare ranking data
        ranked_data = prepare_ranking_data(filtered_data, search_category, top_n)

        # Get category labels based on the search category
        category_labels = get_category_labels(search_category)

        # Create title for the chart
        title = get_top_n_chart_title(top_n, category_labels[0], min_played_seconds, date_range)

        # Generate the chart
        build_top_n_chart_by_listens(ranked_data, search_category, title, category_labels[1])

    except ValueError as e:
        logging.error(e)

def prepare_ranking_data_with_time_units(df: pd.DataFrame, top_n: int, search_category: str) -> pd.DataFrame:
    """
    Prepares data for a ranking chart with playtime in both minutes and hours.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        top_n (int): Number of top entries to display.
        search_category (str): The column to group by (e.g., artist, album, or song).
    
    Returns:
        pd.DataFrame: Aggregated and ranked data with minutes and hours for hover info.
    """
    # Sum playtime by the search category
    grouped_df = df.groupby(search_category)["ms_played"].sum().reset_index(name="total_playtime_ms")
    
    # Calculate minutes and hours
    grouped_df["total_playtime_minutes"] = grouped_df["total_playtime_ms"] / (1000 * 60)
    grouped_df["total_playtime_hours"] = grouped_df["total_playtime_minutes"] / 60

    # Sort by total playtime in minutes (chart scale)
    grouped_df = grouped_df.sort_values(by="total_playtime_minutes", ascending=False)
    
    # Add ranking column
    grouped_df["rank"] = grouped_df["total_playtime_minutes"].rank(method="min", ascending=False).astype(int)
    
    # Return the top n entries
    return grouped_df[grouped_df["rank"] <= top_n]


def get_top_n_playtime_chart_titles(
        top_n: int, 
        min_played_seconds: int, 
        category_label: str,
        date_range: Tuple[str, str] = None,
    ):
    """
    Generates a main title and subtitle for a bar chart showcasing top entities by total playtime.

    Args:
        top_n (int): The number of top-ranked entities to display.
        min_played_seconds (int): The minimum playback time (in seconds) for a track to be considered.
        category_label (str): Search category type.
        date_range (Tuple[str, str] or None): A tuple specifying the start and end dates for filtering the data.

    Returns:
        Tuple[str, str]: A tuple containing:
            - `title` (str): The main title for the chart, indicating the category, time range, and playtime metric.
            - `subtitle` (str): A subtitle providing details about the playback threshold.
    """
    title = f"Top {top_n} {category_label} by Total Playtime (minutes) "
    if date_range:
        title += f"from {date_range[0]} to {date_range[1]}"
    else:
        title = title + "of All Time"
        
    subtitle = f"(Only considers tracks played for at least {min_played_seconds} seconds.)"

    titles = title, subtitle
    return titles


def build_top_n_chart_by_playtime(
        ranked_data: pd.DataFrame, 
        search_category: str, 
        titles: Tuple[str, str],
        category_label: str, 
    ):
    """
    Generates and displays a bar chart for top-ranked entities (e.g., artists, albums, songs) 
    based on total playtime, with enhanced hover information.
    
    Args:
        ranked_data (pd.DataFrame): A DataFrame containing the aggregated and ranked data.
            Must include columns:
                - 'total_playtime_minutes': Total playtime in minutes for chart scaling.
                - 'total_playtime_hours': Total playtime in hours for hover information.
                - 'rank': Rank of each entity.
        search_category (str): The column representing the ranking category 
            (e.g., 'master_metadata_album_artist_name', 'master_metadata_album_album_name', or 'master_metadata_track_name').
        titles (Tuple[str, str]): The title and subtitle for the chart: 
            - Chart title = titles[0].
            - Chart subtitle = titles[1].
        category_label (str): Label for the y-axis.
    
    Returns:
        None: Displays an interactive bar chart using Plotly.
    """
    fig = px.bar(
            ranked_data,
            y=search_category,
            x="total_playtime_minutes",
            title=titles[0],
            color="rank",
            color_continuous_scale=px.colors.sequential.Plasma_r if global_config.get("dark_mode", False) else px.colors.sequential.Plasma,
            template="plotly_dark" if global_config.get("dark_mode", False) else "plotly",
            labels={
                search_category: category_label,
                "total_playtime_minutes": "Total Playtime (minutes)"
            },
            hover_data={
                "total_playtime_minutes": True,
                "total_playtime_hours": ":.2f",  # Show hours with two decimal places
                "rank": True,  # Hide rank in hover (optional)
            }
        )
    fig.update_layout(
        yaxis=dict(autorange="reversed"),  # Reverse bar order for better readability
        title=dict(
            subtitle=dict(
                text=titles[1], 
                font=dict(color="gray", size=13))
        )
    )  
    fig.update_coloraxes(showscale=False)
    fig.show()


def create_top_n_chart_by_playtime(
        df: pd.DataFrame,
        top_n: int,
        search_category: str,
        min_played_seconds: int = 0,
        date_range: Tuple[str, str] = None
    ):
    """
    Creates a ranking bar chart for top artists, albums, or songs with both minutes and hours in hover info.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        top_n (int): Number of top entries to display.
        search_category (str): The column to group by (e.g., artist, album, or song).
        min_played_seconds (int): Minimum playback time (in seconds) to filter by.
        date_range (Tuple[str, str]): Optional date range for filtering data.
    """
    try:
        # Filter data by playback time and date range
        filtered_data = filter_by_playback_time(df, min_played_seconds)
        if date_range:
            filtered_data = filter_by_date_range(filtered_data, date_range)

        # Prepare ranking data with minute and hour units
        ranked_data = prepare_ranking_data_with_time_units(filtered_data, top_n, search_category)

        # Get category labels based on the search category for the chart
        category_labels = get_category_labels(search_category)

        # Create title and subtitle for the chart
        titles = get_top_n_playtime_chart_titles(top_n, min_played_seconds, category_labels[0], date_range)

        # Generate the chart
        build_top_n_chart_by_playtime(ranked_data, search_category, titles, category_labels[1])

    except ValueError as e:
        logging.error(e)


def validate_dataframe_for_heatmap(df: pd.DataFrame):
    """Validates if the DataFrame contains the required columns for creating a heatmap."""
    required_columns = {'ts', 'ms_played'}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"The DataFrame is missing required columns: {missing_columns}")


def preprocess_heatmap_data(df: pd.DataFrame, date_range: Tuple[str, str] = None) -> pd.DataFrame:
    """
    Prepares data for the heatmap by filtering, extracting necessary columns, and aggregating.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing 'ts' and 'ms_played'.
        date_range (tuple): Optional date range ('YYYY-MM-DD', 'YYYY-MM-DD') for filtering.

    Returns:
        pd.DataFrame: Aggregated heatmap data.
    """
    df['datetime'] = pd.to_datetime(df['ts']).dt.tz_localize(None)
    
    # Filter by date range
    if date_range:
        df = filter_by_date_range(df, date_range)
        if df.empty:
            raise ValueError(f"No data available in the date range: {date_range}")

    # Convert ms_played to hours and extract day and hour
    df['hours_played'] = df['ms_played'] / (1000 * 60 * 60)
    df['day_of_week'] = df['datetime'].dt.day_name()
    df['hour'] = df['datetime'].dt.hour

    # Aggregate data for heatmap
    heatmap_data = (
        df.groupby(['day_of_week', 'hour'])['hours_played']
        .sum()
        .reset_index()
        .pivot(index='day_of_week', columns='hour', values='hours_played')
    )

    # Reorder days of the week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(day_order)

    return heatmap_data



def prepare_heatmap_annotations_top_btm_5(flat_data):
    """
    Helper to prapre annotation data for the heatmap.
    """
    top_5 = flat_data.nlargest(5, 'hours_played')
    bottom_5 = flat_data.nsmallest(5, 'hours_played')
    annotations = [
        dict(
            x=row['hour'], y=row['day_of_week'], text=f"{row['hours_played']:.2f}",
            showarrow=False, font=dict(color="black" if row['hours_played'] in top_5.values else "white", size=12)
        )
        for _, row in pd.concat([top_5, bottom_5]).iterrows()
    ]
    
    return annotations


def create_heatmap_figure(heatmap_data: pd.DataFrame, annotations: list, title: str, subtitle: str):
    """
    Creates a heatmap figure using Plotly.

    Args:
        heatmap_data (pd.DataFrame): Aggregated heatmap data.
        annotations (list): List of annotations for the heatmap.
        title (str): Title of the heatmap.
        subtitle (str): Subtitle to show cumulative hours. 
    """
    fig = px.imshow(
        heatmap_data,
        labels={'x': 'Hour of Day', 'y': 'Day of Week', 'color': 'Total Listening Time (hours)'},
        color_continuous_scale=px.colors.sequential.Plasma,
        template="plotly_dark" if global_config.get("dark_mode", False) else "plotly",
        title=title
    )
    fig.update_layout(
        title=dict(
            subtitle=dict(
                text=subtitle, 
                font=dict(color="gray", size=13))
        ),
        xaxis_title='Time of Day (Hourly Intervals)',
        yaxis_title='Day of Week',
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(24)),
            ticktext=[f'{hour}:00' for hour in range(24)]
        ),
        coloraxis_colorbar=dict(title="Total Time (hours)"),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        ),
        annotations=annotations
    )
    fig.show()


def print_heatmap_data_summary(flat_data):
    """
    Helper to print the data summary of the heatmap.
    """
    daily_totals = (
        flat_data.groupby('day_of_week')['hours_played']
        .sum()
        .reset_index()
        .rename(columns={'hours_played': 'Total Hours Played'})
    )
    print("\nSummary Table: Daily Total Listening Times")
    print(daily_totals.to_string(index=False))


def create_heatmap_total_listening_time(df: pd.DataFrame, date_range: tuple = None):
    """
    Generates a heatmap of total listening time in hours and annotates the top and bottom 5 values.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        date_range (Tuple[str, str]): Optional date range for filtering data. Format: ('YYYY-MM-DD', 'YYYY-MM-DD').
    """
    validate_dataframe_for_heatmap(df)
    heatmap_data = preprocess_heatmap_data(df, date_range)

    # Flatten heatmap data for annotations
    flat_data = heatmap_data.stack().reset_index()
    flat_data.columns = ['day_of_week', 'hour', 'hours_played']

    # Annotate top 5 and bottom 5 values
    annotations = prepare_heatmap_annotations_top_btm_5(flat_data)

    # Calculate cumulative total hours
    cumulative_total = flat_data['hours_played'].sum()

    # Generate title
    title = f"{args.user.capitalize()}'s Total Listening Time Heatmap"
    if date_range:
        title += f"\n({date_range[0]} to {date_range[1]})"
    
    subtitle = f"Cumulative Total: {cumulative_total:.2f} hrs"

    # Create heatmap
    create_heatmap_figure(heatmap_data, annotations, title, subtitle)

    # Print daily summary
    print_heatmap_data_summary(flat_data)


# Main execution
if __name__ == "__main__":

    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    # Update dark mode config
    global_config['dark_mode'] = args.dark_mode

    # Configuration
    file_pattern = f"data/{args.user}/Streaming_History_Audio_*[0-9].json"  # Path to json files
    output_file = f"{args.user}_spotify_analysis_output.txt"
    target_artists = ["Coldplay"]

    if not glob.glob(file_pattern):
        logging.warning(f'Check that user {args.user} has an entry in the data directory.')
        raise ValueError(f'File {file_pattern} not found.')

    # Execution
    data = load_files(file_pattern)
    df = convert_to_dataframe(data)
    
    # unique_tracks, unique_artists, total_playback_time_sec = extract_insights(df)
    # write_results(output_file, df, unique_tracks, unique_artists, total_playback_time_sec)
    
    # # Filter by Artist(s)
    # create_histogram_by_listens(
    #     df, 
    #     search_category="master_metadata_album_artist_name", 
    #     values=["HOYO-MiX", "Yu-Peng Chen", "Robin"], 
    #     min_played_seconds=30, 
    #     date_range=("2024-01-01", "2024-12-31")  # Optional
    # )
    
    # # Filter by Album(s)
    # create_histogram_by_listens(
    #     df, 
    #     search_category="master_metadata_album_album_name", 
    #     values=["Over the Garden Wall", "Currents"], 
    #     min_played_seconds=30, 
    #     # date_range=("2023-01-01", "2023-12-31")  # Optional
    # )
    
    # # Filter by Song(s)
    # create_histogram_by_listens(
    #     df, 
    #     search_category="master_metadata_track_name", 
    #     values=["The Highwayman (feat. Jerron 'Blind Boy' Paxton)", "Feather"], 
    #     min_played_seconds=10, 
    #     # date_range=("2023-01-01", "2023-12-31")  # Optional
    # )

    # create_histogram_by_playtime(
    #     df,
    #     search_category="master_metadata_album_artist_name",
    #     values=["BTS"],
    #     min_played_seconds=30,
    #     # date_range=("2024-01-01", "2024-12-31")
    # )

    generate_stacked_area_chart(
        df,  # Your DataFrame with Spotify data
        artist_names=target_artists,  # List of artists to analyze
        date_range=("2019-12", "2024-11"),  # Date range for the chart
        min_played_seconds=30  # Minimum playback time to include
    )

    # create_top_n_chart_by_listens(
    #     df,
    #     top_n=25,
    #     search_category="master_metadata_album_artist_name",
    #     min_played_seconds=30,
    #     # date_range=("2024-01-01", "2024-11-15")  # Optional
    # )

    # create_top_n_chart_by_playtime(
    #     df,
    #     top_n=25,
    #     search_category="master_metadata_album_artist_name",
    #     min_played_seconds=30,
    #     # date_range=("2024-01-01", "2024-11-15")  # Optional
    # )

    # create_heatmap_total_listening_time(df, date_range=('2024-01-01', '2024-12-31'))
