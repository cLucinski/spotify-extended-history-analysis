import argparse
import glob
import json
import logging
import pandas as pd
import plotly.express as px
from typing import List, Dict, Any, Set, Tuple

parser = argparse.ArgumentParser(description='Analyze Spotify streaming history data.')
parser.add_argument('-v', '--verbose',
                    help='Increase output verbosity',
                    action='store_true',
                    default=False)

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
        "\nUnique Tracks:"
    ] + [f"- {track}" for track in sorted(unique_tracks)] + [
        "\nUnique Artists:"
    ] + [f"- {artist}" for artist in sorted(unique_artists)]

    with open(output_file, "w", encoding="utf-8") as file:
        file.write("\n".join(output_lines))

    logging.info(f"Analysis results written to {output_file}")

def filter_data_by_date_range(df: pd.DataFrame, date_range: Tuple[str, str]) -> pd.DataFrame:
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
    df = df.copy()
    
    try:
        # Ensure the 'ts' column is in datetime format
        df["ts"] = pd.to_datetime(df["ts"])

        # Filter by the date range
        filtered_df = df[(df["ts"] >= start_date) & (df["ts"] <= end_date)]

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

def prepare_histogram_data(df: pd.DataFrame, search_category: str) -> pd.DataFrame:
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

    # If searching for an artist, also group by their albums to create an aggregate of that for the chart's bars.
    # Otherwise, just group by month.
    match search_category:
        case "master_metadata_album_artist_name":
            # If searching for artist, include album data
            listens_per_month = df.groupby(["month", "master_metadata_album_album_name"]).size().reset_index(name="num_listens")
        case _:
            # Searching for album, just group by month
            listens_per_month = df.groupby(["month"]).size().reset_index(name="num_listens")

    # TODO: verify the following line isn't needed
    # Group listens per month by the specified column
    listens_per_month = df.groupby(["month", "master_metadata_album_album_name"]).size().reset_index(name="num_listens")
    
    return listens_per_month

def generate_histogram(
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
    if search_category == "master_metadata_album_artist_name":
        fig = px.bar(
            listens_per_month,
            x="month",
            y="num_listens",
            color="master_metadata_album_album_name",
            title=title,
            labels={"month": "Month", "num_listens": "Number of Listens", "master_metadata_album_album_name": "Album"}
        )
    # Otherwise, don't.
    else:
        fig = px.bar(
            listens_per_month,
            x="month",
            y="num_listens",
            title=title,
            labels={"month": "Month", "num_listens": "Number of Listens"}
        )
    
    # Format x-axis
    # fig.update_layout(bargap=0.90)  # In case bars are too few and too wide.
    fig.update_xaxes(dtick="M1", tickformat="%b %Y")
    fig.show()

def create_histogram(
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
            filtered_data = filter_data_by_date_range(filtered_data, date_range)

        # Prepare histogram data
        histogram_data = prepare_histogram_data(filtered_data, search_category)

        # Generate histogram
        generate_histogram(histogram_data, search_category, values, min_played_seconds, date_range)
    except ValueError as e:
        logging.error(e)


def generate_stacked_area_chart(
        data: List[Dict[str, Any]], 
        artist_names: List[str],
        date_range: Tuple[str, str], 
        min_played_seconds: int = 0
        ):
    """
    Creates an interactive stacked area chart for specific artists' listening data.
    The x-axis represents dates grouped by month, and the y-axis represents the cumulative number of listens per album.
    Entries with playback duration shorter than `min_played_seconds` are excluded.
    
    Args:
        data (List[Dict[str, Any]]): The streaming history data.
        artist_names (List[str]): List of artist names to filter by.
        min_played_seconds (int): Minimum playback time (in seconds) to include an entry.
        date_range (Tuple[str, str]): A tuple of start and end dates in 'YYYY-MM-DD' format to force the x-axis range.
    """
    # Filter data for the specified artists
    artist_data = [
        entry for entry in data
        if entry.get("master_metadata_album_artist_name") in artist_names
    ]
    if not artist_data:
        logging.error(f"No data found for artists: {', '.join(artist_names)}")
        return

    # Filter out entries with playback time less than the specified minimum
    artist_data = [
        entry for entry in artist_data
        if entry.get("ms_played", 0) / 1000 >= min_played_seconds
    ]
    if not artist_data:
        logging.error(f"No data remaining after filtering by playback duration for artist(s): {', '.join(artist_names)}")
        return

    # Prepare data for the chart
    df = pd.DataFrame(artist_data)

    # Keep relevant columns
    df = df[["ts", 
            "ms_played", 
            "master_metadata_track_name", 
            "master_metadata_album_artist_name", 
            "master_metadata_album_album_name"]]
    
    # Convert timestamps to datetime and group by month
    if "ts" not in df.columns:
        logging.error("Timestamp field 'ts' is missing in data.")
        return

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

    # Create and show the stacked area chart
    fig = px.area(
        listens_per_month_album_agg,
        x="month",
        y="cumulative_listens",
        color="master_metadata_album_album_name",
        pattern_shape="master_metadata_album_album_name",
        title=f"Monthly Cumulative Listens for {', '.join(artist_names)} (Filtered by {min_played_seconds} seconds)",
        labels={"month": "Month", "cumulative_listens": "Cumulative Listens", "master_metadata_album_album_name": "Album"},
    )

    # Force the x-axis range if date_range is specified
    if date_range:
        fig.update_xaxes(range=[date_range[0], date_range[1]])
    
    # Format the x-axis for better readability
    fig.update_xaxes(dtick="M1", tickformat="%b %Y")  # Format x-axis as 'Month Year'
    fig.show()

def display_top_artists(
        data: List[Dict[str, Any]], 
        top_n: int = 10, 
        min_played_seconds: int = 30, 
        date_range: Tuple[str, str] = None):
    """
    Displays the artists with the highest number of song listens, filtered by playback duration and date range.

    Args:
        data (List[Dict[str, Any]]): The streaming history data.
        top_n (int): Number of top artists to display. Default is 10.
        min_played_seconds (int): Minimum playback time (in seconds) to include an entry. Default is 30.
        date_range (Tuple[str, str]): A tuple of start and end dates in 'YYYY-MM-DD' format to filter data by date range.
    """
    # Prepare data
    df = pd.DataFrame(data)

    # Ensure the necessary columns exist
    required_columns = {"master_metadata_album_artist_name", "master_metadata_track_name", "ms_played", "ts"}
    if not required_columns.issubset(df.columns):
        logging.error(f"The data is missing required fields: {required_columns - set(df.columns)}")
        return

    # Filter out entries with playback time less than the specified minimum
    df = df[df["ms_played"] / 1000 >= min_played_seconds]

    # Convert 'ts' column to datetime and ensure it's timezone-naive
    df["timestamp"] = pd.to_datetime(df["ts"]).dt.tz_localize(None)

    # Filter by date range if specified
    if date_range:
        start_date = pd.to_datetime(date_range[0]).tz_localize(None)
        end_date = pd.to_datetime(date_range[1]).tz_localize(None)
        df = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)]

    # Count the number of listens per artist
    artist_listens = df.groupby("master_metadata_album_artist_name").size().reset_index(name="num_listens")

    # Sort by number of listens in descending order
    top_artists = artist_listens.sort_values(by="num_listens", ascending=False).head(top_n)

    # Add ranking column
    top_artists["rank"] = top_artists["num_listens"].rank(method="min", ascending=False).astype(int)

    # # Display the results
    # logging.info(f"Top {top_n} Artists by Number of Song Listens (Filtered by {min_played_seconds} seconds):")
    # logging.info(top_artists.to_string(index=False))

    title = f"Top {top_n} Artists by Number of Listens ({min_played_seconds} seconds or more) "
    if date_range:
        title = title + f"from {date_range[0]} to {date_range[1]}"
    else:
        title = title + "of All Time"
    
    # Optionally visualize the results
    fig = px.bar(
        top_artists,
        y="master_metadata_album_artist_name",
        x="num_listens",
        title=title,
        color="rank",
        color_continuous_scale=px.colors.sequential.Plasma,
        labels={"master_metadata_album_artist_name": "Artist", "num_listens": "Number of Listens"},
    )
    # fig.update_xaxes(type="category")  # Ensure the x-axis is treated as categorical
    fig.update_layout(yaxis=dict(autorange="reversed")) # Uncomment to reverse bars when graph is horizontal
    fig.update_coloraxes(showscale=False)
    fig.show()


def create_top_n_chart(
        df: pd.DataFrame,
        top_n: int,
        search_category: str,
        min_played_seconds: int = 0,
        date_range: Tuple[str, str] = None
    ):
    """
    Creates a bar chart for ranking top artists, albums, or songs by number of listens.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        top_n (int): Number of top entries to display.
        search_category (str): The column to group by (e.g., 'master_metadata_album_artist_name', 
                        'master_metadata_album_album_name', or 'master_metadata_track_name').
        min_played_seconds (int): Minimum playback time (in seconds) to filter by.
        date_range (Tuple[str, str]): Optional date range for filtering data.
    """
    # Validate required columns
    if search_category not in df.columns or 'ms_played' not in df.columns:
        raise ValueError(f"The data is missing required fields: '{search_category}' or 'ms_played'")
    
    # Filter by playback time
    filtered_df = filter_by_playback_time(df, min_played_seconds)
    
    # Filter by date range if provided
    if date_range:
        filtered_df = filter_data_by_date_range(filtered_df, date_range)
    
    # Count the number of listens by the search category
    grouped_df = filtered_df.groupby(search_category).size().reset_index(name="num_listens")
    
    # Sort and select the top entries
    top_entries = grouped_df.sort_values(by="num_listens", ascending=False).head(top_n)
    
    # Add ranking column
    top_entries["rank"] = range(1, len(top_entries) + 1)

    # A match-case to define the title and y-axis label based on search category
    # category_labels[0] is plural
    # category_labels[1] is singular
    match search_category:
        case "master_metadata_album_artist_name":
            category_labels = ["Artists", "Artist"]
        case "master_metadata_album_album_name":
            category_labels = ["Albums", "Album"]
        case "master_metadata_track_name":
            category_labels = ["Songs", "Song"]
        case _:
            category_labels = ["[undefined]", "[undefined]"]

    
    # Plot the results
    title = f"Top {top_n} {category_labels[0]} by Number of Listens"
    if date_range:
        title += f" ({date_range[0]} to {date_range[1]})"
    
    fig = px.bar(
        top_entries,
        y=search_category,
        x="num_listens",
        title=title,
        color="rank",
        color_continuous_scale=px.colors.sequential.Plasma,
        labels={
            search_category: category_labels[1],
            "num_listens": "Number of Listens"
        }
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))  # Reverse bar order for better readability
    fig.update_coloraxes(showscale=False)
    fig.show()


# Main execution
if __name__ == "__main__":

    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Configuration
    file_pattern = "data/chris/Streaming_History_Audio_*[0-9].json"  # Path to json files
    output_file = "spotify_analysis_output.txt"
    target_artists = ["Coldplay"]

    # Execution
    data = load_files(file_pattern)
    df = convert_to_dataframe(data)
    
    # unique_tracks, unique_artists, total_playback_time_sec = extract_insights(df)
    # write_results(output_file, df, unique_tracks, unique_artists, total_playback_time_sec)
    
    # # Filter by Artist(s)
    # create_histogram(
    #     df, 
    #     search_category="master_metadata_album_artist_name", 
    #     values=["HOYO-MiX", "Yu-Peng Chen", "Robin"], 
    #     min_played_seconds=30, 
    #     date_range=("2024-01-01", "2024-12-31")  # Optional
    # )
    
    # # Filter by Album(s)
    # create_histogram(
    #     df, 
    #     search_category="master_metadata_album_album_name", 
    #     values=["Over the Garden Wall"], 
    #     min_played_seconds=30, 
    #     # date_range=("2023-01-01", "2023-12-31")  # Optional
    # )
    
    # # Filter by Song(s)
    # create_histogram(
    #     df, 
    #     search_category="master_metadata_track_name", 
    #     values=["The Highwayman (feat. Jerron 'Blind Boy' Paxton)"], 
    #     min_played_seconds=10, 
    #     # date_range=("2023-01-01", "2023-12-31")  # Optional
    # )

    # generate_stacked_area_chart(
    #     data,
    #     target_artists,
    #     date_range=("2019-12", "2024-11"),
    #     min_played_seconds=30
    # )

    # display_top_artists(
    #     data, 
    #     top_n=5, 
    #     min_played_seconds=30, 
    #     date_range=("2024-01-01", "2024-12-31")
    # )

    create_top_n_chart(
        df, 
        top_n=5,
        search_category="master_metadata_album_album_name",
        min_played_seconds=30,
        date_range=("2024-01-01", "2024-12-31")  # Optional
    )


