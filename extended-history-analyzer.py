import json
import glob
from typing import List, Dict, Any, Set, Tuple
import plotly.express as px
import pandas as pd


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

    print(f"Analysis results written to {output_file}")

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

def filter_by_artist(df: pd.DataFrame, artist_names: List[str]) -> pd.DataFrame:
    """Filters the DataFrame to include only rows for the specified artists."""
    if "master_metadata_album_artist_name" not in df.columns:
        raise ValueError("Required column 'master_metadata_album_artist_name' is missing in the data.")
    filtered_df = df[df["master_metadata_album_artist_name"].isin(artist_names)]
    if filtered_df.empty:
        raise ValueError(f"No data found for artists: {', '.join(artist_names)}")
    return filtered_df

def filter_by_playback_time(df: pd.DataFrame, min_played_seconds: int) -> pd.DataFrame:
    """Filters the DataFrame to include only rows where playback time meets the minimum threshold."""
    if "ms_played" not in df.columns:
        raise ValueError("Required column 'ms_played' is missing in the data.")
    filtered_df = df[df["ms_played"] / 1000 >= min_played_seconds]
    if filtered_df.empty:
        raise ValueError(f"No data remaining after filtering by playback duration >= {min_played_seconds} seconds.")
    return filtered_df

def filter_data_by_artist_and_playtime(df: pd.DataFrame, artist_names: List[str], min_played_seconds: int) -> pd.DataFrame:
    """Converts raw data to a DataFrame and applies artist and playback time filters."""
    try:
        artist_filtered_df = filter_by_artist(df, artist_names)
        time_filtered_df = filter_by_playback_time(artist_filtered_df, min_played_seconds)
        return time_filtered_df
    except ValueError as e:
        raise ValueError(f"Data filtering failed: {e}")

def prepare_artist_histogram_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepares data for the histogram by grouping listens per month and album."""
    # Ensure required columns are present
    required_columns = ["ts", "master_metadata_album_album_name"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Data is missing required columns: {required_columns}")
    
    # Convert timestamps and group by month
    df["timestamp"] = pd.to_datetime(df["ts"]).dt.tz_localize(None)
    df["month"] = df["timestamp"].dt.to_period("M").dt.to_timestamp()
    
    # Group listens per month and album
    listens_per_month_album_agg = df.groupby(
        ["month", "master_metadata_album_album_name"]
    ).size().reset_index(name="num_listens")
    
    return listens_per_month_album_agg

def generate_artist_histogram(
        listens_per_month_album_agg: pd.DataFrame, 
        artist_names: List[str], 
        min_played_seconds: int, 
        date_range: Tuple[str, str] = None
    ):
    """Generates and displays an interactive histogram."""
    title = (f"Monthly Number of Listens for {', '.join(artist_names)} "
             f"(Filtered by {min_played_seconds} seconds)")
    
    fig = px.bar(
        listens_per_month_album_agg,
        x="month",
        y="num_listens",
        color="master_metadata_album_album_name",
        title=title,
        labels={"month": "Month", "num_listens": "Number of Listens", 
                "master_metadata_album_album_name": "Album"}
    )
    
    # # Adjust x-axis range if a date range is provided
    # if date_range:
    #     start_date, end_date = date_range
    #     fig.update_xaxes(range=[start_date, end_date])
    
    # Format x-axis
    fig.update_xaxes(dtick="M1", tickformat="%b %Y")
    fig.show()

def create_artist_histogram(
        df: pd.DataFrame, 
        artist_names: List[str], 
        min_played_seconds: int = 0, 
        date_range: Tuple[str, str] = None
    ):
    """Creates a histogram for artist listening data."""
    
    try:        
        filtered_data = filter_data_by_artist_and_playtime(df, artist_names, min_played_seconds)
        
        # Filter for date range, if provided
        if date_range:
            filtered_data = filter_data_by_date_range(filtered_data, date_range)
        
        histogram_data = prepare_artist_histogram_data(filtered_data)
        generate_artist_histogram(histogram_data, artist_names, min_played_seconds, date_range)
    except ValueError as e:
        print(e)


def generate_stacked_area_chart(
        data: List[Dict[str, Any]], 
        artist_names: List[str], 
        min_played_seconds: int = 0, 
        date_range: Tuple[str, str] = None):
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
        print(f"No data found for artists: {', '.join(artist_names)}")
        return

    # Filter out entries with playback time less than the specified minimum
    artist_data = [
        entry for entry in artist_data
        if entry.get("ms_played", 0) / 1000 >= min_played_seconds
    ]
    if not artist_data:
        print(f"No data remaining after filtering by playback duration for artist(s): {', '.join(artist_names)}")
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
        print("Timestamp field 'ts' is missing in data.")
        return

    df["timestamp"] = pd.to_datetime(df["ts"])  # Assuming "ts" is the timestamp field
    df["month"] = df["timestamp"].dt.to_period("M").dt.to_timestamp()  # Group by month

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
        title=f"Monthly Cumulative Listens for {', '.join(artist_names)} (Filtered by {min_played_seconds} seconds)",
        labels={"month": "Month", "cumulative_listens": "Cumulative Listens", "master_metadata_album_album_name": "Album"},
    )

    # Force the x-axis range if date_range is specified
    if date_range:
        fig.update_xaxes(range=[date_range[0], date_range[1]])
    
    # Format the x-axis for better readability
    fig.update_xaxes(dtick="M1", tickformat="%b %Y")  # Format x-axis as 'Month Year'
    fig.show()

def display_top_artists(data: List[Dict[str, Any]], top_n: int = 10, min_played_seconds: int = 30, date_range: Tuple[str, str] = None):
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
        print(f"The data is missing required fields: {required_columns - set(df.columns)}")
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

    # Display the results
    print(f"Top {top_n} Artists by Number of Song Listens (Filtered by {min_played_seconds} seconds):")
    print(top_artists.to_string(index=False))

    # Optionally visualize the results
    fig = px.bar(
        top_artists,
        y="master_metadata_album_artist_name",
        x="num_listens",
        title=f"Top {top_n} Artists by Number of Song Listens ({min_played_seconds} seconds or more) from {date_range[0]} to {date_range[1]}",
        color="rank",
        color_continuous_scale=px.colors.sequential.Plasma,
        labels={"master_metadata_album_artist_name": "Artist", "num_listens": "Number of Listens"},
    )
    # fig.update_xaxes(type="category")  # Ensure the x-axis is treated as categorical
    fig.update_layout(yaxis=dict(autorange="reversed")) # Uncomment to reverse bars when graph is horizontal
    fig.update_coloraxes(showscale=False)
    fig.show()


# Main execution
if __name__ == "__main__":
    # Configuration
    file_pattern = "data/chris/Streaming_History_Audio_*[0-9].json"
    output_file = "spotify_analysis_output.txt"
    target_artists = ["HOYO-MiX", "Yu-Peng Chen"]

    # Execution
    data = load_files(file_pattern)
    df = convert_to_dataframe(data)
    unique_tracks, unique_artists, total_playback_time_sec = extract_insights(df)
    write_results(output_file, df, unique_tracks, unique_artists, total_playback_time_sec)
    
    # create_artist_histogram(
    #     df, 
    #     artist_names=["HOYO-MiX", "Yu-Peng Chen", "Robin"], 
    #     min_played_seconds=30, 
    #     date_range=("2023-01-01", "2024-11-29")
    # )

    # generate_stacked_area_chart(
    # data,
    # target_artists,
    # min_played_seconds=30,
    # date_range=("2024-01", "2024-12")
    # )

    # display_top_artists(
    # data, 
    # top_n=25, 
    # min_played_seconds=30, 
    # date_range=("2024-01-01", "2024-11-29")
    # )


