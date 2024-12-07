import json
import glob
from typing import List, Dict, Any, Tuple
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


def extract_insights(data: List[Dict[str, Any]]) -> Tuple[set, set, float]:
    """Extract unique tracks, artists, and total playback time."""
    unique_tracks = {entry.get("master_metadata_track_name") for entry in data if entry.get("master_metadata_track_name")}
    unique_artists = {entry.get("master_metadata_album_artist_name") for entry in data if entry.get("master_metadata_album_artist_name")}
    total_playback_time_sec = sum(entry.get("ms_played", 0) for entry in data) / 1000
    return unique_tracks, unique_artists, total_playback_time_sec


def write_results(output_file: str, combined_data: List[Dict[str, Any]], unique_tracks: set, unique_artists: set, total_playback_time_sec: float):
    """Writes analysis results to a file."""
    output_lines = [
        f"Total track entries read: {len(combined_data)}",
        f"Total unique tracks: {len(unique_tracks)}",
        f"Total unique artists: {len(unique_artists)}",
        f"Total playback time (seconds): {total_playback_time_sec:.2f}",
        "\nUnique Tracks:"
    ] + [f"- {track}" for track in unique_tracks] + [
        "\nUnique Artists:"
    ] + [f"- {artist}" for artist in unique_artists]

    with open(output_file, "w", encoding="utf-8") as file:
        file.write("\n".join(output_lines))

    print(f"Analysis results written to {output_file}")


def generate_histogram(
        data: List[Dict[str, Any]], 
        artist_names: List[str], 
        min_played_seconds: int = 0, 
        date_range: Tuple[str, str] = None):
    """
    Creates an interactive histogram for a specific artist's listening data.
    The x-axis represents dates grouped by month, and the y-axis represents the number of listens.
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
        # if entry.get("master_metadata_album_artist_name") in artist_names
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

    # Prepare data for the histogram
    df = pd.DataFrame(artist_data)

    # Remove erroneous columns
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
    df["album_-_artist"] = df['master_metadata_album_album_name'].map(str) + ' - ' + df['master_metadata_album_artist_name'].map(str) 
    print(df["album_-_artist"][0])
    # Count number of listens per month
    listens_per_month = df.groupby("month").size().reset_index(name="num_listens")

    # Count number listens per month, divided by album
    listens_per_month_album_agg = df.groupby(["month", "album_-_artist"]).size().reset_index(name="num_listens")
    #print(listens_per_month_album_agg.head(5))

    # Line specifically to drop an incredibly long Yu-Peng Chen album name
    # listens_per_month_album_agg.drop(listens_per_month_album_agg.index[listens_per_month_album_agg["master_metadata_album_album_name"] == "A Promise of Dreams - The original soundtrack from the game Project Woolgatherer"], inplace=True)
    # print(test_df)

    # Create and show the histogram
    fig = px.bar(
        listens_per_month_album_agg,
        x="month",
        y="num_listens",
        color="album_-_artist",
        title=f"Monthly Number of Listens for {', '.join(artist_names)} (Filtered by {min_played_seconds} seconds)",
        labels={"month": "Month", "num_listens": "Number of Listens", "album_-_artist": "Album - Artist"},
    )

    # Force the x-axis range if date_range is specified
    if date_range:
        start_date, end_date = date_range
        fig.update_xaxes(range=[start_date, end_date])
    
    # fig.update_yaxes(range=[0, 850])
        
    # Format the x-axis for better readability
    fig.update_xaxes(dtick="M1", tickformat="%b %Y")  # Format x-axis as 'Month Year'
    
    fig.show()


# Main execution
if __name__ == "__main__":
    # Configuration
    file_pattern = "Streaming_History_Audio_*[0-9].json"
    output_file = "spotify_analysis_output.txt"
    target_artists = ["HOYO-MiX", "Yu-Peng Chen", "Robin"]

    # Execution
    data = load_files(file_pattern)
    # unique_tracks, unique_artists, total_playback_time_sec = extract_insights(data)
    # write_results(output_file, data, unique_tracks, unique_artists, total_playback_time_sec)
    
    generate_histogram(
    data,
    target_artists,
    min_played_seconds=30,
    # date_range=("2021-06", "2024-12")
    )
