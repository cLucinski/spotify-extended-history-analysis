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


def generate_histogram(data: List[Dict[str, Any]], artist_name: str):
    """
    Creates an interactive histogram for a specific artist's listening data.
    The x-axis represents dates grouped by month, and the y-axis represents the number of listens.
    """
    # Filter data for the specified artist
    artist_data = [
        entry for entry in data
        if entry.get("master_metadata_album_artist_name") == artist_name
    ]
    if not artist_data:
        print(f"No data found for artist: {artist_name}")
        return

    # Prepare data for the histogram
    df = pd.DataFrame(artist_data)
    
    # Convert timestamps to datetime and group by month
    if "ts" not in df.columns:
        print("Timestamp field 'ts' is missing in data.")
        return

    df["timestamp"] = pd.to_datetime(df["ts"])  # Assuming "ts" is the timestamp field
    df["month"] = df["timestamp"].dt.to_period("M").dt.to_timestamp()  # Group by month

    # Count number of listens per month
    listens_per_month = df.groupby("month").size().reset_index(name="num_listens")

    # Create and show the histogram
    fig = px.bar(
        listens_per_month,
        x="month",
        y="num_listens",
        title=f"Monthly Number of Listens for {artist_name}",
        labels={"month": "Month", "num_listens": "Number of Listens"},
    )
    fig.update_xaxes(dtick="M1", tickformat="%b %Y")  # Format x-axis as 'Month Year'
    fig.show()


# Main execution
if __name__ == "__main__":
    # Configuration
    file_pattern = "Streaming_History_Audio_*[0-9].json"
    output_file = "spotify_analysis_output.txt"
    target_artist = "HOYO-MiX"

    # Execution
    data = load_files(file_pattern)
    unique_tracks, unique_artists, total_playback_time_sec = extract_insights(data)
    write_results(output_file, data, unique_tracks, unique_artists, total_playback_time_sec)
    generate_histogram(data, target_artist)
