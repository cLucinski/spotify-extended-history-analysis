import json
import glob

def get_file_list(pattern):
    """Retrieve and sort files matching the given pattern."""
    return sorted(glob.glob(pattern))

def load_combined_data(file_list):
    """Load and combine data from the given list of JSON files."""
    combined_data = []
    for file_path in file_list:
        with open(file_path, "r", encoding="utf-8") as file:
            combined_data.extend(json.load(file))
    return combined_data

def extract_insights(data):
    """Extract unique tracks, unique artists, and total playback time."""
    unique_tracks = {entry["master_metadata_track_name"] for entry in data if entry["master_metadata_track_name"]}
    unique_artists = {entry["master_metadata_album_artist_name"] for entry in data if entry["master_metadata_album_artist_name"]}
    total_playback_time_sec = sum(entry["ms_played"] for entry in data) / 1000
    return unique_tracks, unique_artists, total_playback_time_sec

def generate_output_lines(data, tracks, artists, playback_time):
    """Generate the lines to be written to the output file."""
    output_lines = [
        f"Total track entries read: {len(data)}",
        f"Total unique tracks: {len(tracks)}",
        f"Total unique artists: {len(artists)}",
        f"Total playback time (seconds): {playback_time:.2f}",
        "\nUnique Tracks:"
    ] + [f"- {track}" for track in tracks] + [
        "\nUnique Artists:"
    ] + [f"- {artist}" for artist in artists]
    return output_lines

def write_output_file(output_file, lines):
    """Write the analysis results to the specified output file."""
    with open(output_file, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))
    print(f"Analysis results written to {output_file}")

def main():
    """Main function to orchestrate the analysis."""
    file_pattern = "Streaming_History_Audio_*[0-9].json"
    output_file = "spotify_analysis_output.txt"
    
    file_list = get_file_list(file_pattern)
    combined_data = load_combined_data(file_list)
    unique_tracks, unique_artists, total_playback_time_sec = extract_insights(combined_data)
    output_lines = generate_output_lines(combined_data, unique_tracks, unique_artists, total_playback_time_sec)
    write_output_file(output_file, output_lines)

if __name__ == "__main__":
    main()
