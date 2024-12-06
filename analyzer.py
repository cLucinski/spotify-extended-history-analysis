import json
import glob

# Regex-like pattern to match all files with the given naming convention
file_pattern = "Streaming_History_Audio_*[0-9].json"

# Initialize an empty list to store all entries
combined_data = []

# Get and sort the list of matching files
file_list = sorted(glob.glob(file_pattern))

# Loop through each sorted file and load its data
for file_path in file_list:
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        combined_data.extend(data)  # Add the contents to the combined list

# Extract insights (unique tracks, artists, total playback time)
unique_tracks = {entry["master_metadata_track_name"] for entry in combined_data if entry["master_metadata_track_name"]}
unique_artists = {entry["master_metadata_album_artist_name"] for entry in combined_data if entry["master_metadata_album_artist_name"]}
total_playback_time_sec = sum(entry["ms_played"] for entry in combined_data) / 1000

# Prepare the results to write to a file
output_lines = [
    f"Total track entries read: {len(combined_data)}",
    f"Total unique tracks: {len(unique_tracks)}",
    f"Total unique artists: {len(unique_artists)}",
    f"Total playback time (seconds): {total_playback_time_sec:.2f}",
    "\nUnique Tracks:"
] + [f"- {track}" for track in unique_tracks] + [
    "\nUnique Artists:"
] + [f"- {artist}" for artist in unique_artists]

# Write the results to an output file
output_file = "spotify_analysis_output.txt"
with open(output_file, "w", encoding="utf-8") as file:
    file.write("\n".join(output_lines))

print(f"Analysis results written to {output_file}")
