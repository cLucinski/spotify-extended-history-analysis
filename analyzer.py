import json

# Load the JSON file
file_path = "Streaming_History_Audio_2019-2021_0.json"

with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# Extract unique tracks and artists
unique_tracks = {entry["master_metadata_track_name"] for entry in data if entry["master_metadata_track_name"]}
unique_artists = {entry["master_metadata_album_artist_name"] for entry in data if entry["master_metadata_album_artist_name"]}

# Calculate total playback time in seconds
total_playback_time_ms = sum(entry["ms_played"] for entry in data)
total_playback_time_sec = total_playback_time_ms / 1000

# Display insights
print(f"Total tracks played: {len(data)}")
print(f"Total unique tracks: {len(unique_tracks)}")
print(f"Total unique artists: {len(unique_artists)}")
print(f"Total playback time (seconds): {total_playback_time_sec:.2f}")

print("\nUnique Tracks:")
for track in unique_tracks:
    print(f"- {track}")

print("\nUnique Artists:")
for artist in unique_artists:
    print(f"- {artist}")
