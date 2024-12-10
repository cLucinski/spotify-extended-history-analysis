# data  

Use this folder to house your downloaded Spotify listening history files.  

**The content of this README is the English segment found in the `ReadMeFirst_ExtendedStreamingHistory.pdf` file which comes included in the data download from Spotify. (2024)**

![](https://web-api.textin.com/ocr_image/external/d9b8d00e54c96774.jpg)

## English

## Read Me First

Thank you for your patience while we gathered your data. This file contains your entire streaming history data for the life of your account. 

For a general description of the data included in your download, please see Understanding My Data. For more granular information about each field in your download, please see below. For information you are entitled to about the processing of your personal data under Article 15 of the GDPR, please see GDPR Article 15 Information. If you are a U.S. resident, and would like information about the categories of sources from which your data was collected and the categories of third parties with whom we share your data, please see the U.S. version of our Privacy Policy.

If you have also requested your account data and/or technical log information, you will be notified separately when those data packages are ready to download. If you have not requested any additional data but would like to do so, please go to your Account Privacy page and follow the instructions or email us at privacy@spotify.com. Please also contact us if you have any questions about your data.

Each stream in the file begins with `{"ts"`.

## Here is an example of one song (end_song) streaming data:
```
{  
  "ts":"YYY-MM-DD13:30:30",  
  "username":"_________",  
  "platform":"_________",  
  "ms_played":_________,  
  "conn_country":"_________",  
  "ip_addr_decrypted":"___.___.___.___",  
  "user_agent_decrypted":"_________",  
  "master_metadata_track_name":"_________,  
  "master_metadata_album_artist_name:_________",  
  "master_metadata_album_album_name:_________",  
  "spotify_track_uri:_________",  
  "episode_name":_________,  
  "episode_show_name":_________,  
  "spotify_episode_uri:_________",  
  "reason_start":"_________",  
  "reason_end":"_________",  
  "shuffle":null/true/false,  
  "skipped":null/true/false,  
  "offline":null/true/false,  
  "offline_timestamp":_________,  
  "incognito_mode":null/true/false,  
 }  
```


## Here is an example of one video (end_video) streaming data:

The following table explains the technical fields:
```
{  
  "ts":"YYY-MM-DD13:30:30",  
  "username":"_________",  
  "platform":"_________",  
  "ms_played":_________,  
  "conn_country":"_________",  
  "ip_addr_decrypted":"___.___.___.___",  
  "user_agent_decrypted":"_________",  
  "master_metadata_track_name":_________,  
  "master_metadata_album_artist_name":_________,  
  "master_metadata_album_album_name":_________,  
  "spotify_track_uri:_________",  
  "episode_name":"___________________________",  
  "episode_show_name":"__________________",  
  "spotify_episode_uri:_________",  
  "reason_start":"_________",  
  "reason_end":"_________",  
  "shuffle":null/true/false,  
  "skipped":null/true/false,  
  "offline":null/true/false,  
  "offline_timestamp":_________,  
  "incognito_mode":null/true/false,  
 }  
```

## The following table explains the technical fields:

| Technical Field | Contains |
| -- | -- |
| ts | This field is a timestamp indicating when the track stopped playing in UTC (Coordinated Universal Time). The order is year month and day followed by a timestamp in military time.
| username | This field is your Spotify username. |
| platform | This field is the platform used when streaming the track (e.g. Android OS, Google Chromecast). |
| ms_played | This field is the number of milliseconds the stream was played. |
| conn_country | This field is the country code of the country where the stream was played (e.g. SE - Sweden). |
| Ip_addr_decrypted | This field contains the IP address logged when streaming the track. |
| user_agent_decrypted | This field contains the user agent used when streaming thetrack (e.g. a browser, like Mozilla Firefox, or Safari) |
| master_metadata_track _name | This field is the name of the track. |
| master_metadata_album_artist_name | This field is the name of the artist, band or podcast. |
| master_metadata_album_album_name | This field is the name of the album of the track. |
| spotify_track_uri | A Spotify URI, uniquely identifying the track in the form of `spotify:track:<base-62string>`. <br> A Spotify URI is a resource identifier that you can enter, forexample, in the Spotify Desktop client’s search box to locatean artist, album, or track. |
| episode_name | This field contains the name of the episode of the podcast. |
| episode_show_name | This field contains the name of the show of the podcast. |
| spotify_episode_uri | A Spotify Episode URI, uniquely identifying the podcastepisode in the form of `spotify:episode:<base-62string>` <br> A Spotify Episode URI is a resource identifier that you canenter, for example, in the Spotify Desktop client’s search boxto locate an episode of a podcast. |
| reason_start | This field is a value telling why the track started (e.g."trackdone") |
| reason_end | This field is a value telling why the track ended (e.g."endplay"). |
| shuffle | This field has the value True or False depending on if shufflemode was used when playing the track. |
| skipped | This field indicates if the user skipped to the next song |
| offline | This field indicates whether the track was played in offlinemode ("True") or not ("False"). |
| offline_timestamp | This field is a timestamp of when offline mode was used, ifused. |
| incognito_mode | This field indicates whether the track was played during aprivate session ("True") or not ("False"). |




