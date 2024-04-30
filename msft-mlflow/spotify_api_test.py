import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Load the existing dataset
url = 'https://media.githubusercontent.com/media/LHydra-Com/ML/main/dataset.csv'
df = pd.read_csv(url)

# Set up Spotify API credentials
client_id = '4187992fdb764829b6b2ce20718027c0'
client_secret = '4adc98b676ed40e1b43c521b355ef809'

# Authenticate with Spotify API
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Function to retrieve the genre for an artist using Spotify API
def get_artist_genre(artist_name):
    """
    Retrieve the genre for an artist using Spotify API.
    """
    results = sp.search(q=artist_name, type='artist', limit=1)
    if results['artists']['items']:
        artist = results['artists']['items'][0]
        genres = artist['genres']
        if genres:
            return genres[0]  # Return the first genre
    return 'Unknown'  # Return 'Unknown' if no genre is found or artist is not found

# Function to retrieve artist popularity and followers using Spotify API
def get_artist_info(artist_name):
    """
    Retrieve artist popularity and followers using Spotify API.
    """
    results = sp.search(q=artist_name, type='artist', limit=1)
    if results['artists']['items']:
        artist = results['artists']['items'][0]
        artist_id = artist['id']
        artist_info = sp.artist(artist_id)
        popularity = artist_info['popularity']
        followers = artist_info['followers']['total']
        return popularity, followers
    return None, None  # Return None if artist is not found

# Function to retrieve track popularity and audio features using Spotify API
def get_track_info(track_name, artist_name):
    """
    Retrieve track popularity and audio features using Spotify API.
    """
    results = sp.search(q=f'track:{track_name} artist:{artist_name}', type='track', limit=1)
    if results['tracks']['items']:
        track = results['tracks']['items'][0]
        track_id = track['id']
        track_info = sp.track(track_id)
        popularity = track_info['popularity']
        audio_features = sp.audio_features([track_id])[0]
        return popularity, audio_features
    return None, None  # Return None if track is not found

# Create a new column 'genre' in the DataFrame
df['genre'] = df['artname'].apply(get_artist_genre)

# Create new columns for artist popularity and followers
df['artist_popularity'], df['artist_followers'] = zip(*df['artname'].apply(get_artist_info))

# Create new columns for track popularity and audio features
df['track_popularity'], df['audio_features'] = zip(*df.apply(lambda x: get_track_info(x['track_name'], x['artname']), axis=1))

# Save the modified dataset to a new CSV file
df.to_csv('dataset_modified.csv', index=False)