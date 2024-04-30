# %%
import os
import requests
import logging
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
# %%

# Set up logging
logging.basicConfig(filename='script.log', level=logging.INFO)
#%% 

# Load the existing dataset
file_path = 'dataset.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError as e:
    print(f"Error accessing the CSV file: {e}")
    df = pd.DataFrame()

df.head()

#%%
#%%
# Set up Spotify API credentials
client_id = os.environ.get('SPOTIFY_CLIENT_ID')
client_secret = os.environ.get('SPOTIFY_CLIENT_SECRET')
#%% 

# Authenticate with Spotify API
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Function to retrieve the genre for an artist using Spotify API
def get_artist_genre(artist_name):
    """
    Retrieve the genre for an artist using Spotify API.
    """
    try:
        results = sp.search(q=artist_name, type='artist', limit=1)
        if results['artists']['items']:
            artist = results['artists']['items'][0]
            genres = artist['genres']
            if genres:
                return genres[0]  # Return the first genre
    except Exception as e:
        print(f"Error retrieving genre for artist {artist_name}: {str(e)}")
    return 'Unknown'  # Return 'Unknown' if no genre is found or artist is not found

# Function to retrieve artist popularity using Spotify API
def get_artist_popularity(artist_name):
    """
    Retrieve artist popularity using Spotify API.
    """
    try:
        results = sp.search(q=artist_name, type='artist', limit=1)
        if results['artists']['items']:
            artist = results['artists']['items'][0]
            artist_id = artist['id']
            artist_info = sp.artist(artist_id)
            popularity = artist_info['popularity']
            return popularity
    except Exception as e:
        print(f"Error retrieving info for artist {artist_name}: {str(e)}")
    return None  # Return None if artist is not found

# Function to retrieve artist followers using Spotify API
def get_artist_followers(artist_name):
    """
    Retrieve artist followers using Spotify API.
    """
    try:
        results = sp.search(q=artist_name, type='artist', limit=1)
        if results['artists']['items']:
            artist = results['artists']['items'][0]
            artist_id = artist['id']
            artist_info = sp.artist(artist_id)
            followers = artist_info['followers']['total']
            return followers
    except Exception as e:
        print(f"Error retrieving info for artist {artist_name}: {str(e)}")
    return None  # Return None if artist is not found

# Function to retrieve track popularity and audio features using Spotify API
def get_track_info(track_name, artist_name):
    """
    Retrieve track popularity and audio features using Spotify API.
    """
    try:
        results = sp.search(q=f'track:{track_name} artist:{artist_name}', type='track', limit=1)
        if results['tracks']['items']:
            track = results['tracks']['items'][0]
            track_id = track['id']
            track_info = sp.track(track_id)
            popularity = track_info['popularity']
            audio_features = sp.audio_features([track_id])[0]
            return popularity, audio_features
    except Exception as e:
        print(f"Error retrieving info for track {track_name} by artist {artist_name}: {str(e)}")
    return None, None  # Return None if track is not found

# Create a new column 'genre' in the DataFrame
df['genre'] = df['artname'].apply(get_artist_genre)

# Create new columns for artist popularity and followers
df['artist_popularity'] = df['artname'].apply(get_artist_popularity)
df['artist_followers'] = df['artname'].apply(get_artist_followers)

# Create new columns for track popularity and audio features
df['track_popularity'], df['audio_features'] = zip(*df.apply(lambda x: get_track_info(x['track_name'], x['artname']), axis=1))

# Save the modified dataset to a new CSV file
df.to_csv('dataset_modified.csv', index=False)