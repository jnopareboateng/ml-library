# %%
import pandas as pd
# !pip install spotipy
import spotipy
# !pip install dask
import dask.dataframe as dd
from spotipy.oauth2 import SpotifyClientCredentials
# from google.colab import userdata
# from google.colab import drive
import time
import os
import configparser

# drive.mount('/content/drive')

# %%
def load_data(file_path):
    return dd.read_csv(file_path)


# %%
def authenticate_spotify():
  # Read configuration file
    config = configparser.ConfigParser()
    config.read('config.ini')
    print(config.sections())

    # Get credentials from config file
    client_id = config.get('SPOTIFY', 'CLIENT_ID')
    client_secret = config.get('SPOTIFY', 'CLIENT_SECRET')

    
    if client_id and client_secret:
        credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        sp = spotipy.Spotify(client_credentials_manager=credentials_manager)
        print("Spotify authentication successful.")
        return sp
    else:
        print("Failed to authenticate with Spotify.")
        return None

# %%
def get_artist_genre(sp, artist_name):
    try:
        if sp:
            results = sp.search(q=artist_name, type='artist', limit=1)
            if results['artists']['items']:
                artist = results['artists']['items'][0]
                genres = artist['genres']
                return genres[0] if genres else 'Unknown'
    except Exception as e:
        print(f"Failed to get genre for artist {artist_name}: {e}")
    return 'Unknown'

# %%
def get_artist_info(sp, artist_name):
    try:
        if sp:
            results = sp.search(q=artist_name, type='artist', limit=1)
            if results['artists']['items']:
                artist = results['artists']['items'][0]
                artist_id = artist['id']
                artist_info = sp.artist(artist_id)
                popularity = artist_info['popularity']
                followers = artist_info['followers']['total']
                genres = artist_info['genres']
                genre = genres[0] if genres else 'Unknown'
                return popularity, followers, genre
    except Exception as e:
        print(f"Failed to get info for artist {artist_name}: {e}")
    return None, None, 'Unknown'

# %%
def apply_functions(df, sp):
    artist_cols = ['artist_popularity', 'artist_followers', 'genre']
    df[artist_cols] = df['artname'].map_partitions(lambda x: x.map(lambda y: get_artist_info(sp, y))).apply(pd.Series)
    track_cols = ['track_popularity', 'audio_features']
    df = df.dropna(subset=artist_cols, how='all').reset_index(drop=True)
    combined_cols = artist_cols + track_cols
    df = pd.concat([df.drop(columns=combined_cols), df[combined_cols]], axis=1)
    df[track_cols] = df.apply(lambda row: get_track_info(sp, row['track_name'], row['artname']), axis=1).apply(pd.Series)
    return df

# %%
def save_data(df, path):
    df.compute().to_csv(path, index=False)

# %%
if __name__ == '__main__':
    file_path = 'dataset.csv'
    output_path = 'dataset_modified.csv'
    sp = authenticate_spotify()
    ddf = load_data(file_path)
    df = apply_functions(ddf, sp)
    save_data(df, output_path)


# %%



