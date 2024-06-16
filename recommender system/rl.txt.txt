import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
import os
import random
import math
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from multiprocessing import Pool, cpu_count
import uuid
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a simple plot
fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[4, 5, 6])])

# Save the plot as an HTML file
pio.write_html(fig, file='plot.html', auto_open=False)

# Assuming the dataset is in the same directory
SONGS_FILE = "Spotify_MPD_Feature_Engineered.csv"
S = 50  # Hyper Parameter
totReco = 0  # Number of total recommendation till now
startConstant = 5  # for low penalty in starting phase

# Read data
Songs = pd.read_csv(SONGS_FILE)
NFEATURE = Songs.shape[1] - 60  # Number of Features (excluding all columns before 'Artiste Popularity')

ratedSongs = set()
userRecommendations = {}  # Store user recommendations

# User data structure
users = {}

def register_user():
    name = input("Enter your name: ")
    age = int(input("Enter your age: "))
    gender = input("Enter your gender (M/F/O): ")
    country = input("Enter your country: ")
    edu_level = input("Enter your education level (Graduate/High School/Middle School/Undergraduate): ")

    if name in users:
        print("User already exists. Please login.")
        return None

    user_id = str(uuid.uuid4())
    users[user_id] = {"name": name, "age": age, "gender": gender, "country": country, "edu_level": edu_level, "features": np.zeros(NFEATURE + 4, dtype=np.float64), "rated_songs": set()}
    userRecommendations[user_id] = []  # Initialize user recommendations
    print(f"User {name} registered successfully. Your user ID is {user_id}")
    return user_id

def login_user():
    user_id = input("Enter your user ID: ")
    if user_id not in users:
        print("Invalid user ID. Please register first.")
        return None
    
    # Load user data
    try:
        with open(f"user_data_{user_id}.json", "r") as file:
            user_data = json.load(file)
            user_data["features"] = np.array(user_data["features"])
            user_data["rated_songs"] = set(user_data["rated_songs"])
            users[user_id] = user_data
    except FileNotFoundError:
        pass
    
    # Load user ratings
    try:
        with open(f"user_ratings_{user_id}.json", "r") as file:
            users[user_id]["rated_songs"].update(json.load(file).keys())
    except FileNotFoundError:
        pass
    
    print(f"Welcome back, {users[user_id]['name']}!")
    return user_id

def get_user_data(user_id):
    if user_id not in users:
        return None
    return users[user_id]

def compute_utility(user_features, song_features, epoch, s=S):
    """ Compute utility U based on user preferences and song preferences """
    user_features = user_features.copy()
    song_features = song_features.copy()
    dot = user_features.dot(song_features)
    ee = (1.0 - 1.0 * math.exp(-1.0 * epoch / s))
    res = dot * ee
    return res

def get_song_features(song):
    """ Feature of particular song """
    if isinstance(song, pd.Series):
        features = song[12:-48].values  # Exclude all columns before 'Artiste Popularity' and after 'Genre_world-music'
        return features.astype(np.float64)  # Convert features to float64
    elif isinstance(song, pd.DataFrame):
        return get_song_features(pd.Series(song.loc[song.index[0]]))
    else:
        raise TypeError("{} should be a Series or DataFrame".format(song))

def get_song_genre(song):
    genres = []
    for col in Songs.columns[-48:]:
        try:
            if isinstance(song, pd.Series):
                value = song[col]
            else:
                value = getattr(song, col)

            if isinstance(value, pd.Series):
                if value.any():  # Check if any value in the Series is True
                    genres.append(col[6:])  # Remove the "Genre_" prefix
            elif value is True:
                genres.append(col[6:])  # Remove the "Genre_" prefix
        except AttributeError:
            pass  # Skip the attribute if it doesn't exist
    return genres

def initialize_q_table():
    """
    Initialize q_table with all values set to 0
    """
    q_table = np.zeros(NFEATURE + 4)
    return q_table

def epsilon_greedy_policy(q_table, epsilon):
    """
    Epsilon-greedy policy:
    With probability epsilon, choose random action
    """
    if np.random.uniform(0, 1) < epsilon:
        return True
    else:
        return False

def update_q_table(q_table, features, reward):
    """
    Update q_table with new q_value
    """
    features = features.astype(np.float64)  # Convert features to float
    q_table += features * reward
    return q_table

def choose_action(q_table, user_features, epsilon, rated_songs):
    """
    Choose an action (recommend a song) based on the Q-table and an exploration strategy,
    excluding songs that have already been rated by the user.
    """
    unrated_songs = Songs.index.difference(rated_songs)
    print(f"Unrated songs: {unrated_songs}")
    print(f"Unrated songs type: {type(unrated_songs)}")
    print(f"Unrated songs length: {len(unrated_songs)}")

    if epsilon_greedy_policy(q_table, epsilon):
        # Choose a random action (song) from the unrated songs
        action = Songs.loc[unrated_songs].sample(1).index[0]
    else:
        # Ensure q_table and user_features have the correct shapes
        q_table = q_table.squeeze()
        user_features = user_features.reshape(1, -1)

        print(f"Q-table shape: {q_table.shape}")
        print(f"User features shape: {user_features.shape}")
        print(f"Q-table values:\n{q_table}")
        print(f"User features values:\n{user_features}")

        # Calculate Q-values for all songs
        q_values = np.dot(q_table, user_features.T).squeeze()
        print(f"Q-values shape: {q_values.shape}")
        
        if q_values.ndim == 0:
            # If q_values is a scalar, broadcast it to an array with the same length as unrated_songs
            q_values = np.full(len(unrated_songs), q_values)
        else:
            # Ensure q_values has the same length as unrated_songs
            q_values = q_values[:len(unrated_songs)]
        
        # Filter out indices in unrated_songs that are out of bounds for q_values
        valid_indices = unrated_songs[unrated_songs < len(q_values)]
        
        # Filter out rated songs from Q-values
        unrated_q_values = q_values[valid_indices]
        
        # Replace NaN values with 0
        unrated_q_values = np.nan_to_num(unrated_q_values, 0)
        
        print(f"Unrated Q-values shape: {unrated_q_values.shape}")
        action = valid_indices[unrated_q_values.argmax()]

    print(f"Chosen action: {action}")
    return action
    
    
def update_features(user_features, song_features, rating, t):
    """
    Update user features based on the song features and user rating
    """
    impact_factor = (rating - 3) / 2  # Scale the impact factor based on the rating (-1 to 1)
    user_features[:-4] = user_features[:-4].astype(np.float64)  # Convert user_features[:-4] to float64
    
    # Check for NaN values in song_features and replace them with 0
    song_features = np.nan_to_num(song_features)
    
    user_features[:-4] += song_features * impact_factor
    return user_features

def get_recommendations(q_table, user_features, user_id, liked_genres):
    """
    Get recommendations based on the learned Q-table or policy
    """
    # Check if user features contain NaN values
    if np.isnan(user_features).any():
        logger.warning("User features contain NaN values. Replacing with 0.")
        user_features = np.nan_to_num(user_features)

    q_values = np.dot(q_table.reshape(1, -1), user_features.reshape(-1, 1))

    # Check if Q-values contain NaN values
    if np.isnan(q_values).any():
        logger.warning("Q-values contain NaN values. Skipping songs with NaN Q-values.")
    
    sorted_indices = q_values.argsort()[0][::-1]  # Sort indices in descending order
    
    recommendations = []
    for index in sorted_indices:
        song = Songs.iloc[index]
        song_genres = get_song_genre(song)
        if not np.isnan(q_values[0, index]) and q_values[0, index] > 0 and any(genre in liked_genres for genre in song_genres):
            recommendations.append(song)
        if len(recommendations) >= 10:
            break
    
    if not recommendations:
        # If no recommendations found based on liked genres, fallback to top Q-value songs
        for index in sorted_indices:
            if not np.isnan(q_values[0, index]) and q_values[0, index] > 0:
                recommendations.append(Songs.iloc[index])
            if len(recommendations) >= 10:
                break
    
    recommendations = pd.DataFrame(recommendations)
    
    # Provide justification for the recommendations
    justification = "The recommendations are based on your preferences, demographics, and the songs you have rated highly. "
    justification += "We have learned from your ratings and selected songs that align with your taste. "
    justification += "Songs that you rated lower have been filtered out to provide more relevant suggestions."
    
    return recommendations, justification

def convert_to_serializable(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {convert_to_serializable(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

def save_data(user_id, user_data, user_ratings, recommendations):
    # Convert all data to a serializable format
    user_data_serializable = convert_to_serializable(user_data)
    user_ratings_serializable = convert_to_serializable(user_ratings)
    recommendations_serializable = convert_to_serializable(recommendations)

    # Save user data
    with open(f"user_data_{user_id}.json", "w") as file:
        json.dump(user_data_serializable, file)

    # Save user ratings
    with open(f"user_ratings_{user_id}.json", "w") as file:
        json.dump(user_ratings_serializable, file)

    # Save recommendations
    with open(f"recommendations_{user_id}.json", "w") as file:
        json.dump(recommendations_serializable, file)
def reinforcement_learning(user_id, s=200, N=5, epsilon=0.5):
    global Songs
    Songs = Songs.copy()

    # Use user's features and rated songs
    user_data = get_user_data(user_id)
    user_features = user_data["features"]
    ratedSongs = user_data["rated_songs"]

    # Initialize Q-table
    q_table = initialize_q_table()

    print("Select song genres that you like")
    Genres = [col[6:] for col in Songs.columns[-48:]]  # Remove "Genre_" prefix from genre names
    for i in range(0, len(Genres)):
        print(str(i + 1) + ". " + Genres[i])
    choice = "y"
    liked_genres = set()
    while (choice.lower().strip() == "y"):
        num = input("Enter number associated with genre: ")
        try:
            genre_index = int(num) - 1
            if 0 <= genre_index < len(Genres):
                liked_genres.add(Genres[genre_index])
            else:
                print("Invalid genre number. Please enter a valid number.")
        except ValueError:
            print("Invalid input. Please enter a valid genre number.")
        choice = input("Do you want to add another genre? (y/n) ")
    for i in range(0, len(Genres)):
        if Genres[i] in liked_genres:
            user_features[i + 12] = 1.0 / len(liked_genres)  # Set feature values for liked genres

    # Update user features with demographic information
    user_data = get_user_data(user_id)
    user_features[-4] = user_data["age"] / 100  # Normalize age to 0-1 range
    user_features[-3] = 1 if user_data["gender"] == "M" else 0  # Binary encoding for gender
    user_features[-2] = 1 if user_data["country"] == "USA" else 0  # Binary encoding for country
    user_features[-1] = 1 if user_data["edu_level"] == "Graduate" else 0  # Binary encoding for education level

    print("\nUser features:")
    print(user_features)
    print("\nQ-table:")
    print(q_table)

    print("\n\nRate following " + str(N) + " songs. So that we can know your taste.\n")
    user_ratings = {}
    for t in range(N):
        # Choose an action (recommend a song) based on the Q-table and an exploration strategy
        action = choose_action(q_table, user_features, epsilon, ratedSongs)
        recommendation = Songs.loc[Songs.index == action]

        # Get user's rating (reward) for the recommended song
        recommendation_features = get_song_features(recommendation)
        featured_artists = recommendation['featured_artists'].iloc[0] if 'featured_artists' in recommendation.columns and not pd.isna(recommendation['featured_artists'].iloc[0]) else ''
        
        while True:
            user_rating = input(f'How much do you like "{recommendation["Music"].iloc[0]}" by {recommendation["artname"].iloc[0]} {f"(feat. {featured_artists})" if featured_artists else ""} (Duration: {str(recommendation["Duration"].iloc[0])} mins) (Genre: {", ".join(get_song_genre(recommendation))}) (1-5): ')
            try:
                user_rating = int(user_rating)
                if 1 <= user_rating <= 5:
                    break
                else:
                    print("Invalid rating. Please enter a rating between 1 and 5.")
            except ValueError:
                print("Invalid input. Please enter a valid rating between 1 and 5.")
        
        user_ratings[recommendation["Music"].iloc[0]] = user_rating
        reward = user_rating / 5.0  # Scale rating from 1-5 to 0-1

        # Update user's features and rated songs (next state)
        user_features = update_features(user_features, recommendation_features, user_rating, t)
        ratedSongs.add(recommendation.index[0])
        next_state = user_features  # The next state is the updated user features

        # Update the Q-table based on the observed reward and next state
        q_table = update_q_table(q_table, user_features, reward)

    # Recommend songs based on the learned Q-table or policy
    recommendations, justification = get_recommendations(q_table, user_features, user_id, liked_genres)
    print("\n\nBased on your preferences, here are some recommendations for you:\n")
    for i, song in enumerate(recommendations.itertuples()):
        featured_artists = song.featured_artists if 'featured_artists' in recommendations.columns and not pd.isna(song.featured_artists) else ''
        print(f"{i+1}. {song.Music} by {song.artname} {f'(feat. {featured_artists})' if featured_artists else ''} (Duration: {str(song.Duration)} mins) (Genre: {', '.join(get_song_genre(song))})")
    
    print(f"\nJustification: {justification}")
    
    print("\nYour ratings:")
    for song, rating in user_ratings.items():
        print(f"{song}: {rating}")
    
    # Save user data, ratings, and recommendations
    save_data(user_id, users[user_id], user_ratings, recommendations)
    
    # Save the Q-table
    with open(f"q_table_{user_id}.json", "w") as file:
        json.dump(q_table.tolist(), file)

def main():
   user_id = register_user()  # You can also use login_user() if the user already exists
   if user_id:
       reinforcement_learning(user_id)

if __name__ == "__main__":
   main()