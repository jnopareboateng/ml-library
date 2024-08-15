import pandas as pd
import numpy as np
import uuid
import logging
import math
import json
import gradio as gr

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# load dataset
SONGS_FILE = "../data/Spotify_MPD_Feature_Engineered.csv"
S = 50  # Hyper Parameter
totReco = 0  # Number of total recommendation till now
startConstant = 5  # for low penalty in starting phase
# Read data
Songs = pd.read_csv(SONGS_FILE)
NFEATURE = Songs.shape[1] - 60  # Number of Features (excluding all columns before 'Artiste Popularity')
ratedSongs = set()
userRecommendations = {}  # Store user recommendations
users = {}

# Function definitions

def register_user_interface(name, age, gender, country, edu_level):
    """
    Registers a new user in the system.
    Parameters:
        name (str): The name of the user.
        age (int): The age of the user.
        gender (str): The gender of the user.
        country (str): The country of the user.
        edu_level (str): The education level of the user.
    Returns:
        str: A message indicating successful registration or an error message.
    """
    # Check for unique user name and ID
    if any(user['name'] == name for user in users.values()):
        return "User name already exists. Please choose a different name."
    user_id = str(uuid.uuid4())
    users[user_id] = {
        "name": name,
        "age": age,
        "gender": gender,
        "country": country,
        "edu_level": edu_level,
        "features": np.zeros(NFEATURE + 4, dtype=np.float64).tolist(),
        "rated_songs": set()
    }
    users[user_id] = list(users[user_id].items())
    
    with open(f'user_data_{user_id}.json', 'w', encoding='utf-8') as file:
        json.dump(users, file)
    return f"User {name} registered successfully. Your user ID is {user_id}"

def login_user_interface(user_id):
    """
    Logs in an existing user.
    Parameters:
        user_id (str): The unique identifier of the user.
    Returns:
        str: A welcome message or an error message if the user ID is invalid.
    """
    # Implementation here...

    if user_id not in users:
        return "Invalid user ID. Please register first."
    try:
        with open(f"user_data_{user_id}.json", "r", encoding='utf-8') as file:
            user_data = json.load(file)
            if "features" in user_data:
                user_data["features"] = np.array(user_data["features"])
            if "rated_songs" in user_data:
                user_data["rated_songs"] = set(user_data["rated_songs"])
            users[user_id] = user_data
    except FileNotFoundError:
        return "User data not found. Please register first."

    return f"Welcome back, {users[user_id]['name']}!"
def compute_utility(user_features, song_features, epoch, s=S):
    """
    Computes the utility score for a song based on user preferences.
    Parameters:
        user_features (np.array): The feature vector of the user.
        song_features (np.array): The feature vector of the song.
        epoch (int): The current epoch of the recommendation system.
        s (int): The hyperparameter S for the utility calculation.
    Returns:
        float: The computed utility score.
    """
    # Implementation here...
    if epoch < 0:
        raise ValueError("Epoch must be a non-negative integer.")
    if s <= 0:
        raise ValueError("S must be a positive integer.")
    
    user_features = user_features.copy()
    song_features = song_features.copy()
    dot = user_features.dot(song_features)
    ee = (1.0 - 1.0 * math.exp(-1.0 * epoch / s))
    res = dot * ee
    return res

def get_song_features(song):
    """
    Retrieves the feature vector for a given song.
    Parameters:
        song (pd.Series or pd.DataFrame): The song data.
    Returns:
        np.array: The feature vector of the song.
    """
    # Implementation here...
    if isinstance(song, pd.Series):
        features = song.iloc[12:-48].values
    elif isinstance(song, pd.DataFrame):
        features = song.iloc[0, 12:-48].values
    else:
        raise TypeError("Song must be a pandas Series or DataFrame.")
    return features.astype(np.float64)


def update_features(user_features, song_features, rating, t):
    """
    Updates the user's feature vector based on a song rating.
    Parameters:
        user_features (np.array): The current feature vector of the user.
        song_features (np.array): The feature vector of the rated song.
        rating (int): The rating given to the song.
        t (int): The total number of recommendations made so far.
    Returns:
        np.array: The updated feature vector of the user.
    """
    # Implementation here...
    if rating < 1 or rating > 5:
        raise ValueError("Rating must be an integer between 1 and 5.")
    if t < 0:
        raise ValueError("Total recommendations must be a non-negative integer.")
    
    user_features = np.array(user_features)
    song_features = np.array(song_features)
    
    impact_factor = (rating - 3) / 2  # Scale the impact factor based on the rating (-1 to 1)
    user_features[:-4] = user_features[:-4].astype(np.float64)  # Convert user_features[:-4] to float64
    
    # Check for NaN values in song_features and replace them with 0
    song_features = np.nan_to_num(song_features)
    
    user_features[:-4] += song_features * impact_factor
    return user_features

def rate_song_interface(user_id, song_id, rating):
    """
    Rates a song and updates the user's preferences.
    Parameters:
        user_id (str): The unique identifier of the user.
        song_id (int): The unique identifier of the song.
        rating (int): The rating given to the song.
    Returns:
        str: A message indicating successful rating or an error message.
    """
    # Implementation here...
    if user_id not in users:
        return "Invalid user ID. Please register or login first."
    if song_id not in Songs.index:
        return "Invalid song ID."
    user_features = users[user_id]["features"]
    song_features = get_song_features(Songs.loc[song_id])
    user_features = update_features(user_features, song_features, rating, totReco)
    users[user_id]["rated_songs"].add(song_id)
    return f"Song rated successfully. Your preferences have been updated."

def get_user_data(user_id):
    """
    Retrieves the data for a given user.
    Parameters:
        user_id (str): The unique identifier of the user.
    Returns:
        dict or None: The user data or None if the user ID is invalid.
    """
    # Implementation here...
    if user_id not in users:
        return None
    return users[user_id]

def get_song_genre(song):
    """
    Determines the genres of a given song.
    Parameters:
        song (pd.Series): The song data.
    Returns:
        list: A list of genres associated with the song.
    """
    # Implementation here...
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

def get_recommendations_interface(user_id):
    """
    Provides song recommendations for a user.
    Parameters:
        user_id (str): The unique identifier of the user.
    Returns:
        list, str: A list of song recommendations and a justification message.
    """
    # Implementation here...
    if user_id not in users:
        return "Invalid user ID. Please register or login first."
    user_features = users[user_id]["features"]
    liked_genres = {genre for genre, value in enumerate(user_features[12:-4]) if value > 0}
    recommendations, justification = get_recommendations(initialize_q_table(), user_features, user_id, liked_genres)
    recommendations_list = recommendations.apply(lambda song: f"{song['Music']} by {song['artname']}", axis=1).tolist()
    return recommendations_list, justification

def choose_action(q_table, user_features, epsilon, rated_songs):
    """
    Chooses an action (song recommendation) based on the Q-table.
    Parameters:
        q_table (np.array): The Q-table for the recommendation system.
        user_features (np.array): The feature vector of the user.
        epsilon (float): The exploration rate for the epsilon-greedy policy.
        rated_songs (set): A set of song IDs that have already been rated.
    Returns:
        int: The chosen action (song ID).
    """
    # Implementation here...
    unrated_songs = Songs.index.difference(rated_songs)
    print(f"Unrated songs: {unrated_songs}")
    print(f"Unrated songs type: {type(unrated_songs)}")
    print(f"Unrated songs length: {len(unrated_songs)}")


def get_recommendations(q_table, user_features, user_id, liked_genres):
    """
    Generates song recommendations based on the Q-table and user preferences.
    Parameters:
        q_table (np.array): The Q-table for the recommendation system.
        user_features (np.array): The feature vector of the user.
        user_id (str): The unique identifier of the user.
        liked_genres (set): A set of genres liked by the user.
    Returns:
        pd.DataFrame, str: A DataFrame of song recommendations and a justification message.
    """
    # Implementation here...
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

def initialize_q_table():
    """
    Initializes the Q-table with zeros.
    Returns:
        np.array: The initialized Q-table.
    """
    # Implementation here...
    q_table = np.zeros((len(Songs), NFEATURE + 4))
    return q_table


# Gradio interface functions

def gradio_register_user(name, age, gender, country, edu_level):
    """
    Gradio interface to register a new user.
    Inputs:
        name: User's name
        age: User's age
        gender: User's gender
        country: User's country
        edu_level: User's education level
    Returns:
        A success message with the user ID or an error message if the user already exists.
    """
    try:
        return register_user_interface(name, age, gender, country, edu_level)
    except Exception as e:
        return f"An error occurred: {str(e)}"

def gradio_login_user(user_id):
    """
    Gradio interface to log in an existing user.
    Inputs:
        user_id: The user's unique identifier
    Returns:
        A welcome message or an error message if the user ID is invalid.
    """
    try:
        return login_user_interface(user_id)
    except Exception as e:
        return f"An error occurred: {str(e)}"

def gradio_rate_song(user_id, song_id, rating):
    """
    Gradio interface to rate a song.
    Inputs:
        user_id: The user's unique identifier
        song_id: The song's unique identifier
        rating: The rating given to the song
    Returns:
        A success message or an error message if the user ID or song ID is invalid.
    """
    try:
        return rate_song_interface(user_id, song_id, rating)
    except Exception as e:
        return f"An error occurred: {str(e)}"

def gradio_get_recommendations(user_id):
    """
    Gradio interface to get song recommendations for a user.
    Inputs:
        user_id: The user's unique identifier
    Returns:
        A list of song recommendations and a justification message or an error message if the user ID is invalid.
    """
    try:
        q_table = initialize_q_table()  # Initialize Q-table
        user_data = get_user_data(user_id)
        if user_data is None:
            return "Invalid user ID. Please register or login first."
        user_features = user_data["features"]
        liked_genres = {genre for genre, value in enumerate(user_features[12:-4]) if value > 0}
        recommendations, justification = get_recommendations(q_table, user_features, user_id, liked_genres)
        recommendations_list = recommendations.apply(lambda song: f"{song['Music']} by {song['artname']}", axis=1).tolist()
        return recommendations_list, justification
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Streamlit app
import streamlit as st
unique_countries = Songs['country'].unique().tolist()
def streamlit_app():
    """
    Streamlit application for the music recommendation system.
    """
    try:
        registration_message = ""
        st.title('Music Recommendation System')
        # User registration
        with st.form("register_user"):
            st.subheader("Register New User")
            name = st.text_input("Name")
            age = st.number_input("Age", min_value=0)
            gender = st.radio("Gender", ["Male", "Female", "Other"])
            country = st.selectbox("Country", unique_countries)
            edu_level = st.text_input("Education Level")
            submit_button = st.form_submit_button("Register")

            if submit_button:
                registration_message = register_user_interface(name, age, gender, country, edu_level)
                if "registered successfully" in registration_message:
                    st.success(registration_message)
                    # Extract user ID from the registration message
                    user_id = registration_message.split()[-1]
                    st.info(f"Your User ID is: {user_id}")
                else:
                    st.error(registration_message)

        # User login
        with st.form("login_user"):
            st.subheader("Login Existing User")
            user_id = st.text_input("User ID")
            login_button = st.form_submit_button("Login")

            if login_button:
                login_message = login_user_interface(user_id)
                if "Welcome back" in login_message:
                    st.success(login_message)
                else:
                    st.error(login_message)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

    # Assume other Streamlit components for rating songs and getting recommendations are here...

if __name__ == "__main__":
    streamlit_app()