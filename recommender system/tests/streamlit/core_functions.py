import pandas as pd
import numpy as np
import uuid
import json
import math

# Load dataset
SONGS_FILE = "../../data/Spotify_MPD_Feature_Engineered.csv"
S = 50  # Hyper Parameter
totReco = 0  # Number of total recommendation till now
startConstant = 5  # for low penalty in starting phase
# Read data
Songs = pd.read_csv(SONGS_FILE)
NFEATURE = Songs.shape[1] - 60  # Number of Features (excluding all columns before 'Artiste Popularity')
ratedSongs = set()
userRecommendations = {}  # Store user recommendations
users = {}
Q = {}  # Initialize Q-table

# Function definitions (same as before)
# ... (include all the function definitions here) ...



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
        "rated_songs": list(set())
    }

    with open(f'user_data_{user_id}.json', 'w', encoding='utf-8') as file:
        json.dump(users[user_id], file)
    return f"User {name} registered successfully. Your user ID is {user_id}"


def login_user_interface(user_id):
    """
    Logs in an existing user.
    Parameters:
        user_id (str): The unique identifier of the user.
    Returns:
        str: A welcome message or an error message if the user ID is invalid.
    """
    if user_id not in users:
        return "Invalid user ID. Please register first."
    try:
        with open(f"user_data_{user_id}.json", "r", encoding='utf-8') as file:
            user_data = json.load(file)
            if "features" in user_data:
                user_data["features"] = np.array(user_data["features"])
            if "rated_songs" in user_data:
                user_data["rated_songs"] = set(user_data["rated_songs"]) # convert list back to set.
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
    global totReco

    if user_id not in users:
        return "Invalid user ID. Please register or login first."
    if song_id not in Songs.index:
        return "Invalid song ID."
    user_features = users[user_id]["features"]
    song_features = get_song_features(Songs.loc[song_id])
    user_features = update_features(user_features, song_features, rating, totReco)
    users[user_id]["features"] = user_features
    users[user_id]["rated_songs"].add(song_id)
    totReco += 1
    return f"Song rated successfully. Your preferences have been updated."


def get_user_data(user_id):
    """
    Retrieves the data for a given user.
    Parameters:
        user_id (str): The unique identifier of the user.
    Returns:
        dict or None: The user data or None if the user ID is invalid.
    """
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
            elif value:
                genres.append(col[6:])  # Remove the "Genre_" prefix
        except (KeyError, AttributeError):
            pass  # Handle missing genre columns or attributes
    return genres


def choose_action(user_id, user_features, epsilon, rated_songs, t):
    """
    Chooses a song recommendation based on the epsilon-greedy policy.
    Parameters:
        user_features (np.array): The feature vector of the user.
        epsilon (float): The exploration rate.
        rated_songs (set): The set of songs rated by the user.
        t (int): The total number of recommendations made so far.
    Returns:
        int: The index of the recommended song.
    """
    if np.random.rand() < epsilon:
        # Explore: Choose a random unrated song
        candidate_songs = set(Songs.index) - rated_songs - {t}  # Exclude already rated songs and current index
        return np.random.choice(list(candidate_songs))
    else:
        # Exploit: Choose the song with the highest Q-value
        max_value = float('-inf')
        best_song = None
        for i in range(len(Songs)):
            if i not in rated_songs and i != t:  # Exclude already rated songs and current index
                key = f"{user_id},{i}"  # Combine user ID and song index for Q-table lookup
                if key in Q:
                    value = Q[key]
                    if value > max_value:
                        max_value = value
                        best_song = i
        return best_song


def get_recommendations_interface(user_id, epsilon):
    """
    Provides song recommendations for a user.
    Parameters:
        user_id (str): The unique identifier of the user.
        epsilon (float): The exploration rate.
    Returns:
        list: A list of recommended song IDs.
    """
    if user_id not in users:
        return "Invalid user ID. Please register or login first."

    user_features = users[user_id]["features"]
    rated_songs = users[user_id]["rated_songs"]

    # Initialize exploration parameters
    alpha = 0.1  # Learning rate
    gamma = 0.9  # Discount factor

    # Generate recommendations
    recommendations = []
    for _ in range(10):  # Generate 10 recommendations
        t = len(Songs)  # Set current index to the end of the song data
        recommended_song_id = choose_action(user_id, user_features, epsilon, rated_songs, t)
        if recommended_song_id is not None:
            recommendations.append(recommended_song_id)

            # Update Q-table (if applicable)
            if recommended_song_id not in rated_songs:
                next_song_features = get_song_features(Songs.loc[recommended_song_id])
                reward = compute_utility(user_features, next_song_features, t)  # Assuming reward based on utility

                prev_q_value = Q.get(f"{user_id},{t - 1}", 0.0)
                updated_q_value = prev_q_value + alpha * (reward + gamma * max(Q.get(f"{user_id},{i}", 0.0) for i in range(len(Songs)) if i != t))
                Q[f"{user_id},{t - 1}"] = updated_q_value

    return [Songs.loc[song_id, 'track_id'] for song_id in recommendations]  # Return track IDs


def initialize_q_table():
    """
    Initializes the Q-table with zeros.
    Returns:
        dict: The initialized Q-table.
    """
    Q = {}
    for user_id in users:
        for i in range(len(Songs)):
            Q[f"{user_id},{i}"] = 0.0
    return Q
