import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import uuid
import json
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

# from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Load the dataset
@st.cache_data
def load_data():
    SONGS_FILE = "data/Spotify_MPD_Feature_Engineered.csv"
    Songs = pd.read_csv(SONGS_FILE)
    return Songs


# User data structure
users = {}


# Function to register a new user
def register_user():
    name = st.text_input("Enter your name: ")
    age = int(st.number_input("Enter your age: "))
    gender = st.selectbox("Enter your gender:", ["Male", "Female", "Other"])
    country = st.text_input("Enter your country: ")
    edu_level = st.selectbox(
        "Enter your education level:",
        ["Graduate", "High School", "Middle School", "Undergraduate"],
    )

    user_id = str(uuid.uuid4())
    users[user_id] = {
        "name": name,
        "age": age,
        "gender": gender,
        "country": country,
        "edu_level": edu_level,
        "features": np.zeros(4),
        "rated_songs": set(),
    }
    st.success(f"User {name} registered successfully. Your user ID is {user_id}")
    return user_id


# Function to login an existing user
def login_user():
    user_id = st.text_input("Enter your user ID: ")
    if user_id in users:
        st.success(f"Welcome back, {users[user_id]['name']}!")
        return user_id
    else:
        st.error("Invalid user ID. Please register first.")
        return None


# Function to compute utility
def compute_utility(user_features, song_features, epoch, s=50):
    user_features = user_features.copy()
    song_features = song_features.copy()
    dot = user_features.dot(song_features)
    ee = 1.0 - 1.0 * np.exp(-1.0 * epoch / s)
    res = dot * ee
    return res


# Function to get song features
def get_song_features(song):
    features = song[12:-48].values.astype(np.float64)
    return features


# Function to get song genre
def get_song_genre(song):
    genres = []
    for col in Songs.columns[-48:]:
        if song[col]:
            genres.append(col[6:])
    return genres


# Function to initialize Q-table
def initialize_q_table():
    q_table = np.zeros(4)
    return q_table


# Epsilon-greedy policy
def epsilon_greedy_policy(q_table, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return True
    else:
        return False


# Function to update Q-table
def update_q_table(q_table, features, reward):
    features = features.astype(np.float64)
    q_table += features * reward
    return q_table


# Function to choose an action (recommend a song)
def choose_action(q_table, user_features, epsilon, rated_songs):
    if len(unrated_songs) == 0:
        logger.warning("No unrated songs. No more actions to choose.")
        return None

    unrated_songs = list(set(Songs.index) - rated_songs)
    if epsilon_greedy_policy(q_table, epsilon):
        action = random.choice(unrated_songs)
    else:
        # Calculate Q-values for all songs
        q_values = np.dot(q_table, user_features.T).squeeze()
        action = unrated_songs[np.argmax(q_values[: len(unrated_songs)])]

    return action


# Function to update user features
def update_features(user_features, song_features, rating, t):
    impact_factor = (rating - 3) / 2
    user_features[:-4] = user_features[:-4].astype(np.float64)
    user_features[:-4] += song_features * impact_factor
    return user_features


# Function to get recommendations
def get_recommendations(q_table, user_features, user_id, liked_genres):
    user_features = np.nan_to_num(user_features)
    q_values = np.dot(q_table.reshape(1, -1), user_features.reshape(-1, 1))

    recommendations = []
    for index in np.argsort(q_values)[0][::-1]:
        song = Songs.iloc[index]
        song_genres = get_song_genre(song)
        if (
            not np.isnan(q_values[0, index])
            and q_values[0, index] > 0
            and any(genre in liked_genres for genre in song_genres)
        ):
            recommendations.append(song)
        if len(recommendations) >= 10:
            break

    if not recommendations:
        for index in np.argsort(q_values)[0][::-1]:
            if not np.isnan(q_values[0, index]) and q_values[0, index] > 0:
                recommendations.append(Songs.iloc[index])
            if len(recommendations) >= 10:
                break

    recommendations = pd.DataFrame(recommendations)
    justification = (
        "Recommendations are based on your preferences, demographics, and ratings."
    )
    return recommendations, justification


# Function to convert data to serializable format
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
        return {
            convert_to_serializable(k): convert_to_serializable(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


# Function to save user data, ratings, and recommendations
def save_data(user_id, user_data, user_ratings, recommendations):
    user_data_serializable = convert_to_serializable(user_data)
    user_ratings_serializable = convert_to_serializable(user_ratings)
    recommendations_serializable = convert_to_serializable(recommendations)

    with open(f"user_data_{user_id}.json", "w") as file:
        json.dump(user_data_serializable, file)

    with open(f"user_ratings_{user_id}.json", "w") as file:
        json.dump(user_ratings_serializable, file)

    with open(f"recommendations_{user_id}.json", "w") as file:
        json.dump(recommendations_serializable, file)


# Main reinforcement learning function
def reinforcement_learning(user_id, s=200, N=5, epsilon=0.5):
    global Songs
    Songs = load_data()

    user_data = users[user_id]
    user_features = user_data["features"]
    rated_songs = user_data["rated_songs"]

    q_table = initialize_q_table()

    st.write("Select song genres that you like:")
    Genres = [col[6:] for col in Songs.columns[-48:]]
    liked_genres = set()
    for i in range(len(Genres)):
        if st.checkbox(f"{i+1}. {Genres[i]} ", value=False):
            liked_genres.add(Genres[i])

    if len(liked_genres) > 0:
        user_features[12 : 12 + len(liked_genres)] = 1.0 / len(liked_genres)
    else:
        logger.warning("liked_genres is empty. Division by zero avoided.")

    user_features[-4] = user_data["age"] / 100
    user_features[-3] = 1 if user_data["gender"] == "Male" else 0
    user_features[-2] = 1 if user_data["country"] == "USA" else 0
    user_features[-1] = 1 if user_data["edu_level"] == "Graduate" else 0

    st.write("User features:")
    st.write(user_features)
    st.write("Q-table:")
    st.write(q_table)

    st.write("Rate the following", N, "songs to learn your taste:")

    user_ratings = {}
    for t in range(N):
        if len(rated_songs) == len(Songs):
            logger.warning("All songs have been rated. No more actions to choose.")
            break
        action = choose_action(q_table, user_features, epsilon, rated_songs)

        recommendation = Songs.loc[action]

        recommendation_features = get_song_features(recommendation)
        st.write(
            f"How much do you like {recommendation['Music']} by {recommendation['artname']}? (1-5):"
        )
        user_rating = st.number_input(min_value=1, max_value=5, step=1)
        user_ratings[recommendation["Music"]] = user_rating
        reward = user_rating / 5.0

        user_features = update_features(
            user_features, recommendation_features, user_rating, t
        )
        rated_songs.add(recommendation.name)
        q_table = update_q_table(q_table, user_features, reward)

    recommendations, justification = get_recommendations(
        q_table, user_features, user_id, liked_genres
    )
    st.write("Based on your preferences, here are some recommendations:")
    for i, song in recommendations.iterrows():
        st.write(
            f"{i+1}. {song['Music']} by {song['artname']} (Duration: {song['Duration']} mins, Genre: {', '.join(get_song_genre(song))})"
        )

    st.write(f"Justification: {justification}")
    st.write("Your ratings:")
    for song, rating in user_ratings.items():
        st.write(f"{song}: {rating}")

    save_data(user_id, user_data, user_ratings, recommendations)


# Main app function
def main():
    st.title("Music Recommender")

    user_id = login_user() or register_user()
    if user_id:
        reinforcement_learning(user_id)


if __name__ == "__main__":
    main()
