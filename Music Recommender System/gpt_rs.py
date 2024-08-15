#%%
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
import implicit
import gradio as gr
import random

#%%
# Load the dataset
df = pd.read_csv('dataset.csv')

# Handle missing values
df['featured_artists'].fillna('No Featured Artists', inplace=True)
df['genre'].fillna('Unknown', inplace=True)

# Convert categorical columns to numerical
df['gender'] = df['gender'].astype('category').cat.codes
df['education'] = df['education'].astype('category').cat.codes

# Normalize 'plays' column to a 0-1 scale to represent ratings
scaler = MinMaxScaler()
df['rating'] = scaler.fit_transform(df[['plays']])

# Handle sparse classes in 'genre'
value_counts = df['genre'].value_counts()
to_remove = value_counts[value_counts <= 10].index
df['genre'].replace(to_remove,'Other', inplace=True)


#%%
# Train-test split
train, test = train_test_split(df, test_size=0.2, random_state=42)

# One-hot encoding for genre and featured artists on training set only
one_hot = OneHotEncoder()
transformed_data_train = one_hot.fit_transform(train[['genre', 'featured_artists']])

# Matrix Creation
def create_user_artists_matrix(df):
    user_artists = df[['usersha1', 'artist_name', 'rating']]
    user_artists.set_index(['usersha1', 'artist_name'], inplace=True)

    # Map string indices to integer indices
    user_index = {user: i for i, user in enumerate(user_artists.index.get_level_values(0).unique())}
    artist_index = {artist: i for i, artist in enumerate(user_artists.index.get_level_values(1).unique())}

    coo = sp.coo_matrix(
        (user_artists.rating.astype(float), 
         ([user_index[user] for user in user_artists.index.get_level_values(0)], 
          [artist_index[artist] for artist in user_artists.index.get_level_values(1)]))
    )
    return coo.tocsr(), user_index, artist_index

train_matrix, user_index, artist_index = create_user_artists_matrix(train)
test_matrix, _, _ = create_user_artists_matrix(test)

#%%
# Instantiate ALS using implicit
implicit_model = implicit.als.AlternatingLeastSquares(factors=50, iterations=20, regularization=0.01)
implicit_model.fit(train_matrix)

# Instantiate recommender
recommender = implicit_model

# Define encoding dictionaries
genre_encoding = {genre: i for i, genre in enumerate(df['genre'].unique())}
education_encoding = {education: i for i, education in enumerate(df['education'].unique())}

# Add 'Unknown' if not present in encoding dictionaries
if 'Unknown' not in genre_encoding:
    genre_encoding['Unknown'] = len(genre_encoding)
if 'Unknown' not in education_encoding:
    education_encoding['Unknown'] = len(education_encoding)

# Define decoding dictionaries
genre_decoding = {i: genre for genre, i in genre_encoding.items()}
education_decoding = {i: education for education, i in education_encoding.items()}

# Function to map user inputs to encoded values
def map_user_inputs_to_encoded(genre, education):
    encoded_genre = genre_encoding.get(genre, genre_encoding['Unknown'])
    encoded_education = education_encoding.get(education, education_encoding['Unknown'])
    return encoded_genre, encoded_education

# Define Reinforcement Learning Recommender class
class ReinforcementLearningRecommender:
    def __init__(self, recommender, learning_rate=0.1, discount_factor=0.9):
        self.recommender = recommender
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = {}  # State-action values

    def get_state(self, user_id):
        user_profile = user_profiles[user_id]
        return tuple(user_profile)

    def get_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(df['artist_name'].unique()))
        return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(df['artist_name'].unique()))
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(len(df['artist_name'].unique()))

        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

    def recommend(self, user_id, n=10):
        state = self.get_state(user_id)
        action = self.get_action(state)
        recommendations, _ = self.recommender.recommend(user_id, train_matrix, n=n)
        return recommendations

# Instantiate RL recommender
rl_recommender = ReinforcementLearningRecommender(recommender)

# Function to handle user feedback
def handle_user_feedback(user_id, feedback):
    state = rl_recommender.get_state(user_id)
    for item, rating in feedback.items():
        action = artist_index[item]
        reward = rating
        next_state = rl_recommender.get_state(user_id)
        rl_recommender.update_q_table(state, action, reward, next_state)

# Initialize user profiles with item features
item_features = train[['music_acousticness', 'danceability', 'energy', 'key', 'loudness', 'mode', 
                    'speechiness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 
                    'explicit', 'duration', 'rating']].values

user_profiles = {user_id: item_features.mean(axis=0) for user_id in train['usersha1'].unique()}

# Function to check if a user exists
def check_user_exists(user_id):
    return user_id in user_profiles

# Function to register a new user
def register_user(user_id, age, gender, education, country, genre, favourite_artists, train_matrix=train_matrix):
    user_profiles[user_id] = {
        'age': age,
        'gender': gender,
        'education': education,
        'country': country,
        'genre': genre,
        'favourite_artists': favourite_artists,
        'ratings': {}
    }
        # Add user preferences to train_matrix
    user_preferences = np.zeros(train_matrix.shape[1])  # Initialize with zeros
    for artist in favourite_artists:
        if artist in artist_index:
            user_preferences[artist_index[artist]] = 1  # Set preference to 1 for favourite artists
    train_matrix = sp.vstack([train_matrix, user_preferences])  # Add user preferences to train_matri
# Function to handle user feedback (ratings)
def handle_user_feedback(user_id, feedback):
    user_profiles[user_id]['ratings'].update(feedback)

# Hybrid Recommendation
def hybrid_recommend_with_rl(user_id: int, n: int = 10) -> list:
    if check_user_exists(user_id):
        num_users = train_matrix.shape[0]
        # num_artists = train_matrix.shape[1]

        if not(0 <= user_id < num_users):
            raise ValueError(f"User ID not found in the training data. Please provide a valid user ID.")
        
        # Get CF recommendations
        cf_artists, cf_scores = recommender.recommend(user_id, train_matrix, N=n)
        
        # Get CBF recommendations
        user_profile = user_profiles[user_id]
        similarity = cosine_similarity([user_profile], item_features)
        cbf_recommendations = np.argsort(-similarity[0])[:n]
        cbf_artists = df.iloc[cbf_recommendations]['artist_name'].tolist()

        # Get RL recommendations
        rl_recommendations = rl_recommender.recommend(user_id, n=n)

        # Get weighted average of CF, CBF, and RL recommendations
        combined_recommendations = []
        for artist in set(cf_artists + cbf_recommendations + rl_recommendations):
            cf_score = cf_scores[cf_artists.index(artist)] if artist in cf_artists else 0
            cbf_score = similarity[0, artist] if artist in cbf_recommendations else 0
            rl_score = 1
            combined_score = 0.5 * cf_score + 0.3 * cbf_score + 0.2 * rl_score  # Weighted average
            combined_recommendations.append((artist, combined_score))

        # Sort combined recommendations by score
        combined_recommendations.sort(key=lambda x: x[1], reverse=True)
        combined_recommendations = [artist for artist, _ in combined_recommendations[:n]]

        return combined_recommendations
    else:
        return []

# Generate unique user ID
def generate_unique_user_id():
    return max(user_profiles.keys(), default=0) + 1

# Gradio App
demo_inputs_rl_with_ratings = [
    gr.Textbox(label="Username", type="text"),
    gr.Slider(minimum=10, maximum=80, step=1, default=18, label="Age"),
    gr.Radio(choices=['Male', 'Female'], label="Gender"),
    gr.Dropdown(label="Education", choices=df['education'].unique().tolist()),
    gr.Dropdown(label="Country", choices=df['country'].unique().tolist()),
    gr.Dropdown(label="Preferred Genre", choices=df['genre'].unique().tolist()),
    gr.Textbox(lines=3, label="Favourite Artists (comma-separated)", type="text"),
    gr.Label("Rate each recommended music (1 to 5):"),
    *[gr.Slider(minimum=1, maximum=5, step=1, default=3, label="Rating") for _ in range(10)]
]

# Function to process inputs and get recommendations
def get_recommendations_and_update_model(username, age, gender, education, country, genre, favourite_artists, *ratings):
    encoded_genre, encoded_education = map_user_inputs_to_encoded(genre, education)
    user_id = generate_unique_user_id()
    register_user(user_id, age, gender, education, country, genre, favourite_artists, train_matrix=train_matrix)  # Add user to user_profiles
    recommendations = hybrid_recommend_with_rl(user_id)
    feedback = {artist: rating for artist, rating in zip(recommendations, ratings)}
    handle_user_feedback(user_id, feedback)
    return recommendations

# Define the Gradio interface function with rating inputs
iface_rl_with_ratings = gr.Interface(fn=get_recommendations_and_update_model,
                                     inputs=demo_inputs_rl_with_ratings,
                                     outputs="text")

# Launch the Gradio interface
iface_rl_with_ratings.launch()

# Evaluate model function
def evaluate_model(test_matrix, recommender, n=10):
    precisions = []
    recalls = []
    for user_id in range(test_matrix.shape[0]):
        if user_id in user_index.values():
            true_items = test_matrix[user_id].indices
            recommendations, _ = recommender.recommend(user_id, train_matrix, n=n)
            true_positives = len(set(recommendations) & set(true_items))
            precision = true_positives / n
            recall = true_positives / len(true_items)
            precisions.append(precision)
            recalls.append(recall)
    return np.mean(precisions), np.mean(recalls)

precision, recall = evaluate_model(test_matrix, recommender)
print(f'Precision: {precision}, Recall: {recall}')