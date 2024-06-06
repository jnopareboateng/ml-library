# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import implicit
import gradio as gr
from scipy.sparse import csr_matrix

# %%
class LinUCB:
    def __init__(self, alpha, d):
        self.alpha = alpha
        self.d = d
        self.A = np.identity(d)
        self.b = np.zeros(d)
        
    def select_arm(self, x):
        A_inv = np.linalg.inv(self.A)
        theta = np.dot(A_inv, self.b)
        p = np.dot(theta, x) + self.alpha * np.sqrt(np.dot(x, np.dot(A_inv, x)))
        return p

    def update(self, x, reward):
        self.A += np.outer(x, x)
        self.b += reward * x

class RLRecommender:
    def __init__(self, alpha, d):
        self.model = LinUCB(alpha, d)
        
    def recommend(self, user_context, items_context, user_latent_features, n=6):
        scores = [self.model.select_arm(np.concatenate([user_context, item, user_latent_features])) for item in items_context]
        item_indices = np.argsort(scores)[-n:]
        return item_indices
    
    def update(self, user_context, item_context, user_latent_features, item_latent_features, rating):
        reward = (rating - 3) / 2  # Normalize the rating to [-1, 1]
        self.model.update(np.concatenate([user_context, item_context, user_latent_features, item_latent_features]), reward)

# %%
# Load and preprocess the data
data = pd.read_csv("Synthetic_Data_With_Spotify_MPD.csv")
data['featured_artists'].fillna('No Featured Artists', inplace=True)
data['Genre'].fillna('Unknown', inplace=True)

# %%
# Split the data into train and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# %%
# Preprocess user demographic features
user_demographic_features = ['age', 'Education', 'gender', 'country']
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['age']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Education', 'gender', 'country'])
    ])

train_data_demographics = preprocessor.fit_transform(train_data[user_demographic_features])
test_data_demographics = preprocessor.transform(test_data[user_demographic_features])

# %%
# Preprocess item features using TF-IDF vectorization
data['combined_features'] = data[['Music', 'artname', 'featured_artists', 'Genre']].fillna('').agg(' '.join, axis=1)
train_data['combined_features'] = train_data[['Music', 'artname', 'featured_artists', 'Genre']].fillna('').agg(' '.join, axis=1)
test_data['combined_features'] = test_data[['Music', 'artname', 'featured_artists', 'Genre']].fillna('').agg(' '.join, axis=1)

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(data['combined_features'])

# %%
# Preprocess user-item interactions
# Keep a reference to the original DataFrame
user_item_df = data.pivot_table(index='usersha1', columns='Music', values='plays', fill_value=0)

# Convert the DataFrame to a CSR matrix
user_item_matrix = csr_matrix(user_item_df.values)

# Collaborative Filtering using Implicit Alternating Least Squares (ALS)
model = implicit.als.AlternatingLeastSquares(factors=50, regularization=0.1, iterations=20)
model.fit(user_item_matrix.T)  # Transpose to fit the model

# %%
# Content-Based Filtering with KNN
content_knn = NearestNeighbors(n_neighbors=10, metric='cosine')
content_knn.fit(tfidf_matrix)

# %%

def create_user_profile(user_data):
    # Implement the function to create a user profile based on user_data
    pass

def hybrid_recommend(user_id, user_data=None, n_recommendations=10):
    if user_id in user_item_df.index:
        # Existing user: use collaborative filtering
        user_idx = user_item_df.index.get_loc(user_id)
        collab_scores = model.recommend(user_idx, user_item_matrix, N=n_recommendations)
    else:
        # New user: use content-based filtering
        # (Assuming user_data contains the user's demographic information and explicit preferences)
        user_profile = create_user_profile(user_data)  # You need to implement this function
        content_scores = cosine_similarity(user_profile, tfidf_matrix)
        collab_scores = [(idx, 0) for idx in range(len(content_scores))]

    # Combine scores
    hybrid_scores = np.array([score for _, score in collab_scores]) + content_scores.flatten()

    # Get top recommendations
    recommended_indices = np.argsort(hybrid_scores)[-n_recommendations:]
    recommended_songs = data.iloc[recommended_indices]

    return recommended_songs[['Music', 'artname', 'Genre', 'plays']]
# %%
alpha = 0.1  # Set the alpha parameter
d = 10  # Set the dimensionality of the context
recommender = RLRecommender(alpha, d)
def recommend_songs_interface(age, education, gender, country, favorite_artist, preferred_genre):
    # Create user profile based on input
    user_data = pd.DataFrame({
        'age': [age],
        'Education': [education],
        'gender': [gender],
        'country': [country],
        'favorite_artist': [favorite_artist],
        'preferred_genre': [preferred_genre]
    })

    user_data_demographics = preprocessor.transform(user_data)
    user_context = user_data_demographics[0]
    
    # Get a user ID (assuming a single user for demonstration purposes)
    user_id = test_data['usersha1'].iloc[0]


    # Get recommendations
    try:
        recommendations = recommender.recommend(user_id, user_context, test_data_demographics, n=10)
    except Exception as e:
        print(f"An error occurred: {e}")
        recommendations = []

    return recommendations.values.tolist()

# Load and preprocess the data
try:
    data = pd.read_csv("Synthetic_Data_With_Spotify_MPD.csv")
except FileNotFoundError:
    print("File not found. Please check the file path and try again.")
    data = pd.DataFrame()


# Define the Gradio interface
iface = gr.Interface(
    fn=recommend_songs_interface, 
    inputs=[
        gr.Slider(minimum=10, maximum=80, step=1, default=25, label="Age"),
        gr.Dropdown(choices=[' Undergraduate ', ' Middle School ', ' Graduate ',
       ' High School '], label="Education"),
        gr.Radio(choices=['Male', 'Female'], label="Gender"),
        gr.Dropdown(choices=list(train_data['country'].unique()), label="Country"),
        gr.Textbox(label="Favorite Artist"),
        gr.Dropdown(choices=list(train_data['Genre'].unique()), label="Preferred Genre"),
    ], 
    outputs=gr.Dataframe(type="pandas", label="Recommended Songs"),
    title="Song Recommender",
    description="Enter your details to get song recommendations. You can provide ratings after receiving recommendations to improve the system."
)

iface.launch()

# %%
