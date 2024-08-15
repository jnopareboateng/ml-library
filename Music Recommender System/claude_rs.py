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
import gradio as gr
from gradio.components import Slider, Dropdown, Radio, Textbox, Dataframe

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
        scores = [self.model.select_arm(np.concatenate([user_context, item.toarray().flatten(), user_latent_features])) for item in items_context]
        item_indices = np.argsort(scores)[-n:]
        return item_indices
    
    def update(self, user_context, item_context, user_latent_features, item_latent_features, rating):
        reward = (rating - 3) / 2  # Normalize the rating to [-1, 1]
        self.model.update(np.concatenate([user_context, item_context.toarray().flatten(), user_latent_features, item_latent_features]), reward)

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
train_data['combined_features'] = train_data[['Music', 'artname', 'featured_artists', 'Genre']].fillna('').agg(' '.join, axis=1)
test_data['combined_features'] = test_data[['Music', 'artname', 'featured_artists', 'Genre']].fillna('').agg(' '.join, axis=1)

tfidf_vectorizer = TfidfVectorizer()
train_tfidf_matrix = tfidf_vectorizer.fit_transform(train_data['combined_features'])
test_tfidf_matrix = tfidf_vectorizer.transform(test_data['combined_features'])

# %%
# Preprocess user-item interactions
user_item_matrix = train_data.pivot_table(index='usersha1', columns='Music', values='plays', fill_value=0)

# %%
# Perform matrix factorization (e.g., SVD)
svd = TruncatedSVD(n_components=50)
user_item_matrix_reduced = svd.fit_transform(user_item_matrix)

# %%
# Initialize the reinforcement learning recommender
rl_recommender = RLRecommender(alpha=0.1, d=train_data_demographics.shape[1] + train_tfidf_matrix.shape[1] + user_item_matrix_reduced.shape[1])

# %%
# Content-Based Filtering with KNN
content_knn = NearestNeighbors(n_neighbors=10, metric='cosine')
content_knn.fit(train_tfidf_matrix)

# %%
# Collaborative Filtering with KNN
collab_knn = NearestNeighbors(n_neighbors=10, metric='cosine')
collab_knn.fit(user_item_matrix_reduced)

# %%
def recommend_songs(age, education, gender, country, user_name, favorite_artist, rating=None, item_id=None):
    user_data = pd.DataFrame({
        'age': [age],
        'Education': [education],
        'gender': [gender],
        'country': [country]
    })

    user_data_demographics = preprocessor.transform(user_data)
    user_context = user_data_demographics[0]
    
    items_context = test_tfidf_matrix[:100]  # Using a subset of items for demonstration
    
    user_id = test_data['usersha1'].iloc[0]  # Assuming a single user for demonstration
    user_latent_features = user_item_matrix_reduced[user_item_matrix.index.get_loc(user_id)]
    
    if rating is not None and item_id is not None:
        # Update the recommender with user feedback
        item_id = int(item_id)
        item_context = items_context[item_id]
        item_latent_features = user_item_matrix_reduced[item_id]
        rl_recommender.update(user_context, item_context, user_latent_features, item_latent_features, rating)
    
    # Content-Based Filtering with KNN
    item_indices = content_knn.kneighbors(items_context, return_distance=False)
    item_indices = item_indices[item_indices < len(test_data)]  # Ensure indices are within bounds
    content_based_items = test_data.iloc[item_indices.flatten()]
    
    # Collaborative Filtering with KNN
    user_indices = collab_knn.kneighbors(user_latent_features.reshape(1, -1), return_distance=False)
    collab_based_items = test_data[test_data['usersha1'].isin(user_item_matrix.index[user_indices.flatten()])]
    
    # Combine content-based and collaborative filtering recommendations
    recommended_items = pd.concat([content_based_items, collab_based_items]).drop_duplicates()
    
    # Reinforcement Learning
    recommended_indices = rl_recommender.recommend(user_context, items_context, user_latent_features)
    rl_recommended_items = test_data.iloc[recommended_indices]
    
    # Combine all recommendations
    final_recommendations = pd.concat([recommended_items, rl_recommended_items]).drop_duplicates()
    
    return final_recommendations[['Music', 'artname', 'Genre', 'plays']]

def recommend_songs_interface(age, education, gender, country, user_name, favorite_artist):
    # Call the original function and convert the output to a list of lists
    recommendations = recommend_songs(age, education, gender, country, user_name, favorite_artist)
    return recommendations.values.tolist()

# Define the Gradio interface
iface = gr.Interface(
    fn=recommend_songs_interface, 
    inputs=[
        Slider(minimum=10, maximum=80, step=1, default=25, label="Age"),
        Dropdown(choices=['High School', 'Graduate', 'Post Graduate', 'Doctorate'], label="Education"),
        Radio(choices=['Male', 'Female'], label="Gender"),
        Dropdown(choices=list(train_data['country'].unique()), label="Country"),
        Textbox(lines=1, placeholder="Enter your name", label="User Name"),
        Textbox(lines=1, placeholder="Enter your favorite artist", label="Favorite Artist")
    ], 
    outputs=Dataframe(type="pandas", label="Recommended Songs"),
    title="Song Recommender",
    description="Enter your details to get song recommendations."
)

# Launch the interface
iface.launch()
