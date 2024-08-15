# Song Recommender System

This project implements a Song Recommender System using a combination of **Reinforcement Learning**, **Content-Based Filtering**, and **Collaborative Filtering** techniques. The recommender system is built using Python, `scikit-learn`, and `Gradio` for the web interface.

## Features

- **Reinforcement Learning**: Users can rate the recommendations on a scale of 1 to 5, and these ratings are used to update the model.
- **Content-Based Filtering**: Recommends songs based on the content features of the items.
- **Collaborative Filtering**: Recommends songs based on user-item interactions.
- **User Input**: Users can input their age, education, gender, country, preferred genre of music, favorite artist, and username.
- **Real-Time Recommendations**: The system provides real-time song recommendations based on the user inputs and feedback.

## Installation

1. **Clone the Repository**
   ```sh
   git clone https://github.com/yourusername/song-recommender-system.git
   cd song-recommender-system
   ```

2. **Create and Activate a Virtual Environment**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Dependencies**
   ```sh
   pip install -r requirements.txt
   ```

## Data

The dataset `Synthetic_Data_With_Spotify_MPD.csv` should be placed in the project directory. Ensure the data contains the following columns:

- `age`
- `Education`
- `gender`
- `country`
- `Music`
- `artname`
- `featured_artists`
- `Genre`
- `usersha1`
- `plays`

## Usage

1. **Run the Recommender System**
   ```sh
   python recommender.py
   ```

2. **Access the Web Interface**
   Open the URL provided by Gradio in your web browser to interact with the recommender system.

## Code Explanation

### Importing Libraries

```python
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
```

### Defining LinUCB Class for Reinforcement Learning

```python
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
```

### Defining RLRecommender Class

```python
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
```

### Data Preprocessing

```python
data = pd.read_csv("Synthetic_Data_With_Spotify_MPD.csv")
data['featured_artists'].fillna('No Featured Artists', inplace=True)
data['Genre'].fillna('Unknown', inplace=True)

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

user_demographic_features = ['age', 'Education', 'gender', 'country']
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['age']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Education', 'gender', 'country'])
    ])

train_data_demographics = preprocessor.fit_transform(train_data[user_demographic_features])
test_data_demographics = preprocessor.transform(test_data[user_demographic_features])

train_data['combined_features'] = train_data[['Music', 'artname', 'featured_artists', 'Genre']].fillna('').agg(' '.join, axis=1)
test_data['combined_features'] = test_data[['Music', 'artname', 'featured_artists', 'Genre']].fillna('').agg(' '.join, axis=1)

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
train_tfidf_matrix = tfidf_vectorizer.fit_transform(train_data['combined_features'])
test_tfidf_matrix = tfidf_vectorizer.transform(test_data['combined_features'])

user_item_matrix = train_data.pivot_table(index='usersha1', columns='Music', values='plays', fill_value=0)

svd = TruncatedSVD(n_components=50)
user_item_matrix_reduced = svd.fit_transform(user_item_matrix)

rl_recommender = RLRecommender(alpha=0.1, d=train_data_demographics.shape[1] + train_tfidf_matrix.shape[1] + user_item_matrix_reduced.shape[1])

content_knn = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute')
content_knn.fit(train_tfidf_matrix)

collab_knn = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute')
collab_knn.fit(user_item_matrix_reduced)
```

### Recommendation Function

```python
def recommend_songs(age, education, gender, country, user_name, favorite_artist, preferred_genre, ratings=None, item_ids=None):
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
    
    if ratings is not None and item_ids is not None:
        for rating, item_id in zip(ratings, item_ids):
            item_id = int(item_id)
            item_context = items_context[item_id]
            item_latent_features = user_item_matrix_reduced[item_id]
            rl_recommender.update(user_context, item_context, user_latent_features, item_latent_features, rating)
    
    filtered_test_data = test_data[test_data['Genre'].str.contains(preferred_genre, case=False, na=False)]
    
    item_indices = content_knn.kneighbors(filtered_test_data[['combined_features']], return_distance=False)
    item_indices = item_indices[item_indices < len(filtered_test_data)]
    content_based_items = filtered_test_data.iloc[item_indices.flatten()]
    
    user_indices = collab_knn.kneighbors(user_latent_features.reshape(1, -1), return_distance=False)
    collab_based_items = filtered_test_data[filtered_test_data['usersha1'].isin(user_item_matrix.index[user_indices.flatten()])]
    
    recommended_items = pd.concat([content_based_items, collab_based_items]).drop_duplicates()
    
    recommended_indices = rl_recommender.recommend(user_context, items_context, user_latent_features)
    rl_recommended_items = filtered_test_data.iloc[recommended_indices]
    
    final_recommendations = pd.concat([recommended_items, rl_recommended_items]).drop_duplicates()
    
    return final_recommendations[['Music', 'artname', 'Genre', 'plays']]

def recommend_songs_interface(age, education, gender, country, user_name, favorite_artist, preferred_genre, ratings=None, item_ids=None):
    recommendations = recommend_songs(age, education, gender, country, user_name, favorite_artist, preferred_genre, ratings, item_ids)
    return recommendations.values.tolist()
```

### Gradio Interface

```python
iface = gr.Interface(
    fn=recommend_songs_interface, 
    inputs=[
        Slider(minimum=10, maximum=80, step=1, default=25, label="Age"),
        Dropdown(choices=['High School', 'Graduate', 'Post Graduate', 'Doctorate'], label="Education"),
        Radio(choices=['Male', 'Female'], label="Gender"),
        Dropdown(choices=list(train_data['country'].unique()), label="Country"),
        Textbox(lines=1, placeholder="Enter your name", label="User Name"),
        Textbox(lines=1, placeholder="Enter your favorite artist", label="Favorite Artist"),
        Dropdown(choices=list(train_data['Genre'].unique()), label="Preferred Genre"),
    ], 
    outputs=Dataframe(type="pandas", label="Recommended

 Songs"),
    title="Song Recommender",
    description="Enter your details to get song recommendations. You can provide ratings after receiving recommendations to improve the system."
)


iface.launch()
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any features, enhancements, or bug fixes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

