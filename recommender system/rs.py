#%%
# Import libraries
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

#%%
# Load the dataset
df = pd.read_csv('dataset.csv')
df.head()

#%%
# Handle missing values
df.isnull().sum()


# %%
df['featured_artists'].fillna('No Featured Artists', inplace=True)
df['genre'].fillna('Unknown', inplace=True)

#%%
# check for missing df
df.isnull().sum()

#%%
# For simplicity, let's convert only 'gender' and 'education'
df['gender'] = df['gender'].astype('category').cat.codes
df['education'] = df['education'].astype('category').cat.codes

# Extract user demographic information
user_demographics = df[['usersha1', 'age', 'education', 'gender', 'country']]


# Display the first few rows of the dataframe
print(user_demographics.head())

# %%
# Extract user listening information
user_listening = df[['usersha1', 'artist_name', 'music', 'genre', 'featured_artists']]
print(user_listening.head())

# %%
# Normalize 'plays' column to a 0-1 scale to represent ratings
scaler = MinMaxScaler()
df['rating'] = scaler.fit_transform(df[['plays']])

# %%
# Display the first few rows of the dataframe
df.head()

# %%
# One-hot encoding
one_hot = OneHotEncoder()
transformed_data = one_hot.fit_transform(df[['genre', 'featured_artists']])

#%%
# Text processing
# df['music'] = df['music'].str.lower().str.replace('[^\w\s]', '')
# df['artist_name'] = df['artist_name'].str.lower().str.replace('[^\w\s]', '')

# Handling sparse classes
value_counts = df['genre'].value_counts()
to_remove = value_counts[value_counts <= 10].index
df['genre'].replace(to_remove, 'Other', inplace=True)

#%%
df.shape
#%%
df.to_csv('preprocessed_data.csv', index=False)

#%%

# Train-test split
train, test = train_test_split(df, test_size=0.2, random_state=42)
# %%

# Item Profiles
item_features = df[['music_acousticness', 'danceability', 'energy',
                     'key', 'loudness', 'mode', 'speechiness', 'instrumentalness',
                       'liveness', 'valence', 'tempo', 'time_signature', 'explicit',
                         'duration', 'rating']]
item_features = MinMaxScaler().fit_transform(item_features)

# User Profiles
user_profiles = df.groupby('usersha1').apply(lambda x: np.average(item_features[x.index], weights=x['rating'], axis=0) if np.sum(x['rating']) != 0 else np.zeros(item_features.shape[1]))
user_profiles = np.array(user_profiles.tolist())
# Recommendation
similarity = cosine_similarity(user_profiles, item_features)
recommendations = np.argsort(-similarity)

# Get top 5 recommendations for the first user
top_5 = recommendations[0, :5]
recommended_songs = df.iloc[top_5]['music']
print(recommended_songs)
# %%
