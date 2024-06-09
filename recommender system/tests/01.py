# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import silhouette_score
from typing import Any, Union, Callable
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import randint as sp_randint

# %%
df = pd.read_csv('../data/dataset.csv')

# %%
# df.set_index('user_id', inplace=True)

# %%
# df.head()

# %%
print(df.shape) # prints the dimensions of the DataFrame

missing_values = np.where(df.isnull(), 1, 0).sum()
print("\nMissing Values:\n", missing_values)

# %%
# Remove rows containing missing values
df_clean = df.dropna()

# prints the dimensions of the cleaned DataFrame
df_clean.shape


# %%
# df

# %%
# df = pd.read_csv('../data/cleaned_data.csv')

# %%
duplicate_records = df.duplicated().sum()
print(duplicate_records)


# %%
df.shape

# %%
# Summary statistics
summary_stats = df.describe()

# Histograms
df['age'].hist()
plt.title('Age Distribution')
plt.show()

# Box plots for audio features
sns.boxplot(data=df[['music_acousticness', 'danceability', 'energy']])
plt.title('Audio Features Distribution')
plt.show()

# Scatter plots for correlations
sns.scatterplot(x='age', y='plays', data=df)
plt.title('Age vs. Plays')
plt.show()


# %%
# AGE GROUPING
# Example: Age groups
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 25, 40, 60, 100], labels=['Teenager', 'Young Adult', 'Adult', 'Mature Adult', 'Elderly'])

df['age_group'].unique()

# %% [markdown]
# Average Plays per User
# 

# %%
# # Reset the index to perform groupby on the original column
# df_reset = df.reset_index()

# # Calculate user profiles
# user_profiles = df_reset.groupby('user_id')['genre'].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown')

# # Map the user profiles back to the original DataFrame
# df_reset['user_favorite_genre'] = df_reset['user_id'].map(user_profiles)

# # Optionally set the index again
# df_reset.set_index('user_id', inplace=True)



# %%
df['avg_plays_per_user'] = df.groupby('user_id')['plays'].transform('mean')
# df['avg_plays_per_user'].unique()

# %% [markdown]
# Aggregrate Popularity Metrics by User Demographics
# 

# %%
df['avg_artiste_popularity'] = df.groupby(['age_group', 'gender'])['artiste_popularity'].transform('mean')
df['avg_audio_popularity'] = df.groupby(['age_group', 'gender'])['audio_popularity'].transform('mean')


# %% [markdown]
# Summary Statistics for Audio Features by User
# 

# %%
audio_features = ['music_acousticness', 'danceability', 'energy', 'loudness', 'speechiness', 'instrumentalness', 'liveness', 'valence', 'tempo']
for feature in audio_features:
    df[f'avg_{feature}_by_demo'] = df.groupby(['age_group', 'gender'])[feature].transform('mean')


# %% [markdown]
# Temporal Features:
# Extracting Year, Month and Day from Release Date
# 

# %%
df['release_year'] = pd.to_datetime(df['release_date']).dt.year
df['release_month'] = pd.to_datetime(df['release_date']).dt.month
df['release_day'] = pd.to_datetime(df['release_date']).dt.day


# %% [markdown]
# Calculate Song's Age Since Release:
# 

# %%
current_year = pd.to_datetime('today').year
df['song_age'] = current_year - df['release_year']


# %% [markdown]
# User Preferences:
# 

# %%
user_profiles = df.groupby('user_id')['genre'].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown')
df['user_favorite_genre'] = df['user_id'].map(user_profiles)


# %% [markdown]
# Calculate Diversity of Genres Each User Listens To:
# 

# %%
df['user_genre_diversity'] = df.groupby('user_id')['genre'].transform(lambda x: x.nunique())

# %% [markdown]
# Interaction Features:
# Capture Interactions Between User Demographics and Song Characteristics:
# 

# %%
df['age_genre_interaction'] = df['age'] * df['genre'].apply(lambda x: hash(x) % 100)


# %% [markdown]
# Create Features that Measure User Affinity Towards Certain Genres or Artists:
# 

# %%
df = df.dropna(subset=['user_id', 'genre'])

user_genre_affinity = df.groupby(['user_id', 'genre'])['plays'].sum().unstack().fillna(0)
df['user_genre_affinity'] = df.apply(lambda row: user_genre_affinity.loc[row['user_id'], row['genre']], axis=1)

# %% [markdown]
# Advanced Audio Features:
# Create Composite Features from Existing Audio Features:
# 

# %%
df['energy_acousticness_ratio'] = df['energy'] / (df['music_acousticness'] + 1e-9)


# %%
# Adding a feature for season based on release month
df['release_season'] = df['release_month'].apply(lambda x: 'Winter' if x in [12, 1, 2] else 'Spring' if x in [3, 4, 5] else 'Summer' if x in [6, 7, 8] else 'Fall')


# %% [markdown]
# ### **Feature Selection**
# 

# %%
# Assuming df is already loaded and preprocessed
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()

# Calculate correlation matrix
correlation_matrix = df[numeric_features].corr()

# Plot the heatmap
plt.figure(figsize=(30, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix of Numeric Features")
plt.show()

# %%
# Drop unnecessary columns
# df.drop(columns=['Unnamed: 0'], inplace=True)

# %%
df.columns

# %%


# Identify features
categorical_features = ['education', 'gender', 'country', 'age_group', 'user_favorite_genre']
numerical_features = df.select_dtypes(include=[float, int]).columns.tolist()

# Remove the target variable from numerical features
numerical_features.remove('plays')  # Assuming 'plays' is the target variable

# Define preprocessing for numerical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Define preprocessing for categorical features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the target variable
X = df.drop(columns=['plays'])
y = df['plays']

# Apply preprocessing to the features
X_preprocessed = preprocessor.fit_transform(X)

# Split the preprocessed data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Checking the shape of the preprocessed data
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)


# %%
# Keep a copy of the DataFrame for later use
X_train_df = pd.DataFrame(X_train)  # Convert numpy array back to DataFrame if needed

# Fit a RandomForestRegressor to analyze feature importance
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Get feature importances
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Get the top 10 features
print('Top 10 Important features')
top_10_features = indices[:10]

# Print the feature ranking
print("Feature ranking:")
for f in top_10_features:
    print(f"{f + 1}. feature {X_train_df.columns[f]} ({importances[f]})")

# Use the original DataFrame to get the feature names
selected_features = X_train_df.columns[top_10_features]
print("Selected Features:")
print(selected_features)

# Plot the feature importances
plt.figure(figsize=(15, 5))
plt.title("Feature Importances")
plt.bar(range(10), importances[top_10_features], color="r", align="center")
# Use feature names as x-ticks
plt.xticks(range(10), selected_features, rotation=45)
plt.xlim([-1, 10])
plt.show()


# %%
# get me the printed array of items indexed by these from the dataframe 

# Assuming 'df' is your DataFrame

selected_columns = df.columns[selected_features]

# Print the names of the selected columns
print(selected_columns)


# %%
df.shape

# %%
# import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

def identify_multicollinearity(df, numerical_features, corr_threshold=0.7, vif_threshold=10):
    """
    Identify multicollinearity in the dataset by calculating the correlation matrix
    and Variance Inflation Factor (VIF).

    Parameters:
    df (DataFrame): The input DataFrame with numerical features.
    numerical_features (list): List of numerical feature names in the DataFrame.
    corr_threshold (float): The threshold for identifying high correlation.
    vif_threshold (float): The threshold for identifying high VIF.

    Returns:
    tuple: A tuple containing sets of highly correlated features and high VIF features.
    """

    # Task 1: Identify Multicollinearity
    correlation_matrix = df[numerical_features].corr()
    highly_correlated_features = set()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > corr_threshold:
                colname = correlation_matrix.columns[i]
                highly_correlated_features.add(colname)

    # Task 2: Implement VIF
    vif_data = df[numerical_features]
    vif = pd.DataFrame()
    vif["Features"] = vif_data.columns
    vif["VIF"] = [variance_inflation_factor(vif_data.values, i) for i in range(vif_data.shape[1])]
    high_vif_features = vif[vif["VIF"] > vif_threshold]["Features"]

    return highly_correlated_features, high_vif_features

# Example usage:
# Assuming 'df' is your DataFrame and 'numerical_features' is your list of numerical feature names
correlated_features, high_vif_features = identify_multicollinearity(df, numerical_features)
print("Highly correlated features:", correlated_features)
print("Features with high VIF (>10):", high_vif_features)


# %%
# List of features to drop based on correlation and VIF
features_to_drop = [
    'avg_audio_popularity',
    'duration', 'avg_loudness_by_demo', 'avg_speechiness_by_demo', 
    'release_year','release_month','avg_music_acousticness_by_demo',
    'avg_danceability_by_demo', 'avg_energy_by_demo','avg_instrumentalness_by_demo',
    'avg_liveness_by_demo', 'avg_valence_by_demo', 'avg_tempo_by_demo', 'release_season'
    
    # Add other highly correlated features identified in the correlation matrix
]
# Dropping the selected features
df_reduced = df.drop(columns=features_to_drop)

# Checking the remaining features
print(df_reduced.columns)


# %%
df_reduced.shape

# %%
# Create a correlation matrix
corr_matrix = df_reduced.corr()

# Plot the correlation heatmap
plt.figure(figsize=(30, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Features with High VIF')
plt.show()

# %%
df.set_index('user_id', inplace=True)

# %%
# Retaining important features
important_features = [
    'age_group','education','country','music','artist_name',
    'featured_artists', 'avg_plays_per_user', 'song_age', 'user_favorite_genre',
    'user_genre_diversity', 'user_genre_affinity', 'energy_acousticness_ratio'
    # Add any other retained features here after checking their importance
]

df_final = df_reduced[important_features]


# %%
df_final

# %%
# df_final.to_csv('../data/important_features.csv')

# %%
df = df_final

# %%

# Select relevant features
subset_features = [
    'age_group', 
    'education', 
    'country', 
    'music', 
    'artist_name', 
    'featured_artists', 
    'avg_plays_per_user', 
    'song_age', 
    'user_favorite_genre', 
    'user_genre_diversity', 
    'user_genre_affinity', 
    'energy_acousticness_ratio'
]

# Create a new DataFrame with the selected features
df_subset = df[subset_features]

# %%
df.info()

# %%
# Further preprocessing

# Define preprocessing for numeric features
numeric_features = [
    'avg_plays_per_user', 
    'song_age', 
    'user_genre_diversity', 
    'user_genre_affinity', 
    'energy_acousticness_ratio'
]
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Define preprocessing for categorical features
categorical_features = ['age_group', 'education', 'country', 'user_favorite_genre']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Define preprocessing for text features
text_features = ['artist_name', 'featured_artists']
text_transformer = Pipeline(steps=[
    ('tfidf', TfidfVectorizer(max_features=500)),
    ('svd', TruncatedSVD(n_components=50))  # Reducing dimensions
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('artist_tfidf', text_transformer, 'artist_name'),
        ('featured_artists_tfidf', text_transformer, 'featured_artists')
    ])


# %%
# Apply preprocessing to the features
df_subset = df_subset.fillna('')

# Apply preprocessing to the features
X = preprocessor.fit_transform(df_subset)


# %%
X.shape

# %%
# Encode the target variable 'music'
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df_subset['music'])


# %%
# Split the data into training and test setsa
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)


# Define the parameter grid to search
param_dist = {
    'n_estimators': [500,1000,1500,2000],
    'max_depth': sp_randint(3, 20),
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': sp_randint(2, 11),
    'min_samples_leaf': sp_randint(2, 11),
    'bootstrap': [True, False],
    'criterion': ['absolute_error', 'squared_error']
}

# Initialize the RandomizedSearchCV object
random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, cv=5,verbose=2, random_state=42)

# Fit the model to the data
random_search.fit(X_train, y_train)

# Print the best parameters and the corresponding score
print(f"Best Parameters: {random_search.best_params_}")
print(f"Best Score: {random_search.best_score_}")

# Predict using the best estimator
y_pred = random_search.best_estimator_.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy with Hyperparameter Tuning: {accuracy}')

# Convert predicted labels back to original music titles
y_pred_labels = label_encoder.inverse_transform(y_pred)

# Display a few predictions
print("Sample Predictions with Hyperparameter Tuning:")
for i in range(10):
    print(f"Predicted: {y_pred_labels[i]}, Actual: {label_encoder.inverse_transform([y_test[i]])[0]}")

# %%
from sklearn.metrics.pairwise import cosine_similarity

# Assuming X is the preprocessed feature matrix
cosine_sim = cosine_similarity(X)

# Function to get recommendations based on content similarity
def get_content_based_recommendations(user_idx, top_n=10):
    sim_scores = list(enumerate(cosine_sim[user_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]  # Exclude the user itself
    item_indices = [i[0] for i in sim_scores]
    return df_subset['music'].iloc[item_indices]

# Example usage
# user_idx = 0  # Replace with actual user index
# content_based_recs = get_content_based_recommendations(user_idx)
# print(content_based_recs)


# %% [markdown]
# Numbers on the left are the `userids`

# %%
user_idx = 25  # Replace with actual user index
content_based_recs = get_content_based_recommendations(user_idx)
print(content_based_recs)

# %% [markdown]
# Collaborative Filtering
# 

# %%
# # Prepare the data for surprise library
# reader = Reader(rating_scale=(1, 5))
# data = Dataset.load_from_df(df_subset[['user_id', 'music_encoded', 'rating']], reader)

# # Train-test split
# trainset, testset = surprise_train_test_split(data, test_size=0.2)

# # Use SVD for collaborative filtering
# svd = SVD()
# svd.fit(trainset)
# predictions = svd.test(testset)

# # Evaluate the model
# accuracy.rmse(predictions)

# # Function to get collaborative filtering recommendations
# def get_collaborative_recommendations(user_id, top_n=10):
#     user_ratings = df_subset[df_subset['user_id'] == user_id]
#     unrated_items = df_subset[~df_subset['music_encoded'].isin(user_ratings['music_encoded'])]
#     predictions = [svd.predict(user_id, item_id) for item_id in unrated_items['music_encoded']]
#     predictions = sorted(predictions, key=lambda x: x.est, reverse=True)
#     top_predictions = predictions[:top_n]
#     top_items = [label_encoder.inverse_transform([pred.iid])[0] for pred in top_predictions]
#     return top_items

# # Example usage
# user_id = 0  # Replace with actual user ID
# collaborative_recs = get_collaborative_recommendations(user_id)
# print(collaborative_recs)


# %% [markdown]
# Reinforcement Learning
# 

# %%
# import random
# from collections import defaultdict

# # Initialize Q-table
# Q = defaultdict(lambda: np.zeros(len(df_subset['music'].unique())))

# # Parameters
# alpha = 0.1  # Learning rate
# gamma = 0.6  # Discount factor
# epsilon = 0.1  # Exploration-exploitation trade-off

# def choose_action(state):
#     if random.uniform(0, 1) < epsilon:
#         return random.choice(range(len(df_subset['music'].unique())))  # Explore
#     else:
#         return np.argmax(Q[state])  # Exploit

# def update_q_table(state, action, reward, next_state):
#     best_next_action = np.argmax(Q[next_state])
#     td_target = reward + gamma * Q[next_state][best_next_action]
#     td_error = td_target - Q[state][action]
#     Q[state][action] += alpha * td_error

# # Example interaction loop
# for _ in range(10000):  # Number of episodes
#     state = user_id
#     action = choose_action(state)
#     next_state = random.choice(range(len(df_subset['user_id'].unique())))  # Simulate next state
#     reward = random.choice([1, -1])  # Simulate reward
#     update_q_table(state, action, reward, next_state)

# # Function to get RL-based recommendations
# def get_rl_recommendations(user_id, top_n=10):
#     state = user_id
#     actions = np.argsort(Q[state])[-top_n:]  # Get top N actions
#     top_items = [label_encoder.inverse_transform([action])[0] for action in actions]
#     return top_items

# # Example usage
# rl_recs = get_rl_recommendations(user_id)
# print(rl_recs)


# %% [markdown]
# Hybrid recommender system
# 

# %%
# def get_hybrid_recommendations(user_id, content_weight=0.3, collaborative_weight=0.4, rl_weight=0.3, top_n=10):
#     content_recs = get_content_based_recommendations(user_id, top_n)
#     collaborative_recs = get_collaborative_recommendations(user_id, top_n)
#     rl_recs = get_rl_recommendations(user_id, top_n)

#     # Aggregate recommendations
#     combined_recs = pd.concat([content_recs, collaborative_recs, rl_recs]).value_counts().index.tolist()
#     return combined_recs[:top_n]

# # Example usage
# hybrid_recs = get_hybrid_recommendations(user_id)
# print(hybrid_recs)


# %%
# df_subset

# %%


# # Load your data
# df = df_subset  

# # Assume your data has the following columns: age_group, education, country, user_favorite_genre, artist_name, avg_plays_per_user, song_age, user_genre_diversity, user_genre_affinity, energy_acousticness_ratio, music (item), user_id

# # Preprocessing pipeline
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', StandardScaler(), ['avg_plays_per_user', 'song_age', 'user_genre_diversity', 'user_genre_affinity', 'energy_acousticness_ratio']),
#         ('cat', OneHotEncoder(), ['age_group', 'education', 'country', 'user_favorite_genre'])
#     ]
# )

# # Fit and transform the demographic data
# demographic_data = df[['age_group', 'education', 'country', 'user_favorite_genre', 'avg_plays_per_user', 'song_age', 'user_genre_diversity', 'user_genre_affinity', 'energy_acousticness_ratio']]
# demographic_data_transformed = preprocessor.fit_transform(demographic_data)

# # Add the user_id column to the transformed data
# user_ids = df['user_id'].values
# demographic_data_transformed = pd.DataFrame(demographic_data_transformed)
# demographic_data_transformed['user_id'] = user_ids


# %%
# from sklearn.model_selection import train_test_split

# # Split the data
# train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# # Transform the training demographic data
# train_demographic_data = train_data[['age_group', 'education', 'country', 'user_favorite_genre', 'avg_plays_per_user', 'song_age', 'user_genre_diversity', 'user_genre_affinity', 'energy_acousticness_ratio']]
# train_demographic_data_transformed = preprocessor.fit_transform(train_demographic_data)
# train_demographic_data_transformed = pd.DataFrame(train_demographic_data_transformed)
# train_demographic_data_transformed['user_id'] = train_data['user_id'].values

# # Train the KNN model
# knn = NearestNeighbors(n_neighbors=5, algorithm='auto', metric='euclidean')
# knn.fit(train_demographic_data_transformed.drop(columns=['user_id']))

# # Evaluate the model
# # (Implement evaluation logic, e.g., precision, recall, etc.)



