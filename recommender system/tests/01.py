# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import silhouette_score
from typing import Any, Union, Callable

# %%
df = pd.read_csv('../data/dataset.csv')

# %%
df.head()

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
df = pd.read_csv('../data/cleaned_data.csv')

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
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 25, 40, 60, 100], labels=['Teen', 'Young Adult', 'Adult', 'Senior', 'Elderly'])

df['age_group'].unique()

# %% [markdown]
# Average Plays per User

# %%
df['avg_plays_per_user'] = df.groupby('usersha1')['plays'].transform('mean')
# df['avg_plays_per_user'].unique()

# %% [markdown]
# Aggregrate Popularity Metrics by User Demographics

# %%
df['avg_artiste_popularity'] = df.groupby(['age_group', 'gender'])['artiste_popularity'].transform('mean')
df['avg_audio_popularity'] = df.groupby(['age_group', 'gender'])['audio_popularity'].transform('mean')


# %% [markdown]
# Summary Statistics for Audio Features by User

# %%
audio_features = ['music_acousticness', 'danceability', 'energy', 'loudness', 'speechiness', 'instrumentalness', 'liveness', 'valence', 'tempo']
for feature in audio_features:
    df[f'avg_{feature}_by_demo'] = df.groupby(['age_group', 'gender'])[feature].transform('mean')


# %% [markdown]
# Temporal Features:
# Extracting Year, Month and Day from Release Date

# %%
df['release_year'] = pd.to_datetime(df['release_date']).dt.year
df['release_month'] = pd.to_datetime(df['release_date']).dt.month
df['release_day'] = pd.to_datetime(df['release_date']).dt.day


# %% [markdown]
# Calculate Song's Age Since Release:

# %%
current_year = pd.to_datetime('today').year
df['song_age'] = current_year - df['release_year']


# %% [markdown]
# User Preferences:

# %%
user_profiles = df.groupby('usersha1')['genre'].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown')
df['user_favorite_genre'] = df['usersha1'].map(user_profiles)


# %% [markdown]
# Calculate Diversity of Genres Each User Listens To:

# %%
df['user_genre_diversity'] = df.groupby('usersha1')['genre'].transform(lambda x: x.nunique())

# %% [markdown]
# Interaction Features:
# Capture Interactions Between User Demographics and Song Characteristics:

# %%
df['age_genre_interaction'] = df['age'] * df['genre'].apply(lambda x: hash(x) % 100)


# %% [markdown]
# Create Features that Measure User Affinity Towards Certain Genres or Artists:

# %%
user_genre_affinity = df.groupby(['usersha1', 'genre'])['plays'].sum().unstack().fillna(0)
df['user_genre_affinity'] = df.apply(lambda row: user_genre_affinity.loc[row['usersha1'], row['genre']], axis=1)


# %% [markdown]
# Advanced Audio Features:
# Create Composite Features from Existing Audio Features:

# %%
df['energy_acousticness_ratio'] = df['energy'] / (df['music_acousticness'] + 1e-9)


# %% [markdown]
# Verification  checks

# %%
# Checking the average plays per user calculation
avg_plays_per_user_check = df.groupby('usersha1')['plays'].mean()
assert df['avg_plays_per_user'].equals(df['usersha1'].map(avg_plays_per_user_check))

# Checking the average artiste popularity by demographics
avg_artiste_popularity_check = df.groupby(['age_group', 'gender'])['artiste_popularity'].mean()
assert df['avg_artiste_popularity'].equals(df[['age_group', 'gender']].apply(lambda row: avg_artiste_popularity_check[row['age_group'], row['gender']], axis=1))

# Checking the average audio popularity by demographics
avg_audio_popularity_check = df.groupby(['age_group', 'gender'])['audio_popularity'].mean()
assert df['avg_audio_popularity'].equals(df[['age_group', 'gender']].apply(lambda row: avg_audio_popularity_check[row['age_group'], row['gender']], axis=1))


# %%
for feature in audio_features:
    avg_feature_by_demo_check = df.groupby(['age_group', 'gender'])[feature].mean()
    assert df[f'avg_{feature}_by_demo'].equals(df[['age_group', 'gender']].apply(lambda row: avg_feature_by_demo_check[row['age_group'], row['gender']], axis=1))


# %%
df['release_year_check'] = pd.to_datetime(df['release_date']).dt.year
df['release_month_check'] = pd.to_datetime(df['release_date']).dt.month
df['release_day_check'] = pd.to_datetime(df['release_date']).dt.day

assert df['release_year'].equals(df['release_year_check'])
assert df['release_month'].equals(df['release_month_check'])
assert df['release_day'].equals(df['release_day_check'])


# %%
df['release_year_check'] = pd.to_datetime(df['release_date']).dt.year
df['release_month_check'] = pd.to_datetime(df['release_date']).dt.month
df['release_day_check'] = pd.to_datetime(df['release_date']).dt.day

assert df['release_year'].equals(df['release_year_check'])
assert df['release_month'].equals(df['release_month_check'])
assert df['release_day'].equals(df['release_day_check'])


# %%
df['song_age_check'] = current_year - df['release_year']
assert df['song_age'].equals(df['song_age_check'])


# %%
# Adding a feature for season based on release month
df['release_season'] = df['release_month'].apply(lambda x: 'Winter' if x in [12, 1, 2] else 'Spring' if x in [3, 4, 5] else 'Summer' if x in [6, 7, 8] else 'Fall')


# %%
user_profiles_check = df.groupby('usersha1')['genre'].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown')
assert df['user_favorite_genre'].equals(df['usersha1'].map(user_profiles_check))


# %%
user_genre_diversity_check = df.groupby('usersha1')['genre'].nunique()
assert df['user_genre_diversity'].equals(df['usersha1'].map(user_genre_diversity_check))


# %%
# Checking the interaction term
df['age_genre_interaction_check'] = df['age'] * df['genre'].apply(lambda x: hash(x) % 100)
assert df['age_genre_interaction'].equals(df['age_genre_interaction_check'])


# %%
# Checking user genre affinity
df['user_genre_affinity_check'] = df.apply(lambda row: user_genre_affinity.loc[row['usersha1'], row['genre']], axis=1)
assert df['user_genre_affinity'].equals(df['user_genre_affinity_check'])


# %%
# Checking the energy_acousticness_ratio calculation
df['energy_acousticness_ratio_check'] = df['energy'] / (df['music_acousticness'] + 1e-9)
assert df['energy_acousticness_ratio'].equals(df['energy_acousticness_ratio_check'])


# %% [markdown]
# **Feature Selection**

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
df.drop(columns=['Unnamed: 0', 'usersha1'], inplace=True)

# Identify features
categorical_features = ['education', 'gender', 'country', 'age_group', 'user_favorite_genre', 'release_season']
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
indices = [35, 42, 17, 33, 12, 9, 11, 10, 16, 13]
selected_columns = df.columns[indices]

# Print the names of the selected columns
print(selected_columns)

# selecte columns are ['avg_music_acousticness_by_demo', 'avg_valence_by_demo', 'mode',
    #    'avg_artiste_popularity', 'music_acousticness', 'plays',
    #    'audio_popularity', 'artiste_popularity', 'loudness', 'danceability']


# %%
df.shape

# %%
df.to_csv('../data/preprocessed_data.csv')

# %%



