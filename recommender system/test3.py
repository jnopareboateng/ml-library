import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

# Load the dataset
df = pd.read_csv('your_dataset.csv')

# Handle missing values
imputer = SimpleImputer(strategy='mean')
df[['plays', 'artiste_popularity']] = imputer.fit_transform(df[['plays', 'artiste_popularity']])

# Normalize numerical features
scaler = StandardScaler()
df[['age', 'plays', 'artiste_popularity']] = scaler.fit_transform(df[['age', 'plays', 'artiste_popularity']])

# Encode categorical variables
categorical_features = ['gender', 'country', 'genre']
one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
transformer = ColumnTransformer(transformers=[('cat', one_hot_encoder, categorical_features)], remainder='passthrough')
df_encoded = transformer.fit_transform(df)

# Feature engineering (example: creating a user engagement level feature)
df['engagement_level'] = df['plays'] * df['rating']

# Dimensionality reduction
pca = PCA(n_components=0.95)  # Retain 95% of variance
df_reduced = pca.fit_transform(df_encoded)

# Split the data train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_reduced, df['rating'], test_size=0.2, random_state=42)


# Now you can proceed with building your hybrid recommender system
