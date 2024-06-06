#%%
# Import libraries
import pandas as pd
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
df['Genre'].fillna('Unknown', inplace=True)

#%%
# check for missing df
df.isnull().sum()

#%%
# For simplicity, let's convert only 'gender' and 'Education'
df['gender'] = df['gender'].astype('category').cat.codes
df['Education'] = df['Education'].astype('category').cat.codes

# Extract user demographic information
user_demographics = df[['usersha1', 'age', 'Education', 'gender', 'country']]

# Display the first few rows of the dataframe
print(user_demographics.head())

# %%
