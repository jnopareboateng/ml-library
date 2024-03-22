#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
%matplotlib inline
# %%
data = pd.read_csv('insurance_claims.csv')
data.head()
# %%
selected_features = ['policy_state','insured_sex', 'insured_education_level', 'insured_occupation', 'auto_make', 'auto_model']
X = data[selected_features] # features
y = data['total_claim_amount'] # target variable
X.head()
# %%
y.head()
# %%
# Check for missing values
print(X.isnull().sum())

# Impute missing values with the mean for numerical features
# data['feature_with_missing'] = data['feature_with_missing'].fillna(data['feature_with_missing'].mean())

# Impute missing values with the mode for categorical features
# data['categorical_feature'] = data['categorical_feature'].fillna(data['categorical_feature'].mode()[0])
# %%
# Label encoding for ordinal features
labeled_feature = X.copy()
label_encoder = LabelEncoder()
labeled_feature.loc[:, 'insured_education_level'] = label_encoder.fit_transform(labeled_feature['insured_education_level'])
# One-hot encoding for nominal features
encoded_features = X.copy()
ohe = OneHotEncoder()
encoded_features = ohe.fit_transform(X[['policy_state', 'insured_sex', 'insured_occupation', 'auto_make', 'auto_model']])
# %%
# encoded_features.head()
# %%

# Histogram of the target variable
plt.figure(figsize=(8, 6))
plt.hist(data['total_claim_amount'], bins=20)
plt.title('Distribution of Total Claim Amount')
plt.show()

# %%
# Boxplot of total claim amount by insured education level
plt.figure(figsize=(8, 6))
sns.boxplot(x='insured_education_level', y='total_claim_amount', data=data)
plt.title('Total Claim Amount by Education Level')
plt.show()
# %%
# Create and fit the model
nb_model = GaussianNB()
nb_model.fit(X, y)
# %%
