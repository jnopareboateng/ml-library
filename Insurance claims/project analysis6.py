# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import scipy.stats as stats
from scipy.stats import skew
from scipy.stats import kurtosis
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler
import statsmodels.api as sm
from sklearn.naive_bayes import GaussianNB
%matplotlib inline

# %%
insurance_claims = pd.read_csv('insurance_claims.csv')

# %%
insurance_claims

# %%
selected_columns = insurance_claims[['months_as_customer', 'age', 'policy_state', 'insured_sex', 'insured_education_level','insured_occupation', 'auto_make', 'auto_model', 'auto_year', 'total_claim_amount']]

# %%
selected_columns

# %%
selected_columns['total_claim_amount'].describe()

# %%
data = selected_columns[['total_claim_amount']]

# %%
skewness = stats.skew(data)

# %%
kurtosis = stats.kurtosis(data)

# %%
print("skewness:",skewness)

# %%
print("Kurtosis:",kurtosis)

# %%
correlation_matrix = np.corrcoef(data, rowvar = False)

# %%
print("Correlation matrix:",correlation_matrix)

# %%
x_values = selected_columns['insured_sex']
y_values = selected_columns['total_claim_amount']
plt.bar(x_values,y_values)
plt.xlabel('insured_sex')
plt.ylabel('total_claim_amount')
plt.title('Bar chart of Total claims by the Sex')
plt.show()

# %%
x_values = selected_columns['policy_state']
y_values = selected_columns['total_claim_amount']
plt.bar(x_values,y_values)
plt.xlabel('policy_state')
plt.ylabel('total_claim_amount')
plt.title('Bar chart of Total claims by the Policy state')
plt.show()

# %%
x_values = selected_columns['insured_education_level']
y_values = selected_columns['total_claim_amount']
plt.bar(x_values,y_values)
plt.xlabel('insured_education_level')
plt.ylabel('total_claim_amount')
plt.title('Bar chart of Total claims by the Insured educational level')
plt.show()

# %%
x_values = selected_columns['insured_occupation']
y_values = selected_columns['total_claim_amount']
plt.bar(x_values,y_values)
plt.xlabel('insured_occupation')
plt.ylabel('total_claim_amount')
plt.xticks(rotation=45)
plt.title('Bar chart of Total claims by the Insured occupation')
plt.show()

# %%
x_values = selected_columns['age']
y_values = selected_columns['total_claim_amount']
plt.bar(x_values,y_values)
plt.xlabel('age')
plt.ylabel('total_claim_amount')
plt.title('Bar chart of Total claims by the Age')
plt.show()

# %%
x_values = selected_columns['auto_year']
y_values = selected_columns['total_claim_amount']
plt.bar(x_values,y_values)
plt.xlabel('auto_year')
plt.ylabel('total_claim_amount')
plt.title('Bar chart of Total claims by the Auto year')
plt.show()

# %%
x_values = selected_columns['months_as_customer']
y_values = selected_columns['total_claim_amount']
plt.bar(x_values,y_values)
plt.xlabel('months_as_customer')
plt.ylabel('total_claim_amount')
plt.title('Bar chart of Total claims by the Months as customer')
plt.show()

# %%
fig = px.scatter(selected_columns, x= 'months_as_customer', y='total_claim_amount',trendline = 'ols')
fig.show()

# %%
x_values = selected_columns['auto_make']
y_values = selected_columns['total_claim_amount']
plt.bar(x_values,y_values)
plt.xlabel('auto_make')
plt.ylabel('total_claim_amount')
plt.xticks(rotation=45)
plt.title('Bar chart of Total claims by the Auto make')
plt.show()

# %%
x_values = selected_columns['auto_model']
y_values = selected_columns['total_claim_amount']
plt.bar(x_values,y_values)
plt.xlabel('auto_model')
plt.ylabel('total_claim_amount')
plt.xticks(rotation=90)
plt.title('Bar chart of Total claims by the Auto model')
plt.show()

# %%
drop = selected_columns[['months_as_customer','age','total_claim_amount']]

# %%
plt.figure
sn.heatmap(drop.corr(),annot=True,cmap='PiYG')
plt.show()

# %%
selected_columns['insured_sex'].unique()

# %%
selected_columns['insured_education_level'].value_counts()

# %%
list = ['policy_state', 'insured_sex', 'insured_education_level', 'insured_occupation', 'auto_make', 'auto_model']
for list in list:
    print('The value counts for', list, 'are:\n', selected_columns[list].value_counts())
    print('==============================')
    

# %%
from sklearn.preprocessing import LabelEncoder

columns_to_encode = ['policy_state', 'insured_sex', 'insured_education_level', 'insured_occupation', 'auto_make', 'auto_model']
label_encoders = {}  # Store the label encoders for later use

for column in columns_to_encode:
    # encode each column with a label encoder
    label_encoder = LabelEncoder()
    transformed_column = label_encoder.fit_transform(selected_columns[column])
    selected_columns[column] = transformed_column
    label_encoders[column] = label_encoder  # Store the label encoder for future reference

# Create a dictionary to store the dataframes
encoded_df_dict = {}

for column, encoder in label_encoders.items():
    # Create a dataframe for each column
    encoded_df = pd.DataFrame({
        'Encoded Value': range(len(encoder.classes_)),
        'Original Class': encoder.classes_
    })
    encoded_df_dict[column] = encoded_df

# Now you can access the dataframe for each column
for column, df in encoded_df_dict.items():
    print(f"For column '{column}':")
    print(df)


# %%
selected_columns

# %%
# Convert categorical variables to numerical representations (one-hot encoding)
selected_columns = pd.get_dummies(selected_columns('insured_sex', 'insured_occupation','policy_state','auto_make',
                                                             'auto_model','insured_education_level','auto_year'))

# Plot the correlation heatmap
plt.figure(figsize=(10, 8))
sn.heatmap(selected_columns.corr(), annot=True, cmap='PiYG')
plt.show()

# %%
#Machine learning Algorithms

# %%
#Logistic Regression

# %%
selected_columns1 = selected_columns.astype(int)

# %%
selected_columns1

# %%
#Decision trees

# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Use pd.get_dummies() to one-hot encode the categorical columns
# df_encoded = pd.get_dummies(selected_columns, columns=['policy_state', 'insured_sex', 'insured_occupation','insured_education_level','auto_make','auto_model'])
# df_encoded = df_encoded.astype(int)

X = selected_columns.drop('total_claim_amount', axis=1)
y = selected_columns[['total_claim_amount']]

# Reshape y to a 1D array
y = np.ravel(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the decision tree classifier
decision_tree = DecisionTreeClassifier(random_state=42)

# Train the decision tree classifier on the training data
decision_tree.fit(X_train, y_train)

# Make predictions on the testing data
predictions = decision_tree.predict(X_test)

# Evaluate the model performance
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions,average='weighted', zero_division=1)
recall = recall_score(y_test, predictions, average='weighted', zero_division=1)

print("Accuracy:", accuracy)
print("Precision:",precision)
print("Recall:",recall)

# %%
#NB

# %%
# Perform one-hot encoding for categorical variables if needed
df_encoded = pd.get_dummies(selected_columns)

# Separate the features (X) and target variable (y)
X = df_encoded.drop('total_claim_amount', axis=1)
y = df_encoded['total_claim_amount']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform oversampling on the training data
oversampler = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

# Initialize and fit the Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test set
predictions = nb_model.predict(X_test)

# Evaluate the model performance
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='weighted', zero_division=1)
recall = recall_score(y_test, predictions, average='weighted', zero_division=1)
f1 = f1_score(y_test, predictions, average='weighted', zero_division=1)

# Print the performance metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# %%


# %%
# Calculate the counts of each class
class_counts = y_train_resampled.value_counts()

# Plot the bar chart
plt.figure(figsize=(10, 6))
class_counts.plot(kind='bar')
plt.title('Distribution of Target Variable after Oversampling')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=270)
plt.show()

# %%


# %%
df_encoded = df_encoded.astype(int)


# %%
print(df_encoded.head())

# %%


# %%


# %%


# %%
#random forest

# %%
from sklearn.ensemble import RandomForestClassifier

# %%
X = selected_columns.drop('total_claim_amount', axis=1)
y = selected_columns[['total_claim_amount']]

# Reshape y to a 1D array
y = np.ravel(y)

#split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the Random Forest classifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Make predictions on the test set
predictions = rf_model.predict(X_test)

# Evaluate the model performance
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='weighted', zero_division=1)
recall = recall_score(y_test, predictions, average='weighted', zero_division=1)
f1 = f1_score(y_test, predictions, average='weighted', zero_division=1)

# Print the performance metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# %%



