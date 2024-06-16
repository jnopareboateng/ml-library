# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('churn.csv')

# Display the first few rows of the dataset
print(data.head())

# Summary statistics
print(data.describe(include='all'))

# Check for missing values
print(data.isnull().sum())

# Visualize the distribution of 'Churn'
plt.figure(figsize=(6, 4))
sns.countplot(data['Churn'])
plt.title('Distribution of Churn')
plt.show()

# Visualize the distribution of 'Gender'
plt.figure(figsize=(6, 4))
sns.countplot(data['Gender'])
plt.title('Distribution of Gender')
plt.show()

# Visualize the distribution of 'College'
plt.figure(figsize=(10, 6))
sns.countplot(y=data['College'])
plt.title('Distribution of College')
plt.show()

# Visualize the distribution of 'Level'
plt.figure(figsize=(10, 6))
sns.countplot(data['Level'])
plt.title('Distribution of Level')
plt.show()

# Visualize the distribution of 'Residence'
plt.figure(figsize=(6, 4))
sns.countplot(data['Residence'])
plt.title('Distribution of Residence')
plt.show()

# Visualize the distribution of 'SIM_Usage'
plt.figure(figsize=(6, 4))
sns.countplot(data['SIM_Usage'])
plt.title('Distribution of SIM Usage')
plt.show()

# Visualize the distribution of 'Usage_Freq'
plt.figure(figsize=(10, 6))
sns.countplot(y=data['Usage_Freq'])
plt.title('Distribution of Usage Frequency')
plt.show()

# Visualize the distribution of 'Network_Strength'
plt.figure(figsize=(10, 6))
sns.countplot(data['Network_Strength'])
plt.title('Distribution of Network Strength')
plt.show()

# Visualize the distribution of 'Data_Exhaustion'
plt.figure(figsize=(6, 4))
sns.countplot(data['Data_Exhaustion'])
plt.title('Distribution of Data Exhaustion')
plt.show()

# Visualize the distribution of 'Other_Networks'
plt.figure(figsize=(6, 4))
sns.countplot(data['Other_Networks'])
plt.title('Distribution of Other Networks')
plt.show()

# Visualize the distribution of 'Monthly_Data_Usage'
plt.figure(figsize=(10, 6))
sns.histplot(data['Monthly_Data_Usage'], kde=True)
plt.title('Distribution of Monthly Data Usage')
plt.show()


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the data
data = pd.read_csv('churn.csv')

# Encoding categorical variables
le = LabelEncoder()
categorical_features = data.select_dtypes(include=['object']).columns
for feature in categorical_features:
    data[feature] = le.fit_transform(data[feature])

# Splitting the data into training and test sets
X = data.drop('Churn', axis=1)
y = data['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data preprocessing completed!")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

# Define the hyperparameters for each model
hyperparameters = {
    'LogisticRegression': {
        'C': uniform(0.1, 10),
        'penalty': ['l1', 'l2']
    },
    'SVC': {
        'C': uniform(0.1, 10),
        'gamma': uniform(0.001, 1)
    },
    'GradientBoostingClassifier': {
        'n_estimators': randint(50, 200),
        'max_depth': randint(1, 10),
        'learning_rate': uniform(0.01, 0.3)
    },
    'MLPClassifier': {
        'hidden_layer_sizes': [(randint(10, 100).rvs(), randint(10, 100).rvs()), (randint(10, 100).rvs(),)],
        'alpha': uniform(0.0001, 0.1)
    },
    'LGBMClassifier': {
        'n_estimators': randint(50, 200),
        'max_depth': randint(1, 10),
        'learning_rate': uniform(0.01, 0.3)
    }
}

# Perform a randomized search for each model
for model in models:
    model_name = model.__class__.__name__
    print(f"\nTuning {model_name}...")
    
    # Initialize a RandomizedSearchCV object
    rs = RandomizedSearchCV(model, hyperparameters[model_name], n_iter=50, cv=5, random_state=42, n_jobs=-1)
    
    # Fit the RandomizedSearchCV object to the data
    rs.fit(X_train, y_train)
    
    # Print the best parameters and the best score
    print(f"Best parameters: {rs.best_params_}")
    print(f"Best score: {rs.best_score_}")
