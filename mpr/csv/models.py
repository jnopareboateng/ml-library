# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd

# Load your data
data = pd.read_csv('your_data.csv')  # replace with your actual csv file

# Define predictors and target
predictors = ['GDP', 'Inflation', 'Unemployment']
target = 'Policy Rates'  # replace with your actual target column

X = data[predictors]
y = data[target]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree model
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)
tree_predictions = tree_model.predict(X_test)
print(f'Decision Tree RMSE: {mean_squared_error(y_test, tree_predictions, squared=False)}')

# Random Forest model
forest_model = RandomForestRegressor(random_state=42)
forest_model.fit(X_train, y_train)
forest_predictions = forest_model.predict(X_test)
print(f'Random Forest RMSE: {mean_squared_error(y_test, forest_predictions, squared=False)}')


# vector auto regression model