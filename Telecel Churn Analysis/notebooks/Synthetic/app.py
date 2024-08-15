import pandas as pd
import numpy as np
import seaborn as sns
import missingno as msno
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb
import streamlit as st

class DataLoader:
    def __init__(self, file_path):
        self.file_path = "reData.csv"

    def load_data(self):
        data = pd.read_csv(self.file_path)
        return data

class DataExplorer:
    def __init__(self, data):
        self.data = data

    def explore_data(self):
        st.write("Data Shape:", self.data.shape)
        st.write("Data Columns:", self.data.columns.values)
        st.write("Data Info:", self.data.info())
        st.write("Data Describe:", self.data.describe())

    def visualize_missing_values(self):
        msno.matrix(self.data)
        st.pyplot()

    def visualize_distribution(self, column):
        fig = px.histogram(self.data, x=column)
        st.plotly_chart(fig)

class ModelTrainer:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        le = LabelEncoder()
        self.X = self.X.apply(lambda col: le.fit_transform(col) if col.dtype == 'object' else col)
    
    
    def train_test_split(self, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test
        
        # le = LabelEncoder()
        # self.X = self.X.apply(le.fit_transform)

    def train_model(self, model, hyperparameters, n_iter=10, cv=5, random_state=42, n_jobs=-1):
        rs = RandomizedSearchCV(model, hyperparameters, n_iter=n_iter, cv=cv, random_state=random_state, n_jobs=n_jobs)
        rs.fit(self.X, self.y)
        return rs

    def get_model(self, model_name):
        if model_name == "Logistic Regression":
            model = LogisticRegression()
        elif model_name == "Random Forest":
            model = RandomForestClassifier()
        elif model_name == "SVM":
            model = SVC()
        elif model_name == "Gradient Boosting":
            model = GradientBoostingClassifier()
        elif model_name == "MLP":
            model = MLPClassifier()
        elif model_name == "KNN":
            model = KNeighborsClassifier()
        elif model_name == "LGBM":
            model = lgb.LGBMClassifier()
        return model

class ModelEvaluator:
    def __init__(self, y_test, y_pred):
        self.y_test = y_test
        self.y_pred = y_pred

    def evaluate_model(self):
        accuracy = accuracy_score(self.y_test, self.y_pred)
        report = classification_report(self.y_test, self.y_pred)
        matrix = confusion_matrix(self.y_test, self.y_pred)
        st.write("Accuracy:", accuracy)
        st.write("Classification Report:", report)
        st.write("Confusion Matrix:", matrix)

    def plot_confusion_matrix(self):
        fig = sns.heatmap(confusion_matrix(self.y_test, self.y_pred), annot=True, fmt='d', cmap='Blues')
        # st.pyplot(fig)

class StreamlitApp:
    def __init__(self):
        self.data_loader = DataLoader('../data/reData.csv')
        self.data = self.data_loader.load_data()
        self.data_explorer = DataExplorer(self.data)

    def run(self):
        st.title("Telecel Churn Prediction")
        st.write("Data Exploration")
        self.data_explorer.explore_data()
        self.data_explorer.visualize_missing_values()

        st.write("Data Visualization")
        column = st.selectbox("Select a column to visualize", self.data.columns)
        self.data_explorer.visualize_distribution(column)

        st.write("Model Training")
        X = self.data.drop('Churn', axis=1)
        y = self.data['Churn']
        model_trainer = ModelTrainer(X, y)
        X_train, X_test, y_train, y_test = model_trainer.train_test_split()

        model_name  = st.selectbox("Select a model", ["Logistic Regression", "Random Forest", "SVM", "Gradient Boosting", "MLP", "KNN", "LGBM"])
        model = model_trainer.get_model(model_name )
        hyperparameters = self.get_hyperparameters(model)
        rs = model_trainer.train_model(model, hyperparameters)
        y_pred = rs.best_estimator_.predict(X_test)

        st.write("Model Evaluation")
        model_evaluator = ModelEvaluator(y_test, y_pred)
        model_evaluator.evaluate_model()
        model_evaluator.plot_confusion_matrix()

    def get_hyperparameters(self, model_name):
        if model_name == "Logistic Regression":
            hyperparameters = {
                'C': uniform(0.1, 10),
                'penalty': ['l1', 'l2']
            }
        elif model_name == "Random Forest":
            hyperparameters = {
                'n_estimators': randint(50, 200),
            'max_depth': randint(1, 10)
            }
        elif model_name == "SVM":
            hyperparameters = {
                'C': uniform(0.1, 10),
                'gamma': uniform(0.001, 1)
            }
        elif model_name == "Gradient Boosting":
            hyperparameters = {
                'n_estimators': randint(50, 200),
            'max_depth': randint(1, 10),
                'learning_rate': uniform(0.01, 0.3)
            }
        elif model_name == "MLP":
            hyperparameters = {
                'hidden_layer_sizes': (randint(10, 100).rvs(), randint(10, 100).rvs()),
                'alpha': uniform(0.0001, 0.1)
            }
        elif model_name == "KNN":
            hyperparameters = {
                'n_neighbors': randint(1, 10)
            }
        elif model_name == "LGBM":
            hyperparameters = {
                'n_estimators': randint(50, 200),
            'max_depth': randint(1, 10),
                'learning_rate': uniform(0.01, 0.3)
            }
        else:
            hyperparameters = {}
        return hyperparameters

if __name__ == "__main__":
    app = StreamlitApp()
    app.run()