# %%
from hmac import new
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np

# %%
df = pd.read_csv("../data/name_gender.csv")

# %%
df.head()

# %%
features = df["name"]
labels = df["gender"]

# %%
# Preprocess and split your data
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)
print(
    f"X_train shape: {X_train.shape}\nX_test shape: {X_test.shape}\ny_train shape: {y_train.shape}\ny_test shape: {y_test.shape}"
)

# %%
# Encode labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# %%
# Feature extraction
vectorizer = CountVectorizer(analyzer="char", ngram_range=(1, 3))
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# %%
# Initialize models
# models = {
#     # "Naive Bayes": MultinomialNB(),
#     # "Random Forest": RandomForestClassifier(),
#     "SVC": SVC(kernel="linear", probability=True),  # Since we're doing classification
#     # "Gradient Boosting": GradientBoostingClassifier(),
# }
model = RandomForestClassifier()

# %%
# Check for shape mismatch
assert X_train_vectorized.shape[0] == len(y_train_encoded), "Mismatched sample sizes"

# %%
# Train and evaluate models
# for name, model in models.items():
#     model.fit(X_train_vectorized, y_train_encoded)
#     scores = cross_val_score(model, X_test_vectorized, y_test_encoded, cv=5)
#     print(f"{name} Accuracy: {scores.mean()}")


model.fit(X_train_vectorized, y_train_encoded)
y_pred = model.predict(X_test_vectorized)
print(classification_report(y_test_encoded, y_pred))
accuracy = accuracy_score(y_test_encoded, y_pred)
print(f"Accuracy: {accuracy}")

# %%
# Predict on new data
# model = models["SVC"]
new_df = pd.read_csv("../data/cleaned_data.csv")

# Predict on new data using the trained SVC model
new_features = new_df["name"]
new_features_vectorized = vectorizer.transform(new_features)
new_predictions_encoded = model.predict(new_features_vectorized)

# Decode the predictions back to the original labels
new_predictions = le.inverse_transform(new_predictions_encoded)

#%%
# Replace existing gender column with newly predicted one
new_df["gender"] = new_predictions

# new_df.drop('predicted_gender', axis=1)

# Save the updated DataFrame to a new CSV file
new_df.to_csv("../data/cleaned_data.csv", index=False)

# Print out the first few rows to verify
new_df.head()

# %%
