# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
)

from scipy.stats import ttest_ind, f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# set the seed for reproducibility
np.random.seed(0)
# set operations to 2 decimal places
pd.set_option("display.float_format", lambda x: "%.2f" % x)

# %%
df = pd.read_csv("data/cleaned_data.csv")

# %%
# Define presets for data visualization
def presets():
    plt.rc('figure', autolayout = True)

    plt.rc('axes', 
        labelsize = 'large',
        labelweight = 'bold',
        titlesize = 14,
        titleweight = 'bold',
        titlepad = 10       
    )

    %config InlineBackend.figure_format = 'retina'
    pd.options.display.max_rows = 10
    sns.set_style('darkgrid')
    warnings.filterwarnings(action = 'ignore', category = UserWarning)
    return 

presets()



# %%
df.describe(include="all").T
df.select_dtypes(include=["object"]).apply(pd.unique)

# %%
df.info()

# %%
# Check for missing values
df.isnull().sum().where(lambda x: x > 0).dropna()

# %%
# Drop the columns that are not needed
df.drop(
    columns=[
        "Unnamed: 0",
    ],
    inplace=True,
)

# %%
df.head()

# %%
# Remove commas and convert to float for specific columns
columns_to_convert = [
    "monthly_premium",
    "policy_value",
    "paid_premium",
    "total_premium",
]
for col in columns_to_convert:
    df[col] = df[col].astype(str).str.replace(",", "").astype(float)

# Convert date columns to datetime
date_columns = ["inception_date", "expiry_date"]
for col in date_columns:
    df[col] = pd.to_datetime(df[col], format="%d-%b-%y")

# %%
df.head()

# %%
df["policy_duration"] = df["expiry_date"] - df["inception_date"]
df["policy_duration"] = df["policy_duration"].dt.days  # Convert to number of days

# %%
df.head()

# %%
# Standardize occupation and fix inconsistencies
occupation_mapping = {
    "TEACHING": "TEACHER",
    "NURSING": "NURSE",
    "CO-ORDINATOR": "COORDINATOR",
    "PUBLICE SERVANT": "PUBLIC SERVANT",
    "TEAHING": "TEACHER",
    "MIDWIFERY": "MIDWIFE",
    "CLEANER": "CLEANER",
    "LOADING OPERATOR": "LOADING OPERATOR",
    "HUMAN RESOURCES": "HUMAN RESOURCE",
    "CARPENTRY": "CARPENTER",
    "SALES OOFICER": "SALES OFFICER",
    "GUARD": "SECURITY GUARD",
    "PHOTOGRAHER": "PHOTOGRAPHER",
    "HEALTH ASSIST": "HEALTH ASSISTANT",
    "MEDICAL LABORATORY S": "MEDICAL LABORATORY SCIENTIST",
    "EDUCATIONIST": "EDUCATOR",
    "CATRER": "CATERER",
    "INSURER": "INSURANCE",
    "PHYSICIAN ASSISTANT": "PHYSICIAN",
    "DATA PROTECTION OFFI": "DATA PROTECTION OFFICER",
    "AGRIC OFFICER": "AGRICULTURAL OFFICER",
    "CONTRACTOR": "CONTRACTOR",
    "AUTO MACHINIES": "AUTO MACHINIST",
    "EXTENSION OFFICER": "EXTENSION WORKER",
    "COOKER": "COOK",
    "BURSE": "NURSE",
    "ANAESTHESIA": "ANESTHESIOLOGIST",
    "SARPOSITORY ATISAN": "SANITARY ARTISAN",
    "CONSERVANCY": "CONSERVATION",
    "STATISCIAN": "STATISTICIAN",
    "PARTRY HANDS": "PANTRY HANDS",
    "LIBARIAN": "LIBRARIAN",
    "GHS": "GOVERNMENT HEALTH SERVICE",
    "CLERGY": "CLERGY",
    "VEGETABLES SELLER": "VEGETABLE SELLER",
    "MTN AGENT": "TELECOM AGENT",
    "DRIVER": "DRIVER",
    "FASHION DESIGNER": "FASHION DESIGNER",
    "SEAMTRESS": "SEAMSTRESS",
}

df["occupation"] = df["occupation"].replace(occupation_mapping)

# %%
occupation_mapping = {
    "TEACHING": "TEACHER",
    "NURSING": "NURSE",
    "CO-ORDINATOR": "COORDINATOR",
    "PUBLICE SERVANT": "PUBLIC SERVANT",
    "TEAHING": "TEACHER",
    "MIDWIFERY": "MIDWIFE",
    "CLEANER": "CLEANER",
    "LOADING OPERATOR": "LOADING OPERATOR",
    "HUMAN RESOURCES": "HUMAN RESOURCE",
    "CARPENTRY": "CARPENTER",
    "SALES OOFICER": "SALES OFFICER",
    "GUARD": "SECURITY GUARD",
    "PHOTOGRAHER": "PHOTOGRAPHER",
    "HEALTH ASSIST": "HEALTH ASSISTANT",
    "MEDICAL LABORATORY S": "MEDICAL LABORATORY SCIENTIST",
    "EDUCATIONIST": "EDUCATOR",
    "CATRER": "CATERER",
    "INSURER": "INSURANCE",
    "PHYSICIAN ASSISTANT": "PHYSICIAN",
    "DATA PROTECTION OFFI": "DATA PROTECTION OFFICER",
    "AGRIC OFFICER": "AGRICULTURAL OFFICER",
    "CONTRACTOR": "CONTRACTOR",
    "AUTO MACHINIES": "AUTO MACHINIST",
    "EXTENSION OFFICER": "EXTENSION WORKER",
    "COOKER": "COOK",
    "BURSE": "NURSE",
    "ANAESTHESIA": "ANESTHESIOLOGIST",
    "SARPOSITORY ATISAN": "SANITARY ARTISAN",
    "CONSERVANCY": "CONSERVATION",
    "STATISCIAN": "STATISTICIAN",
    "PARTRY HANDS": "PANTRY HANDS",
    "LIBARIAN": "LIBRARIAN",
    "GHS": "GOVERNMENT HEALTH SERVICE",
    "CLERGY": "CLERGY",
    "VEGETABLES SELLER": "VEGETABLE SELLER",
    "MTN AGENT": "TELECOM AGENT",
    "DRIVER": "DRIVER",
    "FASHION DESIGNER": "FASHION DESIGNER",
    "SEAMTRESS": "SEAMSTRESS",
}

df["occupation"] = df["occupation"].replace(occupation_mapping)

# %%
# Plot histograms
supt_presets = dict(fontsize=20, fontweight="bold")

df.hist(figsize=(30, 15), color="blue")
plt.suptitle("Feature Distributions", **supt_presets)
plt.show()  # plots the histograms of the DataFrame

# %%
import math

# Assuming df is your DataFrame
numerical_vars = df.select_dtypes(include=["int64", "float64"]).columns

# Calculate the number of rows needed
num_vars = len(numerical_vars)
cols = 2  # Set the number of columns you want
rows = math.ceil(num_vars / cols)  # Calculate rows needed

plt.figure(figsize=(15, 5 * rows))  # Adjust figure size based on rows
for i, var in enumerate(numerical_vars, 1):
    plt.subplot(rows, cols, i)  # Use dynamic rows and cols
    sns.histplot(df[var], kde=True)
    plt.title(f"Distribution of {var}")

plt.tight_layout()
plt.show()

# %%
import math
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming df is your DataFrame
cat_vars = df.select_dtypes(include=["object"]).columns

# Calculate the number of rows needed
num_cat_vars = len(cat_vars)  # Use a different variable for the length
cols = 2  # Set the number of columns you want
rows = math.ceil(num_cat_vars / cols)  # Calculate rows needed

plt.figure(figsize=(15, 5 * rows))  # Adjust figure size based on rows
for i, var in enumerate(cat_vars, 1):
    plt.subplot(rows, cols, i)  # Use dynamic rows and cols
    sns.histplot(df[var], kde=False)
    plt.title(f"Distribution of {var}")

plt.tight_layout()
plt.show()

# %%
df["occupation"].value_counts().min()

# %%
# # Define a threshold for minimum observations per occupation category
# min_observations_per_occupation = 5


# # Count occurrences of each unique value in the "occupation" column
# occupation_counts = df["occupation"].value_counts()

# # Map these counts back to the original DataFrame rows
# df["occupation_count"] = df["occupation"].map(occupation_counts)

# # Use .loc to assign "Other" where the counts are less than the threshold
# df.loc[df["occupation_count"] < min_observations_per_occupation, "occupation_group"] = (
#     "Other"
# )

# # Optionally, you can drop the 'occupation_count' column if it was only needed for this operation
# df.drop("occupation_count", axis=1, inplace=True)

# %%
# Function to add value labels on bars
def add_value_labels(ax):
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="baseline",
            fontsize=10,
            color="black",
            xytext=(0, 5),
            textcoords="offset points",
        )


# Bar charts for categorical variables
categorical_vars = ["branch", "plan", "occupation"]

plt.figure(figsize=(15, 10))
for i, var in enumerate(categorical_vars, 1):
    plt.subplot(2, 2, i)
    ax = sns.countplot(
        data=df, x=var, hue=var, order=df[var].value_counts().index, palette="Set2"
    )
    add_value_labels(ax)
    plt.title(f"Distribution of {var}")
    plt.xticks(rotation=90)

plt.tight_layout()
plt.show()


# Plot the top 10 occupations with value labels
def plot_top_n_occupations(df, n=10):
    top_n_occupations = df["occupation"].value_counts().nlargest(n).index
    plt.figure(figsize=(15, 8))
    ax = sns.countplot(
        data=df[df["occupation"].isin(top_n_occupations)],
        x="occupation",
        hue="occupation",
        order=top_n_occupations,
        palette="Set2",
    )
    add_value_labels(ax)
    plt.title(f"Top {n} Occupations")
    plt.xticks(rotation=90)
    plt.show()


# Plot the top 10 occupations
plot_top_n_occupations(df, n=10)

# %%
# Pie chart for gender distribution
plt.figure(figsize=(6, 4))
gender_counts = df["gender"].value_counts()
plt.pie(
    gender_counts,
    labels=gender_counts.index,
    autopct="%1.1f%%",
    colors=sns.color_palette("pastel"),
)
plt.title("Gender Distribution")
plt.show()

# %%


# %%
# Correlation matrix
plt.figure(figsize=(10, 5))
sns.heatmap(df[numerical_vars].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# %%
df[["plan", "gender"]].groupby("plan").count()

# %%
# Create a pivot table of occupation group by gender and plan
pivot_table = df.pivot_table(
    values="occupation", index="plan", columns="gender", aggfunc="count"
)


display(
    pivot_table.fillna(0)
)  # prints the pivot table of the average plays by education and gender

# %%
# Gender across different plans
fig, ax = plt.subplots()
pivot_table.plot(kind="bar", ax=ax)
ax.set(xlabel="", ylabel="Average Count", title="Plan Across Genders")
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
for container in ax.containers:
    ax.bar_label(container, fmt="%.0f")
ax.legend()  # plots the educational disparity across genders

# %%
# Create a pivot table of occupation group by gender and plan
pivot_table = df.pivot_table(
    values="occupation",
    index="monthly_premium",
    columns="gender",
    aggfunc="count",
)


display(
    pivot_table.fillna(0)
)  # prints the pivot table of the average plays by education and gender

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Gender and Premium Pricing
plt.figure(figsize=(10, 6))
sns.violinplot(x="gender", y="monthly_premium", data=df)
plt.title("Distribution of Monthly Premium by Gender")
plt.show()
.
fig = px.pie(
    df,
    values="total_premium",
    names="gender",
    hole=0.3,
    title="Distribution of Total Premium by Gender",
)
fig.show()

# %% [markdown]
# The width of each ‘violin’ indicates the frequency of data points at different premium levels, with wider sections meaning more data points. The central thick black line represents the interquartile range, and the thin black line indicates the median value for each gender’s premiums
# 

# %%
# Gender across different plans
fig, ax = plt.subplots()
fig.set_size_inches(20, 10)
pivot_table.plot(kind="bar", ax=ax)
ax.set(xlabel="", ylabel="Average Count", title="Monthly premium Across Genders")
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
for container in ax.containers:
    ax.bar_label(container, fmt="%.0f")
ax.legend()  # plots the educational disparity across genders

# %%
plt.figure(figsize=(30, 15))
sns.barplot(x="occupation", y="monthly_premium", data=df)
plt.xticks(rotation=45)
plt.title("Average Monthly Premium by Occupation")
plt.show()

# %%
plt.figure(figsize=(10, 6))
sns.boxplot(x="plan", y="total_premium", data=df)
plt.xticks(rotation=45)
plt.title("Distribution of Total Premium by Plan Type")
plt.show()

# %%
# Existing policy duration (in days)
policy_duration = df["policy_duration"]

# Create new column for policy duration in months
df["policy_duration_months"] = policy_duration / 30

# Create new column for policy duration in years
df["policy_duration_years"] = policy_duration / 365

# Drop the original policy duration column
df.drop(columns=["policy_duration"], inplace=True)

df.head()

# %%
# Policy Value vs Policy Duration
plt.figure(figsize=(10, 5))
sns.scatterplot(
    data=df, x="policy_duration_years", y="policy_value", hue="plan", palette="Set2"
)
plt.title("Policy Value vs Policy Duration")
plt.show()

# %% [markdown]
# From the graph, it appears that all four plans have a higher policy value for longer policy durations. This means that the value of the policy increases as the policy term gets longer.
# 
# Family Security Plan: The policy value starts at around 500 and increases to around 2000 for a 40-year policy.
# 
# Flexi Child Education, Education,and Ultimate Life: These three plans seem to have a similar policy value, starting at around 500 and increasing to around 2000 for a 40-year policy.
# 
# Ultimate Famsec: Starts has only one entry at around 5000
# 

# %%
# Monthly Premium vs Policy Duration
plt.figure(figsize=(10, 5))
sns.scatterplot(
    data=df, x="policy_duration_years", y="monthly_premium", hue="plan", palette="Set2"
)
plt.title("Monthly Premium vs Policy Duration")
plt.show()

# %% [markdown]
# 1. From the graph, it appears that all four plans have a higher monthly premium for longer policy durations.
# 2. This means that you would pay more each month if you choose a longer policy term.
# 
# - The Family Security Plan seems to be the most expensive plan, with a monthly premium around 500 for a 40-year policy.
# - The Ultimate Life plan appears to be the least expensive, with a monthly premium around 100 for a 40-year policy.
# 

# %%
plt.figure(figsize=(30, 12))
sns.barplot(x="occupation", y="monthly_premium", hue="gender", data=df)
plt.xticks(rotation=45)
plt.title("Average Monthly Premium by Occupation and Gender")
plt.show()

# %% [markdown]
# From this image, females seem to be dominating the monthly payment of premiums across multiple occupations
# 

# %% [markdown]
# Statistical Tests
# 

# %%
# 1. Hypothesis testing
# Test for gender differences in monthly premium
female_premiums = df[df["gender"] == "FEMALE"]["monthly_premium"]
male_premiums = df[df["gender"] == "MALE"]["monthly_premium"]
ttest_ind(female_premiums, male_premiums)

# %%
# Test for occupation differences in monthly premium
occupation_groups = df.groupby("occupation")["monthly_premium"].apply(list)
f_oneway(*occupation_groups)

# %% [markdown]
# 1. Independent T-Test
#    `statistic = 0.9756814075971332`:
# 
# - This is the t-statistic calculated from the test.
# 
# `pvalue = 0.3296078624932599`:
# 
# - This is the p-value associated with the t-statistic.
# - Since it's greater than the typical significance level of 0.05, it suggests that there is no statistically significant difference in monthly premiums between females and males in Ghana.
# 
# 2. F-Oneway Test (ANOVA)
#    `statistic = 5.423066525891169`:
# 
# - This is the F-statistic calculated from the test.
# 
# `pvalue = 9.272403792676047e-37`:
# 
# - This is the p-value associated with the F-statistic. Since it's extremely low (almost zero), it suggests that there is a statistically significant difference in monthly premiums across different occupations in Ghana.
# 

# %%
df.head()

# %%
df.info()

# %%
df.columns

# %%
df.head()

# %%
from sklearn.feature_selection import RFE

# Define your features and target variable
X = df.drop(
    columns=[
        "monthly_premium",
        "policyholder",
        "inst",
        "policy_number",
        "inception_date",
        "expiry_date",
        "branch",
    ]
)
y = df["monthly_premium"]

# Define the numerical and categorical features
numerical_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

# Define the preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# Define the model
model = RandomForestRegressor(n_estimators=100, random_state=0)

# Define the pipeline
pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Fit the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}\n Mean Absolute Error: {mae:.2f}")

# Create a preprocessing pipeline for transforming features without including the model
preprocessor_for_rfe = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# Apply preprocessing to X_train
X_train_transformed = preprocessor_for_rfe.fit_transform(X_train)

# Create a new model instance for RFE
rfe_model = RandomForestRegressor(n_estimators=100, random_state=0)

# Create RFE model and select 5 attributes
rfe = RFE(estimator=rfe_model, n_features_to_select=5)
rfe.fit(X_train_transformed, y_train)  # Fit RFE on transformed training data


# Correctly access feature_importances_ from the fitted model within RFE
feature_importances = rfe.estimator_.feature_importances_

# Print the feature importances for the selected features
for i, (importance, selected) in enumerate(zip(feature_importances, rfe.support_)):
    if selected:
        print(f"Transformed Feature: {i}, Importance: {importance}")

# %%
# Get all feature names after preprocessing
all_feature_names = preprocessor_for_rfe.get_feature_names_out()

# Filter the feature names based on rfe.support_
selected_feature_names = all_feature_names[rfe.support_]

# Print the selected feature names
print("Selected feature names:", selected_feature_names)

# %%
df.shape

# %%
# from sklearn.svm import SVR


# x = ["paid_premium", "premium", "total_premium", "policy_duration_months", "occupation"]
# y = ["monthly_premium"]

# X = df[x]
# y = df[y]

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # Define the numerical and categorical features
# numerical_features = X.select_dtypes(include=["int64", "float64"]).columns
# categorical_features = X.select_dtypes(include=["object"]).columns

# # Define the preprocessing pipeline
# preprocessor = ColumnTransformer(
#     transformers=[
#         ("num", StandardScaler(), numerical_features),
#         ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
#     ]
# )

# # Define the models
# models = {
#     "Linear Regression": LinearRegression(),
#     "Decision Tree": DecisionTreeRegressor(random_state=42),
#     "Random Forest": RandomForestRegressor(random_state=42),
#     "SVR": SVR(),
# }

# # Define Hyperparameter grid for models
# param_grid = {
#     "Linear Regression": {},
#     "Decision Tree": {
#         "model__max_depth": [5, 10, 15, 20],
#         "model__min_samples_split": [2, 5, 10],
#     },
#     "Random Forest": {
#         "model__n_estimators": [100, 200, 300],
#         "model__max_depth": [5, 10, 15, 20],
#         "model__min_samples_split": [2, 5, 10],
#     },
#     "SVR": {
#         "model__C": [0.1, 1, 10],
#         "model__gamma": [0.1, 1, 10],
#         "model__kernel": ["linear", "rbf"],
#     },
# }


# # Evaluate the model
# for model_name, model in models.items():
#     pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
#     pipeline.fit(X_train, y_train)
#     y_pred = pipeline.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     mae = mean_absolute_error(y_test, y_pred)
#     print(
#         f"{model_name}:\nMean Squared Error: {mse:.2f}\nMean Absolute Error: {mae:.2f}\n"
#     )

# # Plot feature importances for the Random Forest model

# pipeline = Pipeline(
#     [("preprocessor", preprocessor), ("model", RandomForestRegressor(random_state=42))]
# )
# pipeline.fit(X_train, y_train)
# model = pipeline.named_steps["model"]
# preprocessor = pipeline.named_steps["preprocessor"]

# # Get mask of non-zero feature importances
# mask = model.feature_importances_ > 0.01

# # Get feature importances
# feature_names = preprocessor.get_feature_names_out()
# importances = model.feature_importances_
# filtered_feature_names = feature_names[mask]
# filtered_importances = importances[mask]

# # Create a DataFrame of feature importances
# # feature_importances = pd.DataFrame({"feature": feature_names, "importance": importances})
# # feature_importances = feature_importances.sort_values("importance", ascending=False)
# # Create a DataFrame of feature importances with filtered data
# feature_importances = pd.DataFrame(
#     {"feature": filtered_feature_names, "importance": filtered_importances}
# )
# feature_importances = feature_importances.sort_values("importance", ascending=False)

# # Plot the feature importances
# plt.figure(figsize=(20, 10))
# sns.barplot(
#     data=feature_importances,
#     x="importance",
#     hue="importance",
#     y="feature",
#     palette="viridis",
# )
# plt.title("Feature Importances")
# plt.show()

# %%


# %% [markdown]
# 1. `num_paid_premium`: This feature has an importance value of approximately 0.46 (or 46%). This means that this feature accounts for around 46% of the importance or contribution in predicting the target variable (monthly premium or insurance pricing) in the Random Forest model.
# 
# 2. `num_total_premium`: With an importance value of around 0.24 (or 24%), this feature contributes approximately 24% to the model's predictions.
#    num_premium: This feature has an importance value of around 0.18 (or 18%), contributing roughly 18% to the model's predictions.
# 
# 3. `cat_occupation_LECTURER`: With an importance value of around 0.04 (or 4%), this feature accounts for approximately 4% of the importance or contribution in predicting the target variable.
# 
# 4. `num_policy_duration_months`: This feature has an importance value of around 0.03 (or 3%), contributing roughly 3% to the model's predictions.
# 
# 5. `cat_occupation_COORDINATOR` and cat_occupation_ADMIN: These features have lower importance values, contributing less than 2% to the model's predictions.
# 
# To summarize, the feature num_paid_premium is the most important, contributing around 46% to the model's predictions, followed by num_total_premium (24%), num_premium (18%), and so on.
# It's important to note that these importance values are relative and sum up to 1 (or 100%) across all features. They provide a way to compare the relative importance or contribution of each feature within the context of the Random Forest model.
# 

# %% [markdown]
# 

# %%
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Preprocess the data
def preprocess_data(df, feature_columns, target_column):
    X = df[feature_columns]
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    return X_train, X_test, y_train, y_test, preprocessor


# Train and evaluate models
def train_and_evaluate_models(
    X_train, X_test, y_train, y_test, preprocessor, models, param_grid
):
    results = []
    for model_name, model in models.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        grid_search = GridSearchCV(pipeline, param_grid[model_name], cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        results.append({"model": model_name, "mse": mse, "mae": mae})
        print(
            f"{model_name}:\nMean Squared Error: {mse:.2f}\nMean Absolute Error: {mae:.2f}\n"
        )
    return results


# Plot feature importances for the Random Forest model
def plot_feature_importances(preprocessor, X_train, y_train):
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(random_state=42)),
        ]
    )
    pipeline.fit(X_train, y_train)
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_

    mask = importances > 0.01
    filtered_feature_names = feature_names[mask]
    filtered_importances = importances[mask]

    feature_importances = pd.DataFrame(
        {"feature": filtered_feature_names, "importance": filtered_importances}
    )
    feature_importances = feature_importances.sort_values("importance", ascending=False)

    plt.figure(figsize=(20, 10))
    sns.barplot(
        data=feature_importances,
        x="importance",
        y="feature",
        hue="feature",
        palette="viridis",
        legend=False,
    )
    plt.title("Feature Importances")
    plt.show()


# Main function to run all steps
def main(df):
    feature_columns = [
        "paid_premium",
        "premium",
        "total_premium",
        "policy_duration_months",
        "occupation",
    ]
    target_column = "monthly_premium"

    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(
        df, feature_columns, target_column
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42),
        "SVR": SVR(),
    }

    param_grid = {
        "Linear Regression": {},
        "Decision Tree": {
            "model__max_depth": [5, 10, 15],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        },
        "Random Forest": {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [5, 10, 15],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        },
        "SVR": {
            "model__C": [0.1, 1, 10],
            "model__gamma": [0.01, 0.1, 1],
            "model__kernel": ["linear", "rbf"],
        },
    }

    results = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, preprocessor, models, param_grid
    )
    plot_feature_importances(preprocessor, X_train, y_train)

# %%
main(df)

# %% [markdown]
# 1. Paid Premium (num\_\_paid_premium):
# 
# - This feature has the highest importance, indicating that the amount already paid by the customer has the most significant impact on predicting the monthly insurance premium.
# 
# 2. Total Premium (num\_\_total_premium):
# 
# - The total premium, which is likely the sum of all premiums paid over a period, is the second most important factor. This makes sense as it reflects the overall financial commitment of the policyholder.
# 
# 3. Premium (num\_\_premium):
# 
# - The base premium amount also plays a significant role. This suggests that the inherent cost of the insurance product is crucial in determining the monthly premium.
# 
# 4. Occupation (cat\_\_occupation_LECTURER):
# 
# - Occupation, particularly being a lecturer, is an influential categorical feature. This could imply that certain occupations might have different risk profiles or discounts associated with them.
# 
# 5. Policy Duration (num\_\_policy_duration_months):
# 
# - The length of the policy in months has a moderate impact. This might reflect how long the customer has been with the insurer or the remaining duration of the policy.
# 
# 6. Occupation (cat**occupation_COORDINATOR and cat**occupation_ADMIN):
# 
# - Other occupations such as coordinator and admin also show some importance, though less than the aforementioned factors. This again points to the occupation's role in risk assessment.
# 

# %%
# correlation
# streamlit with inputs gender, occupation, plan type
# output : monthly pay


