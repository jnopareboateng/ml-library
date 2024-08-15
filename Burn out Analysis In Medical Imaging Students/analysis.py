# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

# Set display and visualization settings
pd.set_option("display.float_format", lambda x: "%.3f" % x)
sns.set(style="darkgrid")
plt.rcParams["figure.figsize"] = [10, 5]

# %% [markdown]
# ### Load Data and Summary Statistics
# ---

# %%
# Load the data
df = pd.read_csv("Burnout.csv")
df.head()
# %%
df['Kindly select your university of study'].value_counts()

# rename "Kindly select your university of study" to "University" to "University"
df.rename(columns={"Kindly select your university of study": "University"}, inplace=True)
# %%
# Display information about the data
df.info()

# %%
# Descriptive statistics
df.describe()


# %%
# Check for missing values
def check_missing(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    missing.sort_values(inplace=True)
    missing = pd.DataFrame(missing, columns=["Missing Values"])
    return missing


check_missing(df)

# %% [markdown]
# ### Calculate Burnout Scores
# ---

# %%
# Define burnout score categories (modify as necessary)
ee_columns = [
    "I feel emotionally drained by my studies.",
    "I feel used up at the end of a day at school.",
    "I feel like my courses are breaking me down.",
    "I feel frustrated by my course activities.",
    "I feel studying or attending a class is really a strain on me.",
]

cy_columns = [
    "I have become less interested in my studies since my enrollment at the school.",
    "I have become less enthusiastic about my studies.",
    "I have become more cynical about the potential usefulness of my studies.",
    "I doubt the significance of my studies.",
]

ae_columns = [
    "I can effectively solve the problems that arise in my studies.",
    "I believe that I make an effective contribution to the classes that I attend.",
    "In my opinion, I am a good student.",
    "I have learned many interesting things during the course of my studies.",
    "I feel stimulated when I achieve my study goals.",
    "During class I feel confident that I am effective in getting things done.",
]

# Calculate burnout scores
df["EE_Score"] = df[ee_columns].mean(axis=1)
df["CY_Score"] = df[cy_columns].mean(axis=1)
df["AE_Score"] = df[ae_columns].mean(axis=1)

# Reverse AE_Score (as it measures reduced personal accomplishment)
df["AE_Score"] = 6 - df["AE_Score"]

# Calculate the overall burnout score
df["Burnout_Score"] = 0.4 * df["EE_Score"] + 0.3 * df["CY_Score"] + 0.3 * df["AE_Score"]

# Display summary statistics for burnout scores
print(df[["EE_Score", "CY_Score", "AE_Score", "Burnout_Score"]].describe())

# %% [markdown]
# ### Distribution of Burnout Scores
# ---

# %%
# Plot the distribution of burnout scores
sns.histplot(df["Burnout_Score"], kde=True)

# %% [markdown]
# ### Burnout Prevalence and Demographic Summary
# ---

# %%
# Define burnout threshold (adjust based on your specific criteria)
burnout_threshold = 3.0

# Calculate and display burnout prevalence
burnout_prevalence = (df["Burnout_Score"] >= burnout_threshold).mean()
print(f"Overall burnout prevalence: {burnout_prevalence:.2%}")

# Summarize demographic information
print(df["Gender"].value_counts(normalize=True))
print(df["Age"].value_counts(normalize=True))
print(df['University'].value_counts(normalize=True)) # added university
print(df["Level"].value_counts(normalize=True))
print(df["Religion"].value_counts(normalize=True))

# %% [markdown]
# ### Multiple Regression Analysis
# ---

# %%
# Prepare independent variables
independent_vars = [
    "Gender",
    "Age",
    "Level",
    "University",
    "Religion",
    "Location",
    "Ethnicity",
    "Study's financing",
    "Medications intake due to studies",
]

# Create dummy variables for categorical independent variables
X = pd.get_dummies(df[independent_vars], drop_first=True)

# Add a constant term (intercept)
X = sm.add_constant(X).astype(float)

# Define the dependent variable (Burnout Score)
# y = df["Burnout_Score"].astype(float)
# Define the dependent variables
dependent_vars = ['Burnout_Score', 'EE_Score', 'CY_Score', 'AE_Score']

# Initialize a dictionary to store the results
results = {}

# Perform regression for each dependent variable and store the results
for var in dependent_vars:
    y = df[var].astype(float)
    model = sm.OLS(y, X).fit()
    results[var] = model

# %% [markdown]
# ### Structured Regression Summary Table
# ---

# Create structured summary tables for each dependent variable
summary_tables = {}
for var, model in results.items():
    summary_table = pd.DataFrame({
        "Coefficient": model.params,
        "Std Error": model.bse,
        "t-value": model.tvalues,
        "p-value": model.pvalues,
        "95% CI Lower": model.conf_int()[0],
        "95% CI Upper": model.conf_int()[1]
    })
    summary_tables[var] = summary_table

# Combine the results into one table for easier comparison (like the provided image)
combined_table = pd.concat(summary_tables, axis=1)
combined_table.columns = pd.MultiIndex.from_product([dependent_vars, combined_table.columns])

# Display the combined table
print(combined_table)

# %%
# Optionally, save the table to a CSV or Excel file
combined_table.to_csv("burnout_regression_summary.csv")
# %%
