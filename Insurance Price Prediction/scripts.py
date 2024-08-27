# Gender across different plans
fig, ax = plt.subplots()
pivot_table.plot(kind="bar", ax=ax)
ax.set(xlabel="", ylabel="Average Count", title="Plan Across Genders")
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
for container in ax.containers:
    ax.bar_label(container, fmt="%.0f")
ax.legend()
# plots the educational disparity across genders

# Create a pivot table of occupation group by gender and plan
pivot_table = df.pivot_table(
    values="occupation",
    index="monthly_premium",
    columns="gender",
    aggfunc="count",
)

display(pivot_table.fillna(0))
# prints the pivot table of the average plays by education and gender

# Gender and Premium Pricing
plt.figure(figsize=(10, 6))
sns.violinplot(x="gender", y="monthly_premium", data=df)
plt.title("Distribution of Monthly Premium by Gender")
plt.show()
fig = px.pie(
    df,
    values="total_premium",
    names="gender",
    hole=0.3,
    title="Distribution of Total Premium by Gender",
)
fig.show()

# types of plan vs total premium
plt.figure(figsize=(10, 6))
sns.boxplot(x="plan", y="total_premium", data=df)
plt.title("Total Premium by Plan")
plt.show()

# Existing policy duration (in days)
policy_duration = df["policy_duration"]

# Create new column for policy duration in months
df["policy_duration_months"] = policy_duration / 30

df["policy_duration_years"] = policy_duration / 365

df["policy_duration"] = df["expiry_date"] - df["inception_date"]

df["policy_duration"] = df["policy_duration"].dt.days

df.head()

# Policy Value vs Policy Duration
plt.figure(figsize=(10, 5))
sns.scatterplot(
    data=df, x="policy duration_years", y="policy value", hue="plan",
)

# occupation
# OFFICER
# TEACHER
# TEACHER
# LABOURER
ACCOUNTANT

plt.title("Policy Value vs Policy Duration")
plt.show()

# Monthly Premium vs Policy Duration
plt.figure(figsize=(10, 5))
sns.scatterplot(
    data=df, x="policy_duration_years", y="monthly_premium", hue="plan", palette="S"
)
plt.title("Monthly Premium vs Policy Duration")
plt.show()

# Statistical Tests

# 1. Hypothesis testing

# Test for gender differences in monthly premium
female_premiums = df[df["gender"] == "FEMALE"]["monthly_premium"]
male_premiums = df[df["gender"] == "MALE"]["monthly_premium"]
ttest_ind(female_premiums, male_premiums)

# TtestResult(statistic=0.9756814075971332, pvalue=0.3296078624932599, df=612.0)

# # Test for occupation differences in monthly premium
# occupation_groups = df.groupby("occupation")["monthly_premium"].apply(list)
# f_oneway(*occupation_groups)

# F_onewayResult(statistic=5.423066525891169, pvalue=9.272403792676047e-37)

# 1. Independent T-Test
# statistic = 0.9756814075971332
# :
# e This is the t-statistic calculated from the test.
# pvalue = 0.3296078624932599
# :
# ® This is the p-value associated with the t-statistic.
# e Since it's greater than the typical significance level of 0.05, it suggests that there is no statistically significant difference in monthly premiums between females and males in
# Ghana.
# 2. F-Oneway Test (ANOVA)
# statistic = 5.423066525891169
# :
# ¢ This is the F-statistic calculated from the test.
# pvalue = 9.272403792676047e-37
# :
# e This is the p-value associated with the F-statistic. Since it's extremely low (almost zero), it suggests that there is a statistically significant difference in monthly premiums across different occupations in Ghana.
# Modelling with Linear, Decison Tree, Random Forest and Support Vector Regressors

# All featuers

# Preprocess the data
def preprocess_data(df, feature_columns, target_column):
    X = df[feature_columns]
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

    categorical_transformer = Pipeline(
        steps=[
            (
                "labels",
                OneHotEncoder(handle_unknown="ignore"),
            ),  # Set handle_unknown="ignore'"
        ]
    )

    numeric_transformer = Pipeline(steps=[("scaler", MinMaxScaler())])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return X_train, X_test, y_train, y_test, preprocessor

# Train and evaluate models
def train_and_evaluate_models(
    X_train, X_test, y_train, y_test, preprocessor, models, param_grid
):
    results = []
    for model_name, model in models.items():
        pipeline = Pipeline(
            [("preprocessor", preprocessor), ("model", model)]
        )
        grid_search = GridSearchCV(pipeline, param_grid[model_name], cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        results.append(
            {
                "model": model_name,
                "mse": mse,
                "mae": mae,
            }
        )
        print(
            f"{model_name}:\nMean Squared Error: {mse:.2f}\nMean Absolute Error: {mae:.2f}"
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
        {
            "feature": filtered_feature_names,
            "importance": filtered_importances,
        }
    )
    feature_importances = feature_importances.sort_values("importance", ascending=False)

    fig, ax = plt.subplots()

    sns.barplot(
        data=feature_importances,
        x="importance",
        y="feature",
        hue="feature",
        palette="viridis",
        legend=False,
    )

    for ax in fig.axes:
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        ax.set_title("Feature Importances")
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f")
    plt.show()

# Main function to run all steps
def main(df):
    feature_columns = [
        "gender",
        "paid_premium",
        "premiums_paid",
        "total_premium",
        "policy_duration_years",
        "occupation",
    ]

    target_column = "monthly premium"
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
            "model_min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        },
        "Random Forest": {
            "model_n_estimators": [50, 100, 200],
            "model__max_depth": [5, 10, 15],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        },
        "SVR": {
            "model_C": [0.1, 1, 10],
            "model_gamma": [0.01, 0.1, 1],
            "model__kernel": ["linear", "rbf"],
        },
    }

    results = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, preprocessor, models, param_grid
    )

    plot_feature_importances(preprocessor, X_train, y_train)

main(df)

# Linear Regression:
# Mean Squared Error: 2761.32
# Mean Absolute Error: 38.75

# Decision Tree:
# Mean Squared Error: 3056.39
# Mean Absolute Error: 42.46

# Random Forest:
# Mean Squared Error: 2819.73
# Mean Absolute Error: 39.03

# SVR:
# Mean Squared Error: 2537.41
# Mean Absolute Error: 35.65

# Revised Features
final_features = [
    "gender",
    "occupation",
    "policy_duration_years",
    "plan",
]

def revised_features(df):
    feature_columns = final_features
    target_column = "monthly premium"

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
            "model_min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        },
        "Random Forest": {
            "model_n_estimators": [50, 100, 200],
            "model__max_depth": [5, 10, 15],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        },
        "SVR": {
            "model_C": [0.1, 1, 10],
            "model_gamma": [0.01, 0.1, 1],
            "model__kernel": ["linear", "rbf"],
        },
    }

    results = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, preprocessor, models, param_grid
    )

    plot_feature_importances(preprocessor, X_train, y_train)

revised_features(df)

# Linear Regression:
# Mean Squared Error: 2811.16
# Mean Absolute Error: 39.69

# Decision Tree:
# Mean Squared Error: 3055.32
# Mean Absolute Error: 42.68

# Random Forest:
# Mean Squared Error: 2817.46
# Mean Absolute Error: 39.02

# SVR:
# Mean Squared Error: 2842.80
# Mean Absolute Error: 39.04
