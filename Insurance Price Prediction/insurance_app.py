import os
import time
import numpy as np
import streamlit as st
import joblib
import pandas as pd
from dateutil.relativedelta import relativedelta

# Load the pretrained model
model = joblib.load(os.path.join(os.getcwd(), "best_model.pkl"))

# Add some CSS styling for better appearance
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and description
st.markdown("# Insurance Predictions")
st.markdown("### Predict the cost of insurance based on your specifications")
st.markdown("#### Please fill in the details in the sidebar.")

# Sidebar for user input
st.sidebar.header("User Input Features")


def user_input_features():
    gender = st.sidebar.selectbox("Gender", ("Male", "Female"))

    occupation = st.sidebar.selectbox(
        "Occupation Group",
        (
            "Administration",
            "Education",
            "Labor",
            "Other",
            "Public Service",
            "Healthcare",
            "Culinary",
            "Skilled Trade",
            "Security",
            "Business",
        ),
    )

    plan = st.sidebar.selectbox(
        "Plan",
        ("Family Security Plan", "Education", "Flexi Child Education", "Ultimate Life"),
    )

    inception_date = st.sidebar.date_input(
        "Date of Creating Policy", value=pd.to_datetime("2021-01-01")
    )

    is_life_insurance = st.sidebar.checkbox("Whole Life Insurance")

    if is_life_insurance:
        end_date = inception_date + relativedelta(
            years=100
        )  # Default end date for life insurance
    else:
        end_date = st.sidebar.date_input(
            "End Date of Policy", value=pd.to_datetime("2026-01-01")
        )

    difference = relativedelta(end_date, inception_date)
    policy_duration = difference.years + difference.months / 12.0

    policy_value = st.sidebar.number_input(
        "Policy Value", min_value=0.00, max_value=1e10
    )

    data = {
        "gender": gender,
        "occupation_group": occupation.upper(),
        "plan": plan.upper(),
        "policy_value": policy_value,
        "policy_duration_years": policy_duration,
        "is_life_insurance": is_life_insurance,
    }
    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()


# Display user input
# st.subheader("User Input features")
# Display user input
st.subheader("User Input features")


def highlight_max(s):
    """
    Highlight the maximum value in each column.
    """
    styles = []
    for col in s.index:
        if s[col] == s.max():
            styles.append(
                "background-color: #f0ad4e; color: #fff; font-weight: bold"
            )  # Highlight max value with a shade of orange
        else:
            styles.append("")
    return styles


styled_df = df.style.apply(highlight_max, axis=0)

# Apply additional styling to the dataframe
styled_df.set_table_styles(
    [
        {
            "selector": "th",
            "props": [
                ("background-color", "#5bc0de"),
                ("color", "white"),
                ("font-size", "17px"),
                ("font-weight", "bold"),
            ],
        },  # Header styling
        {"selector": "td", "props": [("font-size", "15px")]},  # Cell content styling
    ]
)

# Display the styled dataframe
st.dataframe(styled_df)

if st.sidebar.button("Get Predictions"):
    with st.spinner("Calculating..."):
        time.sleep(1)  # Simulate the loading time

        # Prediction
        prediction = model.predict(df)

        # Display prediction
        st.subheader("Prediction")
        st.write(f"Your estimated monthly premium is GH₵ {prediction[0]:.2f}")

        # Display a success message
        st.success("Prediction generated successfully!")
