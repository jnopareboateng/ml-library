import streamlit as st
from core_functions import (
    register_user_interface,
    login_user_interface,
    rate_song_interface,
    get_recommendations_interface,
    get_user_data,
)

def register_user_interface_st(st):
    st.subheader("Register User")
    name = st.text_input("Name")
    age = st.slider("Age", min_value=13, max_value=100, value=25)
    gender = st.radio("Gender", ["Male", "Female", "Non-binary"])
    country = st.text_input("Country")
    edu_level = st.selectbox("Education Level", ["High School", "College", "Graduate Degree"])

    if st.button("Register"):
        response = register_user_interface(name, age, gender, country, edu_level)
        st.success(response)

def login_user_interface_st(st):
    st.subheader("Login User")
    user_id = st.text_input("User ID")

    if st.button("Login"):
        response = login_user_interface(user_id)
        return response
    return ""  # return an empty string if the button is not clicked

def rate_song_interface_st(st, user_id):
    user_data = get_user_data(user_id)
    if user_data:
        st.subheader("Rate Song")
        song_to_rate = st.selectbox("Select Song", [Songs.loc[i, 'track_name'] for i in Songs.index])
        rating = st.slider("Rating", min_value=1, max_value=5, value=3)

        if st.button("Rate"):
            selected_song_id = Songs.loc[Songs['track_name'] == song_to_rate].index.values[0]
            response = rate_song_interface(user_id, selected_song_id, rating)
            st.success(response)

def get_recommendations_interface_st(st, user_id):
    user_data = get_user_data(user_id)
    if user_data:
        st.subheader("Get Recommendations")
        epsilon = st.slider("Exploration Rate (Epsilon)", min_value=0.0, max_value=1.0, value=0.1)

        if st.button("Get Recommendations"):
            recommendations = get_recommendations_interface(user_id, epsilon)
            recommended_songs = "\n".join([f"{i+1}. {song}" for i, song in enumerate(recommendations)])
            st.success(f"Recommended Songs:\n{recommended_songs}")


