import streamlit as st
import pyperclip

from core_functions import (
    Songs,
    register_user,
    login_user,
    rate_song,
    get_recommendations,
    get_user_data,
)

def register_user_interface_st(st):
    st.subheader("Register User")
    name = st.text_input("Name")
    age = st.slider("Age", min_value=18, max_value=100, value=25)
    gender = st.radio("Gender", ["Male", "Female", "Non-binary"])
    country = st.text_input("Country")
    edu_level = st.selectbox("Education Level", ["High School", "Undergraduate", "Graduate Degree"])
    fav_genre = st.text_input("Favorite Genre")
    fav_artist = st.text_input("Favorite Artist")

    registration_message = "" # Initialize registration message

    if st.button("Register"):
        registration_message = register_user(name, age, gender, country, edu_level, fav_genre, fav_artist)
        st.success(registration_message)

    # Use session_state to track copy button click
    if "successfully" in registration_message:
        if 'copied' not in st.session_state:
            st.session_state['copied'] = False
            copied = st.session_state['copied']
        
        if st.button("Copy User ID"):
            user_id = registration_message.split(": ")[-1]
            pyperclip.copy(user_id)
            st.session_state['copied'] = True  # Update session state
            copied = True
            st.success("User ID copied to clipboard!")
    
def login_user_interface_st(st):
    st.subheader("Login User")
    user_id = st.text_input("User ID")

    if st.button("Login"):
        login_response = login_user(user_id)
        if "Welcome back" in login_response:
            user_id = login_response.split("Welcome back, ")[1].split("!")[0].strip()
            st.success(login_response[0])
            st.write(login_response[1])  # Display the recommendations table
            rate_song_interface_st(st, user_id)
            get_recommendations_interface_st(st, user_id)
        else:
            st.error(login_response)
    return ""  # return an empty string if the button is not clicked

def rate_song_interface_st(st, user_id):
    user_data = get_user_data(user_id)
    if user_data:
        st.subheader("Rate Song")
        song_to_rate = st.selectbox("Select Song", [Songs.loc[i, 'track_name'] for i in Songs.index])
        rating = st.slider("Rating", min_value=1, max_value=5, value=3)

        if st.button("Rate"):
            selected_song_id = Songs.loc[Songs['track_name'] == song_to_rate].index.values[0]
            response = rate_song(user_id, selected_song_id, rating)
            st.success(response)

def get_recommendations_interface_st(st, user_id):
    user_data = get_user_data(user_id)
    if user_data:
        st.subheader("Get Recommendations")
        epsilon = st.slider("Exploration Rate (Epsilon)", min_value=0.0, max_value=1.0, value=0.1)

        if st.button("Get New Recommendations"):
            recommendations = get_recommendations(user_id, epsilon)
            if recommendations:
                recommendations_df = Songs.loc[recommendations, ['music', 'artist']]

                st.subheader("Recommended Songs")
                for index, row in recommendations_df.iterrows():
                    st.write(f"{row['track_name']} - {row['artiste_name']}")
            else:
                st.warning("No new recommendations available at the moment.")
