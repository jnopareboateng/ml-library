from numpy import rec
import streamlit as st
from core_functions import (
    register_user,
    login_user,
    rate_song,
    get_recommendations,
)
from streamlit_interfaces import (
    register_user_interface_st,
    login_user_interface_st,
    rate_song_interface_st,
    get_recommendations_interface_st,
)

users = {}

def main():
    """
    The main function to run the Streamlit app.
    """
    st.title("Music Recommendation System")

    menu = ["Register", "Login",]
    choice = st.sidebar.selectbox("Menu", menu)


    if choice == "Register":
        register_user_interface_st(st)

    elif choice == "Login":
        login_response = login_user_interface_st(st)
        if "Welcome back" in login_response:
            user_id = login_response.split("Welcome back, ")[1]
            st.success(login_response)
            rate_song_interface_st(st, user_id)
            get_recommendations_interface_st(st, user_id)

    # elif choice == "Recommendation":
    #     st.warning("Please login first.")

if __name__ == "__main__":
    main()