import gradio as gr
from core_functions import (
    register_user_interface,
    login_user_interface,
    rate_song_interface,
    get_recommendations_interface,
    initialize_q_table
)

from gradio_interfaces import (
    register_user_interface as register_user_interface_gr,
    login_user_interface as login_user_interface_gr,
    rate_song_interface as rate_song_interface_gr,
    get_recommendations_interface as get_recommendations_interface_gr
)

# Setup for Gradio app using imported interfaces

# Initialize Q-table
Q = initialize_q_table()

# Track logged-in user
logged_in_user = None

def login_user_wrapper(user_id):
    global logged_in_user
    response = login_user_interface(user_id)
    if "Welcome back" in response:
        logged_in_user = user_id
    gr.Textbox(value=response, editable=False)

def main():
    """
    The main function to run the Gradio app.
    """
    # Define the types of the inputs and outputs for the Gradio interfaces
    register_user_interface = gr.Interface(fn=register_user_interface_gr, inputs="text", outputs="text")
    login_user_wrapper_interface = gr.Interface(fn=login_user_wrapper, inputs="text", outputs="text")

    # Interfaces based on login status
    if logged_in_user is None:
        # Public interfaces: registration, login
        tabs = gr.Tabs(
            [
                gr.Tab("Register User", register_user_interface),  # Pass existing interface objects
                gr.Tab("Login User", login_user_wrapper_interface),
            ]
        )
    else:
        # User-specific interfaces: rate song, get recommendations
        rate_song_interface_wrapper = gr.Interface(fn=rate_song_interface_gr, inputs=["text", "text", "number"], outputs="text")
        get_recommendations_interface_wrapper = gr.Interface(fn=get_recommendations_interface_gr, inputs="text", outputs="text")

        tabs = gr.Tabs(
            [
                gr.Tab("Rate Song", rate_song_interface_wrapper),
                gr.Tab("Get Recommendations", get_recommendations_interface_wrapper),
            ]
        )

    tabs.launch()

if __name__ == "__main__":
    main()

