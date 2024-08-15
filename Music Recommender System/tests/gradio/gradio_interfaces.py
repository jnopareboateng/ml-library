# Gradio interfaces
import gradio as gr


def register_user_interface(gr):
    """
    Gradio interface for user registration.
    """
    with gr.Box(title="Register User"):
        name = gr.Textbox(label="Name")
        age = gr.Slider(label="Age", minimum=13, maximum=100, default=25)
        gender = gr.Radio(label="Gender", choices=["Male", "Female", "Non-binary"])
        country = gr.Textbox(label="Country")
        edu_level = gr.Dropdown(label="Education Level", choices=["High School", "College", "Graduate Degree"])
        submit_button = gr.Button(value="Register")

        def register(name_val, age_val, gender_val, country_val, edu_level_val):
            response = register_user_interface(name_val, age_val, gender_val, country_val, edu_level_val)
            gr.Textbox(value=response, editable=False).show()

        submit_button.click(register, inputs=[name, age, gender, country, edu_level], outputs=None)

def login_user_interface(gr):
    """
    Gradio interface for user login.
    """
    with gr.Box(title="Login User"):
        user_id = gr.Textbox(label="User ID")
        login_button = gr.Button(value="Login")

        def login(user_id_val):
            response = login_user_interface(user_id_val)
            gr.Textbox(value=response, editable=False).show()

        login_button.click(login, inputs=user_id, outputs=None)

def rate_song_interface(user_id):
    """
    Gradio interface for song rating.
    """
    with gr.Box(title="Rate Song"):
        song_to_rate = gr.Dropdown(label="Select Song", choices=[Songs.loc[i, 'track_name'] for i in Songs.index])
        rating = gr.Slider(label="Rating", minimum=1, maximum=5, default=3)
        rate_button = gr.Button(value="Rate")

        def submit_rating(song_id, rating_val):
            response = rate_song_interface(user_id, song_id, rating_val)
            gr.Textbox(value=response, editable=False).show()

        rate_button.click(submit_rating, inputs=[song_to_rate, rating], outputs=None)

def get_recommendations_interface(user_id):
    """
    Gradio interface for song recommendations.
    """
    with gr.Box(title="Get Recommendations"):
        epsilon = gr.Slider(label="Exploration Rate (Epsilon)", minimum=0.0, maximum=1.0, default=0.1)
        recommend_button = gr.Button(value="Get Recommendations")

        def show_recommendations(epsilon_val):
            recommendations = get_recommendations_interface(user_id, epsilon_val)
            recommended_songs = "\n".join([f"{i+1}. {song}" for i, song in enumerate(recommendations)])
            gr.Textbox(value=f"Recommended Songs:\n{recommended_songs}", editable=False).show()

        recommend_button.click(show_recommendations, inputs=epsilon, outputs=None)
