# app.py

import streamlit as st
import pandas as pd

# Load the dataset
@st.cache
def load_data():
    data = pd.read_csv('../../data/Spotify_MPD_Feature_Engineered.CSV')
    return data

data = load_data()

# Display the dataset
st.write("Dataset Overview")
st.dataframe(data.head())

# Basic preprocessing
data.fillna(0, inplace=True)


# app.py (continued)

# Sidebar for user inputs
st.sidebar.header("User Input Features")
selected_genre = st.sidebar.selectbox("Select Genre", data['Genre'].unique())
selected_artist = st.sidebar.selectbox("Select Artist", data['artist_name'].unique())

# Filter data based on user input
filtered_data = data[(data['Genre'] == selected_genre) & (data['artist_name'] == selected_artist)]
st.write(f"Filtered Data for {selected_genre} and {selected_artist}")
st.dataframe(filtered_data.head())


# app.py (continued)

# Sidebar for user inputs
st.sidebar.header("User Input Features")
selected_genre = st.sidebar.selectbox("Select Genre", data['Genre'].unique())
selected_artist = st.sidebar.selectbox("Select Artist", data['artist_name'].unique())

# Filter data based on user input
filtered_data = data[(data['Genre'] == selected_genre) & (data['artist_name'] == selected_artist)]
st.write(f"Filtered Data for {selected_genre} and {selected_artist}")
st.dataframe(filtered_data.head())

# app.py (continued)

import matplotlib.pyplot as plt
import seaborn as sns

# Sidebar for user inputs
st.sidebar.header("User Input Features")
selected_genre = st.sidebar.selectbox("Select Genre", data['Genre'].unique())
selected_artist = st.sidebar.selectbox("Select Artist", data['artist_name'].unique())
selected_country = st.sidebar.selectbox("Select Country", data['country'].unique())
selected_age_group = st.sidebar.selectbox("Select Age Group", sorted(data['age'].unique()))

# Filter data based on user input
filtered_data = data[(data['Genre'] == selected_genre) & 
                     (data['artist_name'] == selected_artist) &
                     (data['country'] == selected_country) &
                     (data['age'] == selected_age_group)]

st.write(f"Filtered Data for {selected_genre}, {selected_artist}, {selected_country}, Age {selected_age_group}")
st.dataframe(filtered_data.head())

# Visualizations
st.write("Data Visualizations")

# Bar chart of plays per artist
plt.figure(figsize=(10,5))
sns.barplot(x='artist_name', y='plays', data=filtered_data)
plt.xticks(rotation=90)
st.pyplot(plt)

# Scatter plot of audio popularity vs. danceability
plt.figure(figsize=(10,5))
sns.scatterplot(x='Audio Popularity', y='Danceability', hue='artist_name', data=filtered_data)
st.pyplot(plt)

# app.py (continued)

from Qtable_RL import QTableRL

# Sidebar for RL parameters
st.sidebar.header("Reinforcement Learning Parameters")
learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.1)
discount_factor = st.sidebar.slider("Discount Factor", 0.1, 1.0, 0.99)
exploration_rate = st.sidebar.slider("Exploration Rate", 0.0, 1.0, 1.0)
episodes = st.sidebar.number_input("Number of Episodes", min_value=1, max_value=10000, value=1000)

# Placeholder environment (to be replaced with actual environment)
class DummyEnv:
    def __init__(self):
        self.state_size = 10
        self.action_size = 4
    def reset(self):
        return 0
    def step(self, action):
        next_state = (action + 1) % self.state_size
        reward = np.random.random()
        done = np.random.choice([True, False])
        return next_state, reward, done, {}

env = DummyEnv()
q_learning_agent = QTableRL(state_size=env.state_size, action_size=env.action_size, learning_rate=learning_rate, discount_factor=discount_factor, exploration_rate=exploration_rate)

if st.sidebar.button("Train RL Agent"):
    rewards = q_learning_agent.train(episodes=episodes, env=env)
    st.write("Training complete!")

    # Visualization of training rewards
    plt.figure(figsize=(10,5))
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    st.pyplot(plt)
