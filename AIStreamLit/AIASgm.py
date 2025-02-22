# Imports
import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import difflib  # Import difflib for fuzzy matching
from sklearn.decomposition import PCA  # Import PCA
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
import seaborn as sns  # Optional: For better visuals
from youtubesearchpython import VideosSearch  # For searching YouTube links

# Load the dataset
data = pd.read_csv("AIStreamLit/spotify_songs.csv")

# Select only the required columns
filtered_data = data[['track_artist', 'track_name', 'danceability', 'energy', 'playlist_subgenre', 'speechiness', 'acousticness', 'instrumentalness', 'valence', 'tempo']]

# Drop rows where 'track_name' or 'track_artist' is NaN
filtered_data = filtered_data.dropna(subset=['track_name', 'track_artist'])
filtered_data['track_name'] = filtered_data['track_name'].astype(str)  # Ensure all track names are strings

#Convert 'playlist_subgenre' to numeric using one-hot encoding
data_encoded = pd.get_dummies(filtered_data['playlist_subgenre'])

# List of features to scale
features_to_scale = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'valence', 'tempo']

# Scale 'speechiness' and 'tempo'
scaler = StandardScaler()
filtered_data[features_to_scale] = scaler.fit_transform(filtered_data[features_to_scale])

# Combine encoded and scaled features
features = pd.concat([data_encoded, filtered_data[features_to_scale]], axis=1)

# Train the k-nearest neighbors model
knn = NearestNeighbors(n_neighbors=10, metric='euclidean')
knn.fit(features)

# Function to find the closest matching song using fuzzy matching
def find_closest_song(song_name, song_list):
    closest_match = difflib.get_close_matches(song_name, song_list, n=1, cutoff=0.6)  # 60% similarity cutoff
    if closest_match:
        return closest_match[0]
    else:
        return None

# Function to search YouTube for a song and return the first video link
def get_youtube_link(song_name, artist_name):
    search_query = f"{song_name} {artist_name} official"
    videos_search = VideosSearch(search_query, limit=1)
    result = videos_search.result()
    
    if result['result']:
        return result['result'][0]['link']  # Return the first YouTube video link
    else:
        return None

def plot_knn(input_features, neighbors_indices):
    # Apply PCA to reduce dimensions
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)
    
    # Plot the points
    plt.figure(figsize=(10, 6))
    
    # Plot all the songs
    sns.scatterplot(x=features_2d[:, 0], y=features_2d[:, 1], alpha=0.5, label="All Songs")
    
    # Plot the input song
    input_2d = pca.transform(input_features)
    plt.scatter(input_2d[:, 0], input_2d[:, 1], color='red', s=100, label="Input Song")
    
    # Plot the neighbors
    for idx in neighbors_indices[0]:
        plt.scatter(features_2d[idx, 0], features_2d[idx, 1], color='green', s=100, label="Recommendation")
    
    plt.title("KNN Song Recommendations")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    st.pyplot(plt)

# Song recommendation function
def recommend_song(song_name, artist_name):
    # Find the closest matching song and artist in the dataset
    closest_song_row = filtered_data[
        (filtered_data['track_name'].str.contains(song_name, case=False)) &
        (filtered_data['track_artist'].str.contains(artist_name, case=False))
    ]
    
    if closest_song_row.empty:
        st.write(f"No close match found for '{song_name}' by '{artist_name}' in the dataset.")
        return
    else:
        # Extract the closest match details
        closest_song = closest_song_row['track_name'].values[0]
        closest_artist = closest_song_row['track_artist'].values[0]
        st.write(f"Closest match found: '{closest_song}' by {closest_artist}")
    
    # Extract the features of the closest matching song
    input_features = features[filtered_data['track_name'] == closest_song]
    
    # Find the nearest neighbors
    distances, indices = knn.kneighbors(input_features)
    
    # Retrieve the recommended songs and artists, excluding the input song
    recommendations = filtered_data.iloc[indices[0]][['track_name', 'track_artist']].values
    st.write(f"Songs similar to '{closest_song}' by {closest_artist}:")
    
    recommended_songs = set()  # Use a set to avoid duplicates
    for rec in recommendations:
        song, artist = rec
        if song != closest_song and (song, artist) not in recommended_songs:  # Avoid duplicates and the input song
            recommended_songs.add((song, artist))
            youtube_link = get_youtube_link(song, artist)
            if youtube_link:
                st.write(f"'{song}' by {artist}: [YouTube Link]({youtube_link})")
            else:
                st.write(f"'{song}' by {artist}: No YouTube link found")

    plot_knn(input_features, indices)

# Streamlit interface
st.title("Spotify Song Recommender")

# Get song name and artist name from the user
input_song = st.text_input("Enter the song name:")
input_artist = st.text_input("Enter the artist name:")

# If both inputs are provided, run the recommendation function
if input_song and input_artist:
    recommend_song(input_song, input_artist)
