# Imports
import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import difflib  # Import difflib for fuzzy matching

#Load the dataset
data = pd.read_csv("spotify_songs.csv")

#Select only the required column
filtered_data = data[['track_name', 'playlist_subgenre', 'speechiness', 'tempo', 'track_artist']]

filtered_data = filtered_data.dropna(subset=['track_name', 'track_artist'])  # Drop rows where 'track_name' is NaN
filtered_data['track_name'] = filtered_data['track_name'].astype(str)  # Ensure all track names are strings

print("Filtered Data Sample (After Cleaning):")
filtered_data.head()

#Convert playlist_genre to numeric using one-hot encoding
data_encoded = pd.get_dummies(filtered_data['playlist_subgenre'])

#Scale tempo and energy
scaler = StandardScaler()

# Apply scaling using .loc
filtered_data.loc[:, 'speechiness'] = scaler.fit_transform(filtered_data[['speechiness']])
filtered_data.loc[:, 'tempo'] = scaler.fit_transform(filtered_data[['tempo']])

features = pd.concat([data_encoded, filtered_data[['speechiness', 'tempo']]], axis=1)

knn = NearestNeighbors(n_neighbors=10, metric='euclidean')
knn.fit(features)


def find_closest_song(song_name, song_list):
    closest_match = difflib.get_close_matches(song_name, song_list, n=1, cutoff=0.6)  # 60% similarity cutoff
    if closest_match:
        return closest_match[0]
    else:
        return None


def recommend_song(song_name, artist_name):
    # Find the closest matching song and artist in the dataset
    closest_song_row = filtered_data[
        (filtered_data['track_name'].str.contains(song_name, case=False)) &
        (filtered_data['track_artist'].str.contains(artist_name, case=False))
    ]
    
    if closest_song_row.empty:
        print(f"No close match found for '{song_name}' by '{artist_name}' in the dataset.")
        return
    else:
        # Extract the closest match details (assuming only one match is needed)
        closest_song = closest_song_row['track_name'].values[0]
        closest_artist = closest_song_row['track_artist'].values[0]
        print(f"Closest match found: '{closest_song}' by {closest_artist} \n\n")
    
    # Extract the features of the closest matching song
    input_features = features[filtered_data['track_name'] == closest_song]
    
    # Find the nearest neighbors
    distances, indices = knn.kneighbors(input_features)
    
    # Retrieve the recommended songs and artists, excluding the input song
    recommendations = filtered_data.iloc[indices[0]][['track_name', 'track_artist']].values
    print(f"Songs similar to '{closest_song}' by {closest_artist}:")
    
    recommended_songs = set()  # Use a set to avoid duplicates
    for rec in recommendations:
        song, artist = rec
        if song != closest_song and (song, artist) not in recommended_songs:  # Avoid duplicates and the input song
            recommended_songs.add((song, artist))
            print(f"'{song}' by {artist}")

# Get both song name and artist name from the user
input_song = input("Enter the song name: ")
input_artist = input("Enter the artist name: ")
recommend_song(input_song, input_artist)