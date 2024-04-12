import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load song data and similarity matrix
song_data = pd.read_csv('musicrec.csv')
similarity_matrix = np.loadtxt('similarities.csv', delimiter=',')

# Streamlit UI
st.title('Music Recommendation System')

# Input field for user to enter song name
song_name = st.text_input('Enter a song name:')

if song_name:
    # Search for the song in the song data
    song = song_data[song_data['title'].str.contains(song_name, case=False)]

    if not song.empty:
        # Get the index of the song in the song data
        song_index = song.index[0]

        # Get similar songs based on the similarity matrix
        similar_songs_indices = np.argsort(similarity_matrix[song_index])[::-1][1:6]
        similar_songs = song_data.iloc[similar_songs_indices]

        st.subheader('Recommended Songs:')
        for i, similar_song in similar_songs.iterrows():
            st.write(f"Name: {similar_song['title']}")

            # Parse details from the 'tags' column
            tags = similar_song['tags'].split()
            artist = ' '.join(tags[:-3])
            genre = tags[-3]
            album = tags[-2]
            rating = tags[-1]

            st.write(f"Artist(s): {artist}")
            st.write(f"Genre: {genre}")
            st.write(f"Album/Movie: {album}")
            st.write(f"User-Rating: {rating}")

            st.write('---')
    else:
        st.write('No results found for the given song name.')
