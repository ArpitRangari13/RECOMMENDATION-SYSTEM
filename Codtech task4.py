# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load the dataset
# Make sure to adjust the path to where your movies.csv file is located
movies = pd.read_csv('movies_metadata.csv')  # Adjust the path

# Display the first few rows of the dataset
print(movies.head())

# Check for missing values in 'title' and 'genres'
print(movies[['title', 'genres']].isnull().sum())

# Fill missing values in 'title' and 'genres' with empty strings
movies['title'] = movies['title'].fillna('')
movies['genres'] = movies['genres'].fillna('')

# Preprocess the genres column
# Convert genres from string to a list
movies['genres'] = movies['genres'].apply(lambda x: x.split('|') if isinstance(x, str) else [])

# Create a new column that combines title and genres for better recommendations
movies['combined_features'] = movies['title'] + ' ' + movies['genres'].apply(lambda x: ' '.join(x))

# Check for any NaN values in the combined_features column
print(movies['combined_features'].isnull().sum())

# Create a TF-IDF Vectorizer to convert the combined features into a matrix of TF-IDF features
# Limit the number of features to reduce memory usage
tfidf = TfidfVectorizer(stop_words='english', max_features=10000)  # Adjust max_features as needed
tfidf_matrix = tfidf.fit_transform(movies['combined_features'])

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get movie recommendations based on the title
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = movies.index[movies['title'] == title].tolist()
    
    if not idx:
        return f"Movie '{title}' not found in the dataset."
    
    idx = idx[0]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]  # Exclude the first one as it is the movie itself

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return movies['title'].iloc[movie_indices]

# Example: Get recommendations for a specific movie
recommended_movies = get_recommendations('The Shawshank Redemption')  # Replace with a movie title from your dataset
print('Recommended movies:')
print(recommended_movies)