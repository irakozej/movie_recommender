import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class MovieRecommender:
    def __init__(self, movies_path):
        self.movies_path = movies_path
        self.movies = None
        self.tfidf_matrix = None
        self.similarity_matrix = None
        self._prepare()

    def _prepare(self):
        # Load data
        self.movies = pd.read_csv(self.movies_path)
        self.movies = self.movies[['title', 'overview']].dropna()

        # TF-IDF Vectorizer
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.movies['overview'])

        # Compute cosine similarity matrix
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

    def recommend(self, title, top_n=5):
        # Check if movie exists
        if title not in self.movies['title'].values:
            return f"Movie '{title}' not found in dataset."

        # Get index of the movie
        idx = self.movies[self.movies['title'] == title].index[0]

        # Get similarity scores
        sim_scores = list(enumerate(self.similarity_matrix[idx]))

        # Sort movies by similarity score
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

        # Get movie titles
        recommendations = [self.movies.iloc[i[0]]['title'] for i in sim_scores]
        return recommendations
