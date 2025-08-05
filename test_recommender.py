from model.recommender import MovieRecommender

recommender = MovieRecommender('data/tmdb_5000_movies.csv')

movie = 'Avatar'
results = recommender.recommend(movie)

print(f"Movies similar to '{movie}':")
for r in results:
    print(" -", r)
