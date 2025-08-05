from flask import Flask, request, render_template, jsonify
from model.recommender import MovieRecommender

app = Flask(__name__)
recommender = MovieRecommender('data/tmdb_5000_movies.csv')

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["GET"])
def recommend():
    movie_title = request.args.get('movie')
    if not movie_title:
        return render_template("index.html", error="Please enter a movie title.")

    recommendations = recommender.recommend(movie_title)

    if isinstance(recommendations, str):
        return render_template("index.html", error=recommendations)

    return render_template("index.html", movie=movie_title, recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
