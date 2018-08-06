from predict.predict import rating_movie, recommend_same_type_movie, recommend_your_favorite_movie, recommend_other_favorite_movie

# rate the movie (USER_ID, MOVIE_ID)
rating_movie(234, 1401)

# recommend 3 same type movies to you (MOVIE_ID)
recommend_same_type_movie(1401)

# recommen 3 movies you have not watched yet (USER_ID)
recommend_your_favorite_movie(234)

# recomen 3 movies that other people will watch after watching this movie (MOVIE_ID)
recommend_other_favorite_movie(1401)
