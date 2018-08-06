### Movies Recommendation System

A movie recommendation system based on Tensorflow which combines FNN and CNN to extract the features of users and movies to compute the ratings

#### Introduction

This is a movie recommendation system based on deep learning, after training the model and extracting the features, it has functions below:

##### rating_movie(USER_ID, MOVIE_ID) : Predict the ratings the USER to the MOVIE

##### recommend_same_type_movie(MOVIE_ID) : recommend 3 same type movies to the MOVIE

##### recommend_your_favorite_movie(USER_ID) : recommend 3 movies USER have not watched yet

##### recommend_other_favorite_movie(MOVIE_ID): recommend 3 movies that ALL USERS who have watched the MOVIE_ID also love to watch

#### Dependencies

```
tensorflow = 1.9.0
pandas = 0.20.3
numpy = 1.14.5
scikit-learn = 0.19.0
matplotlib = 2.0.2
```

#### Instrcution

1- Add this folder to your workspace and open the terminal:


2- Install the dependencies above

3- Open a terminal to get the pre-process data:
```
python data/data.py
```

4- Train the model(Option)(Skip if you do not want to train it by yourself):

I have put the trained model in the files, feel free to train it by yourself or you can use the trained model
```
python model/model.py
```

5- Get the User and Movie features matrix(Option)(Skip if you do not want to train it by yourself):

I have put the trained model in the files, feel free to train it by yourself or you can use the trained model
```
python features/get_features.py
```

6- Enjoy the recommendation system:

In the main file, there are 4 functions which can recommend movies to some users, change the parameters if you want
```
python main.py
```

#### A Quick Show

##### The Training and Test Loss

![image](https://github.com/RenfeiChen/Recommendation-System/blob/master/Loss.png)

##### Some Hints

If you want to know the detail of the model, you can look up the model.py, and here is the structure of the neural network.

![image](https://github.com/RenfeiChen/Recommendation-System/blob/master/Structure.png)
