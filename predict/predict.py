import tensorflow as tf
import pickle
import numpy as np
import random


def get_tensors(loaded_graph):

    uid = loaded_graph.get_tensor_by_name("uid:0")
    user_gender = loaded_graph.get_tensor_by_name("user_gender:0")
    user_age = loaded_graph.get_tensor_by_name("user_age:0")
    user_job = loaded_graph.get_tensor_by_name("user_job:0")
    movie_id = loaded_graph.get_tensor_by_name("movie_id:0")
    movie_categories = loaded_graph.get_tensor_by_name("movie_categories:0")
    movie_titles = loaded_graph.get_tensor_by_name("movie_titles:0")
    targets = loaded_graph.get_tensor_by_name("targets:0")
    dropout_keep_prob = loaded_graph.get_tensor_by_name("dropout_keep_prob:0")
    lr = loaded_graph.get_tensor_by_name("LearningRate:0")
    inference = loaded_graph.get_tensor_by_name("inference/ExpandDims:0")
    movie_combine_layer_flat = loaded_graph.get_tensor_by_name("movie_fc/Reshape:0")
    user_combine_layer_flat = loaded_graph.get_tensor_by_name("user_fc/Reshape:0")
    return uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, lr, dropout_keep_prob, inference, movie_combine_layer_flat, user_combine_layer_flat


"""
    Predict the star the user will rate on the movie:

        Use forward propagation to predicit the predicting ratings based on the trained nn model

"""


def rating_movie(user_id_val, movie_id_val):

    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        # load saved model
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)

        # get tensors from loaded model
        uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, lr, dropout_keep_prob, inference, movie_combine_layer_flat, user_combine_layer_flat = get_tensors(loaded_graph)

        categories = np.zeros([1, 18])
        categories[0] = movies.values[movieid2idx[movie_id_val]][2]

        titles = np.zeros([1, title_count])
        titles[0] = movies.values[movieid2idx[movie_id_val]][1]

        feed = {
              uid: np.reshape(users.values[user_id_val-1][0], [1, 1]),
              user_gender: np.reshape(users.values[user_id_val-1][1], [1, 1]),
              user_age: np.reshape(users.values[user_id_val-1][2], [1, 1]),
              user_job: np.reshape(users.values[user_id_val-1][3], [1, 1]),
              movie_id: np.reshape(movies.values[movieid2idx[movie_id_val]][0], [1, 1]),
              movie_categories: categories,
              movie_titles: titles,
              dropout_keep_prob: 1}

        # get prediction
        results = sess.run([inference], feed)
        print("--------------------------------------------------------------")
        print("You are {}".format(users_orig[user_id_val-1]))
        print("The movie you will see is {}".format(movies_orig[movieid2idx[movie_id_val]]))
        print("And the predicted star you will rate is {}".format(*results))
        print("--------------------------------------------------------------")
        return results


"""
    Get the recommendation movies of the same type the input movie:

        Compute the Cosine Similarity between the input movie and the whole movies features matric.
        Get the top k most similar movies to the input movie, build a probability vector length of 3883 which contains k non-zero values (probabilities)
        Randomly pick 3 movies from the top k movies in order to get different recommendations each time

"""


def recommend_same_type_movie(movie_id_val, top_k=20):

    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        # Load saved model
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)

        norm_movie_matrics = tf.sqrt(tf.reduce_sum(tf.square(movie_matrics), 1, keepdims=True))
        normalized_movie_matrics = movie_matrics / norm_movie_matrics

        # get the same type of the movie
        probs_embeddings = (movie_matrics[movieid2idx[movie_id_val]]).reshape([1, 200])
        probs_similarity = tf.matmul(probs_embeddings, tf.transpose(normalized_movie_matrics))
        sim = (probs_similarity.eval())
        # p is a probability vector which contains k non-zero values
        p = np.squeeze(sim)
        p[np.argsort(p)[:-top_k]] = 0
        p = p / np.sum(p)
        results = set()
        while len(results) != 3:
            c = np.random.choice(3883, 1, p=p)[0]
            results.add(c)

        print("--------------------------------------------------------------")
        print("The movie you have watched is ：{}".format(movies_orig[movieid2idx[movie_id_val]]))
        print("Here are some recommendations based on the same type of the movie above：")

        for val in (results):
            print(movies_orig[val])
        print("--------------------------------------------------------------")
        return results


"""
    Get the recommendation movies based on the input movie:

        Compute the ratings of the input user to all the movies, using the vector to matrix multiply the whole movies features matric
        Get the top k ratings from the result
        Randomly pick 3 movies from the top k ratings movies in order to get different recommendations each time

"""


def recommend_your_favorite_movie(user_id_val, top_k=10):

    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        # load saved model
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)

        # get the user features vector from the matric
        probs_embeddings = (users_matrics[user_id_val-1]).reshape([1, 200])

        # get the ratings by matmul the user vector to the whole movies features matric
        probs_similarity = tf.matmul(probs_embeddings, tf.transpose(movie_matrics))
        sim = (probs_similarity.eval())
        # p is a probability vector which contains k non-zero values
        p = np.squeeze(sim)
        p[np.argsort(p)[:-top_k]] = 0
        p = p / np.sum(p)
        results = set()
        while len(results) != 3:
            c = np.random.choice(3883, 1, p=p)[0]
            results.add(c)

        print("--------------------------------------------------------------")
        print("You are {}".format(users_orig[user_id_val-1]))
        print("Here are some recommendations based on the movies you have watched")
        for val in (results):
            print(movies_orig[val])
        print("--------------------------------------------------------------")
        return results


"""
    Get the recommendation movies based on the other people's ratings to the input movie:

        Compute the top k users who have highest ratings on the input movie, using matmul between the input movie vector and the whole user features matric
        Get the top k users vectors from the user features matric
        Compute the predicted star which the top k users will rate to the all movies
        Get the highest rating as the recommendation of each user so we get k movies
        Randomly pick 3 movies from the k movies in order to get different recommendations each time

"""


def recommend_other_favorite_movie(movie_id_val, top_k=20):
    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        # load saved model
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)

        # get the top k users vectors from the user features matric
        probs_movie_embeddings = (movie_matrics[movieid2idx[movie_id_val]]).reshape([1, 200])
        probs_user_favorite_similarity = tf.matmul(probs_movie_embeddings, tf.transpose(users_matrics))
        favorite_user_id = np.argsort(probs_user_favorite_similarity.eval())[0][-top_k:]

        print("--------------------------------------------------------------")
        print("Some other people like this film too, they are ：{}".format(users_orig[favorite_user_id-1]))

        # get the ratings by matmul the user vectors to the whole movies features matric
        probs_users_embeddings = (users_matrics[favorite_user_id-1]).reshape([-1, 200])
        probs_similarity = tf.matmul(probs_users_embeddings, tf.transpose(movie_matrics))
        sim = (probs_similarity.eval())

        # p is a vector which contains k values that the k users' favorite movie
        p = np.argmax(sim, 1)

        results = set()
        while len(results) != 3:
            c = p[random.randrange(top_k)]
            results.add(c)

        print("The movie you have watched is ：{}".format(movies_orig[movieid2idx[movie_id_val]]))
        print("And they also love these movies：")
        for val in (results):
            print(movies_orig[val])
        print("--------------------------------------------------------------")
        return results


load_dir = pickle.load(open('params.p', mode='rb'))
title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig = pickle.load(open('preprocess.p', mode='rb'))
# the index of the movie index
movieid2idx = {val[0]: i for i, val in enumerate(movies.values)}
users_matrics = pickle.load(open('users_matrics.p', mode='rb'))
movie_matrics = pickle.load(open('movie_matrics.p', mode='rb'))
