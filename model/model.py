import tensorflow as tf
import pickle
import time
import datetime
import os
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


# load data from local
title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig = pickle.load(open('preprocess.p', mode='rb'))

# paramters
# the dimension of the embedding matrix
embed_dim = 32
# number of user_id = 6040
uid_max = max(features.take(0, 1)) + 1
# number of gender = 2
gender_max = max(features.take(2, 1)) + 1
# number of ages = 7
age_max = max(features.take(3, 1)) + 1
# number of jobs = 21
job_max = max(features.take(4, 1)) + 1

# number of movie_id = 3952
movie_id_max = max(features.take(1, 1)) + 1
# number of movie categories = 19
movie_categories_max = max(genres2int.values()) + 1
# number of movie titles = 5216
movie_title_max = len(title_set)

# the way of the embbeding as each movie has 18 rows so we use sum to make it
# into one row
combiner = "sum"

# the length of thte title = 15
sentences_size = title_count
# the size of the sliding windows filters in the CNN
window_sizes = {2, 3, 4, 5}
# the number of the filters
filter_num = 8

# save direction
save_dir = './save'


def get_inputs():
    uid = tf.placeholder(tf.int32, [None, 1], name="uid")
    user_gender = tf.placeholder(tf.int32, [None, 1], name="user_gender")
    user_age = tf.placeholder(tf.int32, [None, 1], name="user_age")
    user_job = tf.placeholder(tf.int32, [None, 1], name="user_job")

    movie_id = tf.placeholder(tf.int32, [None, 1], name="movie_id")
    movie_categories = tf.placeholder(tf.int32, [None, 18], name="movie_categories")
    movie_titles = tf.placeholder(tf.int32, [None, 15], name="movie_titles")
    targets = tf.placeholder(tf.int32, [None, 1], name="targets")
    LearningRate = tf.placeholder(tf.float32, name="LearningRate")
    dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
    return uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, LearningRate, dropout_keep_prob


def get_user_embedding(uid, user_gender, user_age, user_job):
    with tf.name_scope("user_embedding"):
        uid_embed_matrix = tf.Variable(tf.random_uniform([uid_max, embed_dim], -1, 1), name="uid_embed_matrix")
        uid_embed_layer = tf.nn.embedding_lookup(uid_embed_matrix, uid, name="uid_embed_layer")

        gender_embed_matrix = tf.Variable(tf.random_uniform([gender_max, embed_dim // 2], -1, 1), name="gender_embed_matrix")
        gender_embed_layer = tf.nn.embedding_lookup(gender_embed_matrix, user_gender, name="gender_embed_layer")

        age_embed_matrix = tf.Variable(tf.random_uniform([age_max, embed_dim // 2], -1, 1), name="age_embed_matrix")
        age_embed_layer = tf.nn.embedding_lookup(age_embed_matrix, user_age, name="age_embed_layer")

        job_embed_matrix = tf.Variable(tf.random_uniform([job_max, embed_dim // 2], -1, 1), name="job_embed_matrix")
        job_embed_layer = tf.nn.embedding_lookup(job_embed_matrix, user_job, name="job_embed_layer")
    return uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer


def get_user_feature_layer(uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer):
    with tf.name_scope("user_fc"):
        # the first fc layer
        uid_fc_layer = tf.layers.dense(uid_embed_layer, embed_dim, name="uid_fc_layer", activation=tf.nn.relu)
        gender_fc_layer = tf.layers.dense(gender_embed_layer, embed_dim, name="gender_fc_layer", activation=tf.nn.relu)
        age_fc_layer = tf.layers.dense(age_embed_layer, embed_dim, name="age_fc_layer", activation=tf.nn.relu)
        job_fc_layer = tf.layers.dense(job_embed_layer, embed_dim, name="job_fc_layer", activation=tf.nn.relu)

        # the second fc layer
        user_combine_layer = tf.concat([uid_fc_layer, gender_fc_layer, age_fc_layer, job_fc_layer], 2)
        user_combine_layer = tf.contrib.layers.fully_connected(user_combine_layer, 200, tf.tanh)

        user_combine_layer_flat = tf.reshape(user_combine_layer, [-1, 200])
    return user_combine_layer, user_combine_layer_flat


def get_movie_id_embed_layer(movie_id):
    with tf.name_scope("movie_embedding"):
        movie_id_embed_matrix = tf.Variable(tf.random_uniform([movie_id_max, embed_dim], -1, 1), name="movie_id_embed_matrix")
        movie_id_embed_layer = tf.nn.embedding_lookup(movie_id_embed_matrix, movie_id, name="movie_id_embed_layer")
    return movie_id_embed_layer


def get_movie_categories_layers(movie_categories):
    with tf.name_scope("movie_categories_layers"):
        movie_categories_embed_matrix = tf.Variable(tf.random_uniform([movie_categories_max, embed_dim], -1, 1), name="movie_categories_embed_matrix")
        movie_categories_embed_layer = tf.nn.embedding_lookup(movie_categories_embed_matrix, movie_categories, name="movie_categories_embed_layer")
        if combiner == "sum":
            movie_categories_embed_layer = tf.reduce_sum(movie_categories_embed_layer, axis=1, keepdims=True)

    return movie_categories_embed_layer


def get_movie_cnn_layer(movie_titles):
    with tf.name_scope("movie_embedding"):
        movie_title_embed_matrix = tf.Variable(tf.random_uniform([movie_title_max, embed_dim], -1, 1), name="movie_title_embed_matrix")
        movie_title_embed_layer = tf.nn.embedding_lookup(movie_title_embed_matrix, movie_titles, name="movie_title_embed_layer")
        movie_title_embed_layer_expand = tf.expand_dims(movie_title_embed_layer, -1)

    # use different size of filters in the CNN layer
    pool_layer_lst = []
    for window_size in window_sizes:
        with tf.name_scope("movie_txt_conv_maxpool_{}".format(window_size)):
            filter_weights = tf.Variable(tf.truncated_normal([window_size, embed_dim, 1, filter_num], stddev=0.1), name="filter_weights")
            filter_bias = tf.Variable(tf.constant(0.1, shape=[filter_num]), name="filter_bias")

            conv_layer = tf.nn.conv2d(movie_title_embed_layer_expand, filter_weights, [1, 1, 1, 1], padding="VALID", name="conv_layer")
            relu_layer = tf.nn.relu(tf.nn.bias_add(conv_layer, filter_bias), name="relu_layer")

            maxpool_layer = tf.nn.max_pool(relu_layer, [1, sentences_size - window_size + 1, 1, 1], [1, 1, 1, 1], padding="VALID", name="maxpool_layer")
            pool_layer_lst.append(maxpool_layer)

    # dropout layer avoid overfitting
    with tf.name_scope("pool_dropout"):
        pool_layer = tf.concat(pool_layer_lst, 3, name="pool_layer")
        max_num = len(window_sizes) * filter_num
        pool_layer_flat = tf.reshape(pool_layer, [-1, 1, max_num], name="pool_layer_flat")

        dropout_layer = tf.nn.dropout(pool_layer_flat, dropout_keep_prob, name="dropout_layer")
    return pool_layer_flat, dropout_layer


def get_movie_feature_layer(movie_id_embed_layer, movie_categories_embed_layer, dropout_layer):
    with tf.name_scope("movie_fc"):
        # the first fc layer
        movie_id_fc_layer = tf.layers.dense(movie_id_embed_layer, embed_dim, name="movie_id_fc_layer", activation=tf.nn.relu)
        movie_categories_fc_layer = tf.layers.dense(movie_categories_embed_layer, embed_dim, name="movie_categories_fc_layer", activation=tf.nn.relu)

        # the second fc layer
        movie_combine_layer = tf.concat([movie_id_fc_layer, movie_categories_fc_layer, dropout_layer], 2)
        movie_combine_layer = tf.contrib.layers.fully_connected(movie_combine_layer, 200, tf.tanh)

        movie_combine_layer_flat = tf.reshape(movie_combine_layer, [-1, 200])
    return movie_combine_layer, movie_combine_layer_flat


def get_batches(Xs, ys, batch_size):
    for start in range(0, len(Xs), batch_size):
        end = min(start + batch_size, len(Xs))
        yield Xs[start:end], ys[start:end]


# hyper-paremeters
# number of Epochs
num_epochs = 20
# batch Size
batch_size = 256
# dropout probability value
dropout_keep = 0.5
# initial learning rate
initial_learning_rate = 0.001
# show stats for every 20 number of batches
show_every_n_batches = 20

tf.reset_default_graph()
train_graph = tf.Graph()
with train_graph.as_default():
    # get input vectors
    uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, lr, dropout_keep_prob = get_inputs()
    # get the users embedding vectors
    uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer = get_user_embedding(uid, user_gender, user_age, user_job)
    # get the user features
    user_combine_layer, user_combine_layer_flat = get_user_feature_layer(uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer)
    # get the movies embedding vectors
    movie_id_embed_layer = get_movie_id_embed_layer(movie_id)
    # get the movie categories embedding vectors
    movie_categories_embed_layer = get_movie_categories_layers(movie_categories)
    # get the movie title features
    pool_layer_flat, dropout_layer = get_movie_cnn_layer(movie_titles)
    # get the movie features
    movie_combine_layer, movie_combine_layer_flat = get_movie_feature_layer(movie_id_embed_layer, movie_categories_embed_layer, dropout_layer)
    # get the ratings
    with tf.name_scope("inference"):
        # matrix multiply to get the ratings
        inference = tf.reduce_sum(user_combine_layer_flat * movie_combine_layer_flat, axis=1)
        inference = tf.expand_dims(inference, axis=1)

    with tf.name_scope("loss"):
        # MSE Loss
        cost = tf.losses.mean_squared_error(targets, inference)
        loss = tf.reduce_mean(cost)
    global_step = tf.Variable(0, name="global_step", trainable=False)
    lr_decay = tf.train.exponential_decay(lr, global_step, 10000, 0.96, staircase=True)
    optimizer = tf.train.AdamOptimizer(lr_decay)
    gradients = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(gradients, global_step=global_step)


losses = {'train': [], 'test': []}

with tf.Session(graph=train_graph) as sess:

    # keep track of gradient values and sparsity
    grad_summaries = []
    for g, v in gradients:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name.replace(':', '_')), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name.replace(':', '_')), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)

    # output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    # summaries for loss and accuracy
    loss_summary = tf.summary.scalar("loss", loss)

    # train Summaries
    train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # inference summaries
    inference_summary_op = tf.summary.merge([loss_summary])
    inference_summary_dir = os.path.join(out_dir, "summaries", "inference")
    inference_summary_writer = tf.summary.FileWriter(inference_summary_dir, sess.graph)

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    for epoch_i in range(num_epochs):

        # divide data into training and test
        train_X, test_X, train_y, test_y = train_test_split(features, targets_values, test_size=0.2, random_state=0)

        train_batches = get_batches(train_X, train_y, batch_size)
        test_batches = get_batches(test_X, test_y, batch_size)

        # iterations in the training dataset
        for batch_i in range(len(train_X) // batch_size):
            x, y = next(train_batches)

            categories = np.zeros([batch_size, 18])
            for i in range(batch_size):
                categories[i] = x.take(6, 1)[i]

            titles = np.zeros([batch_size, sentences_size])
            for i in range(batch_size):
                titles[i] = x.take(5, 1)[i]

            feed = {
                uid: np.reshape(x.take(0, 1), [batch_size, 1]),
                user_gender: np.reshape(x.take(2, 1), [batch_size, 1]),
                user_age: np.reshape(x.take(3, 1), [batch_size, 1]),
                user_job: np.reshape(x.take(4, 1), [batch_size, 1]),
                movie_id: np.reshape(x.take(1, 1), [batch_size, 1]),
                movie_categories: categories,
                movie_titles: titles,
                targets: np.reshape(y, [batch_size, 1]),
                dropout_keep_prob: dropout_keep,
                lr: initial_learning_rate}

            step, train_loss, summaries, _ = sess.run([global_step, loss, train_summary_op, train_op], feed)
            losses['train'].append(train_loss)
            train_summary_writer.add_summary(summaries, step)

            if (epoch_i * (len(train_X) // batch_size) + batch_i) % show_every_n_batches == 0:
                time_str = datetime.datetime.now().isoformat()
                print('{}: Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(time_str, epoch_i, batch_i, (len(train_X) // batch_size), train_loss))

        # iterations in
        for batch_i in range(len(test_X) // batch_size):
            x, y = next(test_batches)

            categories = np.zeros([batch_size, 18])
            for i in range(batch_size):
                categories[i] = x.take(6, 1)[i]

            titles = np.zeros([batch_size, sentences_size])
            for i in range(batch_size):
                titles[i] = x.take(5, 1)[i]

            feed = {
                uid: np.reshape(x.take(0, 1), [batch_size, 1]),
                user_gender: np.reshape(x.take(2, 1), [batch_size, 1]),
                user_age: np.reshape(x.take(3, 1), [batch_size, 1]),
                user_job: np.reshape(x.take(4, 1), [batch_size, 1]),
                movie_id: np.reshape(x.take(1, 1), [batch_size, 1]),
                movie_categories: categories,
                movie_titles: titles,
                targets: np.reshape(y, [batch_size, 1]),
                dropout_keep_prob: 1,
                lr: initial_learning_rate}

            step, test_loss, summaries = sess.run([global_step, loss, inference_summary_op], feed)
            losses['test'].append(test_loss)
            inference_summary_writer.add_summary(summaries, step)

            if (epoch_i * (len(test_X) // batch_size) + batch_i) % show_every_n_batches == 0:
                time_str = datetime.datetime.now().isoformat()
                print('{}: Epoch {:>3} Batch {:>4}/{}   test_loss = {:.3f}'.format(time_str, epoch_i, batch_i, (len(test_X) // batch_size), test_loss))

    # Save Model
    saver.save(sess, save_dir)
    print('Model Trained and Saved')


pickle.dump((save_dir), open('params.p', 'wb'))

load_dir = pickle.load(open('params.p', mode='rb'))

plt.subplot(2, 1, 1)
plt.plot(losses['train'], label='Training loss')
plt.title('Training and Test Loss')

plt.subplot(2, 1, 2)
plt.plot(losses['test'], label='Test loss')
plt.xlabel('Iterations')

plt.savefig('Loss.png')
plt.show()
