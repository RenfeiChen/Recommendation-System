import pandas as pd
import pickle
import re

'''
# users data, format : UserID::Gender::Age::Occupation::Zip-code
# we only use UserID, Gender, Age and Occupatin in the users

# movies data, format : MovieID::Title::Genres
# MoiveID is category, Title is text and Genres is category

# ratings data, format: UserID::MovieID::Rating::Timestamp
# we don't use the timestamp and the target of the model is the ratings

How to Handle the input data:
1. Do not modify the UserID, Occupation and MovieID
2. In the Gender part, change "F" to 0 and "M" to 1
3. In the Age part, there are 7 different kinds so we turn them into 0 to 6
4. In the Genres part, first we build a dicitionary from string to number then we turn the Genres to number, and we fix the length of number to 18
   as there are 18 types of movies
5. In the Title part, we do same things as Genres, and we remove the year of the title, and we fix the length of number to 15
6. For the Genres and Title, in order to handle it more efficient in the nn, we let them have same length as we fill them with <'PAD'>
'''

# Users
# remove the Zip-code part
users_title = ['UserID', 'Gender', 'Age', 'JobID', 'Zip-code']
users = pd.read_table('./ml-1m/users.dat', sep='::', header=None, names=users_title, engine = 'python')
users = users.filter(regex='UserID|Gender|Age|JobID')
# the original users data
users_orig = users.values
# change the Gender part
gender_map = {'F': 0, 'M': 1}
users['Gender'] = users['Gender'].map(gender_map)
# change the Agr part
age_map = {val: i for i, val in enumerate(set(users['Age']))}
users['Age'] = users['Age'].map(age_map)


# Movies
movies_title = ['MovieID', 'Title', 'Genres']
movies = pd.read_table('./ml-1m/movies.dat', sep='::', header=None, names=movies_title, engine = 'python')
# the original movies data
movies_orig = movies.values
# remove the year from the title
pattern = re.compile(r'^(.*)\((\d+)\)$')
title_map = {val: pattern.match(val).group(1) for i, val in enumerate(set(movies['Title']))}
movies['Title'] = movies['Title'].map(title_map)

# change the genres dictionary string into numeric
genres_set = set()
for val in movies['Genres'].str.split('|'):
    genres_set.update(val)

genres_set.add('<PAD>')
genres2int = {val: i for i, val in enumerate(genres_set)}

# change the genres to integer with the length of 18
genres_count = 18
genres_map = {val: [genres2int[row] for row in val.split('|')] for i, val in enumerate(set(movies['Genres']))}

for key in genres_map:
    for cnt in range(genres_count - len(genres_map[key])):
        # fill up the map to 18 with the number accroding to <'PAD'>
        genres_map[key].insert(len(genres_map[key]) + cnt, genres2int['<PAD>'])

movies['Genres'] = movies['Genres'].map(genres_map)

# change the title dictionary string into numeric
title_set = set()
for val in movies['Title'].str.split():
    title_set.update(val)

title_set.add('<PAD>')
title2int = {val: i for i, val in enumerate(title_set)}

# change the title to integer with the length of 15
title_count = 15
title_map = {val: [title2int[row] for row in val.split()] for i, val in enumerate(set(movies['Title']))}

for key in title_map:
    for cnt in range(title_count - len(title_map[key])):
        # fill up the map to 15 with the number accroding to <'PAD'>
        title_map[key].insert(len(title_map[key]) + cnt, title2int['<PAD>'])

movies['Title'] = movies['Title'].map(title_map)

# ratings
ratings_title = ['UserID','MovieID', 'ratings', 'timestamps']
ratings = pd.read_table('./ml-1m/ratings.dat', sep='::', header=None, names=ratings_title, engine = 'python')
# remove the timestamp part
ratings = ratings.filter(regex='UserID|MovieID|ratings')

# merge users, movies and ratings to a dataset
data = pd.merge(pd.merge(ratings, users), movies)

# divide the data into X and Y
target_fields = ['ratings']
features_pd, targets_pd = data.drop(target_fields, axis=1), data[target_fields]

features = features_pd.values
targets_values = targets_pd.values

# save the data to local
pickle.dump((title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig), open('preprocess.p', 'wb'))
