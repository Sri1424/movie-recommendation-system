import numpy as np
import pandas as pd
movies = pd.read_csv('/content/tmdb_5000_movies.csv')
from google.colab import drive
drive.mount('/content/drive')
credits = pd.read_csv('/content/tmdb_5000_credits.csv', engine='python')
credits.head()['cast'].values
movies.merge(credits,on='title').shape
movies.shape
credits.shape
movies=movies.merge(credits,on='title')
movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]
movies.isnull().sum()
movies.dropna(inplace=True)
movies.duplicated().sum()
movies.iloc[0].genres
def convert(obj):
    L=[]
    for i in obj:
      L.append(  i['name'])
    return L
import ast
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')
import ast
def convert(obj):

    L=[]
    for i in ast.literal_eval(obj):
      L.append(  i['name'])
    return L
movies['genres']=movies['genres'].apply(convert)
movies['keywords']= movies['keywords'].apply(convert)
import ast
def convert3(obj):
    L = []
    counter = 0

    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break

    return L
movies['cast']=movies['cast'].apply(convert3)
def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return Lmovies['crew']=movies['crew'].apply(fetch_director)movies['crew']=movies['crew'].apply(fetch_director
movies.head()
movies['overview']=movies['overview'].apply(lambda x:x.split())
movies.head()
movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])
movies.head()
movies['tags'] = movies['overview'].apply(lambda x: str(x).split()) + \
                 movies['genres'].apply(lambda x: x if isinstance(x, list) else []) + \
                 movies['keywords'].apply(lambda x: x if isinstance(x, list) else []) + \
                 movies['cast'].apply(lambda x: x if isinstance(x, list) else []) + \
                 movies['crew'].apply(lambda x: x if isinstance(x, list) else [])
movies.head()
new_df = movies[['movie_id','title','tags']]
new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))
#new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x) if isinstance(x, list) else x)
new_df.head()
new_df['tags']=new_df['tags'].apply(lambda x:x.lower())
new_df.head()
new_df['tags'][1]
 pip install nltk
import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)
ps.stem('loved')
new_df['tags']=new_df['tags'].apply(stem)
from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer(max_features=5000, stop_words='english')
vectors =cv.fit_transform(new_df['tags']).toarray()
vectors
cv.get_feature_names_out()
from sklearn.metrics.pairwise import cosine_similarity
similarity=cosine_similarity(vectors)
similarity.shape
similarity
def recommend(movie):
    movie_index = new_df[new_df['title']== movie ].index[0]
    distances = similarity[movie_index ]
    movies_list = sorted(list(enumerate(distances)), reverse = True, key= lambda x:x[1])[1:6]
    for i in movies_list:
           print (new_df.iloc[i[0]].title)
recommend('Titanic')

Please ensure all the preceding cells, including the one that creates the 'tags' column, have been successfully executed before running the following cell.










































