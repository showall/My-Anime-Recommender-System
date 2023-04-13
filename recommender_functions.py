import numpy as np
import pandas as pd
import pickle

def get_movie_names(movie_ids, movies_df):
    '''
    INPUT
    movie_ids - a list of movie_ids
    movies_df - original movies dataframe
    OUTPUT
    movies - a list of movie names associated with the movie_ids

    '''
    # Read in the datasets
    movie_lst = list(movies_df[movies_df['movie_id'].isin(movie_ids)]['movie'])

    return movie_lst


def create_ranked_df(movies, reviews):
        '''
        INPUT
        movies - the movies dataframe
        reviews - the reviews dataframe

        OUTPUT
        ranked_movies - a dataframe with movies that are sorted by highest avg rating, more reviews, then time, and must have more than 4 ratings
        '''

        # Pull the average ratings and number of ratings for each movie
        movie_ratings = reviews.groupby('movie_id')['rating']
        avg_ratings = movie_ratings.mean()
        num_ratings = movie_ratings.count()
        last_rating = pd.DataFrame(reviews.groupby('movie_id').max()['date'])
        last_rating.columns = ['last_rating']

        # Add Dates
        rating_count_df = pd.DataFrame({'avg_rating': avg_ratings, 'num_ratings': num_ratings})
        rating_count_df = rating_count_df.join(last_rating)

        # merge with the movies dataset
        movie_recs = movies.set_index('movie_id').join(rating_count_df)

        # sort by top avg rating and number of ratings
        ranked_movies = movie_recs.sort_values(['avg_rating', 'num_ratings', 'last_rating'], ascending=False)

        # for edge cases - subset the movie list to those with only 5 or more reviews
        ranked_movies = ranked_movies[ranked_movies['num_ratings'] > 4]

        return ranked_movies


def find_similar_movies(movie_id, df_content,n=10):
    '''
    INPUT
    movie_id - a movie_id
    movies_df - original movies dataframe
    OUTPUT
    similar_movies - an array of the most similar movies by title
    '''
    df_content_to_select = df_content.loc[:,["anime_id","genre","type"]]
    df_content_to_select["genre"] = df_content_to_select.loc[:,"genre"].fillna("Unknown").copy()
    df_content_to_select["genre1"] = df_content_to_select.loc[:,"genre"].str.split(',').copy()
    df_content_to_select["genre"] = df_content_to_select.loc[:,"genre1"].copy()
    df_content_to_select = df_content_to_select.drop("genre1", axis=1)
    df_model = pd.DataFrame(df_content_to_select["genre"].tolist())
    df_content_to_select = pd.concat([df_content_to_select, df_model], axis=1)
    df_content_to_select = df_content_to_select.drop("genre", axis=1).melt(id_vars=["anime_id","type"]).dropna(subset=["value"])
    df_content_to_select["variable"] = 1
    df_content_to_select = df_content_to_select.drop_duplicates()
    df_content_to_select = df_content_to_select.pivot(index = ["anime_id","type"], columns=["value"]).fillna(0)
    df_content_to_select.columns = [' '.join(col).strip().replace("variable ","").replace("-","_") for col in df_content_to_select.columns.values]
    df_content_to_select.reset_index(inplace=True)
    df_content_to_select = pd.get_dummies(df_content_to_select,["type"])
    movie_content = np.array(df_content_to_select.drop("anime_id", axis=1))
   # print("checkpoint4")   
    #dot_prod_movies = movie_content.dot(np.transpose(movie_content))
    # dot product to get similar movies

    with open("df_content_matrix.pkl","rb") as p :
        dot_prod_movies = pickle.load(p)

    similar_idxs = []
    movie_idx = np.where(df_content['anime_id'] == movie_id)[0][0]
    largest= np.partition(dot_prod_movies[movie_idx].flatten(), -20)[-20]
    new_movie_idx = np.where(dot_prod_movies[movie_idx] >= largest)[0]
    similar_idxs =new_movie_idx[:n] 
    # pull the movie titles based on the indices

    similar_movies = np.array(df_content.iloc[similar_idxs, ]['name'])
    similar_movies = list(similar_movies)
    return similar_movies


def popular_recommendations(user_id, n_top, ranked_movies):
    '''
    INPUT:
    user_id - the user_id (str) of the individual you are making recommendations for
    n_top - an integer of the number recommendations you want back
    ranked_movies - a pandas dataframe of the already ranked movies based on avg rating, count, and time

    OUTPUT:
    top_movies - a list of the n_top recommended movies by movie title in order best to worst
    '''

    top_movies = list(ranked_movies['movie'][:n_top])

    return top_movies


def get_top_anime(n, df_content):
    '''
    INPUT:
    n - (int) the number of top articles to return
    df - (pandas dataframe) df as defined at the top of the notebook 
    
    OUTPUT:
    top_articles - (list) A list of the top 'n' article titles 
    
    '''
    top_anime= list(df_content.sort_values(["rating"],ascending=False,axis=0)["name"].head(n))
    top_anime_id = list(df_content.sort_values(["rating"],ascending=False,axis=0)["anime_id"].head(n))
    return top_anime, top_anime_id # Return the top article titles from df (not df_content)


def find_similar_users(user_id, user_item):
    '''
    INPUT:
    user_id - (int) a user_id
    user_item - (pandas dataframe) matrix of users by articles: 
                1's when a user has interacted with an article, 0 otherwise
    
    OUTPUT:
    similar_users - (list) an ordered list where the closest users (largest dot product users)
                    are listed first
    
    Description:
    Computes the similarity of every pair of users based on the dot product
    Returns an ordered
    
    '''
    # compute similarity of each user to the provided user
    similiarity = pd.DataFrame(np.dot(user_item[user_item.index == user_id],user_item.T).T, columns=["similiarity_score"])
    user_id_list = dict(zip(pd.DataFrame(user_item.index).index,pd.DataFrame(user_item.index)["user_id"]))
    similiarity.index = similiarity.index.map(user_id_list)    
    similiarity = similiarity.reset_index().sort_values(["similiarity_score","index"],ascending=[False,True])
    # sort by similarity
    # create list of just the ids
    most_similar_users = list(similiarity["index"])
    # remove the own user's id
    most_similar_users_exclude_own = [x for x in most_similar_users if x != user_id]  
    return most_similar_users_exclude_own

def get_anime_names(anime_ids, df_content):
    '''
    INPUT:
    article_ids - (list) a list of article ids
    df - (pandas dataframe) df as defined at the top of the notebook
    
    OUTPUT:
    article_names - (list) a list of article names associated with the list of article ids 
                    (this is identified by the title column)
    '''
    # Your code here
    anime_names = []
    for id in anime_ids :
        
     #   article_names.append( df[df.article_id==id]["title"].unique()[0])    
      #  print(id)
        anime_names.append(df_content[df_content.anime_id.astype("float")==id]["name"].values[0])
    
    return anime_names # Return the article names associated with list of article ids


def get_user_animes(user_id, user_item,df_content):
    '''
    INPUT:
    user_id - (int) a user id
    user_item - (pandas dataframe) matrix of users by articles: 
                1's when a user has interacted with an article, 0 otherwise
    
    OUTPUT:
    article_ids - (list) a list of the article ids seen by the user
    article_names - (list) a list of article names associated with the list of article ids 
                    (this is identified by the doc_full_name column in df_content)
    
    Description:
    Provides a list of the article_ids and article titles that have been seen by a user
    '''
    mask = user_item[user_item.index == user_id]
    mask = list(mask.iloc[0])
    anime_id = list(user_item.rating.reset_index().columns[1:])
    id_freq_df = pd.DataFrame([anime_id, mask]).T
    anime_ids = list(id_freq_df[id_freq_df[1]==1][0])
   # article_ids = [str(x) for x in article_ids]
    anime_names = get_anime_names(anime_ids, df_content)
    
    return anime_ids, anime_names # return the ids and names # return a list of the users in order from most to least similar