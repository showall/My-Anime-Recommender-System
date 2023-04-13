import numpy as np
import pandas as pd
import recommender_functions as rf
import sys # can use sys to take command line arguments
import pickle

class Recommender():
    '''
    This Recommender uses FunkSVD to make predictions of exact ratings.  And uses either FunkSVD or a Knowledge Based recommendation (highest ranked) to make recommendations for users.  Finally, if given a movie, the recommender will provide movies that are most similar as a Content Based Recommender.
    '''
    def __init__(self):
        '''
        I didn't have any required attributes needed when creating my class.
        '''


    def fit(self, rating_pth, content_pth, latent_features=3, learning_rate=0.0001, iters=100):
        '''
        This function performs matrix factorization using a basic form of FunkSVD with no regularization

        INPUT:
        reviews_pth - path to csv with at least the four columns: 'user_id', 'movie_id', 'rating', 'timestamp'
        movies_pth - path to csv with each movie and movie information in each row
        latent_features - (int) the number of latent features used
        learning_rate - (float) the learning rate
        iters - (int) the number of iterations

        OUTPUT:
        None - stores the following as attributes:
        n_users - the number of users (int)
        n_movies - the number of movies (int)
        num_ratings - the number of ratings made (int)
        reviews - dataframe with four columns: 'user_id', 'movie_id', 'rating', 'timestamp'
        movies - dataframe of
        user_item_mat - (np array) a user by item numpy array with ratings and nans for values
        latent_features - (int) the number of latent features used
        learning_rate - (float) the learning rate
        iters - (int) the number of iterations
        '''
        # Store inputs as attributes
        self.df = pd.read_csv(rating_pth)
        self.df_content = pd.read_csv(content_pth)
        #self.df = self.df.head(50000)
        #self.df_content = self.df_content

        # Create user-item matrix
        usr_itm = self.df
        self.user_item_df = usr_itm.groupby(['user_id', 'anime_id'])['rating'].max().unstack()
        self.user_item_mat= np.array(self.user_item_df)

        # Store more inputs
        self.latent_features = latent_features
        self.learning_rate = learning_rate
        self.iters = iters

        # Set up useful values to be used through the rest of the function
        self.n_users = self.user_item_mat.shape[0]
        self.n_movies = self.user_item_mat.shape[1]
        self.num_ratings = np.count_nonzero(~np.isnan(self.user_item_mat))
        self.user_ids_series = np.array(self.user_item_df.index)
        self.movie_ids_series = np.array(self.user_item_df.columns)

        # initialize the user and movie matrices with random values
        user_mat = np.random.rand(self.n_users, self.latent_features)
        movie_mat = np.random.rand(self.latent_features, self.n_movies)

        # initialize sse at 0 for first iteration
        sse_accum = 0

        # keep track of iteration and MSE
        print("Optimizaiton Statistics")
        print("Iterations | Mean Squared Error ")

        # for each iteration
        for iteration in range(self.iters):

            # update our sse
            old_sse = sse_accum
            sse_accum = 0

            # For each user-movie pair
            for i in range(self.n_users):
                for j in range(self.n_movies):

                    # if the rating exists
                    if self.user_item_mat[i, j] > 0:

                        # compute the error as the actual minus the dot product of the user and movie latent features
                        diff = self.user_item_mat[i, j] - np.dot(user_mat[i, :], movie_mat[:, j])

                        # Keep track of the sum of squared errors for the matrix
                        sse_accum += diff**2

                        # update the values in each matrix in the direction of the gradient
                        for k in range(self.latent_features):
                            user_mat[i, k] += self.learning_rate * (2*diff*movie_mat[k, j])
                            movie_mat[k, j] += self.learning_rate * (2*diff*user_mat[i, k])

            # print results
            print("%d \t\t %f" % (iteration+1, sse_accum / self.num_ratings))

        # SVD based fit
        # Keep user_mat and movie_mat for safe keeping
        self.user_mat = user_mat
        self.movie_mat = movie_mat

        # Knowledge based fit
  #      self.ranked_movies = rf.create_ranked_df(self.movies, self.reviews)


    def predict_rating(self, user_id, movie_id):
        '''
        INPUT:
        user_id - the user_id from the reviews df
        movie_id - the movie_id according the movies df

        OUTPUT:
        pred - the predicted rating for user_id-movie_id according to FunkSVD
        '''
        try:# User row and Movie Column
            user_row = np.where(self.user_ids_series == user_id)[0][0]
            movie_col = np.where(self.movie_ids_series == movie_id)[0][0]

            # Take dot product of that row and column in U and V to make prediction
            pred = np.dot(self.user_mat[user_row, :], self.movie_mat[:, movie_col])
            pred =  max(min(pred,10),-1)
            movie_name = str(self.df_content[self.df_content['anime_id'] == movie_id]['name'].values[0])
            movie_name = movie_name.replace('\nName: anime, dtype: object', '').strip()
            print("For user {} we predict a {} rating for the movie {}.".format(user_id, round(pred, 2), str(movie_name)))

            return pred

        except:
            print("I'm sorry, but a prediction cannot be made for this user-movie pair.  It looks like one of these items does not exist in our current database.")

            return None

     #  ['New_Viewer', 'Regular_Viewer', 'Premium_Viewer']
    def make_recommendations(self, _id, _id_type='movie', rec_num=12, anime_list=['Fullmetal Alchemist: Brotherhood', 'Steins;Gate', 'Hunter x Hunter (2011)']):
        '''
        INPUT:
        _id - either a user or movie id (int)
        _id_type - "movie" or "user" (str)
        rec_num - number of recommendations to return (int)

        OUTPUT:
        recs - (array) a list or numpy array of recommended movies like the
                       given movie, or recs for a user_id given
        '''
        # if the user is available from the matrix factorization data,
        # I will use this and rank movies based on the predicted values
        # For use with user indexing
        rec_ids, rec_names = None, None
        if _id_type == 'Regular_Viewer':
            if anime_list == None:
                anime_list = ['Fullmetal Alchemist: Brotherhood', 'Steins;Gate', 'Hunter x Hunter (2011)']
            anime_id = list(self.df_content[self.df_content.name.apply(lambda x : x in anime_list)]["anime_id"])
  
            df_to_pivot = self.df[self.df.anime_id.isin(anime_id)]
  
            df_pivot =  df_to_pivot[df_to_pivot.anime_id.isin(anime_id)].pivot(index=["user_id"], columns="anime_id").fillna(0)
            df_pivot.columns = [''.join(str(col[1])).strip().replace("rating_2","").replace("-","_") for col in df_pivot.columns.values]
            df_pivot.reset_index(inplace=True)
            df_pivot_id = df_pivot[["user_id"]]
            df_pivot_id_var = df_pivot.drop(["user_id"], axis=1)
            df_pivot_id.loc[:,"count"] = np.count_nonzero(df_pivot_id_var, axis=1)
            df_final =  df_pivot_id.merge(df_to_pivot.groupby('user_id').agg("mean").reset_index().drop("anime_id", axis=1), on="user_id")
            df_final["score"] = df_final["count"] * df_final["rating"] 
            df_final=  df_final.sort_values("score", ascending = False).reset_index()
            _id = df_final["user_id"][0]

            if _id in self.user_ids_series:
                # Get the index of which row the user is in for use in U matrix
                idx = np.where(self.user_ids_series == _id)[0][0]

                # take the dot product of that row and the V matrix
                preds = np.dot(self.user_mat[idx,:],self.movie_mat)

                # pull the top movies according to the prediction
                indices = preds.argsort()[-rec_num:][::-1] #indices
                rec_ids = self.movie_ids_series[indices]
                rec_names = rf.get_anime_names(rec_ids,self.df_content)

            else:
                rec_names , rec_ids= rf.get_top_anime(rec_num, self.df_content)
                # if we don't have this user, give just top ratings back
                # rec_names = rf.popular_recommendations(_id, rec_num, self.ranked_movies)
        elif _id_type == 'New_Viewer':    
            rec_names , rec_ids= rf.get_top_anime(rec_num, self.df_content)
           # print("Because this user wasn't in our database, we are giving back the top movie recommendations for all users.")

        # Find similar movies if it is a movie that is passed
        else:
            if _id in self.movie_ids_series:
                rec_names = rf.find_similar_movies(_id,self.df_content, rec_num)
                rec_ids = None
            else:
                print("That movie doesn't exist in our database.  Sorry, we don't have any recommendations for you.")

        return rec_names

if __name__ == '__main__':
    import recommender as r

    #instantiate recommender
    rec = r.Recommender()

    # fit recommender
    rec.fit(rating_pth='data/rating2.csv', content_pth= 'data/anime2.csv', learning_rate=.003, iters=10)

    with open("recommender.pkl", "wb") as p:
        pickle.dump(rec, p)
    # predict
    rec.predict_rating(user_id=8, movie_id=2844)

    # make recommendations
    print(rec.make_recommendations(8,'user')) # user in the dataset
    print(rec.make_recommendations(1,'user')) # user not in dataset
    print(rec.make_recommendations(1853728)) # movie in the dataset
    print("hellp")
    print(rec.make_recommendations(1)) # movie not in dataset
    print(rec.n_users)
    print(rec.n_movies)
    print(rec.num_ratings)
