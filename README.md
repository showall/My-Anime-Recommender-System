# Anime Recommendation Engine
This is a recommender system for anime based on user ratings and anime attributes using plotly DASH. The system uses a combination of collaborative filtering with matrix factorization (FunkSVD) and content-based filtering with cosine similarity to make recommendations for anime. 

Dependencies
pandas
numpy
scikit-learn
Installation

# Clone the repository
Install the dependencies with 
`pip install -r requirements.txt`
Run the app with 
`python app.py`

# Usage

#To use the recommender system, follow these steps:

Select your user type.

The system will recommend a list of anime based on your previous ratings and similar users' ratings.

# Files
app.py: main script for running the app
recommender.py: recommender class for making predictions and recommendations
recommender_functions.py: helper functions for the recommender class
anime.csv: dataset of anime with attributes like genre and rating
ratings.csv: dataset of user ratings for anime

# Acknowledgements
This project is based on a template from Udacity's Data Science Nanodegree program. The anime dataset is sourced from MyAnimeList.net.

License
This project is licensed under the MIT License. See LICENSE for more information.
