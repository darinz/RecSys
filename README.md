# 🎬 Movie Recommendation System

This repository contains the code for a **Movie Recommendation System** that is deployed using [Streamlit](https://streamlit.io/). The app provides personalized movie recommendations based on user preferences using a simple and intuitive web interface.

🌐 **Live App**: [Click here to try it out](https://6oujescmadydjnbgygtm9o.streamlit.app)


## About the App

The app recommends movies based on similarity scores computed from movie metadata. It use content-based filtering (e.g., based on genres, keywords, cast, etc.) to suggest similar movies when a user selects a title.

### Features:
- Rate movies to get top recommendations
- Movie posters displayed for visual appeal
- Simple and fast interface powered by Streamlit


## About the Code

This repo includes:

- `movie_recommender.py`: The code that runs the Streamlit web app and logic for computing movie recommendations
- `data/`: Contains preprocessed movie dataset (Similarity Matrix of the top 100 movies)

This code is what powers the **live app**, and can be modified to improve recommendation quality, change UI, or support other datasets.


## Try It Out

No setup required — just visit the app and start exploring movies:

👉 [Movie Recommender App](https://6oujescmadydjnbgygtm9o.streamlit.app)