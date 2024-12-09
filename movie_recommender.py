import streamlit as st
import pandas as pd
import numpy as np

#################
# myIBCF Function
#################

def myIBCF(w: pd.Series, S: pd.DataFrame, top100_ranking: pd.Series, top_N: int = 10) -> list[str]:
    # If w values are not [1,5], set them to NA
    w = w.where((w >= 1) & (w <= 5))

    # Create a copy of w to store the predictions
    w_prime = w.copy()

    # Calculate ratings for empty entries in `w`
    for i in w.index[w.isna()]:
        # Calculate the rating-weighted sum of similarities
        sw_sum = S.loc[i].mul(w, axis=0).sum(skipna=True)
        # Calculate the sum of similarities of rated movies
        s_sum = S.loc[i][w.notna()].sum(skipna=True)
        # If sum of similarities is zero, skip to avoid dividing by zero
        if s_sum == 0:
            continue
        # Calulate the predicted rating for the item
        w_prime[i] = sw_sum / s_sum

    # Sort predictions by descending order and keep the top N
    # Note that these should only be movies that the user has not rated
    pred = w_prime[w.isna()].dropna().sort_values(ascending=False).index[:top_N].to_list()

    # If there are fewer than top_N predictions, use top100_ranking to fill it up
    if len(pred) < top_N:
        # Exclude movies from top100_ranking which the user has ranked or are already in pred
        user_ranked = set(w.dropna().index + pred)
        top100_filtered = [movie for movie in top100_ranking if movie not in user_ranked]
        pred += top100_filtered[:top_N - len(pred)]

    return pred

###############################
# Read the Parquet file for S100
s100_url = "https://github.com/darinz/RecSys/raw/refs/heads/main/data/S100.parquet"
S = pd.read_parquet(s100_url)

# Load movie data from CSV file
top100_url = "https://github.com/darinz/RecSys/raw/refs/heads/main/data/top100_movies.csv"
top100_rated = pd.read_csv(top100_url)
top100_movie_id = top100_rated['MovieID'].astype(str)

# Select 15 movies
# movie_data = top100_rated.sample(n=15)
# Select the first 30 movies
movie_data = top100_rated.head(15)

# Ratings
ratings = {}

# App layout
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("Movie Recommender")

###########################################
# Step 1: Rate as many movies as possible
st.subheader("Step 1: Rate as many movies as possible")

# Splitting data into first two rows and the rest
first_two_rows = movie_data.iloc[:10]  # Assuming 5 columns, first 2 rows = 10 items
remaining_rows = movie_data.iloc[10:]

# Collapsible and scrollable section
with st.expander("Rate movies (click to expand/collapse)", expanded=True):
    # Display the first two rows without scrolling
    st.write("### Movies to Rate")
    cols_top = st.columns(5)
    for i, (_, row) in enumerate(first_two_rows.iterrows()):
        with cols_top[i % 5]:
            st.image(row["image_url"], width=150)
            # Get rating for the movie using radio buttons (1 to 5 stars)
            rating = st.radio(
                row["Title"],
                options=[None, 1, 2, 3, 4, 5],  # Add 'None' option to allow for no selection
                format_func=lambda x: '★' * x if x is not None else "Select a rating",  # Show 'Select a rating' when None
                index=0,  # Set default to 'None' which is at index 0
                key=row["MovieID"]  # Ensure unique keys for each movie
            )
            ratings[row["MovieID"]] = rating if rating is not None else 0

    # Allow scrolling for the rest of the movies
    st.write("### Scroll to Rate More Movies")
    scrollable_section = st.container()
    with scrollable_section:
        cols_scrollable = st.columns(5)
        for i, (_, row) in enumerate(remaining_rows.iterrows()):
            with cols_scrollable[i % 5]:
                st.image(row["image_url"], width=150)
                # Get rating for the movie using radio buttons (1 to 5 stars)
                rating = st.radio(
                    row["Title"],
                    options=[None, 1, 2, 3, 4, 5],  # Add 'None' option to allow for no selection
                    format_func=lambda x: '★' * x if x is not None else "Select a rating",  # Show 'Select a rating' when None
                    index=0,  # Set default to 'None' which is at index 0
                    key=row["MovieID"]  # Ensure unique keys for each movie
                )
                ratings[row["MovieID"]] = rating if rating is not None else 0

########################################
# Step 2: Discover movies you might like
st.subheader("Step 2: Discover movies you might like")

if st.button("Click here to get your recommendations"):
    newuser = pd.Series(index=S.index, data=np.nan)
    
    for movie_id, rating in ratings.items():
        if rating > 0:
            newuser[movie_id] = rating

    recommendations = myIBCF(newuser, S, top100_movie_id)
    st.write("Your movie recommendations:")

    cols = st.columns(5)

    for i in range(5):
        rank = recommendations[i]
        movie_info = top100_rated.loc[top100_rated["MovieID"] == rank]
        with cols[i]:
            st.image(movie_info["image_url"].values[0], width=150, caption=f"Rank {i + 1}")
            st.write(f"{movie_info['Title'].values[0]}")
    
    for i in range(5):
        rank = recommendations[i+5]
        movie_info = top100_rated.loc[top100_rated["MovieID"] == rank]
        with cols[i]:
            st.image(movie_info["image_url"].values[0], width=150, caption=f"Rank {i + 6}")
            st.write(f"{movie_info['Title'].values[0]}")