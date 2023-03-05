# This code scrapes the top 10,000 posts from the Machine Learning subreddit, then cleans and preprocesses the post data.

import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

ml_subreddit = reddit.subreddit('machinelearning')
posts = []

# Collect post data from subreddit
for post in ml_subreddit.hot(limit=10000):
    posts.append([post.title, post.score, post.id, post.subreddit, post.url, post.num_comments, post.selftext, post.created])

# Convert post data into a pandas dataframe
posts = pd.DataFrame(posts,columns=['title', 'score', 'id', 'subreddit', 'url', 'num_comments', 'body', 'created'])

# Remove empty posts and posts with missing data
posts.replace('', np.nan, inplace=True)
posts.dropna(inplace=True)

# Clean and preprocess the post text data
stop = set(stopwords.words('english'))
def clean_text(df, text_field, new_text_field_name):
    """
    Function that takes a dataframe, a text field from that dataframe, and a new text field name and cleans the text data
    by removing unwanted characters, lowercasing, and removing stop words.
    """
    df[new_text_field_name] = df[text_field].str.lower()
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"\d+", "", elem))
    df[new_text_field_name] = df[new_text_field_name].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    
    return df

data_cleans = clean_text(posts, 'title', 'clean_title')

# Tokenize the cleaned post text
data_cleans['text_tokens'] = data_cleans['clean_title'].apply(lambda x: word_tokenize(x))

# Remove unnecessary columns and empty rows
data_clean = data_cleans.drop(['title', "body","text_tokens", "subreddit", "url", "clean_title", "clean_body"], axis=1)
data_clean.dropna(inplace=True)