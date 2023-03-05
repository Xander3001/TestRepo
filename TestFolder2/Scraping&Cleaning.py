# This code snippet scrapes 10000 hot posts from the machine learning subreddit and performs some data cleaning operations on them such as removing special characters, numbers, and stop words.

import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Collecting posts data from subreddit
posts = []
for post in ml_subreddit.hot(limit=10000):
    posts.append([post.title, post.score, post.id, post.subreddit, post.url, post.num_comments, post.selftext, post.created])
posts = pd.DataFrame(posts,columns=['title', 'score', 'id', 'subreddit', 'url', 'num_comments', 'body', 'created'])

# Removing any empty cells in dataframe
posts.replace('', np.nan, inplace=True)
posts.dropna(inplace=True)

# Function to clean text data
def clean_text(df, text_field, new_text_field_name):
    """
    Clean the text data by removing special characters, stop words, and numbers
    
    :param df: the DataFrame containg the text data to be cleaned
    :param text_field: the name of the column containing the original text data
    :param new_text_field_name: the name of the new column where cleaned text data will be stored
    :return: the DataFrame with cleaned text data
    """
    df[new_text_field_name] = df[text_field].str.lower()
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))  
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"\d+", "", elem))
    stop = set(stopwords.words('english'))
    df[new_text_field_name] = df[new_text_field_name].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    
    return df

# Calling the clean_text function on the DataFrame and storing the cleaned data in a new DataFrame called data_cleans
data_cleans = clean_text(posts, 'title', 'clean_title')

# Creating new columns to store tokenized text and dropping unwanted columns
data_cleans['text_tokens'] = data_cleans['clean_body'].apply(lambda x: word_tokenize(x))
data_clean = data_cleans.drop(['title', "body","text_tokens", "subreddit", "url", "clean_title", "clean_body"], axis=1)

# Removing empty cells from data_clean DataFrame
data_clean.dropna(inplace=True)