# This script is used to scrape data from a subreddit related to machine learning and perform text cleaning and preprocessing.

import praw
import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Initializing the Reddit API client using the praw library
reddit = praw.Reddit(client_id='your_client_id',
                     client_secret='your_client_secret',
                     user_agent='your_user_agent',
                     username='your_username',
                     password='your_password')

# Retrieving the top 10000 posts from the machine learning subreddit
ml_subreddit = reddit.subreddit('machinelearning')
posts = []
for post in ml_subreddit.hot(limit=10000):
    posts.append([post.title, post.score, post.id, post.subreddit, post.url, post.num_comments, post.selftext, post.created])

# Creating a pandas DataFrame to store the scraped data
posts = pd.DataFrame(posts,columns=['title', 'score', 'id', 'subreddit', 'url', 'num_comments', 'body', 'created'])

# Removing any empty values in the DataFrame
posts.replace('', np.nan, inplace=True)
posts.dropna(inplace=True)

# Function to clean the text data by removing special characters, URLs, numbers, and converting all letters to lowercase
def clean_text(df, text_field, new_text_field_name):
    df[new_text_field_name] = df[text_field].str.lower()
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))  
    # remove numbers
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"\d+", "", elem))
    
    return df

# Applying the clean_text function to the title field and creating a new field for the cleaned text
data_cleans = clean_text(posts, 'title', 'clean_title')

# Removing stopwords from the cleaned body text using NLTK library
stop = stopwords.words('english')
data_cleans['clean_text'] = data_cleans['clean_body'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# Tokenizing the cleaned text data
data_cleans['text_tokens'] = data_cleans['clean_text'].apply(lambda x: word_tokenize(x))

# Dropping unnecessary columns from the DataFrame
data_clean = data_cleans.drop(['title', "body","text_tokens", "subreddit", "url", "clean_title", "clean_body"], axis=1) 
data_clean.dropna(inplace=True)