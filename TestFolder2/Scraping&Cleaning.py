# This code collects the top 10,000 posts from a specific subreddit and stores them in a Pandas dataframe. 
# It then cleans the text data by removing special characters, URLs, and numbers, and converts the text to lowercase. 
# Finally, it tokenizes the text and stores it in a new column of the dataframe.

import pandas as pd
import numpy as np
import praw
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Initializing the Reddit api
reddit = praw.Reddit(client_id='your_client_id', client_secret='your_client_secret', username='your_username', password='your_password', user_agent='your_user_agent')
ml_subreddit = reddit.subreddit('MachineLearning')

# Collecting the posts
posts = []
for post in ml_subreddit.hot(limit=10000):
    posts.append([post.title, post.score, post.id, post.subreddit, post.url, post.num_comments, post.selftext, post.created])
posts = pd.DataFrame(posts,columns=['title', 'score', 'id', 'subreddit', 'url', 'num_comments', 'body', 'created'])

# Removing empty values and NaN values
posts.replace('', np.nan, inplace=True)
posts.dropna(inplace=True)

# Function to clean text data
def clean_text(df, text_field, new_text_field_name):
    # Convert text to lowercase
    df[new_text_field_name] = df[text_field].str.lower()
    # Remove special characters and URLs
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))
    # Remove numbers
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"\d+", "", elem))
    
    return df

# Applying the clean_text function to the title column
data_cleans = clean_text(posts, 'title', 'clean_title')

# Removing stop words from the text data
stop = stopwords.words('english')
data_cleans['clean_text'] = data_cleans['clean_body'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# Tokenizing the text data
data_cleans['text_tokens'] = data_cleans['clean_text'].apply(lambda x: word_tokenize(x))

# Dropping unnecessary columns and NaN values
data_clean = data_cleans.drop(['title', "body","text_tokens", "subreddit", "url", "clean_title", "clean_body"], axis=1) 
data_clean.dropna(inplace=True)