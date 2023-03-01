
for post in ml_subreddit.hot(limit=10000):
    posts.append([post.title, post.score, post.id, post.subreddit, post.url, post.num_comments, post.selftext, post.created])
posts = pd.DataFrame(posts,columns=['title', 'score', 'id', 'subreddit', 'url', 'num_comments', 'body', 'created'])


posts.replace('', np.nan, inplace=True)
posts.dropna(inplace=True)


def  clean_text(df, text_field, new_text_field_name):
    df[new_text_field_name] = df[text_field].str.lower()
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))  
    # remove numbers
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"\d+", "", elem))
    
    return df
data_cleans = clean_text(posts, 'title', 'clean_title')

data_cleans.head()


data_cleans['clean_text'] = data_cleans['clean_body'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

data_cleans['text_tokens'] = data_cleans['clean_text'].apply(lambda x: word_tokenize(x))


data_clean = data_cleans.drop(['title', "body","text_tokens", "subreddit", "url", "clean_title", "clean_body"], axis=1) 
data_clean.dropna(inplace=True)



# This code collects 10,000 hot posts from a specific subreddit using the PRAW library and saves it as a pandas DataFrame
# It then drops any rows that contain empty fields

import praw
import pandas as pd
import numpy as np

reddit = praw.Reddit(client_id='your_client_id',
                     client_secret='your_client_secret',
                     user_agent='your_user_agent',
                     username='your_username',
                     password='your_password')

ml_subreddit = reddit.subreddit('machinelearning')

posts = []

for post in ml_subreddit.hot(limit=10000):
    posts.append([post.title, post.score, post.id, post.subreddit, post.url, post.num_comments, post.selftext, post.created])

posts = pd.DataFrame(posts,columns=['title', 'score', 'id', 'subreddit', 'url', 'num_comments', 'body', 'created'])

posts.replace('', np.nan, inplace=True)
posts.dropna(inplace=True)

# This function takes a DataFrame, a specific text field, and a new field name for the cleaned text,
# and returns the original DataFrame with the new clean text field added

import regex as re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stop = set(stopwords.words('english'))

def clean_text(df, text_field, new_text_field_name):
    df[new_text_field_name] = df[text_field].str.lower()
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))  
    # remove numbers
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"\d+", "", elem))
    
    return df

# This code applies the clean_text function from above to the 'title' field of the DataFrame and creates a new field called 'clean_title'
# It then applies the NLTK tokenization technique to tokenize the cleaned text into individual words

data_cleans = clean_text(posts, 'title', 'clean_title')

data_cleans['text_tokens'] = data_cleans['clean_title'].apply(lambda x: word_tokenize(x))

# This code removes the 'title', 'body', 'subreddit', 'url' fields from the DataFrame,
# and any rows that contain empty fields for the remaining columns
# It also creates a new field called 'clean_text', which removes stopwords from the 'clean_title' field

data_clean = data_cleans.drop(['title', "body","text_tokens", "subreddit", "url", "clean_title"], axis=1) 
data_clean.dropna(inplace=True)

data_clean['clean_text'] = data_clean['clean_title'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# The final DataFrame has the fields 'score', 'id', 'num_comments', 'created', and 'clean_text' for each post in the subreddit.