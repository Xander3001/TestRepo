'''
Extract data from a given subreddit, clean and preprocess text data, and create a cleaned dataframe.

Args:
    ml_subreddit (object): A subreddit object representing a particular subreddit.
    limit (int): The maximum amount of posts to extract from the subreddit.

Returns:
    A cleaned dataframe with columns for the post title, score, id, subreddit, post url, number of comments, body, and creation date.

'''

import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def extract_data(ml_subreddit, limit=10000):
    posts = []
    for post in ml_subreddit.hot(limit=limit):
        posts.append([post.title, post.score, post.id, post.subreddit, post.url, post.num_comments, post.selftext, post.created])
    posts = pd.DataFrame(posts,columns=['title', 'score', 'id', 'subreddit', 'url', 'num_comments', 'body', 'created'])

    # Remove empty values and drop the respective rows
    posts.replace('', np.nan, inplace=True)
    posts.dropna(inplace=True)
    
    return posts

def clean_text(df, text_field, new_text_field_name):
    '''
    Clean the text in the given dataframe column.

    Args:
        df (pandas DataFrame): The input dataframe.
        text_field (str): The column of the dataframe containing the text data to be cleaned.
        new_text_field_name (str): The new name for the cleaned text column.

    Returns:
       A new dataframe with a cleaned text column. 
    '''
    stop = stopwords.words('english')
    
    df[new_text_field_name] = df[text_field].str.lower()
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"\d+", "", elem))
    df[new_text_field_name] = df[new_text_field_name].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

    return df

def preprocess_text(posts):
    '''
    Preprocess text data in the dataframe.

    Args:
      posts (pandas DataFrame): The input dataframe.

    Returns:
      A new dataframe with the text data preprocessed.
    '''
    # Clean and preprocess the title and body columns
    posts = clean_text(posts, 'title', 'clean_title')
    posts = clean_text(posts, 'body', 'clean_body')
    
    # Tokenize the cleaned text data and create a new column
    posts['text_tokens'] = posts['clean_body'].apply(lambda x: word_tokenize(x))
    
    # Drop unnecessary columns
    data_clean = posts.drop(['title', 'body', 'subreddit', 'url', 'clean_title', 'clean_body'], axis=1) 
    
    # Remove any remaining empty values and drop the respective rows
    data_clean.dropna(inplace=True)
    
    return data_clean

# Demo code to extract data from a subreddit and preprocess the text data
if __name__ == '__main__':
    ml_subreddit = 'MachineLearning'
    subreddit = reddit.subreddit(ml_subreddit)
    posts = extract_data(subreddit)
    data_clean = preprocess_text(posts)
    print(data_clean.head())