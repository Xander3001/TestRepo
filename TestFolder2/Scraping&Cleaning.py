'''
This script collects posts from a specified subreddit, cleans the text, and creates a pandas DataFrame of the resulting data.

Input: 
- ml_subreddit: a subreddit object
- limit: the maximum number of posts to collect, as an int

Output:
- data_clean: a pandas DataFrame containing the cleaned text data, with columns for post score, id, number of comments, and date created

'''

import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def collect_data(ml_subreddit, limit):
    '''
    Collects posts from a specified subreddit
    '''
    posts = []
    for post in ml_subreddit.hot(limit=limit):
        posts.append([post.title, post.score, post.id, post.subreddit, post.url, post.num_comments, post.selftext, post.created])
    posts = pd.DataFrame(posts,columns=['title', 'score', 'id', 'subreddit', 'url', 'num_comments', 'body', 'created'])
    
    return posts

def clean_text(df, text_field, new_text_field_name):
    '''
    Cleans text data in a pandas DataFrame.
    '''
    stop = set(stopwords.words('english'))
    df[new_text_field_name] = df[text_field].str.lower()
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))  
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"\d+", "", elem))
    df['clean_text'] = df[new_text_field_name].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    df['text_tokens'] = df['clean_text'].apply(lambda x: word_tokenize(x))
    
    return df

def main():
    ml_subreddit = reddit.subreddit('machinelearning')
    limit = 10000
    posts = collect_data(ml_subreddit, limit)
    posts.replace('', np.nan, inplace=True)
    posts.dropna(inplace=True)

    data_clean = clean_text(posts, 'title', 'clean_title')    
    data_clean.drop(['title', "body","text_tokens", "subreddit", "url", "clean_title", "clean_body"], axis=1, inplace=True)
    data_clean.dropna(inplace=True)
    
    return data_clean

if __name__ == "__main__":
    main()