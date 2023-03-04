# The overall code fetches the top 10,000 posts from a subreddit named 'ml_subreddit', cleans the text of the post using regular expressions, and creates a new dataset containing only the cleaned text and score of the posts. The new dataset is then further cleaned by removing stop words and tokenized for text analysis.

# Import necessary libraries
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Access the subreddit and get the top 10,000 posts
ml_subreddit = reddit.subreddit('ml')
posts = []
for post in ml_subreddit.hot(limit=10000):
    posts.append([post.title, post.score, post.id, post.subreddit, post.url, post.num_comments, post.selftext, post.created])
posts = pd.DataFrame(posts,columns=['title', 'score', 'id', 'subreddit', 'url', 'num_comments', 'body', 'created'])

# Replace empty cells with NaN and remove them
posts.replace('', np.nan, inplace=True)
posts.dropna(inplace=True)

# Clean the text of the post using regular expressions
def  clean_text(df, text_field, new_text_field_name):
    df[new_text_field_name] = df[text_field].str.lower()
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))  
    # remove numbers
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"\d+", "", elem))
    return df

data_cleans = clean_text(posts, 'title', 'clean_title')

# Remove stop words and tokenize the remaining text
stop = set(stopwords.words('english'))
data_cleans['clean_text'] = data_cleans['clean_body'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
data_cleans['text_tokens'] = data_cleans['clean_text'].apply(lambda x: word_tokenize(x))

# Create a final dataset with cleaned text and score
data_clean = data_cleans.drop(['title', "body","text_tokens", "subreddit", "url", "clean_title", "clean_body"], axis=1) 
data_clean.dropna(inplace=True)