
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



The code above retrieves the 10,000 latest posts from a specific subreddit and stores them in a Pandas DataFrame. The DataFrame includes columns for the post's title, score, ID, subreddit, URL, number of comments, body text, and creation date.

The "clean_text" function takes in the DataFrame, text field to be cleaned, and name for the new cleaned field. The function converts the text to lowercase, removes Twitter handles and URLs, and removes any non-alphanumeric characters. It also removes numbers from the text. The function then adds the cleaned text to a new field in the DataFrame and returns the updated DataFrame.

After cleaning the text, the script creates a new column for the cleaned body text, removes stopwords (common words like "the" and "and"), and tokenizes the remaining words in each post. Finally, unnecessary columns are dropped, and any remaining missing values are removed from the DataFrame.