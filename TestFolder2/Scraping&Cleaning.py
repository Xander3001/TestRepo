# This code fetches the top 10000 posts from a machine learning subreddit and creates a dataframe using specific fields of each post.

posts = []
for post in ml_subreddit.hot(limit=10000):
    posts.append([post.title, post.score, post.id, post.subreddit, post.url, post.num_comments, post.selftext, post.created])
# The posts list is converted into a Pandas dataframe with columns named title, score, id, subreddit, url, num_comments, body, created.

posts = pd.DataFrame(posts,columns=['title', 'score', 'id', 'subreddit', 'url', 'num_comments', 'body', 'created'])

# This code replaces empty values with NaN and drops any row which has NaN values.

posts.replace('', np.nan, inplace=True)
posts.dropna(inplace=True)

# This function takes a dataframe, a specific column containing text, and a new column name to store the cleaned text.
# The function converts text to lowercase and removes special characters, URLs, username mentions (@user),
# the retweet symbol (RT), and any non-alphanumeric characters.
# It also removes numbers from the text.

def  clean_text(df, text_field, new_text_field_name):
    df[new_text_field_name] = df[text_field].str.lower()
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"\d+", "", elem))
    return df

# The clean_text function is applied to the posts dataframe and a new field named 'clean_title' is added to store the cleaned text.

data_cleans = clean_text(posts, 'title', 'clean_title')

# A new column named 'clean_text' is added to the dataframe containing the cleaned text from the 'body' field. Stop words are removed using a list of stop words.

data_cleans['clean_text'] = data_cleans['clean_body'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# The text in the 'clean_text' column is tokenized and added as a new column named 'text_tokens'.

data_cleans['text_tokens'] = data_cleans['clean_text'].apply(lambda x: word_tokenize(x))

# The necessary columns are retained and all rows with NaN values are removed.

data_clean = data_cleans.drop(['title', "body","text_tokens", "subreddit", "url", "clean_title", "clean_body"], axis=1) 
data_clean.dropna(inplace=True)  

# The resulting dataframe is stored in the variable named 'data_clean'.