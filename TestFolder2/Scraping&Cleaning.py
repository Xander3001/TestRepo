# iterate through the top 10000 posts in subreddit 'ml_subreddit' and add the post data to a list
for post in ml_subreddit.hot(limit=10000):
    posts.append([post.title, post.score, post.id, post.subreddit, post.url, post.num_comments, post.selftext, post.created])

# store post data in a pandas dataframe and provide labels for each column
posts = pd.DataFrame(posts,columns=['title', 'score', 'id', 'subreddit', 'url', 'num_comments', 'body', 'created'])

# replace any empty strings with NaN values within the dataframe and drop any NaN values
posts.replace('', np.nan, inplace=True)
posts.dropna(inplace=True)

# define a function to clean text by converting it to lowercase, removing certain characters, and removing numbers
def clean_text(df, text_field, new_text_field_name):
    df[new_text_field_name] = df[text_field].str.lower()
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))  
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"\d+", "", elem))
    
    return df

# apply the clean_text function to the 'title' column in the posts dataframe
data_cleans = clean_text(posts, 'title', 'clean_title')

# create a new 'clean_text' column in the data_cleans dataframe by removing any stop words from the 'clean_body' column
data_cleans['clean_text'] = data_cleans['clean_body'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# create a new 'text_tokens' column in the data_cleans dataframe by tokenizing the 'clean_text' column
data_cleans['text_tokens'] = data_cleans['clean_text'].apply(lambda x: word_tokenize(x))

# drop irrelevant columns from data_cleans and drop any NaN values
data_clean = data_cleans.drop(['title', "body","text_tokens", "subreddit", "url", "clean_title", "clean_body"], axis=1) 
data_clean.dropna(inplace=True)