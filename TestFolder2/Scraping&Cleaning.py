# Loop through the hot posts on ml_subreddit and append relevant information to the 'posts' list
for post in ml_subreddit.hot(limit=10000):
    posts.append([post.title, post.score, post.id, post.subreddit, post.url, post.num_comments, post.selftext, post.created])

# Convert the list of posts into a pandas dataframe with headers
posts = pd.DataFrame(posts,columns=['title', 'score', 'id', 'subreddit', 'url', 'num_comments', 'body', 'created'])

# Replace empty strings in the dataframe with NaN values and drop any rows that contain NaN values
posts.replace('', np.nan, inplace=True)
posts.dropna(inplace=True)

def  clean_text(df, text_field, new_text_field_name):
    # Convert text to lowercase
    df[new_text_field_name] = df[text_field].str.lower()
    # Remove any usernames, special characters, URLs and retweets
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))  
    # Remove any numbers
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"\d+", "", elem))
    
    return df

# Apply the 'clean_text' function to the 'title' column of the 'posts' dataframe and add the cleaned text to a new column named 'clean_title'
data_cleans = clean_text(posts, 'title', 'clean_title')

# Tokenize the cleaned text of each post and add the resulting tokens to a new column named 'text_tokens'
data_cleans['text_tokens'] = data_cleans['clean_title'].apply(lambda x: word_tokenize(x))

# Drop columns that are no longer needed for analysis and drop any rows that contain NaN values
data_clean = data_cleans.drop(['title', "body","text_tokens", "subreddit", "url", "clean_title"], axis=1)
data_clean.dropna(inplace=True)