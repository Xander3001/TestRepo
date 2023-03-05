# Create a list called posts and append to it the data from the hot section of the subreddit with a limit of 10,000
# Each post has a title, score, id, subreddit, url, number of comments, the post itself, and its creation date. 
# Convert the list into a pandas dataframe with columns: title, score, id, subreddit, url, num_comments, body, created.

for post in ml_subreddit.hot(limit=10000):
    posts.append([post.title, post.score, post.id, post.subreddit, post.url, post.num_comments, post.selftext, post.created])
posts = pd.DataFrame(posts,columns=['title', 'score', 'id', 'subreddit', 'url', 'num_comments', 'body', 'created'])

# Replace empty string values with NaN and remove rows with NaN values from data frame
posts.replace('', np.nan, inplace=True)
posts.dropna(inplace=True)

# Function to clean the text data in a DataFrame
def clean_text(df, text_field, new_text_field_name):
    # Convert text to lowercase
    df[new_text_field_name] = df[text_field].str.lower()
    # Replace URL links, Twitter handles, and special characters with empty string
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))  
    # Remove numbers
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"\d+", "", elem))
    
    return df

# Apply the clean_text function to the posts DataFrame on column 'title' and create a new column called 'clean_title'
data_cleans = clean_text(posts, 'title', 'clean_title')

# Create a new column called 'clean_text' that removes the stop words (common words) from the 'clean_body' column 
data_cleans['clean_text'] = data_cleans['clean_body'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# Tokenize (split) the 'clean_text' column into individual words, and create a new column called 'text_tokens'
data_cleans['text_tokens'] = data_cleans['clean_text'].apply(lambda x: word_tokenize(x))

# Drop unnecessary columns from the data frame
data_clean = data_cleans.drop(['title', "body","text_tokens", "subreddit", "url", "clean_title", "clean_body"], axis=1) 
# Remove rows with NaN values from the new data frame
data_clean.dropna(inplace=True)