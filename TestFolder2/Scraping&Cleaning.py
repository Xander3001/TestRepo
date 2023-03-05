# Loop through the top 10,000 posts from the machine learning subreddit and append relevant post data to a "posts" list
for post in ml_subreddit.hot(limit=10000):
    posts.append([post.title, post.score, post.id, post.subreddit, post.url, post.num_comments, post.selftext, post.created])

# Convert the list of post data to a Pandas dataframe and rename columns
posts = pd.DataFrame(posts,columns=['title', 'score', 'id', 'subreddit', 'url', 'num_comments', 'body', 'created'])

# Replace empty cells with NaN and drop all rows with NaN values
posts.replace('', np.nan, inplace=True)
posts.dropna(inplace=True)

# Define a function to clean the text in a given Pandas dataframe column
def clean_text(df, text_field, new_text_field_name):
    # Convert all text to lowercase characters
    df[new_text_field_name] = df[text_field].str.lower()
    # Remove any usernames, URLs, or non-alphanumeric characters 
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))
    # Remove any numerical characters
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"\d+", "", elem))
    return df

# Apply the "clean_text" function to the "title" column of the "posts" dataframe and create a new column with the cleaned text
data_cleans = clean_text(posts, 'title', 'clean_title')

# Create a new column of cleaned text that removes any stop words using NLTK's word_tokenize function
data_cleans['clean_text'] = data_cleans['clean_body'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# Tokenize the cleaned text and create a new column with the tokens
data_cleans['text_tokens'] = data_cleans['clean_text'].apply(lambda x: word_tokenize(x))

# Drop irrelevant columns (including the original title and body) from the dataframe and drop any remaining rows with NaN values
data_clean = data_cleans.drop(['title', "body","text_tokens", "subreddit", "url", "clean_title", "clean_body"], axis=1) 
data_clean.dropna(inplace=True)