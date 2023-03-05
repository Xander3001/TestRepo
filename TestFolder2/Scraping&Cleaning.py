# Loop through top 10,000 posts in the specified subreddit and append relevant data to 'posts' list
for post in ml_subreddit.hot(limit=10000):
    posts.append([post.title, post.score, post.id, post.subreddit, post.url, post.num_comments, post.selftext, post.created])

# Convert 'posts' list to Pandas dataframe and rename columns
posts = pd.DataFrame(posts,columns=['title', 'score', 'id', 'subreddit', 'url', 'num_comments', 'body', 'created'])

# Replace blank cells with NaN values and remove rows with NaN values
posts.replace('', np.nan, inplace=True)
posts.dropna(inplace=True)

# Clean 'title' column and add the cleaned text to a new column 'clean_title'
def  clean_text(df, text_field, new_text_field_name):
    df[new_text_field_name] = df[text_field].str.lower()
    # Remove urls, usernames, and special characters from text
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))
    # Remove numbers from text
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"\d+", "", elem))
    return df

# Apply 'clean_text' function to 'posts' dataframe and save cleaned text in 'clean_title' column
data_cleans = clean_text(posts, 'title', 'clean_title')

# Create new column 'clean_text' by removing stopwords from 'clean_body' column
data_cleans['clean_text'] = data_cleans['clean_body'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# Tokenize 'clean_text' column and save output in new column 'text_tokens'
data_cleans['text_tokens'] = data_cleans['clean_text'].apply(lambda x: word_tokenize(x))

# Drop unnecessary columns and rows with NaN values
data_clean = data_cleans.drop(['title', "body","text_tokens", "subreddit", "url", "clean_title", "clean_body"], axis=1) 
data_clean.dropna(inplace=True)