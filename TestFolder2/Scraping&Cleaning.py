
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



# loop through the top 10,000 hot posts in the ml_subreddit and store their information in a list
# each post's information includes its title, score, id, subreddit, url, number of comments, body, and creation date
posts = []
for post in ml_subreddit.hot(limit=10000):
    posts.append([post.title, post.score, post.id, post.subreddit, post.url, post.num_comments, post.selftext, post.created])

# convert the list of posts to a pandas DataFrame and rename the columns
posts = pd.DataFrame(posts,columns=['title', 'score', 'id', 'subreddit', 'url', 'num_comments', 'body', 'created'])

# replace any empty strings with NaN values and drop rows with NaN values
posts.replace('', np.nan, inplace=True)
posts.dropna(inplace=True)

# define a text cleaning function that takes a DataFrame, a text field, and the name of a new text field as inputs
# the function converts the text to lowercase, removes any usernames or special characters, and removes any numbers 
def clean_text(df, text_field, new_text_field_name):
    df[new_text_field_name] = df[text_field].str.lower()
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))  
    # remove numbers
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"\d+", "", elem))
    
    return df

# apply the clean_text function to the DataFrame to create a new column with cleaned titles
data_cleans = clean_text(posts, 'title', 'clean_title')

# create a new column with the cleaned body text, removing any stop words
stop = set(stopwords.words('english'))
data_cleans['clean_text'] = data_cleans['clean_body'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# create a new column with the tokenized words in the cleaned text
data_cleans['text_tokens'] = data_cleans['clean_text'].apply(lambda x: word_tokenize(x))

# create a new DataFrame with only the necessary columns and drop any rows with NaN values
data_clean = data_cleans.drop(['title', "body","text_tokens", "subreddit", "url", "clean_title", "clean_body"], axis=1) 
data_clean.dropna(inplace=True)