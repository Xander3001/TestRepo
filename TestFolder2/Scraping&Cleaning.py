# This code collects data from the 'ml_subreddit' subreddit, gathering information for the 10,000 hottest posts.
# It then adds that data to a Pandas dataframe called 'posts', with specified column names.
for post in ml_subreddit.hot(limit=10000):
    posts.append([post.title, post.score, post.id, post.subreddit, post.url, post.num_comments, post.selftext, post.created])
posts = pd.DataFrame(posts,columns=['title', 'score', 'id', 'subreddit', 'url', 'num_comments', 'body', 'created'])

# Replace any empty cells in the 'posts' dataframe with 'NaN' using numpy, then remove those rows.
posts.replace('', np.nan, inplace=True)
posts.dropna(inplace=True)

# This function is used to 'clean' data in the Pandas dataframe called 'df' - specifically, the text in the 
# column titled 'text_field'. The cleaned text is then placed in a new column called 'new_text_field_name'.
# The function performs several operations on the text in the 'text_field' column, such as lowercasing, removing
# certain characters, and removing any numbers. The cleaned dataframe is then returned.
def clean_text(df, text_field, new_text_field_name):
    df[new_text_field_name] = df[text_field].str.lower()
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))  
    # remove numbers
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"\d+", "", elem))
    
    return df

# Call the 'clean_text' function on the 'posts' dataframe, specifically the 'title' column, and add the cleaned
# text to a new column called 'clean_title'. The cleaned dataframe is then saved in a variable called 'data_cleans'.
data_cleans = clean_text(posts, 'title', 'clean_title')

# Create a new column in the 'data_cleans' dataframe called 'clean_text', which takes the cleaned text from
# the 'clean_body' column and removes any stop words (words that are considered common and therefore unimportant
# for our analysis).
data_cleans['clean_text'] = data_cleans['clean_body'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# Create a new column in the 'data_cleans' dataframe called 'text_tokens', which tokenizes the cleaned text
# from the 'clean_text' column.
data_cleans['text_tokens'] = data_cleans['clean_text'].apply(lambda x: word_tokenize(x))

# Remove certain columns from the 'data_cleans' dataframe that we do not need for our analysis, and then remove
# any rows with 'NaN' values.
data_clean = data_cleans.drop(['title', "body","text_tokens", "subreddit", "url", "clean_title", "clean_body"], axis=1) 
data_clean.dropna(inplace=True)