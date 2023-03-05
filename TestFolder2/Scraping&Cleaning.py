# Define a loop to retrieve posts from the ml_subreddit subreddit's 'hot' section and store them in a list called 'posts'
for post in ml_subreddit.hot(limit=10000):
    posts.append([post.title, post.score, post.id, post.subreddit, post.url, post.num_comments, post.selftext, post.created])

# Create a pandas dataframe object called 'posts' with the 'posts' list and give the columns appropriate labels
posts = pd.DataFrame(posts,columns=['title', 'score', 'id', 'subreddit', 'url', 'num_comments', 'body', 'created'])

# Replace empty strings in the 'posts' dataframe with NaN and modify 'posts' in place
posts.replace('', np.nan, inplace=True)
posts.dropna(inplace=True)


# Define a function called 'clean_text' with inputs 'df', 'text_field', and 'new_text_field_name'
def  clean_text(df, text_field, new_text_field_name):
    # Convert the specified text_field column to lowercase and assign the result to the new_text_field_name column
    df[new_text_field_name] = df[text_field].str.lower()
    # Apply regex to remove various elements from each element in the new_text_field_name column
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))  
    # remove numbers
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"\d+", "", elem))
    
    # Return the modified dataframe 'df'
    return df

# Call the clean_text function on the 'posts' dataframe and save the result to a variable called 'data_cleans'
data_cleans = clean_text(posts, 'title', 'clean_title')

# Print the first five rows of the 'data_cleans' dataframe
data_cleans.head()


# Create a new column in the 'data_cleans' dataframe called 'clean_text' and apply the lambda function to remove stop words
data_cleans['clean_text'] = data_cleans['clean_body'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# Create a new column in the 'data_cleans' dataframe called 'text_tokens' and tokenize the clean_text column
data_cleans['text_tokens'] = data_cleans['clean_text'].apply(lambda x: word_tokenize(x))


# Create a new dataframe object called 'data_clean' with certain columns removed from the 'data_cleans' dataframe and drop any remaining NaN values
data_clean = data_cleans.drop(['title', "body","text_tokens", "subreddit", "url", "clean_title", "clean_body"], axis=1) 
data_clean.dropna(inplace=True)