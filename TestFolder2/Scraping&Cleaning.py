
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



# #Take the desired number of 'search terms' from reddit
# for post in ml_subreddit.hot(limit=10000):
#     posts.append([post.title, post.score, post.id, post.subreddit, post.url, post.num_comments, post.selftext, post.created])
# posts = pd.DataFrame(posts,columns=['title', 'score', 'id', 'subreddit', 'url', 'num_comments', 'body', 'created'])
# #Replace column values that are empty strings with NaN values
# #As there doesn't seem to be a recommendation anywhere on how to treat empty strings, I have filled them with NaN.
# #Dropping NaN values entire leads to throwing away 60% of the data, hence I will leave them
# #Passing them through automatically shouldn't cause big problems either
# 
# posts.replace('', np.nan, inplace=True)
# #Create a function to clean text
# #Not necessary but will assist in the next few steps.
# posts.dropna(inplace=True)
# 
# 
# def  clean_text(df, text_field, new_text_field_name):
#     df[new_text_field_name] = df[text_field].str.lower()
#     df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))  
#     # remove numbers
# 
# #Clean the title with function defined above and drop unneccesary columns
# #The aim of these next steps is to clean the data.
# #Cleaning the data helps makes the tool you are trying to create (ML classifier) more robust.
#     df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"\d+", "", elem))
#     
#     return df
# data_cleans = clean_text(posts, 'title', 'clean_title')
# 
# data_cleans.head()
# 
# 
# data_cleans['clean_text'] = data_cleans['clean_body'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
# 
# data_cleans['text_tokens'] = data_cleans['clean_text'].apply(lambda x: word_tokenize(x))
# 
# 
# data_clean = data_cleans.drop(['title', "body","text_tokens", "subreddit", "url", "clean_title", "clean_body"], axis=1) 
# data_clean.dropna(inplace=True)