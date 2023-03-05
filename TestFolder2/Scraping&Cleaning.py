# This code retrieves data from the top 10000 posts in a specified subreddit and converts it to a pandas DataFrame. 
# It then performs textual cleaning, removing non-alphanumeric characters, URLs and numbers, and lowercasing all text. 
# Stopwords are also removed and the resulting clean text is tokenized. 
# Finally, unnecessary columns are dropped and any remaining missing values are dropped.