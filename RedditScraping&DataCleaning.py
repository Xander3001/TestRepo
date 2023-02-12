
#Imports
import re
import pandas as pd 
import numpy as np 
import string
string.punctuation
import nltk.corpus
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
import reticker
import praw
import regex as re
##Scrape sub-reddit turn it into df

reddit = praw.Reddit(client_id='Zr8y6Zdcw2n5jA', client_secret='wyGyqvNV4FpYrwmxcswNnYE2ekgd3g', user_agent='StockScrapper')


posts = []
ml_subreddit = reddit.subreddit("SuperStonk")
for post in ml_subreddit.hot(limit=10000):
    posts.append([post.title, post.score, post.id, post.subreddit, post.url, post.num_comments, post.selftext, post.created])
posts = pd.DataFrame(posts,columns=['title', 'score', 'id', 'subreddit', 'url', 'num_comments', 'body', 'created'])

#Drop empty rows
posts.replace('', np.nan, inplace=True)
posts.dropna(inplace=True)

#load list of company stock names

companies = pd.read_csv(r"C:\Users\AlexandruMalanca\Desktop\Reddit Scraper\nasdaq_screener_1646138257324.csv")  

# Extract company/stock names from body & tile column, compare and get final list of tickers to join back to main DF  
stocks = posts['body'].tolist()

ticks= []

extractor = reticker.TickerExtractor()
    
for i in stocks:
    ticks.append(extractor.extract(i))
    
    
#Join Extracted stock names back to original DF

posts['StonkName'] = ticks

aba = posts.apply(pd.Series.explode)


#posts['Company'] = [','.join(map(str, l)) for l in posts['StonkName']]


companies['length'] = companies.Symbol.str.len()

companies = companies[companies.length > 1]

# posts['Companyyyyy'] = 'test'


# for i in posts['Company']:
#     print(i)

# for i in posts['Company']:
#         for j in companies["Symbol"]:
#             if  i == j:
#                    posts['Companyyyyy'] = i
            

posts = pd.merge(aba, companies, how='left', left_on = 'StonkName', right_on = 'Symbol')


##Clean the data


#Clean text data - Normalization
def  clean_text(df, text_field, new_text_field_name):
    df[new_text_field_name] = df[text_field].str.lower()
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))  
    # remove numbers
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"\d+", "", elem))
    
    return df
data_cleans = clean_text(posts, 'title', 'clean_title')
data_cleans = clean_text(posts, 'body', 'clean_body')
data_cleans.head()




#Remove Stop Words

data_cleans['clean_text'] = data_cleans['clean_body'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
data_cleans['clean_titles'] = data_cleans['clean_title'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

#Tokenise for further processing     
data_cleans['text_tokens'] = data_cleans['clean_text'].apply(lambda x: word_tokenize(x))



#Drop unwanted columns
data_clean = data_cleans.drop(['title', "body","text_tokens", "subreddit", "url", "clean_title", "clean_body"], axis=1) 
data_clean.dropna(inplace=True)

#Save to CSV
data_clean.to_csv(r"C:\Users\Alexandru.Malanca\Desktop\Power Bi Demo\RedditComments.csv")
