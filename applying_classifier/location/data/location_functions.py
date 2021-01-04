import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import warnings
warnings.simplefilter("ignore")
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer


df1 = pd.read_csv('./raw_data/date_tweets_day_1.csv', index_col=0)
df2 = pd.read_csv('./raw_data/date_tweets_day_2.csv', index_col=0)
df3 = pd.read_csv('./raw_data/date_tweets_day_3.csv', index_col=0)
df4 = pd.read_csv('./raw_data/date_tweets_day_4.csv', index_col=0)
df5 = pd.read_csv('./raw_data/date_tweets_day_5.csv', index_col=0)

frames = [df1, df2, df3, df4, df5]

data = pd.concat(frames)

class_labels = ['Anti Man-Made','Neutral','Man-Made','News']

def tokenize_single(data, parameters):
    '''
    Function to tokenize any single string.
    
    Input
    -----
    data : str 
    parameters : Regex Filter
        Ex: r'[a-zA-Z]+'
    
    Optional Input
    --------------
    None
        
    Output
    ------
    Tokenized data
    '''   
    tokenizer = RegexpTokenizer(parameters)
    data = tokenizer.tokenize(data)
    return data

def find_us(x):
    '''
    Function to determines whether or not a value possesses an element in the list of states.
    
    Input
    -----
    data : list (str)
    
    Optional Input
    --------------
    None
        
    Output
    ------
    Returns the abbreviation for every element in list of states and not for every element not in the list of states.
    '''
    states_one = ['al', 'ak','az', 'ar', 'ca', 'co', 'ct', 'de', 'dc','fl','ga', 'hi', 'id','il', 'in', 'ia', 'ks', 'ky', 'la', 'me', 'mt', 'ne', 'nv', 'nh', 'nj', 'nm', 'ny', 'nc', 'nd', 'oh', 'ok', 'or', 'md', 'ma', 'mi', 'mn', 'ms', 'mo', 'pa', 'ri', 'sc', 'sd', 'tn', 'tx', 'ut', 'vt', 'va', 'wa', 'wv', 'wi', 'wy']
    new_value = ''
    for i in x:
        if i in states_one:
            new_value = new_value+i
        else:
            new_value = 'not'
    return new_value

def try_split(x):
    '''
    Function that tries to split all elements in a string but passes upon error.
    
    Input
    -----
    data :  str
    
    Optional Input
    --------------
    None
        
    Output
    ------
    List of split words from input string
    '''
    new = []
    try:
        i = x.split()
        new = i
    except:
        new = 'not us'
    
    return new

def states_list():
    '''
    Function to return list of US state abbreviations.
    
    Input
    -----
    None
    
    Optional Input
    --------------
    None
        
    Output
    ------
    List of abbreviations
    '''
    return [ 'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL', 'GA', 'HI', 'ID','IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'MD', 'MA', 'MI', 'MN',
'MS', 'MO', 'PA', 'RI','SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']

def lemmatize_tweet(data):
    '''
    Function to lemmatize tweets

    Input
    -----
    data : str

    Optional Input
    --------------
    None

    Output
    ------
    String containing lemmatized tweets
    '''   
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    lem_data = tokenize_single(data,r'[a-z]+')
    lem_data = [lemmatizer.lemmatize(word) for word in lem_data if word not in stop_words]
    lem_data = [word for word in lem_data if len(word) > 2]
    lem_tweet = untokenize_single(lem_data)
    lem_tweet = lem_tweet.strip()
    
    return lem_tweet

def clean_tweet(data):
    '''
    Function to clean tweets

    Input
    -----
    data : str

    Optional Input
    --------------
    None

    Output
    ------
    Cleaned tweets as strings
    ''' 
    #removing hashtags, hyperlinks, mentions
    data = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",data).split())
    # removing mentions
    data = re.sub('(@[A-Za-z0-9]+)', '', data)
    # removing links
    data = re.sub(r'http\S+', '', data)
    data = re.sub(r'pic\.\S+', '', data)
    # convert contractions
    data = decontracted(data)
    # removing retweets
    data = re.sub("RT",'',data).strip()
    # making lowercase
    data = data.lower()
    
    # filtering for just letters
    data = tokenize_single(data, r'[a-zA-Z]+')
    data = untokenize_single(data)
    
    return data

def decontracted(phrase):
    '''
    Function to convert contractions

    Input
    -----
    data : str

    Optional Input
    --------------
    None

    Output
    ------
    String containing elements from input list
    
    Source
    ------
    https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
    ''' 
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def untokenize_single(data):
    '''
    Function to untokenize a single list.

    Input
    -----
    data : list (str)

    Optional Input
    --------------
    None

    Output
    ------
    String containing elements from input list
    '''
    joined = ','.join(data)
    new_data = joined.replace(',',' ')
    return new_data


def tokenize(data, parameters):
    '''
    Function to tokenize any series of strings.
    
    Input
    -----
    data : str 
    parameters : Regex Filter
        Ex: r'[a-zA-Z]+'
    
    Optional Input
    --------------
    None
        
    Output
    ------
    Tokenized data
    '''
    tokenizer = RegexpTokenizer(parameters)
    data.tweet = data.tweet.apply(lambda x: tokenizer.tokenize(x))
    return data.tweet

def untokenize(data):
    '''
    Function to untokenize a series of lists.

    Input
    -----
    data : list (str)

    Optional Input
    --------------
    None

    Output
    ------
    String containing elements from input list
    '''
    data.tweet = data.tweet.apply(lambda x: ','.join(x))
    data.tweet = data.tweet.apply(lambda x: x.replace(',',' '))
    return data.head()


def lowercase(word_list):
    '''
    Function to lowercase all words in a list.
    
    Input
    -----
    data : list (str)
    
    Optional Input
    --------------
    None
        
    Output
    ------
    Same input list now lowercased
    '''
    lowered = []
    for x in word_list:
        x = x.lower()
        lowered.append(x)
    return lowered