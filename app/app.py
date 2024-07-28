import uvicorn
import pickle
import nltk
import numpy as np
from fastapi import FastAPI, HTTPException
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


# Download necessary nltk resources
nltk.download('stopwords')
nltk.download('wordnet')
app = FastAPI()
# Load the pickled model
with open('mnb_model.pkl', 'rb') as file:
    model = pickle.load(file)
# Load the pickled vectorizer
with open('tfidvectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)
#test tweet
tweet = "@test Loving the new iphone, it's the best. #iphonelife"

#index route
@app.get("/")
def read_root():
    return True


@app.post("/predict")
def predict(tweet: str):
    try:
        # TODO: implement sentiment analysis using a pre-trained model
        # for now, we'll just return the input tweet
        return {"result": analyze_sentiment(tweet).tolist()[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error: " + str(e))

def analyze_sentiment(tweet: str):
    data = preprocess_tweet(tweet)
    return model.predict(data)

def preprocess_tweet(tweet:str):
    """
    Preprocesses a given tweet by tokenizing, stemming, and lemmatizing the words.

    Parameters:
    tweet (str): The input tweet to be preprocessed.

    Returns:
    str: The preprocessed tweet as a string of space-separated tokens.
    """
    tokens = tokenize_tweet(tweet)
    tokens = stem_tweet(tokens)
    tokens = lemmatize_tweet(tokens)
    tweet_vector = vectorizer.transform(tokens)
    return tweet_vector

def tokenize_tweet(tweet:str):
    """
    Tokenizes a given tweet into individual words using the TweetTokenizer from the NLTK library.
    Stopwords and punctuation are removed from the tokenized words.

    Parameters:
    tweet (str): The input tweet to be tokenized.

    Returns:
    list: A list of strings representing the tokenized words after removing stopwords and punctuation.
    """
    tokenizer = TweetTokenizer(strip_handles = True, preserve_case = False, )
    pptweet = tokenizer.tokenize(tweet)
    #load stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    stopwords_list = list(stop_words)
    punctuation_list = list(punctuation)
    # Combine stopwords and punctuation into a single list
    stop_list = stopwords_list + punctuation_list
    stop_list = stop_list + ['#sxsw', 'sxsw', 'sxswi', '#sxswi', 'rt', 'ipad', 'google', 'apple', 'iphone', 'amp',
             'android', 'sxswi', 'link', '#apple',
             '#google', '...', '\x89', '#ipad2',
             '0','1','2','3','4','5','6','7','8','9',
             '#iphone', '#android', 'store', 'austin', '#ipad']
    #remove stopwords and punctuation from the tweet
    return [word for word in pptweet if word not in stop_list]

def stem_tweet(tokens: list):
    """
    This function takes a list of tokens (words) as input and applies stemming to each token.
    Stemming reduces words to their base or root form, which helps in reducing the vocabulary size and improving the efficiency of text analysis.

    Parameters:
    tokens (list): A list of strings representing the tokens (words) to be stemmed.

    Returns:
    list: A list of strings representing the stemmed tokens.
    """
    stemmer = nltk.PorterStemmer()
    return [stemmer.stem(word) for word in tokens]

def lemmatize_tweet(tokens: list):
    """
    This function takes a list of tokens (words) as input and applies lemmatization to each token.
    Lemmatization reduces words to their base or root form, which helps in reducing the vocabulary size and improving the efficiency of text analysis.
    It uses the WordNetLemmatizer from the NLTK library.

    Parameters:
    tokens (list): A list of strings representing the tokens (words) to be lemmatized.

    Returns:
    list: A list of strings representing the lemmatized tokens.
    """
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in tokens]


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)