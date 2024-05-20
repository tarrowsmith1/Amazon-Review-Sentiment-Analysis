import spacy
import pandas as pd
import re
from spacytextblob.spacytextblob import SpacyTextBlob


# 'en_core_web_sm' is a pre-trained english language model provided by spaCy
nlp = spacy.load("en_core_web_sm")

# Adding spacytextblob extension that integrates TextBlob's sentiment analysis capabilities into the spaCy pipeline. This component allows spaCy to perform sentiment analysis on text using TextBlob's pre-trained sentiment analysis model.
nlp.add_pipe("spacytextblob")

def preprocess_text(review: str):
    '''
    Preprocess text data for sentiment analysis
    Returns preprocessed text/review
    '''
    # Lowercase the text
    review = review.lower()

    # Remove whitespaces
    review = review.strip()

    # Remove punctuation using regex
    pattern = re.compile(r"[^a-zA-Z0-9 ]")
    review = re.sub(pattern, '', review)

    # Remove stopwords using spaCy
    doc = nlp(review)
    # For each token (word) check if it's a stop word, if it is not then add to list and finally join all the non-stopwords together; filtering the stopwords
    review = ' '.join([token.text for token in doc if not token.is_stop])

    return review

def sentiment_prediction(review: str):
    '''
    Predicts the sentiment of a review
    Returns the sentiment score - if it is close to 1 then it is positive, -1 then it is negative, 0 then it is neutral
    '''
    # Preprocess the review text
    processed_text = preprocess_text(review)

    # Load the preprocessed text into spaCy
    doc = nlp(processed_text)

    # Get the sentiment score
    #sentiment_score = doc.sentiment
    sentiment_score = doc._.blob.polarity

    return sentiment_score

# Load the dataset
df = pd.read_csv(r"Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv")

# Drop rows with an empty reviews.text column; without a review
df = df.dropna(subset=['reviews.text'])

# Selecting only the reviews column from the dataset - feature variable to be used for sentiment analysis
reviews_data = df['reviews.text']

# Apply sentiment analysis to each review using the function
#df["sentiment"] = reviews_data.apply(sentiment_prediction)

# Test
testing_reviews = ["This product is very bad!", "I love this product!", "This product can go do one!!!!", "This product is OK!"]

for review in testing_reviews:
  sentiment = sentiment_prediction(review)
  print(f"Review: {review} - Sentiment: {sentiment}")