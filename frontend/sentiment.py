

from textblob import TextBlob
# Define function to get sentiment label from score
def getSentiment(score):
  if (score < 0 ):
    return 'negative'
  elif (score == 0):
    return 'neutral'
  else:
    return 'positive'

# Define functions to get polarity and subjectivity of text
def get_polarity(tweet):
  return TextBlob(tweet).sentiment.polarity

def get_subjectivity(tweet):
  return TextBlob(tweet).sentiment.subjectivity

def get_polarity_subjective_sentimet(tweet):
    subjectivity = get_subjectivity(tweet)
    polarity = get_polarity(tweet)
    sentiment = getSentiment(polarity)
    return subjectivity,polarity,sentiment
