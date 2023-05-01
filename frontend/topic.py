import streamlit as st
import gensim
from gensim.utils import simple_preprocess



# Load the LDA model and the dictionary
lda_model = gensim.models.ldamodel.LdaModel.load('models/lda_model/lda_model.pickle')
dictionary = gensim.corpora.Dictionary.load('models/lda_model/lda_model.id2word')

def get_tweet_topic(tweet_text):
    # Clean the tweet text
    cleaned_text = simple_preprocess(tweet_text)
    
    # Convert the cleaned text to a bag of words representation using the dictionary
    bow_vector = dictionary.doc2bow(cleaned_text)
    
    # Use the LDA model to get the topic distribution for the tweet
    topic_distribution = lda_model.get_document_topics(bow_vector)
    
    # Return the topic with the highest probability
    top_topic = max(topic_distribution, key=lambda x: x[1])
    return top_topic[0]

