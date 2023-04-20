import string
import streamlit as st
import pandas as pd
import re
import neattext.functions as nfx
from textblob import TextBlob
import numpy as np
from keybert import KeyBERT
# For Flair (Keybert)
from flair.embeddings import TransformerDocumentEmbeddings
import seaborn as sns
import os
import json
from transformers import RobertaModel, RobertaTokenizer
from transformers import RobertaModel, RobertaTokenizer, AutoTokenizer


# Load pre-trained model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')

def getSentiment(score):
  if (score < 0 ):
    return 'negative'
  elif (score == 0):
    return 'neutral'
  else:
    return 'positive'


def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove mentions (@)
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags (#)
    text = re.sub(r'#\w+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase
    text = text.lower()
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_polarity(tweet):
  return TextBlob(tweet).sentiment.polarity

def get_subjectivity(tweet):
  return TextBlob(tweet).sentiment.subjectivity

@st.cache(hash_funcs={RobertaTokenizer: lambda x: 0})
def load_model():
    return KeyBERT(model=model)



st.set_page_config(
    page_title="Social Lifter",
    page_icon="🌟",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown("""
            <style>
            .css-cio0dv.egzxvld1{
                visibility:hidden;
            }
            </style>
            """,unsafe_allow_html=True)

st.markdown("<h1 style = 'text-align: center;'>Social Lifter</h1>", unsafe_allow_html=True)

st.markdown("---")

st.markdown("<h2>Tweet Reach Predictor</h2>", unsafe_allow_html=True)

form = st.form("tweet")

text = form.text_area("Enter your Tweet")
date = form.date_input("Enter your Date")
time = form.time_input("Enter your Time")

isTagged = form.selectbox("Users Tagged?", options=("True", "False"))
isLocation = form.selectbox("Location Provided?", options=("True", "False"))

followers = form.number_input("Enter No. of Followers")
following = form.number_input("Enter No. of Following")
isVerified = form.selectbox("Is your account verified?", options=("Verified", "Not Verified"))
account_age = form.number_input("How old is your account?")
average_like = form.number_input("Whats the average likes that you get?")

btn = form.form_submit_button("Predict Reach")

if btn:
    kw_model = load_model()
    english_pattern = re.compile(r'[a-zA-Z]')
    # Use str.contains() method to keep only rows where the 'Tweet' column contains English letters
    cleaned_text = re.findall(r'[a-zA-Z]+', text)
    cleaned_text_series = pd.Series(cleaned_text)
    cleaned_text_series = cleaned_text_series.apply(lambda x: nfx.remove_multiple_spaces(x))
    cleaned_text_series = cleaned_text_series.str.cat(sep=' ')
    cleaned_text_series = clean_text(cleaned_text_series)
    polarity = get_polarity(cleaned_text_series)
    subjectivity = get_subjectivity(cleaned_text_series)
    sentiment = getSentiment(polarity)
    keywords = kw_model.extract_keywords(
    cleaned_text_series,
    keyphrase_ngram_range=(1, 2),
    use_mmr=True,
    stop_words="english",
    top_n=5,
    diversity=0.5,
)
    print(keywords)
    inputs = {
        
        'sentiment': [sentiment],
        'subjectivity': [subjectivity],
        'polarity': [polarity],
        'Text': [cleaned_text_series],
        'Date': [date],
        'Time': [time],
        'Users Tagged?': [isTagged],
        'Location Provided?': [isLocation],
        'No. of Followers': [followers],
        'No. of Following': [following],
        'Verified Account': [isVerified],
        'Account Age': [account_age],
        'Average Likes': [average_like]
    }

    df = pd.DataFrame(inputs)
    st.dataframe(df)

