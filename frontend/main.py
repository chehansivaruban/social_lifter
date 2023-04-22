import string
import streamlit as st
import pandas as pd
import re
import neattext.functions as nfx
from textblob import TextBlob
from keybert import KeyBERT
import seaborn as sns
from transformers import RobertaModel, RobertaTokenizer
from transformers import RobertaModel, RobertaTokenizer
from pandas import DataFrame
import gensim
from gensim.utils import simple_preprocess
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import joblib

# Hide the Streamlit tag in the footer
def hide_streamlit_footer():
    st.markdown("""
                <style>
                .css-cio0dv.egzxvld1{
                    visibility:hidden;
                }
                </style>
                """,unsafe_allow_html=True)
    
# Display page header
def display_page_header():
    st.markdown("<h1 style = 'text-align: center;'>Social Lifter</h1>", unsafe_allow_html=True)
    st.markdown("---")

# Display tweet reach predictor section header
def display_tweet_reach_predictor_section_header():
    st.markdown("<h2>Tweet Reach Predictor</h2>", unsafe_allow_html=True)



label_encoder_day_of_week = joblib.load("label_encoder_day_of_week.pkl")
label_encoder_language = joblib.load("label_encoder_language.pkl")
label_encoder_clean_tweet = joblib.load("label_encoder_clean_tweet.pkl")
label_encoder_sentiment = joblib.load("label_encoder_sentiment.pkl")
label_encoder_key_words = joblib.load("label_encoder_key_words.pkl")

# Load pre-trained model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')
# Load the LDA model and the dictionary
lda_model = gensim.models.ldamodel.LdaModel.load('lda_topic_detection_model_10.lda')
dictionary = gensim.corpora.Dictionary.load('lda_topic_detection_model_10.lda.id2word')
rf_reg = joblib.load("rf_reg_model.pkl")
# Define function to get sentiment label from score
def getSentiment(score):
  if (score < 0 ):
    return 'negative'
  elif (score == 0):
    return 'neutral'
  else:
    return 'positive'


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

def get_tweet_topic_list(tweet_text):
    # Clean the tweet text
    cleaned_text = simple_preprocess(tweet_text)
    
    # Convert the cleaned text to a bag of words representation using the dictionary
    bow_vector = dictionary.doc2bow(cleaned_text)
    
    # Use the LDA model to get the topic distribution for the tweet
    topic_distribution = lda_model.get_document_topics(bow_vector)
    
    # Return the topic with the highest probability
    top_topic = max(topic_distribution, key=lambda x: x[1])
    return top_topic[0]

# Define function to clean text data
def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Define functions to get polarity and subjectivity of text
def get_polarity(tweet):
  return TextBlob(tweet).sentiment.polarity

def get_subjectivity(tweet):
  return TextBlob(tweet).sentiment.subjectivity

# Define function to load KeyBERT model
@st.cache_data()
def load_model():
    return KeyBERT(model=model)

# Extract tweet keywords
def extract_keywords(text):
    kw_model = load_model()
    # Use str.contains() method to keep only rows where the 'Tweet' column contains English letters
    cleaned_text = re.findall(r'[a-zA-Z]+', text)
    cleaned_text_series = pd.Series(cleaned_text)
    cleaned_text_series = cleaned_text_series.apply(lambda x: nfx.remove_multiple_spaces(x))
    cleaned_text_series = cleaned_text_series.str.cat(sep=' ')
    cleaned_text_series = clean_text(cleaned_text_series)
    keywords = kw_model.extract_keywords(
    cleaned_text_series,
    keyphrase_ngram_range=(1, 2),
    use_mmr=True,
    stop_words="english",
    top_n=5,
    diversity=0.5,)
    
    return keywords

def keyword_dataframe(keywords):
    df = (
        DataFrame(keywords, columns=["Keyword/Keyphrase", "Relevancy"])
        .sort_values(by="Relevancy", ascending=False)
        .reset_index(drop=True)
    )
    return df
def style_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cmGreen = sns.light_palette("green", as_cmap=True)
    cmRed = sns.light_palette("red", as_cmap=True)
    df = df.style.background_gradient(
        cmap=cmGreen,
        subset=["Relevancy"],
    )
    return df
# Define the styling functions

def style_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cmGreen = sns.light_palette("green", as_cmap=True)
    cmRed = sns.light_palette("red", as_cmap=True)
    df = df.style.background_gradient(
        cmap=cmGreen,
        subset=["Relevancy"],
    )
    return df


def display_dataframe(df: pd.DataFrame) -> None:
    c1, c2, c3 = st.columns([1, 3, 1])

    format_dictionary = {
        "Relevancy": "{:.1%}",
    }

    df_key = df.format(format_dictionary)

    with c2:
        st.table(df_key)


# Configure Streamlit page
st.set_page_config(
    page_title="Social Lifter",
    page_icon="🌟",
    layout="centered",
    initial_sidebar_state="expanded",
)
hide_streamlit_footer()
display_page_header()
display_tweet_reach_predictor_section_header()

# Create form for inputting data
form = st.form("tweet")

text = form.text_area("Enter your Tweet")
date = form.date_input("Enter your Date")
time = form.time_input("Enter your Time")

isTagged = form.selectbox("Users Tagged ?", options=("True", "False"))
isLocation = form.selectbox("Location Provided ?", options=("True", "False"))
isHashtag = form.selectbox("Is Hashtag available ?", options=("True", "False"))
isCashtag = form.selectbox("Is Cashtag available ?", options=("True", "False"))

followers = form.number_input("Enter No. of Followers")
following = form.number_input("Enter No. of Following")
isVerified = form.selectbox("Is your account verified?", options=("Verified", "Not Verified"))
account_age = form.number_input("How old is your account?")
average_like = form.number_input("Whats the average likes that you get?")

btn = form.form_submit_button("Predict Reach")

if btn:
  isTagged = 1 if isTagged == "True" else 0  
  isLocation = 1 if isLocation == "True" else 0  
  isVerified = 1 if isVerified == "True" else 0  
  isHashtag = 1 if isHashtag == "True" else 0  
  isCashtag = 1 if isCashtag == "True" else 0  
  date = date.strftime("%A")
  time = datetime.strptime(str(time), '%H:%M:%S').strftime('%H')
  cleaned_text_series = clean_text(text)
  polarity = get_polarity(cleaned_text_series)
  subjectivity = get_subjectivity(cleaned_text_series)
  sentiment = getSentiment(polarity)
  # print(keywords)
  extracted_keywords = extract_keywords(text)
  df = keyword_dataframe(extracted_keywords)
  df.index += 1
  df =style_dataframe(df)
  display_dataframe(df)
  topic = get_tweet_topic(cleaned_text_series)
  topic_list = get_tweet_topic_list(cleaned_text_series)
  # Initialize LabelEncoder object
  label_encoder = LabelEncoder()

  # Convert categorical features to numerical values
  day_of_week_encoded = label_encoder.fit_transform([date])
  language_encoded = label_encoder.fit_transform(["English"])
  clean_tweet_encoded = label_encoder.fit_transform([cleaned_text_series])
  sentiment_encoded = label_encoder.fit_transform([sentiment])
  key_words_encoded = label_encoder.fit_transform([topic_list])
  inputs = pd.DataFrame({
    "time": [time],  # add missing value
    "Day of week": [day_of_week_encoded[0]],
    "Cashtags": [isCashtag],  # add missing value
    "Hashtags": [isHashtag],  # add missing value
    "Language": [language_encoded[0]],
    "Location": [isLocation],  # add missing value
    "Mentioned_users": [isTagged],  # add missing value
    "Followers": [followers],
    "Following": [following],
    "Verified": [isVerified],
    "Average_favourite_count": [average_like],
    "account_age": [account_age],
    "clean_tweet": [clean_tweet_encoded[0]],
    "subjectivity": [subjectivity],
    "polarity": [polarity],
    "sentiment": [sentiment_encoded[0]],
    "topics": [topic],  # add missing value
    "key_words": [key_words_encoded[0]]
})

  df = pd.DataFrame(inputs)
  
  prediction = rf_reg.predict(df)
  st.write(prediction)




