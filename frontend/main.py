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
import plotly.graph_objs as go
from sidebar import get_user_inputs
import numpy as np
import pickle

from display import (
    hide_streamlit_footer,
    display_page_header,
    style_dataframe,
    display_dataframe,
    display_container
)


pick_read = open("label_encoder_day_of_week.pkl",'rb')
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
rf_reg = joblib.load("models/rf_reg_model.pkl")
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

# Configure Streamlit page
st.set_page_config(
    page_title="Social Lifter",
    page_icon="🌟",
    layout="centered",
    initial_sidebar_state="expanded",
)
hide_streamlit_footer()
display_page_header()
tab1, tab2, tab3= st.tabs(["Tweet Reach Predictor", "Generate Tweet", "Info"])


with tab1:
  text, date, time, isTagged, isLocation, isHashtag, isCashtag, followers, following, isVerified, account_age, average_like, btn = get_user_inputs()
  if btn:
    display_container()
    st.markdown("<br>",unsafe_allow_html=True)
    st.markdown("---")
    predCol,predCol1, predCol2 = st.columns([5,8, 10])
    with predCol1:
        st.markdown("<br>",unsafe_allow_html=True)
        st.write("Likes")
    with predCol2:
        st.metric("Likes", "70", "-1.2", label_visibility="hidden")
    st.markdown("---")
    predCol,predCol3, predCol4 = st.columns([5,8, 10])
    with predCol3:
        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        st.write("Comments")
    with predCol4:
        st.metric("Comments", "70", "-1.2", label_visibility="hidden")
    st.markdown("---")
    predCol,predCol5, predCol6 = st.columns([5,8, 10])
    with predCol5:
        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        st.write("Retweets")
    with predCol6:
        st.metric("Retweets", "70", "-1.2", label_visibility="hidden")
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
    df_key = keyword_dataframe(extracted_keywords)
    df_key.index += 1
    df_key_style =style_dataframe(df_key)
    display_dataframe(df_key_style)
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
    likes = prediction[0][0]
    comments = prediction[0][1]
    retweets = prediction[0][2]
    
    
    

    st.write(prediction)
    # Create a bar chart using Plotly
    fig = go.Figure(
    data=[go.Bar(x=df_key['Keyword/Keyphrase'], y=df_key['Relevancy'])])
    fig.update_layout(
                    title = dict(text = 'Keywords'),
                    xaxis_title="Keyword/Keyphras",
                    yaxis_title="Relevancy"
                )
    # Display the chart in Streamlit
    st.plotly_chart(fig)
    st.write(likes)
    st.write(comments)
    st.write(retweets)
  else: 
    st.markdown("---")
    st.markdown("<h1>Get your Tweet predictions</h1>", unsafe_allow_html=True)
    st.markdown("---")
    html_string = "<div style='border-radius: 10%; overflow: hidden;'><img src='https://digiday.com/wp-content/uploads/sites/3/2023/01/twitter-flatline-digiday-gif.gif?w=800&h=466&crop=1'></div>"
    st.markdown(html_string, unsafe_allow_html=True)


   
    
    
    
css = '''
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:1.5rem;
    margin-right: 100px;
    position: relative;
    margin-left: 15px;
    }
</style>
'''
st.markdown(css, unsafe_allow_html=True)




# ------------------------info tab ------------------

with tab3:
    st.markdown("<br><h1>What are the steps to create an effective marketing plan on Twitter that can lead to success for a business?</h1>", unsafe_allow_html=True)
    st.markdown("""<p>
                Deciding which social media platform to focus on and how to optimize your content can be challenging. Every network has its unique style, and you must
                understand how each one works to create a successful social media strategy for your small business. <br><br>
                Twitter was launched in 2006 and has undergone many changes since its inception, from simple 140-character messages to a powerful tool for businesses. Today, it remains one of the most popular social networks.
                </p>
                <h3>Crafting the Perfect Twitter Marketing Strategy for Business Success</h3>
                <p>
                Twitter is one of the most powerful social media platforms available to businesses. It offers a unique opportunity to engage with customers, build brand awareness, and drive traffic to your website. However,
                crafting a successful Twitter marketing strategy can be challenging. In this article, we'll explore some tips and tricks for creating a perfect Twitter marketing strategy for your business.
                </p>
                <h3>1. Define Your Twitter Marketing Goals</h3>
                <p>
                The first step in creating a successful Twitter marketing strategy is defining your goals. What do you want to achieve with your Twitter presence? Do you want to increase brand awareness, drive website traffic, or generate leads? Defining your goals will help you focus your efforts and measure your success.
                </p>
                <h3>2. Know Your Audience</h3>
                <p>
                To create content that resonates with your audience, you need to know who they are. Conduct research to understand the demographics and interests of your Twitter followers. This information will help you create content that speaks to their needs and interests.
                </p>
                <h3>3. Develop a Content Strategy</h3>
                <p>
                Once you know your audience and your goals, it's time to develop a content strategy. Your content strategy should focus on providing value to your audience while achieving your marketing goals. Consider the types of content that resonate with your audience, such as images, videos, or blog posts.
                </p>
                <h3>4. Engage with Your Audience</h3>
                <p>
                Twitter is all about engagement. To build a following on Twitter, you need to engage with your audience regularly. Respond to comments and messages, retweet relevant content, and participate in Twitter chats.
                </p>
                <h3>5. Use Twitter Analytics</h3>
                <p>
                Twitter offers a powerful analytics tool that can help you measure the success of your Twitter marketing strategy. Use Twitter Analytics to track engagement, follower growth, and website clicks. This information can help you refine your strategy and optimize your content.
                </p>
                <h3>6. Experiment with Twitter Advertising</h3>
                <p>
                Twitter offers a range of advertising options that can help you reach a broader audience. Experiment with different types of Twitter ads, such as promoted tweets or promoted accounts, to see what works best for your business.
                </p>
                <h3>7. Monitor Your Competitors</h3>
                <p>
                Finally, keep an eye on your competitors on Twitter. Monitor their content and engagement to understand what's working for them. Use this information to refine your own Twitter marketing strategy and stay ahead of the competition.
                <br><br>
                In conclusion, crafting a perfect Twitter marketing strategy takes time and effort, but the results can be incredibly rewarding. By defining your goals, knowing your audience, developing a content strategy, engaging with your audience, using Twitter Analytics, experimenting with Twitter advertising, and monitoring your competitors, you can create a Twitter presence that drives business success.
                </p>
                """, unsafe_allow_html=True)




