import streamlit as st
import tweepy
from textblob import TextBlob
# from wordcloud import WordCloud
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import plotly.graph_objs as go
import plotly.express as px
import streamlit as st
from collections import Counter
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')

from text_processing import (
    clean_text
)

from sentiment import (
    get_polarity,
    getSentiment
)


consumerKey = "bjIuWGDX6sFYHJkRGRBcIwL39"
consumerSecret = 'GwmgitXfUFxTS0337DDN6mq7h0PhbG8D4dh2MbLMsNKgNWz9TE'
accessToken = '1435565451501768710-Nvph3hIyWxljix0fpoMJ5FgIbtmGfo'
accessTokenSecret = 'nKQ13yb9v1C6i6zpTXUdGnozkrgi5XNxv2NCijLAdREF0'

#Create the authentication object
authenticate = tweepy.OAuthHandler(consumerKey, consumerSecret) 
    
# Set the access token and access token secret
authenticate.set_access_token(accessToken, accessTokenSecret) 
    
# Creating the API object while passing in auth information
api = tweepy.API(authenticate, wait_on_rate_limit = True)

def display_analyze():
    st.title("Twitter Analysis Tool")
    st.subheader("This tool provides a way to analyze the tweets of any desired Twitter account. It does the following:")
    st.write("1. Retrieves the 5 most recent tweets from the specified Twitter handle")
    st.write("2. Creates a word density based on the content of the tweets")
    st.write("3. Conducts a sentiment analysis of the tweets and presents the results in the form of a bar graph.")
    raw_text = st.text_input("Please input the precise Twitter username of the individual you are interested in without including the '@' symbol.")
    Analyzer_choice = st.selectbox("Select the Activities",  ["Show Recent Tweets","Generate WordCloud" ,"Visualize the Sentiment Analysis"])
    if st.button("Analyze"):
        if Analyzer_choice == "Show Recent Tweets":
            st.success("Fetching last 5 Tweets")
            def Show_Recent_Tweets(raw_text):
                with st.spinner("Fetching last 5 Tweets..."):
                    posts = api.user_timeline(screen_name=raw_text, count = 100, lang ="en", tweet_mode="extended")
                def get_tweets():
                    l=[]
                    i=1
                    for tweet in posts[:5]:
                        l.append(tweet.full_text)
                        i= i+1
                    return l
                recent_tweets=get_tweets()
                return recent_tweets
            recent_tweets= Show_Recent_Tweets(raw_text)
            st.write(recent_tweets)
    elif Analyzer_choice=="Generate WordCloud":
        st.success("Generating Word Cloud")
        def gen_wordcloud():
            with st.spinner("Generating Word density..."):
                posts = api.user_timeline(screen_name=raw_text, count = 100, lang ="en", tweet_mode="extended")
            # Create a dataframe with a column called Tweets
            df = pd.DataFrame([tweet.full_text for tweet in posts], columns=['Tweets'])
            allWords = ' '.join([twts for twts in df['Tweets']])
            allWords = re.sub(r'[^a-zA-Z\s]', '', allWords)
            allWords_token = nltk.word_tokenize(allWords)
             # Filter out stopwords          
            filtered_words = [word for word in allWords_token if word.lower() not in stopwords.words('english')]
            
            # Join the filtered words back into a paragraph
            filtered_paragraph = ' '.join(filtered_words)
            filtered_paragraph_words = filtered_paragraph.split()
            word_counts = Counter(filtered_paragraph_words)
            # get the top 10 most common words
            top_words = dict(word_counts.most_common(10))
            # create a bar chart of the word frequency
            # Define the color palette
            colors = px.colors.qualitative.Pastel
            labels = list(top_words.keys())
            values = list(top_words.values())
            fig = go.Figure([go.Bar(x=labels,
                                    y=values,
                                    marker_color=colors[:len(labels)])])
            # Update the layout
            fig.update_layout(
                title='Word Density',
                xaxis_title='Words',
                yaxis_title='Density',
                xaxis_tickangle=-45
            )
            # Display the chart
            st.plotly_chart(fig)
        img=gen_wordcloud()
    elif Analyzer_choice=="Visualize the Sentiment Analysis":
        def Plot_Analysis():
            st.success("Generating Visualisation for Sentiment Analysis")
            with st.spinner("Generating Visualisation........"):
                posts = api.user_timeline(screen_name=raw_text, count = 100, lang ="en", tweet_mode="extended")
            df = pd.DataFrame([tweet.full_text for tweet in posts], columns=['Tweets'])
            df['Tweets'] = df['Tweets'].apply(clean_text)
            df['polarity'] = df['Tweets'].apply(get_polarity)
            df['Analysis'] = df['polarity'].apply(getSentiment)
            return df
        df= Plot_Analysis()
        sentiment_counts = df['Analysis'].value_counts()
        fig = go.Figure(go.Bar(
            x=sentiment_counts.values,
            y=sentiment_counts.index,
            orientation='h',
            marker=dict(color=['green', 'blue', 'red'])
        ))
        fig.update_layout(
            title='Sentiment Analysis',
            xaxis_title='Count',
            yaxis_title='Sentiment',
            yaxis=dict(autorange="reversed")
            )
        st.plotly_chart(fig, use_container_width=True)
        # st.write(sns.countplot(x=df["Analysis"],data=df))
        # st.pyplot(use_container_width=True)

            
     



