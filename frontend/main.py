import streamlit as st
import plotly.graph_objs as go
from sidebar import get_user_inputs
import numpy as np
import tensorflow as tf


from display import (
    hide_streamlit_footer,
    display_page_header,
    style_dataframe,
    display_dataframe,
    display_container
)

from encoders import (
    load_label_encoders
)
from info import (
    create_twitter_marketing_plan
)
from topic import (
    get_tweet_topic
)

from regression_model import (
    get_dt_pred
)

from emotion import (
    get_emotion
)
from sentiment import (
    get_polarity_subjective_sentimet
)
from tweet_keyword import (
    extract_keywords,
    keyword_dataframe
)

from text_processing import (
    clean_text
)

from embed_tweet import (
    get_embedded_tweet
)

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
    cleaned_text_series = clean_text(text)
    print(f'clean : {cleaned_text_series}')
    subjectivity,polarity,sentiment = get_polarity_subjective_sentimet(cleaned_text_series)
    emotion = get_emotion(cleaned_text_series)
    st.write(emotion)
    extracted_keywords = extract_keywords(text)
    topic = get_tweet_topic(cleaned_text_series)
    st.write(topic)
    prediction = get_dt_pred(date, time, isTagged, isLocation, isHashtag, isCashtag, followers, following, isVerified, account_age, average_like,subjectivity,polarity,sentiment,topic)
    likes = round(int(prediction[0]))       
    st.write()
    display_container(text,likes)
    st.markdown("<br>",unsafe_allow_html=True)
    st.markdown("---")
    predCol,predCol1, predCol2 = st.columns([5,8, 10])
    with predCol1:
        st.markdown("<br>",unsafe_allow_html=True)
        st.write("Likes")
    with predCol2:
        st.metric("Likes", likes, likes - average_like, label_visibility="hidden")
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
    df_key = keyword_dataframe(extracted_keywords)
    df_key.index += 1
    df_key_style =style_dataframe(df_key)
    display_dataframe(df_key_style)
    
    
    

    # st.write(prediction)
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
    # st.write(likes)
    # st.write(comments)
    # st.write(retweets)
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

with tab2:
    get_embedded_tweet()


# ------------------------info tab ------------------

with tab3:
    create_twitter_marketing_plan()
    




