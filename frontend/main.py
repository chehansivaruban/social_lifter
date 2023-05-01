import streamlit as st
import plotly.graph_objs as go
from sidebar import get_user_inputs


from display import (
    hide_streamlit_footer,
    display_page_header,
    style_dataframe,
    display_dataframe,
    display_container
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


from tweet_generator import (
    tweet_generator
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
from tweet_analyzer import (
    display_analyze
)

# Configure Streamlit page
st.set_page_config(
    page_title="Social Lifter",
    page_icon="🌟",
    layout="centered",
    initial_sidebar_state="expanded",
)
st.markdown("""
    <style>
        

        .stButton button {
            background-color: #1DA1F2;
            color: white;
            border-radius: 5px;
            padding: 0.5em 1em;
            transition: background-color 0.5s ease, color 0.5s ease;
        }
        .stButton button:hover {
        background-color: #9e1f63;
        transition: background-color 0.5s ease, color 0.5s ease;
        color: black !important;
        border: 1px solid #9e1f63;
        box-shadow: 0px 3px 5px rgba(0, 0, 0, 0.2);
        }
        
        .stTextInput input {
            border: 2px solid #1DA1F2;
            border-radius: 5px;
            padding: 0.5em;
        }

        .stSelectbox select {
            background-color: #f5f5f5;
            border: 2px solid #1DA1F2;
            border-radius: 5px;
            padding: 0.5em;
            color: #1DA1F2;
            font-weight: bold;
        }

        .stGraph {
            height: 500px;
        }
    </style>
""", unsafe_allow_html=True)


hide_streamlit_footer()
display_page_header()
tab1, tab2, tab3, tab4,tab5= st.tabs(["Tweet Reach Predictor","Analyze Account", "Generate Tweet","View Tweet" ,"Info"])

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
    prediction_likes, prediction_comments, prediction_retweets = get_dt_pred(date, time, isTagged, isLocation, isHashtag, isCashtag, followers, following, isVerified, account_age, average_like,subjectivity,polarity,sentiment,topic)
    likes = round(int(prediction_likes[0]))    
    comments = round(int(prediction_comments[0])) 
    retweets = round(int(prediction_retweets[0])) 
    st.write()
    display_container(text,likes,comments,retweets)
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
        st.metric("Comments", comments, "", label_visibility="hidden")
    st.markdown("---")
    predCol,predCol5, predCol6 = st.columns([5,8, 10])
    with predCol5:
        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        st.write("Retweets")
    with predCol6:
        st.metric("Retweets", retweets, "", label_visibility="hidden")
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
    font-size:1rem;
    margin-right: 50;
    position: relative;
    margin-left: 15px;
    }
</style>
'''
st.markdown(css, unsafe_allow_html=True)

with tab2:
    display_analyze()
    


with tab3:
    tweet_generator()            
   


with tab4:
    get_embedded_tweet()

with tab5:
    create_twitter_marketing_plan()
    




