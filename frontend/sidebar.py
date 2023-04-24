import streamlit as st

def get_user_inputs():
    with st.sidebar.form("tweet"):
        text = st.text_area("Enter your Tweet")
        date = st.date_input("Enter your Date")
        time = st.time_input("Enter your Time")

        isTagged = st.selectbox("Users Tagged ?", options=("True", "False"))
        isLocation = st.selectbox("Location Provided ?", options=("True", "False"))
        isHashtag = st.selectbox("Is Hashtag available ?", options=("True", "False"))
        isCashtag = st.selectbox("Is Cashtag available ?", options=("True", "False"))

        followers = st.number_input("Enter No. of Followers",step=1)
        following = st.number_input("Enter No. of Following",step=1)
        isVerified = st.selectbox("Is your account verified?", options=("Verified", "Not Verified"))
        account_age = st.number_input("How old is your account?",step=1)
        average_like = st.number_input("Whats the average likes that you get?",step=1)

        btn = st.form_submit_button("Predict Reach")
        
    return text, date, time, isTagged, isLocation, isHashtag, isCashtag, followers, following, isVerified, account_age, average_like, btn
