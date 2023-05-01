import joblib
from datetime import datetime
import pandas as pd


dt_reg_likes = joblib.load("models/final_model/dt/dt_reg_likes_best_30_04_02.pkl")
dt_reg_comments = joblib.load("models/final_model/random/rf_reg_comments_01_05_01.pkl")
dt_reg_retweets = joblib.load("models/final_model/dt/dt_reg_retweets_best_01_05_01.pkl")

def get_dt_pred(date, time, isTagged, isLocation, isHashtag, isCashtag, followers, following, isVerified, account_age, average_like,subjectivity,polarity,sentiment,topic):
    isTagged = 1 if isTagged == "True" else 0  
    isLocation = 1 if isLocation == "True" else 0  
    isVerified = 1 if isVerified == "True" else 0  
    isHashtag = 1 if isHashtag == "True" else 0  
    isCashtag = 1 if isCashtag == "True" else 0  
    date = date.strftime("%A")
    time = datetime.strptime(str(time), '%H:%M:%S').strftime('%H')
    day_of_week_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    day_of_week_encoded = day_of_week_map[date]
    # Define the mapping from sentiment string to integer
    sentiment_map = {'positive': 0, 'negative': 1, 'neutral': 2}
    sentiment_encoded = sentiment_map[sentiment]
    Followers_to_Following = followers / following
    inputs = pd.DataFrame({
    "time": [time],  # add missing value
    "Day of week": [day_of_week_encoded],
    "Cashtags": [isCashtag],  # add missing value
    "Hashtags": [isHashtag],  # add missing value
    "Location": [isLocation],  # add missing value
    "Mentioned_users": [isTagged],  # add missing value
    "Followers": [followers],
    "Following": [following],
    "Verified": [isVerified],
    "Average_favourite_count": [average_like],
    "account_age": [account_age],
    "subjectivity": [subjectivity],
    "polarity": [polarity],
    "sentiment": [sentiment_encoded],
    "topics": [topic],  # add missing value
    "Followers_to_Following": [Followers_to_Following.round(2)],  # add missing value
    })

    df = pd.DataFrame(inputs)
    prediction_likes = dt_reg_likes.predict(df)
    prediction_comments = dt_reg_comments.predict(df)
    prediction_retweets = dt_reg_retweets.predict(df)
    return prediction_likes, prediction_comments, prediction_retweets
    
    