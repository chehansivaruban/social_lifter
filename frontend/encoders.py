import joblib
from transformers import RobertaModel, RobertaTokenizer

def load_label_encoders():
    label_encoder_day_of_week = joblib.load("label_encoder_day_of_week.pkl")
    label_encoder_language = joblib.load("label_encoder_language.pkl")
    label_encoder_clean_tweet = joblib.load("label_encoder_clean_tweet.pkl")
    label_encoder_sentiment = joblib.load("label_encoder_sentiment.pkl")
    label_encoder_key_words = joblib.load("label_encoder_key_words.pkl")
    # Load pre-trained model and tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base')
    
    return label_encoder_day_of_week, label_encoder_language, label_encoder_clean_tweet, label_encoder_sentiment, label_encoder_key_words, tokenizer, model




