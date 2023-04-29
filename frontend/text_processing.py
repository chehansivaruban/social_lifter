import string
from keras.utils.data_utils import pad_sequences
import re
import pickle
from gensim.utils import simple_preprocess


with open('models/emotion/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    
def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_tweet_padded_sequence(tweet_text):
    # Define the tokenizer and maxlen
   
    def preprocess_sentence(tweet_text):
    # Tokenize the sentence
        sequences = tokenizer.texts_to_sequences([tweet_text])
        # Pad the sequence
        print(sequences)
        padded_sequence = pad_sequences(sequences, maxlen=50, padding='post', truncating='post')
        return padded_sequence
    # Define the sentence to predict the emotion
    new_sentence = tweet_text

    # Preprocess the new sentence
    padded_sequence = preprocess_sentence(tweet_text)
    return padded_sequence

