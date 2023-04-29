
import tensorflow as tf

from text_processing import (
    clean_text,
    get_tweet_padded_sequence
)

model = tf.keras.models.load_model('models/emotion/emotion_h5.h5')


def get_emotion(cleaned_text_series):
    padded_sequence = get_tweet_padded_sequence(cleaned_text_series)
    # Predict the emotion of the new sentence using the loaded model
    prediction_emotion = model.predict(padded_sequence)
        # Get the index of the predicted emotion class
    predicted_class_index = tf.argmax(prediction_emotion, axis=1).numpy()[0]
    # Define a dictionary to map the predicted class index to the emotion label
    index_to_class = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
    # Get the predicted emotion label
    predicted_emotion = index_to_class[predicted_class_index]
    return predicted_emotion