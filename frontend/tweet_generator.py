import openai
import streamlit as st

# Set OpenAI API key
openai.api_key = "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"


# Function to generate tweet
def generate_tweet(prompt):
    completions = openai.Completion.create(
        engine="text-davinci-002", #"text-davinci-003"
        prompt=prompt,
        max_tokens=280,
        n=1,
        stop=None,
        temperature=0.5,
    )

    message = completions.choices[0].text
    return message.strip()

def tweet_generator():
    # Front-end using Streamlit syntax
    st.title("Tweet Generator")
    content = st.text_input("Enter the content of tweet you want to generate")
    mood_of_content = st.text_input('Enter the type of tweet:')
    fprompt = f"Write a {mood_of_content} tweet about {content} within 120 characters with hashtags."

    # Generate Tweet
    # check box so that app's state is preserved
    if st.button("Generate"): 
        tweet = generate_tweet(fprompt)
        st.info(tweet)