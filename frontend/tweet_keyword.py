import streamlit as st
from keybert import KeyBERT
import re
import pandas as pd
import neattext.functions as nfx
from pandas import DataFrame


from text_processing import (
    clean_text,
)
# Define function to load KeyBERT model
@st.cache_data()
def load_model():
    return KeyBERT()

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
