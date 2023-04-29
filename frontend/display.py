import streamlit as st
import pandas as pd
import seaborn as sns

def hide_streamlit_footer():
    st.markdown("""
                <style>
                .css-cio0dv.egzxvld1{
                    visibility:hidden;
                }
                </style>
                """,unsafe_allow_html=True)

#  Display page header
def display_page_header():
    st.markdown("<h1 style = 'text-align: center;'>Social Lifter</h1>", unsafe_allow_html=True)
    st.markdown("---")

# Display tweet reach predictor section header
def display_tweet_reach_predictor_section_header():
    st.markdown("<h2>Tweet Reach Predictor</h2>", unsafe_allow_html=True)
    
# Define the styling functions

def style_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cmGreen = sns.light_palette("green", as_cmap=True)
    cmRed = sns.light_palette("red", as_cmap=True)
    df = df.style.background_gradient(
        cmap=cmGreen,
        subset=["Relevancy"],
    )
    return df


def display_dataframe(df: pd.DataFrame) -> None:
    c1, c2, c3 = st.columns([1, 3, 1])

    format_dictionary = {
        "Relevancy": "{:.1%}",
    }

    df_key = df.format(format_dictionary)

    with c2:
        st.table(df_key)

def display_container(text,likes):
  text = text.replace('\n', ' ')
  st.markdown(f"""
    <style>
        .tweet-container {{
          display: flex;
          align-items: flex-start;
          border: 1px solid #ccc;
          padding: 10px;
          border-radius: 8px;
          position: relative;
        }}

        .tweet-container img {{
          width: 50px;
          height: 50px;
          border-radius: 50%;
          margin-right: 10px;
          margin-top: 10px;
        }}
        .tweet-header h2 {{
          margin-right: -40px;
          
        }}

        .tweet-content {{
          display: flex;
          flex-direction: column;
          flex-grow: 1;
        }}

        .tweet-header {{
          display: flex;
          align-items: center;
          flex-direction:row;
          justify-content:left;
        }}

        .profile-name {{
          font-size: 16px;
          margin-left: 20px;
        }}

        .username {{
          font-size: 14px;
          color: #555;
        }}

        .tweet-text {{
          font-size: 16px;
          margin-top: 10px;
          position: relative;
        }}

        .tweet-metrics {{
          display: flex;
          justify-content: space-between;
          margin-top: 10px;
        }}

        .metric {{
            font-size: 14px;
            color: #fff;
            display: flex;
            align-items: center;
            margin-right: 100px;
            margin-left: 10px;
        }}
    </style>

    <div class="tweet-container">
      <img src="https://picsum.photos/50" alt="Profile picture">
      <div class="tweet-content">
        <div class="tweet-header">
          <h2 class="profile-name">John Doe</h2>
          <span class="username">@johndoe</span>
        </div>
        <p class="tweet-text">{text}</p>
        <div class="tweet-metrics">
          <span class="metric">{likes} likes &#x2764;</span>
          <span class="metric">50 comments &#x1F4AC;</span>
          <span class="metric">20 retweets &#x1F501;&#xFE0E;</span>
        </div>
      </div>
    </div>
""", unsafe_allow_html=True)
    