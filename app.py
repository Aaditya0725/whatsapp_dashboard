import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import emoji
from collections import Counter

# Function to parse WhatsApp chat
def parse_whatsapp_chat(file):
    with open(file, "r", encoding="utf-8") as f:
        content = f.read()

    # Regex for extracting date, time, name, and message
    messages = re.findall(r"(\d{1,2}/\d{1,2}/\d{4}, \d{1,2}:\d{2} [APM]+) - (.*?): (.*)", content)
    
    # Create a DataFrame
    df = pd.DataFrame(messages, columns=["datetime", "user", "message"])
    
    # Convert datetime column to datetime type
    df["datetime"] = pd.to_datetime(df["datetime"], format="%m/%d/%Y, %I:%M %p")
    
    return df

# Function to get the most common words
def get_most_common_words(df, num=10):
    all_messages = " ".join(df["message"].dropna())
    words = re.findall(r'\w+', all_messages.lower())
    word_counts = Counter(words)
    most_common = word_counts.most_common(num)
    return most_common

# Function to extract emojis
def extract_emojis(df):
    emojis = []
    for msg in df["message"].dropna():
        emojis.extend([char for char in msg if char in emoji.UNICODE_EMOJI['en']])
    emoji_counts = Counter(emojis)
    return emoji_counts.most_common()

# Main Streamlit UI
st.title("WhatsApp Chat Analysis Dashboard")

# Upload file
uploaded_file = st.file_uploader("Upload your WhatsApp chat file (.txt)", type=["txt"])

if uploaded_file is not None:
    # Parse chat data
    df = parse_whatsapp_chat(uploaded_file)
    
    # Show basic info
    st.subheader("Chat Summary")
    st.write(f"Total Messages: {df.shape[0]}")
    st.write(f"Users: {df['user'].nunique()}")
    st.write(f"Date Range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    # Most frequent words
    st.subheader("Most Frequent Words")
    most_common_words = get_most_common_words(df)
    word_df = pd.DataFrame(most_common_words, columns=["Word", "Frequency"])
    st.write(word_df)
    
    # Plot Most Frequent Words
    st.subheader("Word Frequency Plot")
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Frequency", y="Word", data=word_df)
    st.pyplot()

    # Most used emojis
    st.subheader("Most Used Emojis")
    emoji_counts = extract_emojis(df)
    emoji_df = pd.DataFrame(emoji_counts, columns=["Emoji", "Frequency"])
    st.write(emoji_df)
    
    # Plot Most Used Emojis
    st.subheader("Emoji Frequency Plot")
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Frequency", y="Emoji", data=emoji_df)
    st.pyplot()

    # Sentiment analysis
    st.subheader("Sentiment Analysis")
    sentiments = df["message"].apply(lambda x: TextBlob(x).sentiment.polarity)
    avg_sentiment = sentiments.mean()
    st.write(f"Average Sentiment: {avg_sentiment:.2f} (positive if > 0, negative if < 0)")

    # Optional: Display a sample of the messages
    st.subheader("Sample Messages")
    st.write(df.head(10))
