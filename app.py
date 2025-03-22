import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import emoji
from collections import Counter
from textblob import TextBlob
import altair as alt

# Function to parse WhatsApp chat
def parse_whatsapp_chat(file):
    content = file.getvalue().decode("utf-8")

    # Updated regex pattern
    pattern = r"(\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}) - ([^:]+): (.+)"
    messages = re.findall(pattern, content)

    if not messages:
        st.error("No messages found! Check chat format.")
        return pd.DataFrame(columns=["datetime", "user", "message"])

    # Create DataFrame
    df = pd.DataFrame(messages, columns=["datetime", "user", "message"])

    # Convert datetime column and format it
    df["datetime"] = pd.to_datetime(df["datetime"], format="%m/%d/%y, %H:%M", errors="coerce")
    return df.dropna()

# Function to get the most common words (excluding "Media omitted")
def get_most_common_words(df, num=10):
    all_messages = " ".join(df["message"].dropna())
    words = re.findall(r'\w+', all_messages.lower())

    # Remove unwanted words
    ignore_words = ["media", "omitted"]
    words = [word for word in words if word not in ignore_words]

    word_counts = Counter(words)
    return word_counts.most_common(num)

# Function to extract top 10 emojis
def extract_emojis(df, top_n=10):
    emojis = []
    for msg in df["message"].dropna():
        emojis.extend([char for char in msg if emoji.is_emoji(char)])
    emoji_counts = Counter(emojis)
    return emoji_counts.most_common(top_n)

# Function to get top message contributors
def get_top_contributors(df, top_n=5):
    user_counts = df["user"].value_counts().head(top_n)
    return user_counts

# Function to get daily message count
def get_daily_activity(df):
    return df.groupby(df["datetime"].dt.date).size().reset_index(name="message_count")

# Main Streamlit UI
st.title("📊 WhatsApp Chat Analysis Dashboard")

# Upload file
uploaded_file = st.file_uploader("📁 Upload your WhatsApp chat file (.txt)", type=["txt"])

if uploaded_file is not None:
    df = parse_whatsapp_chat(uploaded_file)

    if df.empty:
        st.warning("⚠️ No messages found! Try a different file.")
    else:
        st.subheader("📌 Chat Summary")

        # Format dates
        start_date = df["datetime"].min().strftime("%d %B %Y")
        end_date = df["datetime"].max().strftime("%d %B %Y")

        st.write(f"💬 **Total Messages:** {df.shape[0]}")
        st.write(f"👥 **Total Users:** {df['user'].nunique()} ({', '.join(df['user'].unique())})")
        st.write(f"📅 **Date Range:** {start_date} - {end_date}")

        # Most frequent words
        st.subheader("📖 Most Frequent Words")
        most_common_words = get_most_common_words(df)
        word_df = pd.DataFrame(most_common_words, columns=["Word", "Frequency"])
        st.write(word_df)

        # Plot Most Frequent Words
        st.subheader("📊 Word Frequency Plot")
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Frequency", y="Word", data=word_df)
        st.pyplot()

        # Most used emojis
        st.subheader("😀 Most Used Emojis")
        emoji_counts = extract_emojis(df)
        emoji_df = pd.DataFrame(emoji_counts, columns=["Emoji", "Frequency"])
        st.write(emoji_df)

        # Plot Most Used Emojis
        st.subheader("🎭 Emoji Frequency Plot")
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Frequency", y="Emoji", data=emoji_df)
        st.pyplot()

        # Top contributors
        st.subheader("🏆 Top Message Contributors")
        top_users = get_top_contributors(df)
        st.write(top_users)

        # Pie chart for top contributors
        st.subheader("📊 Top Contributors Pie Chart")
        fig, ax = plt.subplots()
        ax.pie(top_users, labels=top_users.index, autopct="%1.1f%%", startangle=90, colors=sns.color_palette("pastel"))
        st.pyplot(fig)

        # Message activity over time
        st.subheader("📈 Message Activity Over Time")
        daily_activity = get_daily_activity(df)
        line_chart = alt.Chart(daily_activity).mark_line().encode(
            x="datetime:T", y="message_count:Q"
        ).properties(width=700, height=400)
        st.altair_chart(line_chart, use_container_width=True)

        # Sentiment analysis
        st.subheader("😊 Sentiment Analysis")
        sentiments = df["message"].apply(lambda x: TextBlob(x).sentiment.polarity)
        avg_sentiment = sentiments.mean()
        st.write(f"📊 **Average Sentiment Score:** {avg_sentiment:.2f} (Positive if > 0, Negative if < 0)")

        # Optional: Display a sample of the messages
        st.subheader("📜 Sample Messages")
        st.write(df.head(10))


        st.write("Made by Chichhore 😈😎")