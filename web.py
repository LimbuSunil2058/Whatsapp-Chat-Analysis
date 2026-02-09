import streamlit as st
import pandas as pd
from urlextract import URLExtract
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from collections import Counter
import emoji
import re

st.sidebar.title("Chat Analysis")

# Preprocess
def preprocess(chat):
    data = []
    android_pattern = r"(\d{2}/\d{2}/\d{4}),\s(\d{2}:\d{2})\s-\s([^:]+):\s(.*)" 
    iphone_pattern = r"\[(\d{2}/\d{2}/\d{4}), (\d{2}:\d{2}:\d{2})\] ([^:]+): (.*)"  
    match_found=False
    for line in chat.splitlines():
        combined_pattern = re.match(android_pattern, line) or re.match(iphone_pattern,line)
        if combined_pattern:
            data.append(combined_pattern.groups())
            match_found=True
    if not match_found:
            st.warning(f"Unrecognized WhatsApp format")
            st.stop()   
        


    df = pd.DataFrame(data, columns=["Date", "Time", "User", "Messages"])


    try:
         df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"],
                                    format="%d/%m/%Y %H:%M")
    except:
        df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"],
                                        format="%d/%m/%Y %H:%M:%S")


    df["Year"] = df["Datetime"].dt.year
    df['DateOnly']=df['Datetime'].dt.date
    df["Month"] = df["Datetime"].dt.month_name()
    df['Month_Name']=df['Datetime'].dt.month
    df["Day"] = df["Datetime"].dt.day
    df['Week']=df['Datetime'].dt.day_name()
    df["Hour"] = df["Datetime"].dt.hour
    df["Minute"] = df["Datetime"].dt.minute
    # Extract emojis from each message
    def extract_emojis(text):
        return "".join([ch for ch in text if ch in emoji.EMOJI_DATA])
    df['Emojis'] = df['Messages'].apply(extract_emojis)


    return df

# remove unwanted texts, stopwords and return Frequent words in DF
@st.cache_resource
def load_nltk():
    nltk.download('stopwords')
    nltk.download('vader_lexicon')
load_nltk()
from nltk.corpus import stopwords

def remove_unwanted_messages(df):
    temp_df = df.copy()
    temp_df = df[
    (df['Messages'] != '<Media omitted>') &
    (df['Messages'] != 'This message was deleted')]

    words = []
    stop_words = set(stopwords.words('english'))
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002700-\U000027BF"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE)

    for message in temp_df['Messages']:

        message = re.sub(r'@\u2068~.*?\u2069', '', message)
        message = re.sub(r'@\u2068.*?\u2069', '', message)

        message = re.sub(r'http\S+', '', message)

        message = emoji_pattern.sub('', message)


        for word in message.lower().split():
            if len(word) > 2 and word not in stop_words:
                words.append(word)

    common=Counter(words).most_common(30)
   
    result=pd.DataFrame(common,columns=['Messages','Frequency'])
    return result

# Text sentiment analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sentiment=SentimentIntensityAnalyzer()
def sentiment_analysis(message):
    
    scores=sentiment.polarity_scores(message)['compound']
    if scores>=0.05:
        return 'Positive'
    elif scores<= -0.05:
        return 'Negative'
    else:
        return 'Neutral'
# Emoji Sentiment Analysis
def emoji_sentiment(emojies):
    emoji_sentiment_map = {

    # 🔥 Positive Emojis
    "😂": "Positive",
    "🤣": "Positive",
    "😅": "Positive",
    "😆": "Positive",
    "😊": "Positive",
    "😆": "Positive",
    "😆": "Positive",
    "😁": "Positive",
    "😄": "Positive",
    "😃": "Positive",
    "😍": "Positive",
    "🥰": "Positive",
    "😘": "Positive",
    "❤️": "Positive",
    "💖": "Positive",
    "💕": "Positive",
    "💙": "Positive",
    "💚": "Positive",
    "💜": "Positive",
    "👍": "Positive",
    "👏": "Positive",
    "🙌": "Positive",
    "🔥": "Positive",
    "🎉": "Positive",
    "🥳": "Positive",
    "😎": "Positive",
    "🤩": "Positive",
    "✨": "Positive",
    "😹": "Positive",

    #  Negative Emojis
    "😭": "Negative",
    "😢": "Negative",
    "❌": "Negative",
    "😞": "Negative",
    "😔": "Negative",
    "😡": "Negative",
    "🤬": "Negative",
    "😠": "Negative",
    "💔": "Negative",
    "😒": "Negative",
    "🙄": "Negative",
    "😤": "Negative",
    "😩": "Negative",
    "😫": "Negative",
    "🥲": "Negative",
    "😣": "Negative",
    "😖": "Negative",

    # Neutral Emojis
    "😐": "Neutral",
    "🙂": "Neutral",
    "🤔": "Neutral",
    "😶": "Neutral",
    "🙃": "Neutral",
    "🤷": "Neutral",
    "👌": "Neutral",
    "✌️": "Neutral",
    "👀": "Neutral"}
  
    sentiments = []

    for i in emojies:
            if i in emoji_sentiment_map:
                sentiments.append(emoji_sentiment_map[i])

        
    if "Negative" in sentiments:
            return "Negative"
    elif "Positive" in sentiments:
            return "Positive"
    else:
            return "Neutral"

    
    
# Statics
def fetch_stats(selected_user, df):
    words = []
    for message in df["Messages"]:
        words.extend(message.split())
    # Cuont total number of messages
    num_messages = df.shape[0]

    # Average message per day
    daily_avg=round(df.groupby(['Date']).count()['Messages'].mean(),2)

    # Average message per month
    monthly_avg=round(df.groupby(['Month']).count()['Messages'].mean(),2)

    # Count total Media shared
    media_count = df[df["Messages"] == "<Media omitted>"].shape[0]

    # Count total number of links shared
    extractor = URLExtract()
    links = []
    for message in df["Messages"]:
        links.extend(extractor.find_urls(message))

    # Find total inactive and active days
    active_days=df.groupby(['DateOnly']).count()['Messages']
    total_dates=overall_all_dates
    active_dates_only=pd.to_datetime(active_days.index)
    inactive_days=len(total_dates.difference(active_dates_only))
    total_dates=len(total_dates)
    num_active_days=len(active_days)


    top_users = None
    top_user_percentage = None

    # Find top users
    if selected_user == "Overall":
        top_users = df["User"].value_counts().head(10)
        top_user_percentage=(round(df['User'].value_counts()/df.shape[0]*100,2)).reset_index().rename(columns={'count':'percentage'})

    # Average Response Time 
    df = df.sort_values('Datetime').reset_index(drop=True)
    df['Response_Time']=df['Datetime'].diff().dt.total_seconds()/60
    df['Response_Time']=df['Response_Time'].fillna(0)
    df['is_reply'] = (df['User']!=df['User'].shift(1)) & (df['Response_Time'] > 0) & (df['Response_Time']<=60)
    replies=df[df['is_reply']==True]
    avg_reply=replies.groupby('User')['Response_Time'].mean().reset_index()
    avg_reply.columns = ['User', 'Avg Response Time (Mins)']
    avg_reply = avg_reply.sort_values(by='Avg Response Time (Mins)', ascending=True)
    avg_reply=avg_reply.head(10)

    # Create a wordcloud and df of common words
    clean_frequent_words = remove_unwanted_messages(df)
    wc = WordCloud(
    width=500,
    height=500,
    background_color="white",
    min_font_size=10)
    word_freq = dict(zip(clean_frequent_words["Messages"], clean_frequent_words["Frequency"]))# Convert word-frequency DF to dict
    if len(word_freq) > 0:
        common_words = wc.generate_from_frequencies(word_freq)
    else:
        common_words = None  


    # Show emojies 
    emojies=[]
    for message in df['Messages']:
        emojies.extend([i for i in message if i in emoji.EMOJI_DATA])
    result_emoji=pd.DataFrame(emojies).value_counts().reset_index().rename(columns={0:'Emoji','count':'Frequency'})

    
   # Show Chat Time Line
    timeline = df.groupby(['Year', 'Month', 'Month_Name']).count()['Messages'].reset_index()
    timeline['DateTime'] = pd.to_datetime(
        timeline['Year'].astype(str) + '-' + timeline['Month'].astype(str))

    timeline['DateWithMonth'] = timeline['DateTime'].dt.strftime('%B-%Y')
    timeline = timeline.sort_values('DateTime')
    timeline=timeline.tail(12)

    # Show piechart fo weekends
    weekends=df.groupby(['Week']).count()['Messages'].reset_index()

    # Busy horus of the day 
    busy_hours=df.groupby(['Hour']).count()['Messages'].reset_index().sort_values(by='Hour',ascending=True)
    def hour_conveter(hour):
        if hour==0:
            return '12 AM'
        elif hour==12:
            return "12 PM"
        elif hour<12:
            return f'{hour} AM'
        elif hour>12:
            return f'{hour-12} PM'
    busy_hours["Hour_Label"] =busy_hours['Hour'].apply(hour_conveter)

    # Convertation Stater of the day
    conversation_starter=df.groupby(['DateOnly']).first()['User'].value_counts().reset_index()
    conversation_starter=conversation_starter.head(10)

    # Daily Activity
    daily_activity=df.groupby(['DateOnly']).count()['Messages']
    
    # Text Bsed Sentiment Analysis 
    df['Sentiment']=df['Messages'].apply(sentiment_analysis) #create coloum calles sentiment

    # Emoji based Sentiment Analysis
    df['Emoji_Sentiment'] = df['Emojis'].apply(emoji_sentiment)

    # Combine Both Text sentiment and Emoji Senntiment 
    def final_sentiment(row):

        text_s = row['Sentiment']
        emoji_s = row['Emoji_Sentiment']

        if "Negative" in [text_s, emoji_s]:
            return "Negative"
        elif "Positive" in [text_s, emoji_s]:
            return "Positive"
        else:
            return "Neutral"
    df['Final_Sentiment'] = df.apply(final_sentiment, axis=1)

    count_sentiment=df['Final_Sentiment'].value_counts().reset_index()
    count_sentiment.columns = ['Final_Sentiment', 'Count']
    daily_sentiment = df.groupby(['DateOnly', 'Final_Sentiment']).size().unstack(fill_value=0)

    # Sentiment Analysis based on week
    heatmap_data = df.pivot_table(
    index='Week',
    columns='Final_Sentiment',
    values='Messages',
    aggfunc='count',
    fill_value=0)




    # Return all
    return num_messages, words,daily_avg,monthly_avg, media_count, links,inactive_days,total_dates, num_active_days, top_users,top_user_percentage,avg_reply,common_words,clean_frequent_words,result_emoji,timeline,weekends,busy_hours,conversation_starter,daily_activity,count_sentiment,daily_sentiment,heatmap_data




#  FILE UPLOAD 
uploaded_file = st.sidebar.file_uploader("Choose a WhatsApp chat file", type=["txt"])


if uploaded_file is not None:
    chat_text = uploaded_file.getvalue().decode("utf-8")
    df = preprocess(chat_text)

    
    #  USER SELECTION 
    user_list = ["Overall"] + sorted(df["User"].unique().tolist())
    selected_user = st.sidebar.selectbox("Select User", user_list)

    # Fetch total dates from starting to end before filtering
    overall_start_date = df['DateOnly'].min()
    overall_end_date = df['DateOnly'].max()
    overall_all_dates = pd.date_range(start=overall_start_date, end=overall_end_date)


    if selected_user != "Overall":
        df = df[df["User"] == selected_user]

    #  ANALYSIS
    if st.sidebar.button("Show Analysis"):
        st.title("Your Whatsapp Chat Statistics Is Here")
        with st.spinner("Analyzing chat..."):
            num_msgs, words,daily_avg,monthly_avg, media, links, inactive_days, total_dates, num_active_days, top_users,top_user_percentage,avg_reply,common_words,clean_frequent_words,result_emoji,timeline,weekends,busy_hours,conversation_starter,daily_activity,count_sentiment,daily_sentiment,heatmap_data= fetch_stats(selected_user, df)
        st.success("Analysis completed")
        col1, col2, col3, col4= st.columns(4)

        col1.metric("Total Messages", num_msgs)
        col2.metric("Total Words", len(words))
        col3.metric("Number Of Media Shared", media)
        col4.metric("Number Of Links Shared", len(links))

        col1, col2,col3,col4=st.columns(4)
        col1.metric("Average Message Per Day",(daily_avg))
        col2.metric("Average Message Per Month",monthly_avg)
        col3.metric("Total Active Days",f"{num_active_days}/{total_dates}")
        col4.metric("Total Inactive Days",f"{inactive_days}/{total_dates}")
        

       
        # Show Daily Activites
        st.text("."*176)
        st.header("Activity Over Time")
        fig,ax=plt.subplots()
        sns.lineplot(
        x=daily_activity.index,
        y=daily_activity.values,
        ax=ax)
        plt.xticks(rotation=45)
        plt.xlabel('Date')
        st.pyplot(fig)

        
        # Show busy hours of a day 
        st.text("."*176)
        st.header("Busiest Time of The Day")
        fig,ax=plt.subplots()
        sns.lineplot(busy_hours,x=busy_hours['Hour_Label'],y=busy_hours['Messages'],color='#FF6B6B',ax=ax)
        # plt.title("Most Busy Hours")
        plt.xlabel('Time')
        plt.xticks(rotation=80)
        plt.grid()
        st.pyplot(fig)


        # Timeline of last 12 months and Busiest day of the week
        st.text("."*176)
        col1,col2=st.columns(2)

        with col1:
            # Show timeline
            st.header("Time Line of last 12 Months")
            fig,ax=plt.subplots()
            sns.lineplot(timeline,x=timeline['DateWithMonth'],y=timeline['Messages'],ax=ax)
            plt.xlabel('Date')
            plt.ylabel('Frequency of Messages')
            plt.grid()
            plt.xticks(rotation=45)
            st.pyplot(fig)

        with col2:
            # Show piechart of busy weekends
            st.header("Busiest Day of The Week")
            fig,ax=plt.subplots(figsize=(4,4))
            ax.pie(weekends['Messages'],labels=weekends['Week'],autopct='%1.1f%%',colors=sns.color_palette('Set2'))
            # ax.set_title("Weekly Chat Activity Distribution")
            ax.axis('equal')
            st.pyplot(fig)



        
        #  TOP USERS GRAPH 
        if selected_user == "Overall":
            st.text("."*176)
            col1,col2=st.columns(2)
            with col1:
                st.subheader("Top Active Users")
                fig, ax = plt.subplots(figsize=(3,3))
                ax.bar(top_users.index, top_users.values,color='b')
                ax.set_xlabel("Users")
                ax.set_ylabel("Message Count")
                ax.set_title("User vs Message Count")
                plt.xticks(rotation=90)

                st.pyplot(fig)
            with col2:
                st.subheader("Top Chat Contributor In Percentage")
                st.data_editor(top_user_percentage)
        
        
        # Average Response Time 

        if selected_user=='Overall':
            st.text("."*176)
            st.header("Average Response Time of Users")
            
            col1 , col2 = st.columns(2)
            
            with col1:
                fig,ax=plt.subplots(figsize=(3,3))
                sns.barplot(x=avg_reply['User'],y=avg_reply['Avg Response Time (Mins)'])
                plt.xticks(rotation=90)
                st.pyplot(fig)
            with col2:
                st.data_editor(avg_reply)
         # Show conversation stater
        if selected_user=='Overall':
            st.text("."*176)
            st.header("Conversation Staters")
            col1,col2=st.columns(2)
            with col1:
                fig,ax=plt.subplots(figsize=(3,3))
                sns.barplot(conversation_starter,x=conversation_starter['User'],y=conversation_starter['count'],color='g')
                plt.xlabel("User Name")
                plt.xticks(rotation=90)
                st.pyplot(fig)
            with col2:
                st.data_editor(conversation_starter)

        
        # show common words and emojies
        st.text("."*176)
        col1, col2=st.columns(2)
        # Common words
        with col1:
            st.header("Common Words")
            st.data_editor(clean_frequent_words)


        # Emojies
        with col2:
            st.header("Common emojies")
            st.data_editor(result_emoji)

        
        
        # Show Common Words In WordCloud
        st.text("."*176)
        st.header("Most Common Words in WordCloud")
        fig,ax=plt.subplots()
        if common_words is not None:
            ax.imshow(common_words)
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.info("No words available to generate WordCloud for this user.")


       
        # Sentiment Analysis show 

        st.text("."*176)
        st.header("Sentiment Analysis")
        st.subheader("Emotion Insight (Text + Emoji Combined)")
        col1,col2=st.columns(2)
        # Pie chart
        with col1:
            fig, ax = plt.subplots(figsize=(10,10))

            colors = [ '#4D96FF', '#6BCB77','#FF6B6B'] 
            ax.pie(
                count_sentiment['Count'],
                labels=count_sentiment['Final_Sentiment'],
                autopct='%1.1f%%',
                colors=colors,
                startangle=90,
                labeldistance=1.05,
                pctdistance=0.75,
                textprops={'fontsize':14, 'weight':'bold'} 
            )
            ax.set_title("Sentiment Distribution", fontsize=18, weight='bold')
            ax.axis('equal') 
            st.pyplot(fig)

        # line graph
        with col2:
            fig, ax = plt.subplots(figsize=(6,6))
            daily_sentiment.plot(
                ax=ax,
                marker='o',
                linewidth=2,
                markersize=5
            )
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Messages', fontsize=12)
            ax.set_title("Daily Sentiment Trend", fontsize=16, weight='bold')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

        # Sentiment Analysis in Heatmap
        st.text("."*176)
        st.header("Sentiment Analysis Based on Week")
        fig,ax=plt.subplots()
        sns.heatmap(heatmap_data,ax=ax)
        ax.set_xlabel("Sentiment")
        st.pyplot(fig)