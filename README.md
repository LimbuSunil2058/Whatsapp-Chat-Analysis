# 📊 WhatsApp Chat Analyzer

A powerful **WhatsApp Chat Analysis Web App** built using **Python, Streamlit, NLP, and Data Visualization**.  
This project allows you to upload exported WhatsApp chat files and generate deep insights such as activity trends, sentiment analysis, emoji usage, response time, and word frequency.

## 🚀 Live Demo: **[Click Here](https://whatsapp-chat-analysis-3rxrchqhawjiysoqgs2kyj.streamlit.app/)**

---

## 🌟 Features

### 📈 General Statistics
* **Total Messages, Words, Media, & Links:** Get a quick summary of the chat volume.
* **Activity Metrics:** Average messages per day/month and tracking of active vs. inactive days.

### ⏳ Time Series Analysis
* **Activity Over Time:** Line charts showing message volume trends over specific dates.
* **Monthly Timeline:** Visualize chat frequency over the last 12 months.
* **Busiest Hours:** Identify the "peak hours" of the day for your group.
* **Weekly Activity:** Pie chart breakdown of the busiest days of the week.

### 👥 User Insights
* **Top Contributors:** Bar charts identifying the most active members.
* **Response Time:** Analysis of how fast users reply to messages (Average Response Time).
* **Conversation Starters:** Identify who initiates conversations the most.

### 📝 Content Analysis
* **Word Cloud:** Visual representation of the most frequently used words.
* **Common Words:** Tabular view of top words (excluding stop words).
* **Emoji Analysis:** Frequency count of the most used emojis.

### 😃 Sentiment Analysis
* **Hybrid Sentiment Scoring:** Uses **NLTK VADER** (for text) and a custom **Emoji Sentiment Map** to classify messages.
* **Sentiment Distribution:** Pie chart showing the ratio of Positive, Negative, and Neutral messages.
* **Daily Trend:** Line graph tracking how sentiment changes day-to-day.
* **Weekly Heatmap:** A heatmap showing sentiment intensity across different days of the week.

---

## 🛠️ Tech Stack

**Frontend**
- Streamlit

**Backend & Analysis**
- Python
- Pandas
- Regex

**Visualization**
- Matplotlib
- Seaborn
- WordCloud

**NLP**
- NLTK (VADER Sentiment Analysis)
- Emoji Processing

**Utilities**
- URLExtract
---

## 📥 How to Use

### 1. Export WhatsApp Chat
To analyze your chat, you first need to export it from WhatsApp:
1.  Open a chat (individual or group) in WhatsApp on your phone.
2.  Tap on the **three dots** (Android) or **Contact Info** (iOS).
3.  Select **More** > **Export Chat**.
4.  Choose **Without Media** (this app analyzes text data; media files increase upload size unnecessarily).
5.  Save the generated `.txt` file.

### 2. Run the App
1.  Open the [Live Demo](https://whatsapp-chat-analysis-3rxrchqhawjiysoqgs2kyj.streamlit.app/) or run it locally.
2.  Upload the `.txt` file in the sidebar.
3.  Select a specific user or choose "Overall" for group statistics.
4.  Click **"Show Analysis"**.

---

## 💻 Local Installation

If you want to run this project on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
    cd YOUR_REPO_NAME
    ```

2.  **Create a virtual environment (Optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the app:**
    ```bash
    streamlit run app.py
    ```

---

## 📂 Project Structure

```text
├── app.py                # Main Streamlit application code
├── requirements.txt      # List of Python dependencies
├── README.md             # Project documentation
└── .gitignore            # Files to ignore in version control
