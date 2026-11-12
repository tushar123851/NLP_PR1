
# ============================================================
# AIRLINE TWEET SENTIMENT ANALYZER
# NLP FUNDAMENTALS - PR 1
# Developer: Tushar Vala
# Institute: Red & White Skill Education
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import nltk
import re
import os

from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import pos_tag
from nltk.tokenize import wordpunct_tokenize

from nlp_utils import clean_text, heuristic_sentiment


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Airline Tweet Sentiment Analyzer",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* Main application background */

.stApp {
    background: linear-gradient(
        135deg,
        #101d42 0%,
        #202c68 55%,
        #342c83 100%
    );
    color: white;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #475569;
}

section[data-testid="stSidebar"] * {
    color: #ffffff;
}

/* Main heading */

.main-title {
    font-size: 36px;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 10px;
}

.subtitle {
    font-size: 19px;
    font-weight: 600;
    color: #ffffff;
}

.description {
    font-size: 14px;
    color: #e2e8f0;
}

/* Section headings */

h1, h2, h3, h4 {
    color: #ffffff !important;
}

/* Labels */

.stTextArea label,
.stTextInput label,
.stSelectbox label,
.stRadio label,
.stCheckbox label {
    color: #ffffff !important;
    font-weight: 600 !important;
}

/* IMPORTANT: Text area input */

.stTextArea textarea {
    background-color: #ffffff !important;
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    caret-color: #000000 !important;
    font-size: 15px !important;
    border-radius: 8px !important;
    border: 1px solid #cbd5e1 !important;
}

/* Text input */

.stTextInput input {
    background-color: #ffffff !important;
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    caret-color: #000000 !important;
}

/* Placeholder */

.stTextArea textarea::placeholder,
.stTextInput input::placeholder {
    color: #777777 !important;
    -webkit-text-fill-color: #777777 !important;
    opacity: 1 !important;
}

/* Dropdown */

.stSelectbox div[data-baseweb="select"] {
    background-color: #ffffff !important;
    color: #000000 !important;
}

.stSelectbox div[data-baseweb="select"] * {
    color: #000000 !important;
}

/* Dropdown options */

div[data-baseweb="popover"] ul {
    background-color: #ffffff !important;
}

div[data-baseweb="popover"] li {
    color: #000000 !important;
}

/* Buttons */

.stButton > button {
    background: linear-gradient(
        90deg,
        #6255ef,
        #9255f5
    );
    color: #ffffff !important;
    border: none;
    border-radius: 9px;
    padding: 8px 18px;
    font-weight: 600;
    transition: 0.3s;
}

.stButton > button:hover {
    background: linear-gradient(
        90deg,
        #8b5cf6,
        #a855f7
    );
    transform: translateY(-2px);
}

/* Divider */

hr {
    border-color: #64748b !important;
    margin-top: 30px;
    margin-bottom: 30px;
}

/* Result cards */

.result-card {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 12px;
    padding: 18px;
    margin-top: 12px;
    margin-bottom: 12px;
}

/* Cleaned text */

.cleaned-text {
    background-color: #ffffff;
    color: #000000;
    padding: 15px;
    border-radius: 8px;
    font-size: 15px;
    min-height: 40px;
    overflow-wrap: anywhere;
}

/* Sentiment result */

.positive-result {
    background-color: #dcfce7;
    color: #166534;
    padding: 16px;
    border-radius: 9px;
    font-weight: bold;
}

.negative-result {
    background-color: #fee2e2;
    color: #991b1b;
    padding: 16px;
    border-radius: 9px;
    font-weight: bold;
}

.neutral-result {
    background-color: #fef3c7;
    color: #92400e;
    padding: 16px;
    border-radius: 9px;
    font-weight: bold;
}

/* Dataframe */

[data-testid="stDataFrame"] {
    background-color: #ffffff;
}

[data-testid="stDataFrame"] * {
    color: #000000 !important;
}

/* Footer */

.footer {
    text-align: center;
    color: #ffffff;
    padding: 20px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. NLTK RESOURCE SETUP
# ============================================================

@st.cache_resource
def load_nltk_resources():

    resources = [
        ("sentiment/vader_lexicon.zip", "vader_lexicon")
    ]

    for path, package in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)

    try:
        return SentimentIntensityAnalyzer()
    except Exception:
        nltk.download("vader_lexicon", quiet=True)
        return SentimentIntensityAnalyzer()


try:
    sia = load_nltk_resources()
except Exception:
    sia = None


# ============================================================
# 4. LOAD TRAINED MACHINE LEARNING MODEL
# ============================================================

@st.cache_resource
def load_ml_model():

    model_path = "sentiment_model.joblib"

    if not os.path.exists(model_path):
        return None

    try:
        model = joblib.load(model_path)
        return model

    except Exception as error:
        st.error(f"Model loading error: {error}")
        return None


model = load_ml_model()


# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def normalize_sentiment(value):

    value = str(value).strip().lower()

    if value in ["positive", "1", "pos"]:
        return "positive"

    elif value in ["negative", "0", "neg"]:
        return "negative"

    elif value in ["neutral", "2", "neu"]:
        return "neutral"

    return value


def get_sentiment_style(sentiment):

    sentiment = normalize_sentiment(sentiment)

    if sentiment == "positive":
        return "positive-result", "😊"

    elif sentiment == "negative":
        return "negative-result", "😠"

    return "neutral-result", "😐"


def analyze_heuristic(tweet):

    prediction, positive_score, negative_score = (
        heuristic_sentiment(tweet)
    )

    prediction = normalize_sentiment(prediction)

    positive_score = float(positive_score)
    negative_score = float(negative_score)

    return prediction, positive_score, negative_score


def analyze_vader(tweet):

    if sia is None:
        raise RuntimeError("VADER sentiment analyzer is unavailable.")

    scores = sia.polarity_scores(tweet)

    compound = scores["compound"]

    if compound >= 0.05:
        prediction = "positive"

    elif compound <= -0.05:
        prediction = "negative"

    else:
        prediction = "neutral"

    probabilities = {
        "negative": scores["neg"],
        "neutral": scores["neu"],
        "positive": scores["pos"]
    }

    return prediction, probabilities, scores


def analyze_machine_learning(tweet):

    if model is None:
        raise FileNotFoundError(
            "sentiment_model.joblib was not found. "
            "Please train and save your model first."
        )

    prediction = model.predict([tweet])[0]

    prediction = normalize_sentiment(prediction)

    probabilities = {}

    if hasattr(model, "predict_proba"):

        values = model.predict_proba([tweet])[0]

        classes = model.classes_

        for label, probability in zip(classes, values):

            probabilities[normalize_sentiment(label)] = (
                float(probability)
            )

    return prediction, probabilities


def get_pos_tags(tweet):

    try:

        words = wordpunct_tokenize(tweet)

        words = [
            word for word in words
            if re.search(r"[A-Za-z]", word)
        ]

        if not words:
            return pd.DataFrame(columns=["Word", "POS Tag"])

        try:
            nltk.data.find("taggers/averaged_perceptron_tagger_eng")
        except LookupError:
            nltk.download(
                "averaged_perceptron_tagger_eng",
                quiet=True
            )

        tagged_words = pos_tag(words)

        return pd.DataFrame(
            tagged_words,
            columns=["Word", "POS Tag"]
        )

    except Exception as error:

        st.warning(f"POS tagging unavailable: {error}")

        return pd.DataFrame(columns=["Word", "POS Tag"])


def build_probability_dataframe(probabilities):

    labels = ["negative", "neutral", "positive"]

    data = []

    for label in labels:

        probability = probabilities.get(label, 0.0)

        data.append({
            "Sentiment": label,
            "Probability (%)": round(probability * 100, 2)
        })

    return pd.DataFrame(data)


# ============================================================
# 6. SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ✈️ NLP Project")

    st.markdown("**Project:** Airline Tweet Sentiment Analyzer")

    st.markdown("**Dataset:** Twitter US Airline Sentiment")

    st.markdown("**Institute:** Red & White Skill Education")

    st.markdown("**Developer:** Tushar Vala")

    st.divider()

    st.markdown("### NLP Techniques")

    st.markdown("""
    - Text Cleaning
    - RegEx
    - Tokenization
    - Stopword Removal
    - Bag of Words
    - TF-IDF
    - VADER
    - Word2Vec
    - Machine Learning
    - POS Tagging
    """)

    st.divider()

    st.markdown("### Analysis Approaches")

    st.markdown("""
    1. Heuristic Sentiment
    2. VADER Sentiment
    3. Machine Learning
    """)

    st.divider()

    st.caption("NLP Fundamentals | PR 1")


# ============================================================
# 7. MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '✈️ Airline Tweet Sentiment Analyzer'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'NLP Fundamentals: From Raw Text to Predictions'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="description">'
    'Analyse airline customer feedback using Heuristic, '
    'VADER and Machine Learning approaches.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# 8. EXAMPLE TWEETS
# ============================================================

st.markdown("### 📝 Enter Your Airline Tweet")

col1, col2, col3 = st.columns(3)

with col1:

    if st.button("Positive Example", use_container_width=True):

        st.session_state.tweet_input = (
            "I love this airline! The staff was very helpful "
            "and the flight experience was amazing."
        )

with col2:

    if st.button("Negative Example", use_container_width=True):

        st.session_state.tweet_input = (
            "This was bad. My flight was delayed and "
            "the customer service was terrible."
        )

with col3:

    if st.button("Neutral Example", use_container_width=True):

        st.session_state.tweet_input = (
            "My flight from New York to Chicago is scheduled "
            "for tomorrow afternoon."
        )


# ============================================================
# 9. USER INPUT
# ============================================================

tweet = st.text_area(
    "Write or paste your tweet:",
    key="tweet_input",
    height=120,
    placeholder="Enter your airline experience here..."
)

approach = st.selectbox(
    "Choose Sentiment Analysis Approach",
    [
        "Machine Learning",
        "VADER",
        "Heuristic"
    ]
)

analyze_button = st.button(
    "🚀 Analyse Sentiment",
    use_container_width=False
)


# ============================================================
# 10. ANALYSIS
# ============================================================

if analyze_button:

    if not tweet.strip():

        st.warning("Please enter a tweet before analysing.")

    elif len(tweet) > 500:

        st.warning(
            "Your tweet is longer than 500 characters. "
            "It will be shortened for analysis."
        )

        tweet = tweet[:500]

    if tweet.strip():

        try:

            # -----------------------------------------------
            # TEXT CLEANING
            # -----------------------------------------------

            cleaned = clean_text(tweet)

            st.divider()

            st.markdown("## 📊 Analysis Results")

            st.markdown("### Cleaned Text")

            st.markdown(
                f'<div class="cleaned-text">'
                f'{cleaned if cleaned else "No text remaining after cleaning."}'
                f'</div>',
                unsafe_allow_html=True
            )

            # -----------------------------------------------
            # SELECT ANALYSIS METHOD
            # -----------------------------------------------

            probabilities = {}

            vader_scores = None

            if approach == "Machine Learning":

                prediction, probabilities = (
                    analyze_machine_learning(cleaned)
                )

            elif approach == "VADER":

                prediction, probabilities, vader_scores = (
                    analyze_vader(tweet)
                )

            else:

                prediction, positive_score, negative_score = (
                    analyze_heuristic(tweet)
                )

                total = positive_score + negative_score

                if total > 0:

                    probabilities = {
                        "positive": positive_score / total,
                        "negative": negative_score / total,
                        "neutral": 0.0
                    }

                else:

                    probabilities = {
                        "positive": 0.0,
                        "negative": 0.0,
                        "neutral": 1.0
                    }

            # -----------------------------------------------
            # PREDICTED SENTIMENT
            # -----------------------------------------------

            prediction = normalize_sentiment(prediction)

            css_class, emoji = get_sentiment_style(prediction)

            st.markdown("### 🎯 Predicted Sentiment")

            st.markdown(
                f'<div class="{css_class}">'
                f'{emoji} {prediction.upper()} SENTIMENT'
                f'</div>',
                unsafe_allow_html=True
            )

            st.caption(f"Analysis method: {approach}")

            # -----------------------------------------------
            # CLASS PROBABILITIES
            # -----------------------------------------------

            st.markdown("### 📈 Class Probabilities")

            probability_df = build_probability_dataframe(
                probabilities
            )

            chart_df = probability_df.set_index("Sentiment")

            st.bar_chart(
                chart_df,
                y="Probability (%)",
                height=300
            )

            st.dataframe(
                probability_df,
                use_container_width=True,
                hide_index=True
            )

            # -----------------------------------------------
            # VADER DETAILS
            # -----------------------------------------------

            if vader_scores is not None:

                st.markdown("### 🔍 VADER Scores")

                vader_df = pd.DataFrame(
                    [
                        {
                            "Score": key,
                            "Value": round(value, 4)
                        }
                        for key, value in vader_scores.items()
                    ]
                )

                st.dataframe(
                    vader_df,
                    use_container_width=True,
                    hide_index=True
                )

            # -----------------------------------------------
            # POS TAGGING
            # -----------------------------------------------

            st.markdown("### 🔎 POS Tagging")

            pos_df = get_pos_tags(tweet)

            if not pos_df.empty:

                st.dataframe(
                    pos_df,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.info("No POS tagging results available.")

            # -----------------------------------------------
            # DOWNLOAD ANALYSIS
            # -----------------------------------------------

            report = pd.DataFrame([
                {
                    "Original Tweet": tweet,
                    "Cleaned Text": cleaned,
                    "Analysis Approach": approach,
                    "Predicted Sentiment": prediction
                }
            ])

            csv_data = report.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="📥 Download Analysis Report",
                data=csv_data,
                file_name="airline_sentiment_report.csv",
                mime="text/csv"
            )

        except FileNotFoundError as error:

            st.error(str(error))

        except Exception as error:

            st.error(
                f"Something went wrong during analysis: {error}"
            )


# ============================================================
# 11. FOOTER
# ============================================================

st.divider()

st.markdown("""
<div class="footer">

<strong>Developed by Tushar Vala</strong>

<p>Natural Language Processing | Machine Learning | Text Classification</p>

<p>Red & White Skill Education – PR 1</p>

</div>
""", unsafe_allow_html=True)
