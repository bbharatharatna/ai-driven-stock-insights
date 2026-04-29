import streamlit as st
import feedparser
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import spacy
from wordcloud import WordCloud
from collections import Counter
import re
import yake
from sklearn.feature_extraction.text import CountVectorizer
from fpdf import FPDF
import unicodedata
import pandas as pd

# Professional Color Scheme
TEAL_GREEN = "#008080"
CHARCOAL_GREY = "#36454F"
LIGHT_TEAL = "#B2DFDB"
ACCENT_GREY = "#F5F5F5"
MUTED_GREY = "#E8E8E8"  # Softer, more muted grey for the help box
TEXT_MUTED = "#666666"  # Muted text color

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main app styling */
    .main > div {
        padding-top: 2rem;
    }
   
    /* Headers styling */
    .css-10trblm {
        color: #008080;
    }
   
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #F5F5F5;
    }
   
    /* Button styling */
    .stButton > button {
        background-color: #008080;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
   
    .stButton > button:hover {
        background-color: #36454F;
        color: white;
    }
   
    /* Download button styling */
    .stDownloadButton > button {
        background-color: #36454F;
        color: white;
        border: none;
        border-radius: 5px;
    }
   
    .stDownloadButton > button:hover {
        background-color: #008080;
    }
   
    /* Slider styling */
    .stSlider > div > div > div > div {
        background-color: #008080;
    }
   
    /* Text input styling */
    .stTextInput > div > div > input {
        border: 2px solid #B2DFDB;
        border-radius: 5px;
    }
   
    .stTextInput > div > div > input:focus {
        border-color: #008080;
        box-shadow: 0 0 0 0.2rem rgba(0, 128, 128, 0.25);
    }
   
    /* Dataframe styling */
    .dataframe {
        border: 1px solid #B2DFDB;
    }
   
    /* Success message styling */
    .stSuccess {
        background-color: #B2DFDB;
        border-left: 4px solid #008080;
    }
   
    /* Warning message styling */
    .stWarning {
        border-left: 4px solid #36454F;
    }
   
    /* Info message styling */
    .stInfo {
        background-color: #F5F5F5;
        border-left: 4px solid #008080;
    }
</style>
""", unsafe_allow_html=True)

# Set matplotlib and seaborn styling
plt.style.use('default')
sns.set_palette([TEAL_GREEN, CHARCOAL_GREY, LIGHT_TEAL, "#4DB6AC", "#80CBC4"])
sns.set_style("whitegrid")
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': CHARCOAL_GREY,
    'axes.labelcolor': CHARCOAL_GREY,
    'xtick.color': CHARCOAL_GREY,
    'ytick.color': CHARCOAL_GREY,
    'text.color': CHARCOAL_GREY,
    'grid.color': '#E0E0E0',
    'font.family': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
})

@st.cache_resource
def load_finbert():
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return nlp

@st.cache_resource
def load_bart():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")

@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

finbert_nlp = load_finbert()
summarizer = load_bart()
spacy_nlp = load_spacy()

def get_article_links_from_rss(rss_url, max_articles=3):
    feed = feedparser.parse(rss_url)
    return [entry.link for entry in feed.entries[:max_articles]]

def scrape_article_bs4(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        title = soup.title.string.strip() if soup.title else "No Title"
        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text().strip() for p in paragraphs if len(p.get_text()) > 50])
        return title + ". " + content[:1500]
    except Exception as e:
        return f"Error: {str(e)}"

def filter_articles_by_company(articles, company_name, ticker):
    filtered = []
    company_query = company_name.lower()
    ticker_query = ticker.lower()
    for art in articles:
        text = art.lower()
        if company_query in text or ticker_query in text:
            filtered.append(art)
    return filtered

def analyze_finbert(texts):
    results = finbert_nlp(texts)
    return [(text, res['label'], res['score']) for text, res in zip(texts, results)]

def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="7d")
    return hist

def extract_keywords(text, top_n=10):
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    stop_words = set([
        'company', 'market', 'share', 'stock', 'price', 'would', 'could', 'said', 'this', 'that', 'about', 'with', 'from'
    ])
    filtered = [w for w in words if w not in stop_words]
    common = Counter(filtered).most_common(top_n)
    return common

def extract_keywords_yake(text, max_keywords=10):
    kw_extractor = yake.KeywordExtractor(top=max_keywords, stopwords=None)
    keywords = kw_extractor.extract_keywords(text)
    return [kw for kw, _ in keywords]

def extract_top_keywords(articles, top_n=10):
    vectorizer = CountVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(articles)
    sum_words = X.sum(axis=0)
    keywords = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    keywords = sorted(keywords, key=lambda x: x[1], reverse=True)[:top_n]
    return keywords

def extract_named_entities(text):
    doc = spacy_nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def summarize_article_hf(text, max_length=130, min_length=30):
    try:
        if not text or not text.strip():
            return "No content to summarize."
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        if isinstance(summary, list) and len(summary) > 0 and 'summary_text' in summary[0]:
            return summary[0]['summary_text']
        else:
            return "Summarization failed (unexpected output)."
    except Exception as e:
        return f"Error summarizing: {str(e)}"

def avg_sentiment_score(articles):
    sentiment_map = {'POSITIVE': 1, 'NEUTRAL': 0, 'NEGATIVE': -1}
    scores = []
    for article in articles:
        try:
            result = finbert_nlp(article[:512])[0]
            scores.append(sentiment_map[result['label']] * result['score'])
        except Exception:
            continue
    if scores:
        return round(sum(scores) / len(scores), 3)
    else:
        return 0.0

def clean_text_for_pdf(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')

def export_report_to_pdf(articles, summary_text, filename="Stock_Report.pdf"):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, "AI-Driven Stock Insights: Final Report", ln=True)
        pdf.ln(10)
        clean_summary = clean_text_for_pdf(summary_text)
        pdf.multi_cell(0, 10, "Summary:\n" + clean_summary + "\n")
        pdf.ln(5)
        for i, article in enumerate(articles[:3], 1):
            preview = clean_text_for_pdf(article[:700].replace('\n', ' ') + "...")
            pdf.multi_cell(0, 10, f"Article {i} Preview:\n{preview}\n")
            pdf.ln(3)
        pdf.output(filename)
        return filename
    except Exception as e:
        return f"Error generating PDF: {str(e)}"

def plot_volatility(stock_data):
    if 'Close' not in stock_data.columns:
        st.warning("No stock data for volatility plot.")
        return
    changes = stock_data['Close'].pct_change().dropna() * 100
    fig, ax = plt.subplots(figsize=(10, 6))
    changes.plot(kind='bar', color=TEAL_GREEN, ax=ax, alpha=0.8)
    ax.set_ylabel('Daily % Change', fontweight='bold')
    ax.set_title('Daily Volatility (%)', fontweight='bold', color=CHARCOAL_GREY)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

def display_entity_types(entities):
    ent_df = pd.DataFrame(entities, columns=["Entity", "Type"])
    st.write("**Named Entities by Type:**")
    for ent_type in ["ORG", "PERSON", "GPE", "DATE"]:
        matches = ent_df[ent_df["Type"] == ent_type]["Entity"].unique()
        if len(matches) > 0:
            st.write(f"**{ent_type}:**", ", ".join(matches[:5]))

st.set_page_config(
    page_title="AI-Driven Stock Insights",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title with professional styling
st.markdown(f"""
<h1 style='color: {TEAL_GREEN}; text-align: center; margin-bottom: 2rem;'>
    🚀 AI-Driven Stock Insights: Analyzing Financial News with LLMs
</h1>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown(f"""
    <h2 style='color: {CHARCOAL_GREY}; border-bottom: 2px solid {TEAL_GREEN}; padding-bottom: 10px;'>
        ⚙️ Configuration
    </h2>
    """, unsafe_allow_html=True)
   
    company_name = st.text_input("Company Name", value="Apple")
    ticker = st.text_input("Stock Ticker", value="AAPL")
    st.markdown("Optionally, enter a ticker-specific RSS feed for best results (otherwise, filtering will be used):")
    rss_url = st.text_input(
        "News RSS Feed",
        value=""
    )
    default_general_feed = "https://finance.yahoo.com/news/rssindex"
    max_articles = st.slider("Number of News Articles", 1, 10, 3)

if st.button("🔍 Analyze", key="analyze_btn"):
    # --- News Scraping ---
    st.markdown(f"<h3 style='color: {TEAL_GREEN};'>📰 1. Latest Financial News</h3>", unsafe_allow_html=True)
    effective_rss = rss_url if rss_url.strip() else default_general_feed
    news_urls = get_article_links_from_rss(effective_rss, max_articles=max_articles)
    articles = [scrape_article_bs4(url) for url in news_urls]

    # Filter for company
    filtered_articles = filter_articles_by_company(articles, company_name, ticker)
    if filtered_articles:
        st.warning("Showing news matched to company name/ticker.")
        display_articles = filtered_articles
    else:
        st.info("No company-matched articles found. Showing latest finance news.")
        display_articles = articles

    for i, art in enumerate(display_articles):
        st.markdown(f"**Article {i+1} Preview:** {art[:300]}...")

    # --- Sentiment Analysis ---
    st.markdown(f"<h3 style='color: {TEAL_GREEN};'>📊 2. Sentiment Analysis (FinBERT)</h3>", unsafe_allow_html=True)
    sentiments = analyze_finbert(display_articles)
    df_sent = pd.DataFrame(sentiments, columns=["Article", "Sentiment", "Score"])
    st.dataframe(df_sent[["Sentiment", "Score"]])

    # --- Stock Data ---
    st.markdown(f"<h3 style='color: {TEAL_GREEN};'>📈 3. {company_name} ({ticker}) Stock Data (Last 7 Days)</h3>", unsafe_allow_html=True)
    stock_data = fetch_stock_data(ticker)
    st.dataframe(stock_data.tail(7))

    # --- Visualization: Stock Price ---
    st.markdown(f"<h3 style='color: {TEAL_GREEN};'>📉 4. Stock Price Trend</h3>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    stock_data['Close'].plot(ax=ax, title=f"{company_name} Stock Price", color=TEAL_GREEN, linewidth=2)
    ax.set_title(f"{company_name} Stock Price", fontweight='bold', color=CHARCOAL_GREY)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    # --- Feature 1: Stock Volatility Visualization ---
    st.markdown(f"<h3 style='color: {TEAL_GREEN};'>⚡ 5. Stock Volatility</h3>", unsafe_allow_html=True)
    plot_volatility(stock_data)

    # --- Visualization: Sentiment Distribution ---
    st.markdown(f"<h3 style='color: {TEAL_GREEN};'>🎯 6. Sentiment Distribution</h3>", unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.countplot(data=df_sent, x="Sentiment", palette=[TEAL_GREEN, CHARCOAL_GREY, LIGHT_TEAL], ax=ax2)
    ax2.set_title('Sentiment Distribution', fontweight='bold', color=CHARCOAL_GREY)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2)

    # --- Keyword Extraction ---
    st.markdown(f"<h3 style='color: {TEAL_GREEN};'>🔑 7. Top Keywords (Basic, YAKE, CountVectorizer)</h3>", unsafe_allow_html=True)
    merged_text = " ".join(display_articles)
    basic_keywords = extract_keywords(merged_text)
    yake_keywords = extract_keywords_yake(merged_text)
    vectorizer_keywords = extract_top_keywords(display_articles)
    st.write("**Basic:**", basic_keywords)
    st.write("**YAKE:**", yake_keywords)
    st.write("**CountVectorizer:**", vectorizer_keywords)

    # --- Named Entity Extraction ---
    st.markdown(f"<h3 style='color: {TEAL_GREEN};'>🏷️ 8. Named Entities (spaCy)</h3>", unsafe_allow_html=True)
    entities = extract_named_entities(merged_text)
    st.write(entities[:20])

    # --- Feature 2: Dates and Categories of Named Entities ---
    display_entity_types(entities)

    # --- Word Cloud ---
    st.markdown(f"<h3 style='color: {TEAL_GREEN};'>☁️ 9. Word Cloud</h3>", unsafe_allow_html=True)
    wc = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(merged_text)
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    ax3.imshow(wc, interpolation='bilinear')
    ax3.axis("off")
    plt.tight_layout()
    st.pyplot(fig3)

    # --- Summarization ---
    st.markdown(f"<h3 style='color: {TEAL_GREEN};'>📋 10. News Summary (BART)</h3>", unsafe_allow_html=True)
    summary_text = summarize_article_hf(merged_text[:1000])
    st.write(summary_text)

    # --- Sentiment vs Stock Price ---
    st.markdown(f"<h3 style='color: {TEAL_GREEN};'>⚖️ 11. Sentiment vs. Stock Price Change</h3>", unsafe_allow_html=True)
    avg_score = avg_sentiment_score(display_articles)
    pct_change = stock_data['Close'].pct_change().dropna().mean() * 100
   
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Avg Sentiment Score", f"{avg_score}")
    with col2:
        st.metric("Avg Stock Price Change (7d)", f"{pct_change:.2f}%")

    # --- PDF Report Export ---
    st.markdown(f"<h3 style='color: {TEAL_GREEN};'>📄 12. Export Report</h3>", unsafe_allow_html=True)
    pdf_file = export_report_to_pdf(display_articles, summary_text)
    if pdf_file and not pdf_file.startswith("Error"):
        with open(pdf_file, "rb") as f:
            st.download_button("📥 Download PDF Report", f, file_name=pdf_file, mime="application/pdf")
    else:
        st.write("Error generating PDF.")
    st.success("✅ Analysis Complete!")

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style='background-color: {MUTED_GREY}; padding: 15px; border-radius: 10px; border-left: 3px solid {TEAL_GREEN}; margin-top: 10px;'>
<h4 style='color: {TEXT_MUTED}; margin-top: 0; font-size: 14px;'>📖 How to use:</h4>
<p style='color: {TEXT_MUTED}; margin-bottom: 5px; font-size: 12px;'>1. Enter company name, ticker, and optionally a company-specific RSS feed.</p>
<p style='color: {TEXT_MUTED}; margin-bottom: 5px; font-size: 12px;'>2. Click 'Analyze' to run the workflow.</p>
<p style='color: {TEXT_MUTED}; margin-bottom: 0; font-size: 12px;'>3. Download the PDF report if needed.</p>
</div>
""", unsafe_allow_html=True)