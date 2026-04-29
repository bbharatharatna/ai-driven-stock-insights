AI-Driven Stock Insights

An intelligent stock analysis web application that combines real-time financial data
with Natural Language Processing (NLP) to deliver AI-powered news sentiment analysis,
stock price visualization, and financial summaries.

Live Demo: https://bbharatharatna-ai-driven-stock-insights-app-43h2bm.streamlit.app/
GitHub: https://github.com/bbharatharatna/ai-driven-stock-insights


FEATURES

- Real-Time Stock Data: Fetches live stock prices and historical data using Yahoo Finance
- News Sentiment Analysis: Classifies financial news as Positive, Negative, or Neutral using FinBERT
- AI News Summarization: Summarizes news articles using DistilBART (distilbart-cnn-6-6)
- Keyword Extraction: Extracts key financial topics from news using YAKE
- Named Entity Recognition: Identifies companies, people, and entities using spaCy
- Stock Price Chart: Visualizes historical closing prices using Matplotlib
- Stock Volatility Chart: Plots daily percentage change to highlight market volatility
- PDF Report Generation: Exports the full analysis as a downloadable PDF using FPDF
- Word Cloud: Generates a visual word cloud from news article content


AI MODELS USED

Model                          Purpose                         Source
ProsusAI/finbert               Financial sentiment analysis    Hugging Face
sshleifer/distilbart-cnn-6-6   News summarization              Hugging Face
en_core_web_sm                 Named entity recognition        spaCy


TECH STACK

Category           Tools
Frontend/UI        Streamlit
Data               yfinance, feedparser, BeautifulSoup4
NLP and AI         Transformers, FinBERT, DistilBART, spaCy, YAKE
Visualization      Matplotlib, Seaborn, WordCloud
Report             FPDF
Language           Python 3.11


INSTALLATION

1. Clone the repository
   git clone https://github.com/bbharatharatna/ai-driven-stock-insights.git
   cd ai-driven-stock-insights

2. Install dependencies
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm

3. Run the app
   streamlit run app.py


HOW TO USE

1. Open the app using the Live Demo link or run it locally
2. In the Configuration sidebar, enter the Company Name, Stock Ticker,
   optional News RSS Feed URL, and number of articles to analyze
3. Click Analyze to generate sentiment analysis, article summaries,
   stock charts, word cloud, keyword extraction, and a downloadable PDF report


KNOWN LIMITATIONS

- Yahoo Finance may occasionally rate-limit requests. Wait 1 to 2 minutes and retry.
- First load may take 30 to 60 seconds as AI models are downloaded and cached.
- Streamlit Community Cloud free tier may put the app to sleep after 7 days of
  inactivity. It will wake up automatically on the next visit.


AUTHOR

B. Bharatha Ratna (Birudu Bharatha Ratna)
GitHub: https://github.com/bbharatharatna


LICENSE

This project is for educational and portfolio purposes.
