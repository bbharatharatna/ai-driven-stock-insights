# AI-Driven Stock Insights

An intelligent stock analysis web application that combines real-time financial data with Natural Language Processing (NLP) to deliver AI-powered news sentiment analysis, stock price visualization, and financial summaries.

[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://bbharatharatna-ai-driven-stock-insights-app-43h2bm.streamlit.app/)
[![GitHub](https://img.shields.io/badge/Source%20Code-GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/bbharatharatna/ai-driven-stock-insights)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)

---

## Overview

AI-Driven Stock Insights fetches live stock data from Yahoo Finance and runs it through a pipeline of AI models to deliver sentiment classification, article summarization, keyword extraction, and named entity recognition — all in one downloadable PDF report.

---

## Key Features

- Real-time stock prices and historical data via Yahoo Finance
- News sentiment classification (Positive, Negative, Neutral) using FinBERT
- AI news summarization using DistilBART (distilbart-cnn-6-6)
- Keyword extraction from financial news using YAKE
- Named entity recognition for companies and people using spaCy
- Stock price chart with historical closing prices using Matplotlib
- Stock volatility chart showing daily percentage change
- Word cloud generated from news article content
- PDF report export of the full analysis using FPDF

---

## AI Models Used

| Model                          | Purpose                        | Source        |
|--------------------------------|--------------------------------|---------------|
| ProsusAI/finbert               | Financial sentiment analysis   | Hugging Face  |
| sshleifer/distilbart-cnn-6-6   | News summarization             | Hugging Face  |
| en_core_web_sm                 | Named entity recognition       | spaCy         |

---

## Tech Stack

| Category         | Tools                                              |
|------------------|----------------------------------------------------|
| Frontend / UI    | Streamlit                                          |
| Data             | yfinance, feedparser, BeautifulSoup4               |
| NLP and AI       | Transformers, FinBERT, DistilBART, spaCy, YAKE     |
| Visualization    | Matplotlib, Seaborn, WordCloud                     |
| Report           | FPDF                                               |
| Language         | Python 3.11                                        |

---

## Live Demo

https://bbharatharatna-ai-driven-stock-insights-app-43h2bm.streamlit.app/

---

## Getting Started

### Installation

    git clone https://github.com/bbharatharatna/ai-driven-stock-insights.git
    cd ai-driven-stock-insights
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    streamlit run app.py

### Requirements

    streamlit>=1.32.0
    yfinance
    feedparser
    beautifulsoup4
    transformers
    torch==2.3.0
    spacy
    yake
    matplotlib
    seaborn
    wordcloud
    fpdf

---

## Usage

1. Open the app using the Live Demo link or run locally
2. In the Configuration sidebar, enter the Company Name and Stock Ticker
3. Optionally add a News RSS Feed URL and number of articles to analyze
4. Click Analyze to generate the full report
5. View sentiment analysis, summaries, charts, word cloud, and download the PDF

### Sample Inputs to Try

- Company: Apple | Ticker: AAPL
- Company: Tesla | Ticker: TSLA
- Company: Infosys | Ticker: INFY.NS
- Company: Reliance | Ticker: RELIANCE.NS

---

## How It Works

    +-------------------------------+
    |   Enter Company + Ticker      |
    +-------------------------------+
                   |
                   v
    +-------------------------------+
    |  Fetch Live Stock Data        |
    |  via Yahoo Finance            |
    +-------------------------------+
                   |
                   v
    +-------------------------------+
    |  Fetch News Articles          |
    |  via RSS Feed / Feedparser    |
    +-------------------------------+
                   |
                   v
    +-------------------------------+
    |  NLP Pipeline                 |
    |  FinBERT  -> Sentiment        |
    |  DistilBART -> Summary        |
    |  YAKE     -> Keywords         |
    |  spaCy    -> Entities         |
    +-------------------------------+
                   |
                   v
    +-------------------------------+
    |  Visualizations               |
    |  Price Chart, Volatility,     |
    |  Word Cloud                   |
    +-------------------------------+
                   |
                   v
    +-------------------------------+
    |  Export Full PDF Report       |
    +-------------------------------+

---

## Known Limitations

- Yahoo Finance may occasionally rate-limit requests. Wait 1 to 2 minutes and retry.
- First load may take 30 to 60 seconds as AI models are downloaded and cached.
- Streamlit Community Cloud free tier may sleep after 7 days of inactivity and will wake automatically on next visit.

---

## Developer

B Bharatha Ratna
GitHub: https://github.com/bbharatharatna

---

This project is for educational and portfolio purposes.
