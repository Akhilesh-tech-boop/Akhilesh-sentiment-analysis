from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

def analyze_sentiment(tweets_df):
    """
    Perform sentiment analysis using VADER.
    """
    analyzer = SentimentIntensityAnalyzer()
    
    # Apply sentiment analysis to each tweet
    sentiment_scores = tweets_df['cleaned_tweet'].apply(lambda tweet: analyzer.polarity_scores(tweet)['compound'])
    
    # Add sentiment scores to the DataFrame
    tweets_df['sentiment_score'] = sentiment_scores
    tweets_df['sentiment'] = tweets_df['sentiment_score'].apply(
        lambda x: 'positive' if x > 0 else ('neutral' if x == 0 else 'negative')
    )
    return tweets_df

if __name__ == "__main__":
    tweets_df = pd.read_csv('data/cleaned_tweets.csv')
    sentiment_df = analyze_sentiment(tweets_df)
    sentiment_df.to_csv('data/sentiment_tweets.csv', index=False)
    print("Sentiment analysis complete. Results saved to 'sentiment_tweets.csv'.")

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd

def plot_sentiment_trend(tweets_df):
    """
    Plot sentiment trend over time.
    """
    tweets_df['timestamp'] = pd.to_datetime(tweets_df['timestamp'])
    tweets_df.set_index('timestamp', inplace=True)
    
    # Resample by day and plot sentiment score trend
    sentiment_daily = tweets_df.resample('D').mean()['sentiment_score']
    plt.figure(figsize=(10, 6))
    plt.plot(sentiment_daily, color='blue')
    plt.title('Sentiment Trend Over Time')
    plt.xlabel('Date')
    plt.ylabel('Average Sentiment Score')
    plt.grid(True)
    plt.show()

def generate_wordcloud(tweets_df, sentiment='positive'):
    """
    Generate a word cloud for either positive or negative tweets.
    """
    sentiment_tweets = tweets_df[tweets_df['sentiment'] == sentiment]['cleaned_tweet']
    text = ' '.join(sentiment_tweets)
    
    wordcloud = WordCloud(width=800, height=400).generate(text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'{sentiment.capitalize()} Sentiment Word Cloud')
    plt.show()

if __name__ == "__main__":
    sentiment_df = pd.read_csv('data/sentiment_tweets.csv')
    plot_sentiment_trend(sentiment_df)
    generate_wordcloud(sentiment_df, sentiment='positive')
    generate_wordcloud(sentiment_df, sentiment='negative')

    # Social Media Sentiment Analysis

This project performs sentiment analysis on social media data, specifically Twitter, to understand public sentiment towards specific topics, products, or events using **Natural Language Processing (NLP)** techniques.

## Requirements

- Python 3.x
- `pip install -r requirements.txt`

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/social-media-sentiment-analysis.git
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Create a `.env` file with your Twitter API credentials:
   

