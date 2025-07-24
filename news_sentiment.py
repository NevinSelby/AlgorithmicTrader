import requests
import feedparser
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import re
import time
import warnings
from urllib.parse import quote_plus

# Sentiment Analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

warnings.filterwarnings('ignore')


class NewsSource:
    """Base class for news sources."""
    
    def __init__(self, name, base_url):
        self.name = name
        self.base_url = base_url
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def fetch_news(self, ticker, max_articles=10):
        """Fetch news articles for a ticker. To be implemented by subclasses."""
        raise NotImplementedError


class RSSNewsSource(NewsSource):
    """RSS-based news source."""
    
    def __init__(self, name, rss_url_template):
        super().__init__(name, rss_url_template)
        self.rss_url_template = rss_url_template
    
    def fetch_news(self, ticker, max_articles=10):
        """Fetch news from RSS feeds."""
        try:
            # Format URL with ticker
            url = self.rss_url_template.format(ticker=ticker.upper())
            
            # Parse RSS feed
            feed = feedparser.parse(url)
            
            articles = []
            for entry in feed.entries[:max_articles]:
                article = {
                    'title': entry.get('title', ''),
                    'description': entry.get('description', '') or entry.get('summary', ''),
                    'link': entry.get('link', ''),
                    'published': self._parse_date(entry.get('published', '')),
                    'source': self.name
                }
                articles.append(article)
            
            return articles
            
        except Exception as e:
            print(f"Error fetching from {self.name}: {e}")
            return []
    
    def _parse_date(self, date_str):
        """Parse date from RSS feed."""
        try:
            # Try multiple date formats
            formats = [
                '%a, %d %b %Y %H:%M:%S %Z',
                '%a, %d %b %Y %H:%M:%S %z',
                '%Y-%m-%dT%H:%M:%S%z',
                '%Y-%m-%d %H:%M:%S'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except:
                    continue
            
            # If all fail, return current time
            return datetime.now()
            
        except:
            return datetime.now()


class WebScrapingNewsSource(NewsSource):
    """Web scraping-based news source."""
    
    def __init__(self, name, base_url, search_url_template, selectors):
        super().__init__(name, base_url)
        self.search_url_template = search_url_template
        self.selectors = selectors
    
    def fetch_news(self, ticker, max_articles=10):
        """Fetch news by web scraping."""
        try:
            # Format search URL
            search_term = f"{ticker} stock news"
            url = self.search_url_template.format(
                query=quote_plus(search_term),
                ticker=ticker.upper()
            )
            
            # Make request
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            articles = []
            
            # Extract articles based on selectors
            article_elements = soup.select(self.selectors.get('article_container', 'article'))
            
            for element in article_elements[:max_articles]:
                try:
                    title_elem = element.select_one(self.selectors.get('title', 'h1, h2, h3'))
                    desc_elem = element.select_one(self.selectors.get('description', 'p'))
                    link_elem = element.select_one(self.selectors.get('link', 'a'))
                    
                    article = {
                        'title': title_elem.get_text(strip=True) if title_elem else '',
                        'description': desc_elem.get_text(strip=True) if desc_elem else '',
                        'link': self._resolve_url(link_elem.get('href') if link_elem else ''),
                        'published': datetime.now() - timedelta(hours=1),  # Approximate
                        'source': self.name
                    }
                    
                    if article['title']:  # Only add if we have a title
                        articles.append(article)
                        
                except Exception as e:
                    continue
            
            return articles
            
        except Exception as e:
            print(f"Error scraping from {self.name}: {e}")
            return []
    
    def _resolve_url(self, url):
        """Resolve relative URLs."""
        if url.startswith('http'):
            return url
        elif url.startswith('/'):
            return self.base_url + url
        else:
            return self.base_url + '/' + url


class RedditNewsSource(NewsSource):
    """Reddit-based news source (free alternative to Twitter)."""
    
    def __init__(self):
        super().__init__("Reddit", "https://www.reddit.com")
    
    def fetch_news(self, ticker, max_articles=10):
        """Fetch sentiment from Reddit financial subreddits."""
        try:
            # Search multiple financial subreddits
            subreddits = ['stocks', 'investing', 'SecurityAnalysis', 'ValueInvesting']
            articles = []
            
            for subreddit in subreddits:
                try:
                    # Use Reddit's JSON API (no authentication required for public posts)
                    url = f"https://www.reddit.com/r/{subreddit}/search.json"
                    params = {
                        'q': ticker,
                        'restrict_sr': '1',
                        'sort': 'new',
                        'limit': max_articles // len(subreddits) + 1
                    }
                    
                    response = requests.get(url, headers=self.headers, params=params, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        for post in data.get('data', {}).get('children', []):
                            post_data = post.get('data', {})
                            
                            article = {
                                'title': post_data.get('title', ''),
                                'description': post_data.get('selftext', '')[:500],  # Limit length
                                'link': f"https://reddit.com{post_data.get('permalink', '')}",
                                'published': datetime.fromtimestamp(post_data.get('created_utc', 0)),
                                'source': f"r/{subreddit}",
                                'score': post_data.get('score', 0),
                                'comments': post_data.get('num_comments', 0)
                            }
                            
                            if article['title']:
                                articles.append(article)
                    
                    time.sleep(1)  # Be respectful to Reddit's servers
                    
                except Exception as e:
                    print(f"Error fetching from r/{subreddit}: {e}")
                    continue
            
            return articles[:max_articles]
            
        except Exception as e:
            print(f"Error fetching from Reddit: {e}")
            return []


class SentimentAnalyzer:
    """Sentiment analysis for financial news."""
    
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Financial sentiment words (weights)
        self.financial_positive = {
            'bullish': 2.0, 'rally': 1.5, 'surge': 1.8, 'soar': 1.8, 'jump': 1.3,
            'gain': 1.2, 'rise': 1.1, 'up': 0.8, 'positive': 1.0, 'strong': 1.2,
            'outperform': 1.5, 'beat': 1.3, 'exceed': 1.4, 'upgrade': 1.6,
            'buy': 1.4, 'growth': 1.2, 'profit': 1.3, 'revenue': 1.1
        }
        
        self.financial_negative = {
            'bearish': -2.0, 'crash': -2.5, 'plunge': -2.2, 'tumble': -1.8,
            'fall': -1.2, 'drop': -1.3, 'decline': -1.4, 'down': -0.8,
            'negative': -1.0, 'weak': -1.2, 'underperform': -1.5, 'miss': -1.3,
            'downgrade': -1.6, 'sell': -1.4, 'loss': -1.5, 'debt': -1.1
        }
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of text using multiple methods."""
        if not text:
            return {
                'compound': 0.0,
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'financial_score': 0.0,
                'confidence': 0.0
            }
        
        # Clean text
        text = self._clean_text(text)
        
        # VADER sentiment
        vader_scores = self.vader_analyzer.polarity_scores(text)
        
        # TextBlob sentiment
        try:
            blob = TextBlob(text)
            textblob_polarity = blob.sentiment.polarity
            textblob_subjectivity = blob.sentiment.subjectivity
        except:
            textblob_polarity = 0.0
            textblob_subjectivity = 0.5
        
        # Financial-specific sentiment
        financial_score = self._calculate_financial_sentiment(text)
        
        # Combine scores
        compound_score = (
            vader_scores['compound'] * 0.4 +
            textblob_polarity * 0.3 +
            financial_score * 0.3
        )
        
        # Calculate confidence based on agreement
        scores = [vader_scores['compound'], textblob_polarity, financial_score]
        confidence = 1.0 - (np.std(scores) / 2.0)  # Lower std = higher confidence
        confidence = max(0.0, min(1.0, confidence))
        
        return {
            'compound': compound_score,
            'positive': vader_scores['pos'],
            'negative': vader_scores['neg'],
            'neutral': vader_scores['neu'],
            'financial_score': financial_score,
            'confidence': confidence,
            'textblob_polarity': textblob_polarity,
            'textblob_subjectivity': textblob_subjectivity
        }
    
    def _clean_text(self, text):
        """Clean and preprocess text."""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _calculate_financial_sentiment(self, text):
        """Calculate financial-specific sentiment score."""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        score = 0.0
        total_words = len(words)
        
        if total_words == 0:
            return 0.0
        
        # Score positive financial terms
        for word in words:
            if word in self.financial_positive:
                score += self.financial_positive[word]
            elif word in self.financial_negative:
                score += self.financial_negative[word]
        
        # Normalize by text length
        normalized_score = score / max(total_words, 1) * 10
        
        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, normalized_score))


class NewsSentimentAggregator:
    """Main class for aggregating news and sentiment analysis."""
    
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Initialize news sources
        self.news_sources = [
            # Yahoo Finance RSS
            RSSNewsSource(
                "Yahoo Finance",
                "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
            ),
            
            # MarketWatch RSS  
            RSSNewsSource(
                "MarketWatch",
                "https://feeds.marketwatch.com/marketwatch/companynews/{ticker}/"
            ),
            
            # Reddit
            RedditNewsSource(),
        ]
    
    def get_sentiment_data(self, ticker, days_back=7, max_articles_per_source=5):
        """Get aggregated sentiment data for a ticker."""
        try:
            all_articles = []
            
            # Fetch from all sources
            for source in self.news_sources:
                try:
                    articles = source.fetch_news(ticker, max_articles_per_source)
                    all_articles.extend(articles)
                    time.sleep(0.5)  # Be respectful to servers
                except Exception as e:
                    print(f"Error fetching from {source.name}: {e}")
                    continue
            
            if not all_articles:
                return self._get_neutral_sentiment()
            
            # Filter articles by date
            cutoff_date = datetime.now() - timedelta(days=days_back)
            recent_articles = [
                article for article in all_articles 
                if article['published'] >= cutoff_date
            ]
            
            if not recent_articles:
                recent_articles = all_articles[:10]  # Fallback to recent articles
            
            # Analyze sentiment for each article
            sentiments = []
            for article in recent_articles:
                text = f"{article['title']} {article['description']}"
                sentiment = self.sentiment_analyzer.analyze_sentiment(text)
                sentiment['article'] = article
                sentiments.append(sentiment)
            
            # Aggregate sentiments
            return self._aggregate_sentiments(sentiments)
            
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            return self._get_neutral_sentiment()
    
    def _aggregate_sentiments(self, sentiments):
        """Aggregate multiple sentiment scores."""
        if not sentiments:
            return self._get_neutral_sentiment()
        
        # Weight by confidence
        total_weight = 0
        weighted_compound = 0
        weighted_financial = 0
        
        compounds = []
        financials = []
        confidences = []
        
        for sentiment in sentiments:
            confidence = sentiment['confidence']
            weight = confidence if confidence > 0.3 else 0.3  # Minimum weight
            
            weighted_compound += sentiment['compound'] * weight
            weighted_financial += sentiment['financial_score'] * weight
            total_weight += weight
            
            compounds.append(sentiment['compound'])
            financials.append(sentiment['financial_score'])
            confidences.append(confidence)
        
        if total_weight == 0:
            return self._get_neutral_sentiment()
        
        # Calculate aggregated scores
        avg_compound = weighted_compound / total_weight
        avg_financial = weighted_financial / total_weight
        avg_confidence = np.mean(confidences)
        
        # Calculate final sentiment score (blend of compound and financial)
        final_score = (avg_compound * 0.6 + avg_financial * 0.4)
        
        # Determine sentiment label
        if final_score > 0.1:
            sentiment_label = "Positive"
        elif final_score < -0.1:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"
        
        return {
            'sentiment_score': final_score,
            'sentiment_label': sentiment_label,
            'confidence': avg_confidence,
            'compound_score': avg_compound,
            'financial_score': avg_financial,
            'article_count': len(sentiments),
            'std_deviation': np.std(compounds),
            'individual_scores': compounds,
            'sources': list(set([s['article']['source'] for s in sentiments]))
        }
    
    def _get_neutral_sentiment(self):
        """Return neutral sentiment when no data is available."""
        return {
            'sentiment_score': 0.0,
            'sentiment_label': "Neutral",
            'confidence': 0.5,
            'compound_score': 0.0,
            'financial_score': 0.0,
            'article_count': 0,
            'std_deviation': 0.0,
            'individual_scores': [],
            'sources': []
        }
    
    def get_historical_sentiment(self, ticker, days=30):
        """Get historical sentiment data (simplified version)."""
        try:
            # For this implementation, we'll return mock historical data
            # In a real system, you'd store daily sentiment in a database
            dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
            
            # Generate realistic sentiment pattern (random walk with mean reversion)
            np.random.seed(hash(ticker) % 2147483647)  # Deterministic but ticker-specific
            sentiment_values = np.random.normal(0, 0.3, days)
            
            # Apply some smoothing and mean reversion
            for i in range(1, len(sentiment_values)):
                sentiment_values[i] = sentiment_values[i-1] * 0.7 + sentiment_values[i] * 0.3
            
            # Clamp values
            sentiment_values = np.clip(sentiment_values, -1, 1)
            
            return pd.DataFrame({
                'date': dates,
                'sentiment_score': sentiment_values,
                'confidence': np.random.uniform(0.4, 0.9, days)
            })
            
        except Exception as e:
            print(f"Error getting historical sentiment: {e}")
            return pd.DataFrame()


# Test function
def test_sentiment_analysis(ticker="AAPL"):
    """Test the sentiment analysis system."""
    print(f"Testing sentiment analysis for {ticker}...")
    
    aggregator = NewsSentimentAggregator()
    sentiment_data = aggregator.get_sentiment_data(ticker, days_back=3, max_articles_per_source=3)
    
    print(f"\nSentiment Results for {ticker}:")
    print(f"Sentiment Score: {sentiment_data['sentiment_score']:.3f}")
    print(f"Sentiment Label: {sentiment_data['sentiment_label']}")
    print(f"Confidence: {sentiment_data['confidence']:.3f}")
    print(f"Articles Analyzed: {sentiment_data['article_count']}")
    print(f"Sources: {', '.join(sentiment_data['sources'])}")
    
    return sentiment_data


if __name__ == "__main__":
    # Test the system
    result = test_sentiment_analysis("TSLA")
    print("\nTest completed!") 