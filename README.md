# Algorithmic Trading Strategy Backtester

A comprehensive Python-based web application built with Streamlit that allows users to backtest multiple algorithmic trading strategies on historical stock data.

## Features

- **Multiple Trading Strategies**: 6 professional algorithmic trading strategies including ML-enhanced
- **Machine Learning Integration**: LSTM neural networks for price prediction and signal generation
- **NLP Sentiment Analysis**: Real-time news scraping and sentiment scoring from multiple sources
- **Interactive Web Interface**: Professional Streamlit interface with dynamic controls
- **Real-time Data Fetching**: Uses `yfinance` to fetch live historical stock data
- **Professional Backtesting Engine**: Powered by `backtrader` library
- **ML Prediction Demo**: Live demonstration of machine learning analysis
- **Sentiment Analysis Demo**: Real-time market sentiment visualization and interpretation
- **Comprehensive Analytics**: Includes Sharpe ratio, maximum drawdown, win rate, and more
- **Strategy-Specific Parameters**: Dynamic parameter configuration for each strategy
- **Object-Oriented Design**: Well-structured, modular codebase following OOP principles

## Project Structure

```
AlgorithmicTrading/
├── app.py              # Main Streamlit application
├── data_handler.py     # Data fetching and preprocessing
├── strategies.py       # Multiple trading strategy implementations
├── ml_models.py        # LSTM and ML components for price prediction
├── news_sentiment.py   # NLP sentiment analysis and news scraping
├── backtester.py       # Backtrader engine wrapper
├── requirements.txt    # Python dependencies (includes PyTorch + NLP)
└── README.md          # Project documentation
```

## Available Trading Strategies

### 1. SMA Crossover Strategy
- **Logic**: Buy when fast MA crosses above slow MA, sell when it crosses below
- **Parameters**: Fast period, Slow period
- **Best for**: Trend following, capturing major price movements

### 2. RSI Strategy
- **Logic**: Buy when RSI is oversold (below threshold), sell when overbought (above threshold)
- **Parameters**: RSI period, Upper threshold, Lower threshold
- **Best for**: Range-bound markets, identifying overbought/oversold conditions

### 3. MACD Strategy
- **Logic**: Buy when MACD line crosses above signal line, sell when it crosses below
- **Parameters**: Fast EMA, Slow EMA, Signal line period
- **Best for**: Momentum trading, trend confirmation

### 4. Bollinger Bands Strategy
- **Logic**: Buy when price touches lower band, sell when it touches upper band
- **Parameters**: Moving average period, Standard deviation factor
- **Best for**: Volatility-based trading, mean reversion

### 5. Mean Reversion Strategy
- **Logic**: Buy when price deviates significantly below average, sell when above
- **Parameters**: Moving average period, Deviation threshold
- **Best for**: Sideways markets, contrarian trading

### 6. ML Enhanced Strategy
- **Logic**: Combines LSTM price predictions with multiple technical indicators
- **Parameters**: Prediction threshold, Confidence threshold, Technical indicator periods
- **Best for**: Advanced users, AI-powered decision making, adaptive trading

## Quick Start

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. **Clone or download the project files**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** to the URL shown in the terminal (typically `http://localhost:8501`)

## How to Use

### 1. Select Trading Strategy
Choose from 5 available algorithmic trading strategies in the dropdown menu.

### 2. Configure Parameters
Use the sidebar to set:
- **Stock Ticker**: Enter any valid stock symbol (e.g., MSFT, GOOGL, TSLA)
- **Date Range**: Select start and end dates for historical data
- **Starting Cash**: Set your initial portfolio value
- **Strategy Parameters**: Configure strategy-specific parameters (dynamically updated)
- **Commission**: Set trading commission percentage

### 3. Run Backtest
Click the "Run Backtest" button to execute the selected strategy.

### 4. Analyze Results
Review the comprehensive results including:
- Final portfolio value and total return
- Risk metrics (Sharpe ratio, maximum drawdown)
- Trading statistics (total trades, win rate)
- Strategy-specific parameter summary

## Strategy Implementation

All strategies are implemented as classes inheriting from `backtrader.Strategy` with comprehensive logging and trade tracking. Each strategy includes:

- **Signal Generation**: Clear buy/sell signal logic
- **Order Management**: Proper order tracking and execution
- **Trade Logging**: Detailed trade execution logs
- **Performance Tracking**: Trade counting and PnL calculation

## Machine Learning Features

The platform includes advanced ML capabilities for sophisticated trading analysis:

### LSTM Price Prediction
- **PyTorch Neural Network**: Multi-layer LSTM built with PyTorch for time series forecasting
- **Feature Engineering**: 20+ technical indicators and price patterns
- **Training Pipeline**: Custom PyTorch training loop with validation splits
- **Prediction Engine**: Real-time price forecasting with confidence scores

### ML-Enhanced Strategy
- **Hybrid Approach**: Combines ML predictions with traditional technical analysis
- **Confidence-Based Trading**: Only executes trades above confidence threshold
- **Dynamic Adaptation**: Learns from market patterns and adjusts parameters
- **Ensemble Voting**: Weighs ML predictions against multiple technical indicators

## Natural Language Processing Features

The platform includes comprehensive NLP capabilities for sentiment-driven trading analysis:

### News Sentiment Analysis
- **Multi-Source Scraping**: Aggregates news from Yahoo Finance, MarketWatch, and Reddit
- **Free Data Sources**: No paid APIs required - uses RSS feeds and web scraping
- **Sentiment Scoring**: VADER sentiment + TextBlob + Financial-specific vocabulary
- **Real-time Analysis**: Live sentiment tracking for any stock ticker

### Sentiment Integration
- **LSTM Enhancement**: Sentiment features integrated into price prediction models
- **11 Sentiment Features**: Score, confidence, moving averages, volatility, momentum, correlation indicators
- **Dynamic Weighting**: Confidence-based sentiment incorporation
- **Multi-timeframe Analysis**: Short-term news impact + historical sentiment trends

## Architecture Overview

### Core Classes

1. **DataHandler** (`data_handler.py`)
   - Fetches historical stock data using yfinance with multiple fallback methods
   - Converts data to backtrader-compatible format
   - Handles data validation and error management

2. **Multiple Strategy Classes** (`strategies.py`)
   - Implements 5 different algorithmic trading strategies
   - Each inherits from `backtrader.Strategy`
   - Modular design with strategy registry system

3. **Backtester** (`backtester.py`)
   - Wraps the backtrader Cerebro engine
   - Supports dynamic strategy parameter configuration
   - Configures analyzers for comprehensive performance metrics

4. **StreamlitApp** (`app.py`)
   - Professional UI with strategy selection capabilities
   - Dynamic parameter configuration based on selected strategy
   - Comprehensive results display and visualization

## Configuration Options

### Strategy Parameters
- **Dynamic Configuration**: Each strategy has its own configurable parameters
- **SMA Crossover**: Fast MA (5-50), Slow MA (20-200)
- **RSI**: Period (5-30), Upper threshold (60-90), Lower threshold (10-40)
- **MACD**: Fast EMA (5-20), Slow EMA (15-40), Signal period (5-15)
- **Bollinger Bands**: Period (10-50), Deviation factor (1.0-3.0)
- **Mean Reversion**: Period (10-50), Threshold (0.01-0.10)
- **ML Enhanced**: Prediction threshold (0.5-5.0%), Confidence threshold (0.3-0.9), Technical periods

## Performance Metrics

The application provides comprehensive performance analysis:

- **Portfolio Value**: Starting vs. final portfolio value
- **Total Return**: Percentage gain/loss
- **Sharpe Ratio**: Risk-adjusted return measure
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Trade Statistics**: Total trades, win rate, profit/loss per trade
