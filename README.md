# Trading Platform

A modern, interactive web platform for learning, exploring, and practicing trading and investing.

## Features

- ✨ **Dynamic Stock Dashboard**: Search any stock ticker and view live data, price charts, and performance metrics
- 📊 **Technical Indicators**: 10+ indicators including SMA, EMA, RSI, MACD, Bollinger Bands, Stochastic, ADX, and more
- 📈 **Interactive Charts**: Plotly.js powered charts with zoom, hover, and exploration capabilities
- 🔄 **Backtesting**: Test trading strategies on historical data with detailed performance metrics
- 🤖 **Machine Learning Predictions**: LSTM-based model for price predictions
- 🌓 **Modern UI**: Dark/light mode, responsive design, smooth animations
- 📚 **Learning Resources**: Educational content about trading, investing, and risk management

## Tech Stack

- **Frontend**: Next.js 14, React, TailwindCSS, Plotly.js
- **Backend**: FastAPI, Python
- **Data**: yfinance (Yahoo Finance API)
- **Charts**: Plotly
- **ML**: Simplified LSTM model for predictions

## Quick Start

### Option 1: Automated Setup (Recommended)

```bash
# Run the setup script
chmod +x setup.sh
./setup.sh
```

### Option 2: Manual Setup

### Prerequisites

- Node.js 18+ and npm
- Python 3.8+
- pip

### Installation

1. **Install Backend Dependencies**
```bash
cd backend
pip install -r requirements.txt
```

2. **Install Frontend Dependencies**
```bash
cd ../frontend
npm install
```

## Running the Application

You need to run both the backend and frontend servers.

### Terminal 1: Backend
```bash
cd backend
python main.py
# Or use the run script: ./run.sh
```
Backend will run on http://localhost:8000

### Terminal 2: Frontend
```bash
cd frontend
npm run dev
```
Frontend will run on http://localhost:3000

### Usage

1. Open your browser to http://localhost:3000
2. Search for any stock symbol (e.g., AAPL, TSLA, MSFT, GOOGL)
3. View real-time data and charts
4. Toggle technical indicators on/off
5. Explore backtesting and ML prediction features

## Features

### 📊 Chart Analysis
- Interactive candlestick charts
- Toggleable technical indicators (SMA, EMA, RSI, MACD, Volume, etc.)
- Zoom and pan functionality
- Dark/light mode support

### 🔄 Strategy Backtesting
- Test trading strategies on historical data
- Configure entry and exit conditions
- View performance metrics: Total Return, Win Rate, Sharpe Ratio, Max Drawdown, CAGR
- Equity curve visualization

### 🤖 AI Predictions
- LSTM-based price predictions for 7-day and 30-day horizons
- Confidence scoring
- Trend analysis (bullish/bearish/neutral)
- Momentum calculations

## API Endpoints

- `GET /`: Health check
- `GET /stock/{symbol}`: Get current stock data
- `GET /stock/{symbol}/chart?period={period}&indicators={ind}`: Get historical chart with indicators
- `GET /news`: Get market news
- `GET /backtest/{symbol}?entry_condition={cond}&exit_condition={cond}&initial_capital={amount}`: Run strategy backtest
- `GET /ml/predict/{symbol}`: Get ML price predictions
- `GET /health`: API health status

## Project Structure

```
AlgorithmicTrading/
├── frontend/          # Next.js frontend
│   ├── app/          # App router pages
│   ├── components/   # React components
│   ├── package.json
│   └── tailwind.config.js
├── backend/           # FastAPI backend
│   ├── main.py       # Main API server
│   ├── models/       # ML models
│   ├── requirements.txt
│   └── run.sh
├── setup.sh          # Setup script
└── README.md
```

## Technical Indicators

The platform supports the following technical indicators:

1. **SMA (Simple Moving Average)** - 20, 50 day periods
2. **EMA (Exponential Moving Average)** - 12, 26 day periods
3. **RSI (Relative Strength Index)** - 14 day period
4. **MACD (Moving Average Convergence Divergence)**
5. **Bollinger Bands**
6. **Stochastic Oscillator**
7. **ADX (Average Directional Index)**
8. **Volume Bars**

## Backtesting

Currently implements a simple RSI-based strategy:
- Buy when RSI < 30 (oversold)
- Sell when RSI > 70 (overbought)

The system calculates:
- Total return percentage
- Number of trades
- Win rate
- Sharpe ratio
- Maximum drawdown
- CAGR (Compound Annual Growth Rate)

## Machine Learning

The platform includes a simplified LSTM-based predictor:
- 7-day price predictions
- 30-day price predictions
- Confidence scoring based on volatility
- Trend analysis using technical indicators

For production use, you can enhance this with a properly trained PyTorch or TensorFlow model.

## Contributing

Contributions are welcome! This project uses:
- Free APIs (yfinance)
- Open-source libraries
- No paid services required

## License

MIT License - Feel free to use this project for learning and practice!

## Notes

- This platform is for educational and practice purposes only
- Past performance does not guarantee future results
- Always do your own research before making investment decisions

