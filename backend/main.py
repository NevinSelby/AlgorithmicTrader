from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objs as go
from typing import Optional

# Try to import ML predictor
try:
    from models.predictor import StockPredictor
    ml_predictor = StockPredictor()
except ImportError:
    print("Warning: ML predictor not available")
    ml_predictor = None

app = FastAPI(title="IterAI Finance Trading Platform API")

# CORS middleware
import os
ALLOWED_ORIGINS_STR = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
# Handle both comma-separated and single origin
if ALLOWED_ORIGINS_STR == "*":
    # When using wildcard, cannot use allow_credentials=True
    ALLOWED_ORIGINS = ["*"]
    ALLOW_CREDENTIALS = False
else:
    ALLOWED_ORIGINS = [origin.strip() for origin in ALLOWED_ORIGINS_STR.split(",") if origin.strip()]
    ALLOW_CREDENTIALS = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Indicator calculation functions
def calculate_sma(data: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average"""
    return data.rolling(window=period).mean()

def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average"""
    return data.ewm(span=period, adjust=False).mean()

def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Moving Average Convergence Divergence"""
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    macd = ema_fast - ema_slow
    signal_line = calculate_ema(macd, signal)
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(data: pd.Series, period: int = 20, std: float = 2):
    """Bollinger Bands"""
    sma = calculate_sma(data, period)
    std_dev = data.rolling(window=period).std()
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    return upper, sma, lower

def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3):
    """Stochastic Oscillator"""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_period).mean()
    return k_percent, d_percent

def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    """Average Directional Index"""
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr1 = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1)
    tr = tr1.max(axis=1)
    
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    
    return adx

@app.get("/")
def read_root():
    return {"message": "IterAI Finance Trading Platform API", "status": "running"}

@app.get("/stock/{symbol}")
async def get_stock_data(symbol: str):
    try:
        ticker = yf.Ticker(symbol)
        
        # Get current data
        info = ticker.info
        current_price = info.get('currentPrice', 0)
        previous_close = info.get('previousClose', 0)
        change = current_price - previous_close
        change_percent = (change / previous_close * 100) if previous_close else 0
        volume = info.get('volume', 0)
        
        return {
            "symbol": symbol,
            "price": current_price,
            "change": change,
            "changePercent": change_percent,
            "volume": volume,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/stock/{symbol}/chart")
async def get_stock_chart(
    symbol: str,
    period: str = "1y",
    indicators: Optional[str] = None
):
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        
        if hist.empty:
            raise HTTPException(status_code=404, detail="No data found for symbol")
        
        # Create candlestick chart
        fig = go.Figure()
        
        # Candlesticks
        dates = hist.index.strftime('%Y-%m-%d').tolist()
        fig.add_trace(go.Candlestick(
            x=dates,
            open=hist['Open'].tolist(),
            high=hist['High'].tolist(),
            low=hist['Low'].tolist(),
            close=hist['Close'].tolist(),
            name=symbol
        ))
        
        # Parse indicators (comma-separated string like "SMA_20,RSI,MACD")
        indicator_list = indicators.split(',') if indicators else []
        
        close_prices = hist['Close']
        
        if 'SMA_20' in indicator_list:
            sma20 = calculate_sma(close_prices, 20)
            fig.add_trace(go.Scatter(
                x=dates,
                y=sma20.tolist(),
                mode='lines',
                name='SMA 20',
                line=dict(color='#f97316', width=2)  # Muted orange
            ))
        
        if 'SMA_50' in indicator_list:
            sma50 = calculate_sma(close_prices, 50)
            fig.add_trace(go.Scatter(
                x=dates,
                y=sma50.tolist(),
                mode='lines',
                name='SMA 50',
                line=dict(color='#9333ea', width=2)  # Muted purple
            ))
        
        if 'EMA_12' in indicator_list:
            ema12 = calculate_ema(close_prices, 12)
            fig.add_trace(go.Scatter(
                x=dates,
                y=ema12.tolist(),
                mode='lines',
                name='EMA 12',
                line=dict(color='#06b6d4', width=2)  # Muted cyan
            ))
        
        if 'EMA_26' in indicator_list:
            ema26 = calculate_ema(close_prices, 26)
            fig.add_trace(go.Scatter(
                x=dates,
                y=ema26.tolist(),
                mode='lines',
                name='EMA 26',
                line=dict(color='#c026d3', width=2)  # Muted magenta
            ))
        
        if 'RSI' in indicator_list:
            rsi = calculate_rsi(close_prices)
            # RSI in separate subplot
            fig.add_trace(go.Scatter(
                x=dates,
                y=rsi.tolist(),
                mode='lines',
                name='RSI',
                line=dict(color='#3b82f6', width=2),  # Muted blue
                yaxis='y2'
            ))
        
        if 'MACD' in indicator_list:
            macd, signal, hist_macd = calculate_macd(close_prices)
            fig.add_trace(go.Scatter(x=dates, y=macd.tolist(), mode='lines', name='MACD', line=dict(color='#22c55e', width=2)))  # Muted green
            fig.add_trace(go.Scatter(x=dates, y=signal.tolist(), mode='lines', name='Signal', line=dict(color='#ef4444', width=2)))  # Muted red
        
        if 'Volume' in indicator_list:
            # Volume bars in separate subplot
            colors = ['green' if hist['Close'].iloc[i] >= hist['Open'].iloc[i] else 'red' 
                     for i in range(len(hist))]
            fig.add_trace(go.Bar(
                x=dates,
                y=hist['Volume'].tolist(),
                name='Volume',
                marker_color=colors,
                yaxis='y3'
            ))
        
        if 'Bollinger' in indicator_list:
            try:
                upper_bb, middle_bb, lower_bb = calculate_bollinger_bands(close_prices, 20, 2)
                # Fill NaN values for plotting
                upper_bb = upper_bb.bfill().ffill()
                middle_bb = middle_bb.bfill().ffill()
                lower_bb = lower_bb.bfill().ffill()
                fig.add_trace(go.Scatter(x=dates, y=upper_bb.tolist(), mode='lines', name='Upper BB', line=dict(color='#eab308', width=1, dash='dash')))  # Muted yellow
                fig.add_trace(go.Scatter(x=dates, y=middle_bb.tolist(), mode='lines', name='Middle BB', line=dict(color='#6b7280', width=1)))  # Muted gray
                fig.add_trace(go.Scatter(x=dates, y=lower_bb.tolist(), mode='lines', name='Lower BB', line=dict(color='#eab308', width=1, dash='dash')))  # Muted yellow
            except Exception as e:
                print(f"Error adding Bollinger Bands: {e}")
        
        if 'Stochastic' in indicator_list:
            try:
                high = hist['High']
                low = hist['Low']
                k_percent, d_percent = calculate_stochastic(high, low, close_prices)
                # Fill NaN values
                k_percent = k_percent.bfill().ffill()
                d_percent = d_percent.bfill().ffill()
                fig.add_trace(go.Scatter(x=dates, y=k_percent.tolist(), mode='lines', name='%K', line=dict(color='#3b82f6', width=2), yaxis='y2'))  # Muted blue
                fig.add_trace(go.Scatter(x=dates, y=d_percent.tolist(), mode='lines', name='%D', line=dict(color='#ef4444', width=2), yaxis='y2'))  # Muted red
            except Exception as e:
                print(f"Error adding Stochastic: {e}")
        
        if 'ADX' in indicator_list:
            try:
                high = hist['High']
                low = hist['Low']
                adx = calculate_adx(high, low, close_prices)
                # Fill NaN values
                adx = adx.bfill().ffill()
                fig.add_trace(go.Scatter(x=dates, y=adx.tolist(), mode='lines', name='ADX', line=dict(color='#ec4899', width=2), yaxis='y2'))  # Muted pink
            except Exception as e:
                print(f"Error adding ADX: {e}")
        
        # Layout with subplots
        fig.update_layout(
            title=f"{symbol} Stock Chart",
            xaxis=dict(title="Date"),
            yaxis=dict(title="Price ($)"),
            yaxis2=dict(title="RSI/Stoch/ADX", overlaying='y', side='right', range=[0, 100]),
            yaxis3=dict(title="Volume", overlaying='y', side='right', anchor='free', position=0.95, range=[0, hist['Volume'].max() * 2]),
            hovermode='x unified',
            template='plotly_dark',
            height=600
        )
        
        # Convert to JSON using Plotly's JSON encoder
        import json
        fig_json = fig.to_json()
        fig_dict = json.loads(fig_json)
        return fig_dict
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/news")
async def get_market_news():
    try:
        # Get news from multiple sources
        news_items = []
        
        # S&P 500 News
        try:
            ticker = yf.Ticker("^GSPC")
            sp_news = ticker.news[:5] if hasattr(ticker, 'news') else []
            news_items.extend(sp_news)
        except:
            pass
        
        # Tech stocks news (AAPL, MSFT, GOOGL)
        for symbol in ["AAPL", "MSFT", "GOOGL"]:
            try:
                ticker = yf.Ticker(symbol)
                if hasattr(ticker, 'news') and ticker.news:
                    news_items.extend(ticker.news[:2])
            except:
                continue
        
        # Deduplicate and format
        seen_titles = set()
        unique_news = []
        for item in news_items[:10]:
            title = item.get('title', 'Market News')
            if title not in seen_titles:
                seen_titles.add(title)
                unique_news.append({
                    "title": title,
                    "link": item.get('link', '#'),
                    "publisher": item.get('publisher', 'Financial News'),
                    "providerPublishTime": item.get('providerPublishTime', 0)
                })
        
        return {"news": unique_news}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/market-overview")
async def get_market_overview():
    """
    Get overall market sentiment and key indices
    """
    try:
        indices = {
            "^GSPC": "S&P 500",
            "^IXIC": "NASDAQ",
            "^DJI": "Dow Jones",
            "^VIX": "Vix"
        }
        
        overview = []
        for symbol, name in indices.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="5d")
                
                if not hist.empty:
                    current = hist['Close'].iloc[-1]
                    prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current
                    change = current - prev_close
                    change_pct = (change / prev_close * 100) if prev_close else 0
                    
                    overview.append({
                        "symbol": symbol,
                        "name": name,
                        "value": round(float(current), 2),
                        "change": round(float(change), 2),
                        "changePercent": round(float(change_pct), 2)
                    })
            except:
                continue
        
        return {"overview": overview}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def evaluate_condition(condition: str, indicators: dict) -> bool:
    """Evaluate a trading condition with indicator values"""
    try:
        # Parse condition (e.g., "RSI < 30", "SMA20 > SMA50", "MACD > Signal")
        condition = condition.replace(' ', '')
        
        # Handle comparison operators
        for op in ['>=', '<=', '>', '<', '==', '!=']:
            if op in condition:
                left, right = condition.split(op, 1)
                left_val = indicators.get(left.upper(), float(left) if left.replace('.', '', 1).replace('-', '', 1).isdigit() else 0)
                right_val = indicators.get(right.upper(), float(right) if right.replace('.', '', 1).replace('-', '', 1).isdigit() else 0)
                
                if op == '>':
                    return left_val > right_val
                elif op == '<':
                    return left_val < right_val
                elif op == '>=':
                    return left_val >= right_val
                elif op == '<=':
                    return left_val <= right_val
                elif op == '==':
                    return left_val == right_val
        return False
    except:
        return False

@app.get("/backtest/{symbol}")
async def backtest_strategy(
    symbol: str,
    entry_condition: str = "RSI < 30",
    exit_condition: str = "RSI > 70",
    initial_capital: float = 100000
):
    """
    Backtest a trading strategy on historical data with full indicator support
    """
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="2y")
        
        if hist.empty:
            raise HTTPException(status_code=404, detail="No data found")
        
        # Calculate ALL indicators
        close = hist['Close']
        high = hist['High']
        low = hist['Low']
        
        # Calculate basic indicators
        rsi = calculate_rsi(close, 14)
        sma20 = calculate_sma(close, 20)
        sma50 = calculate_sma(close, 50)
        ema20 = calculate_ema(close, 20)
        macd, signal, _ = calculate_macd(close)
        upper_bb, middle_bb, lower_bb = calculate_bollinger_bands(close, 20)
        k_percent, d_percent = calculate_stochastic(high, low, close)
        adx = calculate_adx(high, low, close)
        
        # Store all indicators
        hist['RSI'] = rsi
        hist['SMA20'] = sma20
        hist['SMA50'] = sma50
        hist['EMA20'] = ema20
        hist['MACD'] = macd
        hist['SIGNAL'] = signal
        hist['UPPER_BB'] = upper_bb
        hist['LOWER_BB'] = lower_bb
        hist['K_PERCENT'] = k_percent
        hist['D_PERCENT'] = d_percent
        hist['ADX'] = adx
        
        # Backtest engine
        capital = float(initial_capital)
        position = None  # 'long' or None
        shares = 0
        trades = []
        equity_curve = []
        entry_prices = []
        
        for i in range(50, len(hist)):  # Start from 50 to ensure indicators are calculated
            current_price = float(close.iloc[i])
            
            # Get all indicator values for current row
            indicators = {
                'RSI': float(rsi.iloc[i]) if not np.isnan(rsi.iloc[i]) else 50,
                'SMA20': float(sma20.iloc[i]) if not np.isnan(sma20.iloc[i]) else current_price,
                'SMA50': float(sma50.iloc[i]) if not np.isnan(sma50.iloc[i]) else current_price,
                'EMA20': float(ema20.iloc[i]) if not np.isnan(ema20.iloc[i]) else current_price,
                'MACD': float(macd.iloc[i]) if not np.isnan(macd.iloc[i]) else 0,
                'SIGNAL': float(signal.iloc[i]) if not np.isnan(signal.iloc[i]) else 0,
                'UPPER_BB': float(upper_bb.iloc[i]) if not np.isnan(upper_bb.iloc[i]) else current_price,
                'LOWER_BB': float(lower_bb.iloc[i]) if not np.isnan(lower_bb.iloc[i]) else current_price,
                'K_PERCENT': float(k_percent.iloc[i]) if not np.isnan(k_percent.iloc[i]) else 50,
                'D_PERCENT': float(d_percent.iloc[i]) if not np.isnan(d_percent.iloc[i]) else 50,
                'ADX': float(adx.iloc[i]) if not np.isnan(adx.iloc[i]) else 0,
                'PRICE': current_price
            }
            
            # Calculate current equity
            current_equity = capital + (shares * current_price if position == 'long' else 0)
            equity_curve.append({"date": hist.index[i].isoformat(), "value": round(current_equity, 2)})
            
            # Evaluate entry condition
            if position != 'long' and evaluate_condition(entry_condition, indicators):
                # Buy signal
                shares = capital / current_price
                entry_prices.append(current_price)
                trades.append({
                    "type": "BUY",
                    "price": round(current_price, 2),
                    "date": hist.index[i].isoformat(),
                    "reason": entry_condition
                })
                capital = 0
                position = 'long'
            # Evaluate exit condition
            elif position == 'long' and evaluate_condition(exit_condition, indicators):
                # Sell signal
                capital = shares * current_price
                entry_price = entry_prices.pop(0) if entry_prices else current_price
                pnl = ((current_price - entry_price) / entry_price) * 100
                trades.append({
                    "type": "SELL",
                    "price": round(current_price, 2),
                    "date": hist.index[i].isoformat(),
                    "reason": exit_condition,
                    "pnl": round(pnl, 2)
                })
                shares = 0
                position = None
        
        # Close final position
        if position == 'long':
            final_price = float(close.iloc[-1])
            capital = shares * final_price
        
        # Calculate metrics
        final_capital = capital
        total_return = ((final_capital - initial_capital) / initial_capital) * 100
        
        # Count trades and calculate win rate
        buy_trades = [t for t in trades if t['type'] == 'BUY']
        sell_trades = [t for t in trades if t['type'] == 'SELL']
        num_trades = len(buy_trades)
        
        winning_trades = len([t for t in sell_trades if 'pnl' in t and t['pnl'] > 0])
        win_rate = (winning_trades / num_trades * 100) if num_trades > 0 else 0
        
        # Calculate Sharpe ratio
        returns = []
        for i in range(1, len(equity_curve)):
            if equity_curve[i-1]['value'] > 0:
                ret = (equity_curve[i]['value'] / equity_curve[i-1]['value']) - 1
                returns.append(ret)
        
        avg_return = np.mean(returns) if returns else 0
        std_return = np.std(returns) if returns else 1
        sharpe = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0
        
        # Calculate max drawdown
        peak = initial_capital
        max_drawdown = 0
        for point in equity_curve:
            if point['value'] > peak:
                peak = point['value']
            drawdown = ((point['value'] - peak) / peak) * 100
            if drawdown < max_drawdown:
                max_drawdown = drawdown
        
        # Calculate CAGR (assuming 2 years)
        years = 2.0
        if final_capital > 0 and initial_capital > 0:
            cagr = (((final_capital / initial_capital) ** (1/years)) - 1) * 100
        else:
            cagr = 0
        
        return {
            "symbol": symbol,
            "initialCapital": initial_capital,
            "finalCapital": round(final_capital, 2),
            "totalReturn": round(total_return, 2),
            "trades": num_trades,
            "winRate": round(win_rate, 2),
            "sharpeRatio": round(sharpe, 2),
            "maxDrawdown": round(max_drawdown, 2),
            "cagr": round(cagr, 2),
            "buyAndHold": round(((close.iloc[-1] - close.iloc[50]) / close.iloc[50] * 100), 2),
            "equity": equity_curve[-min(30, len(equity_curve)):]  # Last 30 points
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/ml/predict/{symbol}")
async def ml_predict(symbol: str):
    """
    Predict future stock prices using ML model
    """
    try:
        ticker = yf.Ticker(symbol)
        
        # Get current info
        info = ticker.info
        current_price = info.get('currentPrice', 0)
        
        # Get historical data for ML model
        hist = ticker.history(period="2y")
        if hist.empty:
            raise HTTPException(status_code=404, detail="No data found")
        
        close_prices = hist['Close']
        
        # Use the predictor
        close_prices_list = close_prices.tolist()
        
        # Calculate indicators for enhanced prediction
        sma_20 = float(calculate_sma(close_prices, 20).iloc[-1])
        sma_50 = float(calculate_sma(close_prices, 50).iloc[-1])
        rsi = calculate_rsi(close_prices, 14).iloc[-1]
        
        # Get predictions using ML predictor
        if ml_predictor:
            try:
                predictions_7d = ml_predictor.predict(close_prices_list, periods=7)
                predictions_30d = ml_predictor.predict(close_prices_list, periods=30)
                confidence = ml_predictor.get_confidence(close_prices_list)
                
                # Fallback if predictions are None
                if predictions_7d is None or predictions_30d is None:
                    predictions_7d = [float(current_price * (1 + 0.002 * i)) for i in range(1, 8)]
                    predictions_30d = [float(current_price * (1 + 0.002 * i)) for i in range(1, 31)]
                    confidence = 65
            except Exception as e:
                print(f"Prediction error: {e}")
                # Fallback if predictor fails
                predictions_7d = [float(current_price * (1 + 0.002 * i)) for i in range(1, 8)]
                predictions_30d = [float(current_price * (1 + 0.002 * i)) for i in range(1, 31)]
                confidence = 65
        else:
            # Fallback prediction
            predictions_7d = [float(current_price * (1 + 0.002 * i)) for i in range(1, 8)]
            predictions_30d = [float(current_price * (1 + 0.002 * i)) for i in range(1, 31)]
            confidence = 65
        
        # Calculate momentum
        price_change = close_prices.iloc[-1] - close_prices.iloc[-20]
        momentum = float((price_change / close_prices.iloc[-20]) * 100) if close_prices.iloc[-20] > 0 else 0
        
        # Determine trend
        if sma_20 > sma_50 and momentum > 0:
            trend = "bullish"
        elif sma_20 < sma_50 and momentum < 0:
            trend = "bearish"
        else:
            trend = "neutral"
        
        
        # Calculate predictions - ensure all are Python floats
        predicted_price_7d = float(predictions_7d[-1]) if predictions_7d else float(current_price)
        predicted_price_30d = float(predictions_30d[-1]) if predictions_30d else float(current_price)
        
        predicted_change_7d = ((predicted_price_7d - current_price) / current_price) * 100
        
        return {
            "symbol": symbol,
            "currentPrice": round(float(current_price), 2),
            "predictedPrice7d": round(predicted_price_7d, 2),
            "predictedPrice30d": round(predicted_price_30d, 2),
            "predictedChange7d": round(float(predicted_change_7d), 2),
            "confidence": round(float(confidence), 1),
            "trend": trend,
            "momentum": round(float(momentum), 2),
            "predictionData": {
                "7d": [round(float(p), 2) for p in predictions_7d] if predictions_7d else [],
                "30d": [round(float(p), 2) for p in predictions_30d] if predictions_30d else []
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/market-stats/{symbol}")
async def get_market_stats(symbol: str):
    """
    Get comprehensive market statistics for a symbol
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        hist = ticker.history(period="1y")
        
        if hist.empty:
            raise HTTPException(status_code=404, detail="No data found")
        
        # Calculate various statistics
        close = hist['Close']
        high_52w = close.max()
        low_52w = close.min()
        current = close.iloc[-1]
        
        # Volatility
        returns = close.pct_change().dropna()
        volatility = returns.std() * (252 ** 0.5) * 100  # Annualized volatility
        
        # Beta (simplified)
        market_ticker = yf.Ticker("^GSPC")
        market_hist = market_ticker.history(period="1y")
        if not market_hist.empty:
            market_returns = market_hist['Close'].pct_change().dropna()
            common_dates = returns.index.intersection(market_returns.index)
            if len(common_dates) > 1:
                covariance = returns.loc[common_dates].cov(market_returns.loc[common_dates])
                variance = market_returns.var()
                beta = float(covariance / variance) if variance > 0 else 1.0
            else:
                beta = 1.0
        else:
            beta = 1.0
        
        return {
            "symbol": symbol,
            "currentPrice": round(float(current), 2),
            "high52Week": round(float(high_52w), 2),
            "low52Week": round(float(low_52w), 2),
            "volatility": round(float(volatility), 2),
            "beta": round(float(beta), 2),
            "marketCap": info.get('marketCap', 0),
            "peRatio": info.get('trailingPE', 0),
            "dividendYield": info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

