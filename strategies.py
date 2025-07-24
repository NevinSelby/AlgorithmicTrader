import backtrader as bt
from datetime import datetime
import pandas as pd
import numpy as np


class SMACrossoverStrategy(bt.Strategy):
    """
    Simple Moving Average Crossover Strategy.
    
    Buy when fast SMA crosses above slow SMA.
    Sell when fast SMA crosses below slow SMA.
    """
    
    params = (
        ('fast', 20),  # Fast moving average period
        ('slow', 50),  # Slow moving average period
    )
    
    def __init__(self):
        """Initialize the strategy with indicators and tracking variables."""
        # Calculate the moving averages
        self.fast_ma = bt.indicators.SimpleMovingAverage(
            self.datas[0].close, period=self.params.fast
        )
        self.slow_ma = bt.indicators.SimpleMovingAverage(
            self.datas[0].close, period=self.params.slow
        )
        
        # Create crossover signal
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)
        
        # Track orders and trades
        self.order = None
        self.buy_price = None
        self.buy_comm = None
        self.trade_count = 0
        
    def log(self, txt, dt=None):
        """Logging function for strategy events."""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}: {txt}')
    
    def notify_order(self, order):
        """Notification method for order status changes."""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED: Price: {order.executed.price:.2f}, '
                        f'Cost: {order.executed.value:.2f}, '
                        f'Commission: {order.executed.comm:.2f}')
                self.buy_price = order.executed.price
                self.buy_comm = order.executed.comm
            else:
                self.log(f'SELL EXECUTED: Price: {order.executed.price:.2f}, '
                        f'Cost: {order.executed.value:.2f}, '
                        f'Commission: {order.executed.comm:.2f}')
                
                gross_pnl = (order.executed.price - self.buy_price) * order.executed.size
                net_pnl = gross_pnl - self.buy_comm - order.executed.comm
                self.log(f'OPERATION PROFIT: Gross: {gross_pnl:.2f}, Net: {net_pnl:.2f}')
                
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        
        self.order = None
    
    def notify_trade(self, trade):
        """Notification method for completed trades."""
        if not trade.isclosed:
            return
        
        self.trade_count += 1
        self.log(f'TRADE #{self.trade_count} CLOSED: PnL: {trade.pnl:.2f}, '
                f'Net PnL: {trade.pnlcomm:.2f}')
    
    def next(self):
        """Main strategy logic executed on each bar."""
        if self.order:
            return
        
        current_close = self.datas[0].close[0]
        
        if not self.position:
            if self.crossover > 0:
                self.log(f'BUY SIGNAL: Fast MA: {self.fast_ma[0]:.2f}, '
                        f'Slow MA: {self.slow_ma[0]:.2f}, Close: {current_close:.2f}')
                self.order = self.buy()
        else:
            if self.crossover < 0:
                self.log(f'SELL SIGNAL: Fast MA: {self.fast_ma[0]:.2f}, '
                        f'Slow MA: {self.slow_ma[0]:.2f}, Close: {current_close:.2f}')
                self.order = self.sell()
    
    def stop(self):
        """Called when the strategy execution is complete."""
        self.log(f'Strategy completed with final portfolio value: {self.broker.getvalue():.2f}')
        self.log(f'Total trades executed: {self.trade_count}')


class RSIStrategy(bt.Strategy):
    """
    RSI (Relative Strength Index) Strategy.
    
    Buy when RSI is oversold (below lower threshold).
    Sell when RSI is overbought (above upper threshold).
    """
    
    params = (
        ('period', 14),      # RSI calculation period
        ('upper', 70),       # Overbought threshold
        ('lower', 30),       # Oversold threshold
    )
    
    def __init__(self):
        """Initialize the strategy with RSI indicator."""
        self.rsi = bt.indicators.RelativeStrengthIndex(
            self.datas[0].close, period=self.params.period
        )
        
        self.order = None
        self.buy_price = None
        self.buy_comm = None
        self.trade_count = 0
    
    def log(self, txt, dt=None):
        """Logging function for strategy events."""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}: {txt}')
    
    def notify_order(self, order):
        """Notification method for order status changes."""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED: Price: {order.executed.price:.2f}, RSI: {self.rsi[0]:.2f}')
                self.buy_price = order.executed.price
                self.buy_comm = order.executed.comm
            else:
                self.log(f'SELL EXECUTED: Price: {order.executed.price:.2f}, RSI: {self.rsi[0]:.2f}')
                
                gross_pnl = (order.executed.price - self.buy_price) * order.executed.size
                net_pnl = gross_pnl - self.buy_comm - order.executed.comm
                self.log(f'OPERATION PROFIT: Gross: {gross_pnl:.2f}, Net: {net_pnl:.2f}')
        
        self.order = None
    
    def notify_trade(self, trade):
        """Notification method for completed trades."""
        if not trade.isclosed:
            return
        
        self.trade_count += 1
        self.log(f'TRADE #{self.trade_count} CLOSED: PnL: {trade.pnl:.2f}')
    
    def next(self):
        """Main strategy logic executed on each bar."""
        if self.order:
            return
        
        current_close = self.datas[0].close[0]
        current_rsi = self.rsi[0]
        
        if not self.position:
            # Buy when RSI is oversold
            if current_rsi < self.params.lower:
                self.log(f'BUY SIGNAL: RSI: {current_rsi:.2f}, Close: {current_close:.2f}')
                self.order = self.buy()
        else:
            # Sell when RSI is overbought
            if current_rsi > self.params.upper:
                self.log(f'SELL SIGNAL: RSI: {current_rsi:.2f}, Close: {current_close:.2f}')
                self.order = self.sell()
    
    def stop(self):
        """Called when the strategy execution is complete."""
        self.log(f'RSI Strategy completed with final portfolio value: {self.broker.getvalue():.2f}')
        self.log(f'Total trades executed: {self.trade_count}')


class MACDStrategy(bt.Strategy):
    """
    MACD (Moving Average Convergence Divergence) Strategy.
    
    Buy when MACD line crosses above signal line.
    Sell when MACD line crosses below signal line.
    """
    
    params = (
        ('me1', 12),    # Fast EMA period
        ('me2', 26),    # Slow EMA period
        ('signal', 9),  # Signal line period
    )
    
    def __init__(self):
        """Initialize the strategy with MACD indicator."""
        self.macd = bt.indicators.MACD(
            self.datas[0].close,
            me1=self.params.me1,
            me2=self.params.me2,
            signal=self.params.signal
        )
        
        # MACD crossover signal
        self.crossover = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)
        
        self.order = None
        self.buy_price = None
        self.buy_comm = None
        self.trade_count = 0
    
    def log(self, txt, dt=None):
        """Logging function for strategy events."""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}: {txt}')
    
    def notify_order(self, order):
        """Notification method for order status changes."""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED: Price: {order.executed.price:.2f}, '
                        f'MACD: {self.macd.macd[0]:.4f}, Signal: {self.macd.signal[0]:.4f}')
                self.buy_price = order.executed.price
                self.buy_comm = order.executed.comm
            else:
                self.log(f'SELL EXECUTED: Price: {order.executed.price:.2f}, '
                        f'MACD: {self.macd.macd[0]:.4f}, Signal: {self.macd.signal[0]:.4f}')
                
                gross_pnl = (order.executed.price - self.buy_price) * order.executed.size
                net_pnl = gross_pnl - self.buy_comm - order.executed.comm
                self.log(f'OPERATION PROFIT: Gross: {gross_pnl:.2f}, Net: {net_pnl:.2f}')
        
        self.order = None
    
    def notify_trade(self, trade):
        """Notification method for completed trades."""
        if not trade.isclosed:
            return
        
        self.trade_count += 1
        self.log(f'TRADE #{self.trade_count} CLOSED: PnL: {trade.pnl:.2f}')
    
    def next(self):
        """Main strategy logic executed on each bar."""
        if self.order:
            return
        
        current_close = self.datas[0].close[0]
        
        if not self.position:
            # Buy when MACD crosses above signal line
            if self.crossover > 0:
                self.log(f'BUY SIGNAL: MACD: {self.macd.macd[0]:.4f}, '
                        f'Signal: {self.macd.signal[0]:.4f}, Close: {current_close:.2f}')
                self.order = self.buy()
        else:
            # Sell when MACD crosses below signal line
            if self.crossover < 0:
                self.log(f'SELL SIGNAL: MACD: {self.macd.macd[0]:.4f}, '
                        f'Signal: {self.macd.signal[0]:.4f}, Close: {current_close:.2f}')
                self.order = self.sell()
    
    def stop(self):
        """Called when the strategy execution is complete."""
        self.log(f'MACD Strategy completed with final portfolio value: {self.broker.getvalue():.2f}')
        self.log(f'Total trades executed: {self.trade_count}')


class BollingerBandsStrategy(bt.Strategy):
    """
    Bollinger Bands Strategy.
    
    Buy when price touches lower band.
    Sell when price touches upper band.
    """
    
    params = (
        ('period', 20),    # Moving average period
        ('devfactor', 2),  # Standard deviation factor
    )
    
    def __init__(self):
        """Initialize the strategy with Bollinger Bands indicator."""
        self.bollinger = bt.indicators.BollingerBands(
            self.datas[0].close,
            period=self.params.period,
            devfactor=self.params.devfactor
        )
        
        self.order = None
        self.buy_price = None
        self.buy_comm = None
        self.trade_count = 0
    
    def log(self, txt, dt=None):
        """Logging function for strategy events."""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}: {txt}')
    
    def notify_order(self, order):
        """Notification method for order status changes."""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED: Price: {order.executed.price:.2f}, '
                        f'Lower Band: {self.bollinger.lines.bot[0]:.2f}')
                self.buy_price = order.executed.price
                self.buy_comm = order.executed.comm
            else:
                self.log(f'SELL EXECUTED: Price: {order.executed.price:.2f}, '
                        f'Upper Band: {self.bollinger.lines.top[0]:.2f}')
                
                gross_pnl = (order.executed.price - self.buy_price) * order.executed.size
                net_pnl = gross_pnl - self.buy_comm - order.executed.comm
                self.log(f'OPERATION PROFIT: Gross: {gross_pnl:.2f}, Net: {net_pnl:.2f}')
        
        self.order = None
    
    def notify_trade(self, trade):
        """Notification method for completed trades."""
        if not trade.isclosed:
            return
        
        self.trade_count += 1
        self.log(f'TRADE #{self.trade_count} CLOSED: PnL: {trade.pnl:.2f}')
    
    def next(self):
        """Main strategy logic executed on each bar."""
        if self.order:
            return
        
        current_close = self.datas[0].close[0]
        upper_band = self.bollinger.lines.top[0]
        lower_band = self.bollinger.lines.bot[0]
        
        if not self.position:
            # Buy when price touches or goes below lower band
            if current_close <= lower_band:
                self.log(f'BUY SIGNAL: Close: {current_close:.2f}, Lower Band: {lower_band:.2f}')
                self.order = self.buy()
        else:
            # Sell when price touches or goes above upper band
            if current_close >= upper_band:
                self.log(f'SELL SIGNAL: Close: {current_close:.2f}, Upper Band: {upper_band:.2f}')
                self.order = self.sell()
    
    def stop(self):
        """Called when the strategy execution is complete."""
        self.log(f'Bollinger Bands Strategy completed with final portfolio value: {self.broker.getvalue():.2f}')
        self.log(f'Total trades executed: {self.trade_count}')


class MeanReversionStrategy(bt.Strategy):
    """
    Mean Reversion Strategy.
    
    Buy when price is significantly below moving average.
    Sell when price is significantly above moving average.
    """
    
    params = (
        ('period', 20),       # Moving average period
        ('threshold', 0.02),  # Percentage threshold for mean reversion (2%)
    )
    
    def __init__(self):
        """Initialize the strategy with moving average indicator."""
        self.sma = bt.indicators.SimpleMovingAverage(
            self.datas[0].close, period=self.params.period
        )
        
        self.order = None
        self.buy_price = None
        self.buy_comm = None
        self.trade_count = 0
    
    def log(self, txt, dt=None):
        """Logging function for strategy events."""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}: {txt}')
    
    def notify_order(self, order):
        """Notification method for order status changes."""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED: Price: {order.executed.price:.2f}, SMA: {self.sma[0]:.2f}')
                self.buy_price = order.executed.price
                self.buy_comm = order.executed.comm
            else:
                self.log(f'SELL EXECUTED: Price: {order.executed.price:.2f}, SMA: {self.sma[0]:.2f}')
                
                gross_pnl = (order.executed.price - self.buy_price) * order.executed.size
                net_pnl = gross_pnl - self.buy_comm - order.executed.comm
                self.log(f'OPERATION PROFIT: Gross: {gross_pnl:.2f}, Net: {net_pnl:.2f}')
        
        self.order = None
    
    def notify_trade(self, trade):
        """Notification method for completed trades."""
        if not trade.isclosed:
            return
        
        self.trade_count += 1
        self.log(f'TRADE #{self.trade_count} CLOSED: PnL: {trade.pnl:.2f}')
    
    def next(self):
        """Main strategy logic executed on each bar."""
        if self.order:
            return
        
        current_close = self.datas[0].close[0]
        current_sma = self.sma[0]
        
        # Calculate percentage deviation from mean
        deviation = (current_close - current_sma) / current_sma
        
        if not self.position:
            # Buy when price is significantly below SMA (mean reversion opportunity)
            if deviation < -self.params.threshold:
                self.log(f'BUY SIGNAL: Close: {current_close:.2f}, SMA: {current_sma:.2f}, '
                        f'Deviation: {deviation:.2%}')
                self.order = self.buy()
        else:
            # Sell when price is significantly above SMA or returns close to mean
            if deviation > self.params.threshold or abs(deviation) < 0.005:
                self.log(f'SELL SIGNAL: Close: {current_close:.2f}, SMA: {current_sma:.2f}, '
                        f'Deviation: {deviation:.2%}')
                self.order = self.sell()
    
    def stop(self):
        """Called when the strategy execution is complete."""
        self.log(f'Mean Reversion Strategy completed with final portfolio value: {self.broker.getvalue():.2f}')
        self.log(f'Total trades executed: {self.trade_count}')


class MLEnhancedStrategy(bt.Strategy):
    """
    ML-Enhanced Strategy combining LSTM price predictions with technical analysis.
    
    Uses a trained LSTM model to predict future prices and combines these predictions
    with multiple technical indicators for more accurate trading signals.
    """
    
    params = (
        ('prediction_threshold', 2.0),    # ML prediction threshold (%)
        ('confidence_threshold', 0.6),    # Minimum confidence for trades
        ('rsi_period', 14),               # RSI period
        ('bb_period', 20),                # Bollinger Bands period
        ('bb_std', 2.0),                  # Bollinger Bands standard deviation
    )
    
    def __init__(self):
        """Initialize the ML-enhanced strategy."""
        # Technical indicators
        self.rsi = bt.indicators.RelativeStrengthIndex(
            self.datas[0].close, period=self.params.rsi_period
        )
        
        self.bollinger = bt.indicators.BollingerBands(
            self.datas[0].close,
            period=self.params.bb_period,
            devfactor=self.params.bb_std
        )
        
        self.macd = bt.indicators.MACD(self.datas[0].close)
        
        # Moving averages for additional signals
        self.sma_20 = bt.indicators.SimpleMovingAverage(self.datas[0].close, period=20)
        self.sma_50 = bt.indicators.SimpleMovingAverage(self.datas[0].close, period=50)
        
        # Track orders and trades
        self.order = None
        self.buy_price = None
        self.buy_comm = None
        self.trade_count = 0
        
        # ML prediction storage
        self.ml_predictions = {}
        self.last_ml_signal = 'HOLD'
        self.ml_confidence = 0.0
        
        # Initialize data collection for ML
        self.price_data = []
        self.min_data_points = 100  # Minimum points needed for ML
        
    def log(self, txt, dt=None):
        """Logging function for strategy events."""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}: {txt}')
    
    def notify_order(self, order):
        """Notification method for order status changes."""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED: Price: {order.executed.price:.2f}, '
                        f'ML Signal: {self.last_ml_signal}, Confidence: {self.ml_confidence:.2f}')
                self.buy_price = order.executed.price
                self.buy_comm = order.executed.comm
            else:
                self.log(f'SELL EXECUTED: Price: {order.executed.price:.2f}, '
                        f'ML Signal: {self.last_ml_signal}, Confidence: {self.ml_confidence:.2f}')
                
                gross_pnl = (order.executed.price - self.buy_price) * order.executed.size
                net_pnl = gross_pnl - self.buy_comm - order.executed.comm
                self.log(f'OPERATION PROFIT: Gross: {gross_pnl:.2f}, Net: {net_pnl:.2f}')
        
        self.order = None
    
    def notify_trade(self, trade):
        """Notification method for completed trades."""
        if not trade.isclosed:
            return
        
        self.trade_count += 1
        self.log(f'TRADE #{self.trade_count} CLOSED: PnL: {trade.pnl:.2f}')
    
    def collect_data(self):
        """Collect current market data for ML prediction."""
        current_data = {
            'Open': self.datas[0].open[0],
            'High': self.datas[0].high[0],
            'Low': self.datas[0].low[0],
            'Close': self.datas[0].close[0],
            'Volume': self.datas[0].volume[0],
            'Date': self.datas[0].datetime.date(0)
        }
        self.price_data.append(current_data)
        
        # Keep only recent data to manage memory
        if len(self.price_data) > 300:
            self.price_data = self.price_data[-250:]
    
    def get_ml_signal(self):
        """Get ML-based trading signal using simplified heuristics."""
        if len(self.price_data) < self.min_data_points:
            return 'HOLD', 0.5, "Insufficient data"
        
        try:
            # Convert to DataFrame for analysis
            df = pd.DataFrame(self.price_data)
            df.set_index('Date', inplace=True)
            
            # Calculate technical features similar to LSTM
            df['Price_Change'] = df['Close'].pct_change()
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['Volatility'] = df['Close'].rolling(window=20).std()
            
            # Simple ML-like prediction using recent trends and volatility
            recent_data = df.tail(20)
            
            if len(recent_data) < 20:
                return 'HOLD', 0.5, "Insufficient recent data"
            
            # Calculate momentum and trend strength
            price_momentum = recent_data['Price_Change'].tail(5).mean()
            trend_strength = abs(recent_data['Close'].iloc[-1] - recent_data['SMA_20'].iloc[-1]) / recent_data['SMA_20'].iloc[-1]
            volatility_norm = recent_data['Volatility'].iloc[-1] / recent_data['Close'].iloc[-1]
            
            # Simulate ML prediction based on patterns
            current_price = recent_data['Close'].iloc[-1]
            sma_20 = recent_data['SMA_20'].iloc[-1]
            sma_50 = recent_data['SMA_50'].iloc[-1]
            
            # Prediction logic (simplified neural network-like decision)
            prediction_score = 0
            confidence_factors = []
            
            # Momentum factor
            if price_momentum > 0.01:
                prediction_score += 2
                confidence_factors.append(0.2)
            elif price_momentum < -0.01:
                prediction_score -= 2
                confidence_factors.append(0.2)
            
            # Trend factor
            if current_price > sma_20 and sma_20 > sma_50:
                prediction_score += 1.5
                confidence_factors.append(0.15)
            elif current_price < sma_20 and sma_20 < sma_50:
                prediction_score -= 1.5
                confidence_factors.append(0.15)
            
            # Volatility factor (lower volatility = higher confidence)
            if volatility_norm < 0.02:
                confidence_factors.append(0.2)
            else:
                confidence_factors.append(0.1)
            
            # Calculate predicted change
            predicted_change = prediction_score * 0.5  # Scale to percentage
            
            # Determine signal
            if predicted_change > self.params.prediction_threshold:
                signal = 'BUY'
            elif predicted_change < -self.params.prediction_threshold:
                signal = 'SELL'
            else:
                signal = 'HOLD'
            
            # Calculate confidence
            confidence = min(0.9, sum(confidence_factors) + 0.3)
            
            reason = f"ML Prediction: {predicted_change:.2f}%, Momentum: {price_momentum:.3f}"
            
            return signal, confidence, reason
            
        except Exception as e:
            return 'HOLD', 0.3, f"ML Error: {str(e)}"
    
    def next(self):
        """Main strategy logic executed on each bar."""
        if self.order:
            return
        
        # Collect data for ML analysis
        self.collect_data()
        
        # Get current market data
        current_close = self.datas[0].close[0]
        current_rsi = self.rsi[0]
        bb_upper = self.bollinger.lines.top[0]
        bb_lower = self.bollinger.lines.bot[0]
        bb_middle = self.bollinger.lines.mid[0]
        
        # Get ML signal
        ml_signal, ml_confidence, ml_reason = self.get_ml_signal()
        self.last_ml_signal = ml_signal
        self.ml_confidence = ml_confidence
        
        # Skip trading if confidence is too low
        if ml_confidence < self.params.confidence_threshold:
            return
        
        # Generate technical signals
        rsi_signal = 'BUY' if current_rsi < 30 else 'SELL' if current_rsi > 70 else 'HOLD'
        bb_signal = 'BUY' if current_close <= bb_lower else 'SELL' if current_close >= bb_upper else 'HOLD'
        macd_signal = 'BUY' if self.macd.macd[0] > self.macd.signal[0] else 'SELL'
        trend_signal = 'BUY' if self.sma_20[0] > self.sma_50[0] else 'SELL'
        
        # Combine all signals with ML having higher weight
        buy_votes = 0
        sell_votes = 0
        
        # ML signal (weight: 2)
        if ml_signal == 'BUY':
            buy_votes += 2
        elif ml_signal == 'SELL':
            sell_votes += 2
        
        # Technical signals (weight: 1 each)
        for signal in [rsi_signal, bb_signal, macd_signal, trend_signal]:
            if signal == 'BUY':
                buy_votes += 1
            elif signal == 'SELL':
                sell_votes += 1
        
        # Decision logic
        if not self.position:
            # Enter long position
            if buy_votes >= 3 and buy_votes > sell_votes:
                self.log(f'BUY SIGNAL: ML={ml_signal}({ml_confidence:.2f}), '
                        f'RSI={rsi_signal}, BB={bb_signal}, MACD={macd_signal}, '
                        f'Votes: Buy={buy_votes}, Sell={sell_votes}, {ml_reason}')
                self.order = self.buy()
        else:
            # Exit position
            if sell_votes >= 3 and sell_votes > buy_votes:
                self.log(f'SELL SIGNAL: ML={ml_signal}({ml_confidence:.2f}), '
                        f'RSI={rsi_signal}, BB={bb_signal}, MACD={macd_signal}, '
                        f'Votes: Buy={buy_votes}, Sell={sell_votes}, {ml_reason}')
                self.order = self.sell()
    
    def stop(self):
        """Called when the strategy execution is complete."""
        self.log(f'ML-Enhanced Strategy completed with final portfolio value: {self.broker.getvalue():.2f}')
        self.log(f'Total trades executed: {self.trade_count}')
        self.log(f'Data points collected: {len(self.price_data)}')


# Strategy registry for easy access
STRATEGIES = {
    'SMA Crossover': SMACrossoverStrategy,
    'RSI': RSIStrategy,
    'MACD': MACDStrategy,
    'Bollinger Bands': BollingerBandsStrategy,
    'Mean Reversion': MeanReversionStrategy,
    'ML Enhanced': MLEnhancedStrategy,
}

# Strategy parameter configurations
STRATEGY_PARAMS = {
    'SMA Crossover': {
        'fast': {'min': 5, 'max': 50, 'default': 20, 'step': 1, 'help': 'Fast moving average period'},
        'slow': {'min': 20, 'max': 200, 'default': 50, 'step': 1, 'help': 'Slow moving average period'},
    },
    'RSI': {
        'period': {'min': 5, 'max': 30, 'default': 14, 'step': 1, 'help': 'RSI calculation period'},
        'upper': {'min': 60, 'max': 90, 'default': 70, 'step': 1, 'help': 'Overbought threshold'},
        'lower': {'min': 10, 'max': 40, 'default': 30, 'step': 1, 'help': 'Oversold threshold'},
    },
    'MACD': {
        'me1': {'min': 5, 'max': 20, 'default': 12, 'step': 1, 'help': 'Fast EMA period'},
        'me2': {'min': 15, 'max': 40, 'default': 26, 'step': 1, 'help': 'Slow EMA period'},
        'signal': {'min': 5, 'max': 15, 'default': 9, 'step': 1, 'help': 'Signal line period'},
    },
    'Bollinger Bands': {
        'period': {'min': 10, 'max': 50, 'default': 20, 'step': 1, 'help': 'Moving average period'},
        'devfactor': {'min': 1.0, 'max': 3.0, 'default': 2.0, 'step': 0.1, 'help': 'Standard deviation factor'},
    },
    'Mean Reversion': {
        'period': {'min': 10, 'max': 50, 'default': 20, 'step': 1, 'help': 'Moving average period'},
        'threshold': {'min': 0.01, 'max': 0.10, 'default': 0.02, 'step': 0.001, 'help': 'Reversion threshold (as decimal)'},
    },
    'ML Enhanced': {
        'prediction_threshold': {'min': 0.5, 'max': 5.0, 'default': 2.0, 'step': 0.1, 'help': 'ML prediction threshold (%)'},
        'confidence_threshold': {'min': 0.3, 'max': 0.9, 'default': 0.6, 'step': 0.05, 'help': 'Minimum confidence for trades'},
        'rsi_period': {'min': 5, 'max': 30, 'default': 14, 'step': 1, 'help': 'RSI period'},
        'bb_period': {'min': 10, 'max': 50, 'default': 20, 'step': 1, 'help': 'Bollinger Bands period'},
        'bb_std': {'min': 1.0, 'max': 3.0, 'default': 2.0, 'step': 0.1, 'help': 'Bollinger Bands standard deviation'},
    },
} 