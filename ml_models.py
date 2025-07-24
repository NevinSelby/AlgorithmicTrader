import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# Try to import sentiment analysis
try:
    from news_sentiment import NewsSentimentAggregator
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False
    print("Sentiment analysis not available. Install required NLP packages.")


class LSTMModel(nn.Module):
    """
    PyTorch LSTM model for time series prediction.
    """
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=dropout)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True, dropout=dropout)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size//2, batch_first=True, dropout=dropout)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size//2, 25)
        self.fc2 = nn.Linear(25, output_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # LSTM layers with residual connections
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.dropout(lstm1_out)
        
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm2_out = self.dropout(lstm2_out)
        
        lstm3_out, _ = self.lstm3(lstm2_out)
        lstm3_out = self.dropout(lstm3_out)
        
        # Take the last timestep output
        last_output = lstm3_out[:, -1, :]
        
        # Fully connected layers
        fc1_out = self.relu(self.fc1(last_output))
        fc1_out = self.dropout(fc1_out)
        output = self.fc2(fc1_out)
        
        return output


class LSTMPredictor:
    """
    LSTM model for stock price prediction based on historical data.
    """
    
    def __init__(self, sequence_length=60, prediction_days=5, include_sentiment=True):
        """
        Initialize LSTM predictor.
        
        Args:
            sequence_length (int): Number of days to look back for prediction
            prediction_days (int): Number of days to predict ahead
            include_sentiment (bool): Whether to include sentiment analysis
        """
        self.sequence_length = sequence_length
        self.prediction_days = prediction_days
        self.include_sentiment = include_sentiment and SENTIMENT_AVAILABLE
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_trained = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.optimizer = None
        self.criterion = None
        
        # Initialize sentiment aggregator if available
        if self.include_sentiment:
            try:
                self.sentiment_aggregator = NewsSentimentAggregator()
                print("✅ Sentiment analysis enabled")
            except Exception as e:
                print(f"❌ Sentiment analysis failed to initialize: {e}")
                self.include_sentiment = False
                self.sentiment_aggregator = None
        else:
            self.sentiment_aggregator = None
        
    def create_features(self, data, ticker=None):
        """
        Create technical indicator features from OHLCV data.
        
        Args:
            data (pd.DataFrame): OHLCV data
            ticker (str): Stock ticker for sentiment analysis
            
        Returns:
            pd.DataFrame: Enhanced data with technical indicators and sentiment
        """
        df = data.copy()
        
        # Price-based features
        df['Price_Change'] = df['Close'].pct_change()
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Open_Close_Ratio'] = df['Open'] / df['Close']
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'Price_to_SMA_{window}'] = df['Close'] / df[f'SMA_{window}']
        
        # Volatility features
        df['Volatility_10'] = df['Close'].rolling(window=10).std()
        df['Volatility_20'] = df['Close'].rolling(window=20).std()
        
        # Volume features
        df['Volume_SMA_10'] = df['Volume'].rolling(window=10).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_10']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        bb_window = 20
        bb_std = 2
        df['BB_Middle'] = df['Close'].rolling(window=bb_window).mean()
        bb_std_dev = df['Close'].rolling(window=bb_window).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std_dev * bb_std)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std_dev * bb_std)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Add sentiment features if available
        if self.include_sentiment and ticker:
            df = self._add_sentiment_features(df, ticker)
        
        # Remove NaN values
        df = df.dropna()
        
        return df
    
    def _add_sentiment_features(self, df, ticker):
        """
        Add sentiment analysis features to the dataframe.
        
        Args:
            df (pd.DataFrame): Dataframe with OHLCV and technical indicators
            ticker (str): Stock ticker for sentiment analysis
            
        Returns:
            pd.DataFrame: Dataframe with added sentiment features
        """
        try:
            print(f"🔍 Fetching sentiment data for {ticker}...")
            
            # Get recent sentiment data
            current_sentiment = self.sentiment_aggregator.get_sentiment_data(
                ticker, days_back=7, max_articles_per_source=3
            )
            
            # Get historical sentiment (simplified - in practice you'd store this)
            historical_sentiment = self.sentiment_aggregator.get_historical_sentiment(
                ticker, days=len(df)
            )
            
            # If we have historical data, merge it
            if not historical_sentiment.empty and len(historical_sentiment) >= len(df):
                # Align historical sentiment with our data
                sentiment_values = historical_sentiment['sentiment_score'].tail(len(df)).values
                confidence_values = historical_sentiment['confidence'].tail(len(df)).values
            else:
                # Use current sentiment for all days (simplified approach)
                sentiment_values = np.full(len(df), current_sentiment['sentiment_score'])
                confidence_values = np.full(len(df), current_sentiment['confidence'])
            
            # Add sentiment features
            df['Sentiment_Score'] = sentiment_values
            df['Sentiment_Confidence'] = confidence_values
            
            # Create derived sentiment features
            df['Sentiment_SMA_5'] = df['Sentiment_Score'].rolling(window=5).mean()
            df['Sentiment_SMA_10'] = df['Sentiment_Score'].rolling(window=10).mean()
            df['Sentiment_Volatility'] = df['Sentiment_Score'].rolling(window=10).std()
            df['Sentiment_Momentum'] = df['Sentiment_Score'].diff()
            
            # Sentiment-price correlation features
            df['Sentiment_Price_Correlation'] = df['Sentiment_Score'].rolling(window=20).corr(df['Close'])
            
            # Binary sentiment indicators
            df['Sentiment_Positive'] = (df['Sentiment_Score'] > 0.1).astype(int)
            df['Sentiment_Negative'] = (df['Sentiment_Score'] < -0.1).astype(int)
            df['Sentiment_Strong_Positive'] = (df['Sentiment_Score'] > 0.3).astype(int)
            df['Sentiment_Strong_Negative'] = (df['Sentiment_Score'] < -0.3).astype(int)
            
            print(f"✅ Added sentiment features. Current sentiment: {current_sentiment['sentiment_score']:.3f} ({current_sentiment['sentiment_label']})")
            
        except Exception as e:
            print(f"❌ Error adding sentiment features: {e}")
            # Add neutral sentiment features if error occurs
            df['Sentiment_Score'] = 0.0
            df['Sentiment_Confidence'] = 0.5
            df['Sentiment_SMA_5'] = 0.0
            df['Sentiment_SMA_10'] = 0.0
            df['Sentiment_Volatility'] = 0.0
            df['Sentiment_Momentum'] = 0.0
            df['Sentiment_Price_Correlation'] = 0.0
            df['Sentiment_Positive'] = 0
            df['Sentiment_Negative'] = 0
            df['Sentiment_Strong_Positive'] = 0
            df['Sentiment_Strong_Negative'] = 0
        
        return df
    
    def prepare_data(self, data):
        """
        Prepare data for LSTM training.
        
        Args:
            data (pd.DataFrame): Enhanced OHLCV data with features
            
        Returns:
            tuple: (X_sequences, y_target, feature_names)
        """
        # Select features for training
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'Price_Change', 'High_Low_Ratio', 'Open_Close_Ratio',
            'Price_to_SMA_5', 'Price_to_SMA_10', 'Price_to_SMA_20', 'Price_to_SMA_50',
            'Volatility_10', 'Volatility_20',
            'Volume_Ratio', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'BB_Position'
        ]
        
        # Add sentiment features if available
        if self.include_sentiment:
            sentiment_features = [
                'Sentiment_Score', 'Sentiment_Confidence', 'Sentiment_SMA_5', 'Sentiment_SMA_10',
                'Sentiment_Volatility', 'Sentiment_Momentum', 'Sentiment_Price_Correlation',
                'Sentiment_Positive', 'Sentiment_Negative', 'Sentiment_Strong_Positive', 'Sentiment_Strong_Negative'
            ]
            feature_columns.extend(sentiment_features)
        
        # Filter available columns
        available_features = [col for col in feature_columns if col in data.columns]
        feature_data = data[available_features].values
        
        # Scale features
        scaled_features = self.feature_scaler.fit_transform(feature_data)
        
        # Scale target (Close price)
        close_prices = data['Close'].values.reshape(-1, 1)
        scaled_prices = self.scaler.fit_transform(close_prices)
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_features) - self.prediction_days + 1):
            X.append(scaled_features[i-self.sequence_length:i])
            y.append(scaled_prices[i+self.prediction_days-1][0])
        
        return np.array(X), np.array(y), available_features
    
    def build_model(self, input_size):
        """
        Build LSTM model architecture using PyTorch.
        
        Args:
            input_size (int): Number of features per timestep
        """
        self.model = LSTMModel(input_size=input_size, hidden_size=100, num_layers=3, output_size=1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
    
    def train(self, data, ticker=None, validation_split=0.2, epochs=50, batch_size=32, verbose=0):
        """
        Train the LSTM model.
        
        Args:
            data (pd.DataFrame): OHLCV data
            ticker (str): Stock ticker for sentiment analysis
            validation_split (float): Fraction of data to use for validation
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            verbose (int): Verbosity level
            
        Returns:
            dict: Training history and metrics
        """
        try:
            # Create features
            enhanced_data = self.create_features(data, ticker)
            
            if len(enhanced_data) < self.sequence_length + self.prediction_days:
                raise ValueError(f"Insufficient data. Need at least {self.sequence_length + self.prediction_days} days.")
            
            # Prepare data
            X, y, feature_names = self.prepare_data(enhanced_data)
            
            if len(X) < 10:  # Minimum samples needed
                raise ValueError("Insufficient training samples after preprocessing.")
            
            # Build model
            self.build_model(X.shape[2])
            
            # Convert to PyTorch tensors
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.FloatTensor(y).to(self.device)
            
            # Split data for validation
            split_idx = int(len(X_tensor) * (1 - validation_split))
            X_train, X_val = X_tensor[:split_idx], X_tensor[split_idx:]
            y_train, y_val = y_tensor[:split_idx], y_tensor[split_idx:]
            
            # Create data loaders
            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Training loop
            train_losses = []
            val_losses = []
            
            self.model.train()
            for epoch in range(epochs):
                # Training phase
                epoch_train_loss = 0
                for batch_X, batch_y in train_loader:
                    self.optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    self.optimizer.step()
                    epoch_train_loss += loss.item()
                
                # Validation phase
                self.model.eval()
                epoch_val_loss = 0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self.model(batch_X)
                        loss = self.criterion(outputs.squeeze(), batch_y)
                        epoch_val_loss += loss.item()
                
                train_losses.append(epoch_train_loss / len(train_loader))
                val_losses.append(epoch_val_loss / len(val_loader))
                
                if verbose > 0 and epoch % 10 == 0:
                    print(f'Epoch {epoch}/{epochs}, Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}')
                
                self.model.train()
            
            self.is_trained = True
            
            # Calculate metrics on training data
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(X_tensor).cpu().numpy()
            
            predictions_rescaled = self.scaler.inverse_transform(predictions.reshape(-1, 1))
            y_rescaled = self.scaler.inverse_transform(y.reshape(-1, 1))
            
            mse = mean_squared_error(y_rescaled, predictions_rescaled)
            mae = mean_absolute_error(y_rescaled, predictions_rescaled)
            
            return {
                'history': {'loss': train_losses, 'val_loss': val_losses},
                'mse': mse,
                'mae': mae,
                'samples_trained': len(X),
                'features_used': feature_names
            }
            
        except Exception as e:
            print(f"Training failed: {str(e)}")
            return {'error': str(e)}
    
    def predict_next_prices(self, data, ticker=None, num_predictions=5):
        """
        Predict future prices.
        
        Args:
            data (pd.DataFrame): Recent OHLCV data
            ticker (str): Stock ticker for sentiment analysis
            num_predictions (int): Number of future prices to predict
            
        Returns:
            dict: Predictions and confidence information
        """
        if not self.is_trained:
            return {'error': 'Model not trained yet'}
        
        try:
            # Create features
            enhanced_data = self.create_features(data, ticker)
            
            # Get the last sequence
            feature_columns = self.feature_scaler.feature_names_in_ if hasattr(self.feature_scaler, 'feature_names_in_') else None
            if feature_columns is None:
                # Fallback to manual feature selection
                feature_columns = [col for col in enhanced_data.columns if col != 'Close']
            
            last_sequence = enhanced_data[feature_columns].tail(self.sequence_length).values
            last_sequence_scaled = self.feature_scaler.transform(last_sequence)
            
            # Convert to PyTorch tensor and reshape for prediction
            last_sequence_tensor = torch.FloatTensor(last_sequence_scaled).unsqueeze(0).to(self.device)
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                prediction_scaled = self.model(last_sequence_tensor).cpu().numpy()
            
            prediction = self.scaler.inverse_transform(prediction_scaled.reshape(-1, 1))[0][0]
            
            # Get current price for comparison
            current_price = data['Close'].iloc[-1]
            predicted_change = ((prediction - current_price) / current_price) * 100
            
            return {
                'predicted_price': prediction,
                'current_price': current_price,
                'predicted_change_percent': predicted_change,
                'confidence': 'medium',  # Could be enhanced with prediction intervals
                'prediction_days_ahead': self.prediction_days
            }
            
        except Exception as e:
            return {'error': f'Prediction failed: {str(e)}'}
    
    def get_feature_importance(self, data, ticker=None):
        """
        Calculate feature importance for the model (simplified version).
        
        Args:
            data (pd.DataFrame): OHLCV data
            ticker (str): Stock ticker for sentiment analysis
            
        Returns:
            dict: Feature importance scores
        """
        if not self.is_trained:
            return {'error': 'Model not trained yet'}
        
        try:
            enhanced_data = self.create_features(data, ticker)
            
            # Simple feature importance based on correlation with target
            correlations = enhanced_data.corr()['Close'].abs().sort_values(ascending=False)
            
            # Remove 'Close' itself and get top features
            feature_importance = correlations.drop('Close').head(10).to_dict()
            
            return {
                'feature_importance': feature_importance,
                'top_features': list(feature_importance.keys())[:5]
            }
            
        except Exception as e:
            return {'error': f'Feature importance calculation failed: {str(e)}'}


class MLSignalGenerator:
    """
    Generate trading signals using ML predictions and technical analysis.
    """
    
    def __init__(self, lstm_predictor):
        """
        Initialize signal generator with LSTM predictor.
        
        Args:
            lstm_predictor (LSTMPredictor): Trained LSTM model
        """
        self.lstm_predictor = lstm_predictor
    
    def generate_signal(self, data, threshold=2.0):
        """
        Generate trading signal based on ML prediction and technical indicators.
        
        Args:
            data (pd.DataFrame): Recent OHLCV data
            threshold (float): Percentage threshold for signal generation
            
        Returns:
            dict: Signal information
        """
        # Get ML prediction
        ml_prediction = self.lstm_predictor.predict_next_prices(data)
        
        if 'error' in ml_prediction:
            return {'signal': 'HOLD', 'confidence': 0, 'reason': ml_prediction['error']}
        
        # Get latest technical indicators
        enhanced_data = self.lstm_predictor.create_features(data)
        latest = enhanced_data.iloc[-1]
        
        # ML signal
        predicted_change = ml_prediction['predicted_change_percent']
        ml_signal = 'BUY' if predicted_change > threshold else 'SELL' if predicted_change < -threshold else 'HOLD'
        
        # Technical signals
        rsi_signal = 'BUY' if latest['RSI'] < 30 else 'SELL' if latest['RSI'] > 70 else 'HOLD'
        bb_signal = 'BUY' if latest['BB_Position'] < 0.2 else 'SELL' if latest['BB_Position'] > 0.8 else 'HOLD'
        macd_signal = 'BUY' if latest['MACD'] > latest['MACD_Signal'] else 'SELL'
        
        # Combine signals
        buy_votes = sum([1 for s in [ml_signal, rsi_signal, bb_signal, macd_signal] if s == 'BUY'])
        sell_votes = sum([1 for s in [ml_signal, rsi_signal, bb_signal, macd_signal] if s == 'SELL'])
        
        if buy_votes >= 2:
            final_signal = 'BUY'
            confidence = buy_votes / 4
        elif sell_votes >= 2:
            final_signal = 'SELL'
            confidence = sell_votes / 4
        else:
            final_signal = 'HOLD'
            confidence = 0.5
        
        return {
            'signal': final_signal,
            'confidence': confidence,
            'ml_prediction': ml_prediction,
            'technical_signals': {
                'rsi': rsi_signal,
                'bollinger_bands': bb_signal,
                'macd': macd_signal
            },
            'vote_count': {'buy': buy_votes, 'sell': sell_votes}
        } 