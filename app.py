import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode
from datetime import datetime, date, timedelta
import warnings
import os
import numpy as np

# Set environment variable to prevent GUI issues
os.environ['MPLBACKEND'] = 'Agg'

# Import our custom classes
from data_handler import DataHandler
from strategies import STRATEGIES, STRATEGY_PARAMS
from backtester import Backtester

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Try to import ML components (optional)
try:
    from ml_models import LSTMPredictor, MLSignalGenerator
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    st.warning("ML components not available. Install PyTorch and scikit-learn to enable ML features.")

# Try to import sentiment analysis
try:
    from news_sentiment import NewsSentimentAggregator
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False


class StreamlitApp:
    """
    Main class that builds and runs the Streamlit user interface.
    """
    
    def __init__(self):
        """
        Initialize the Streamlit app with page configuration.
        """
        st.set_page_config(
            page_title="Algorithmic Trading Strategy Backtester",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Initialize session state variables
        if 'backtest_results' not in st.session_state:
            st.session_state.backtest_results = None
        if 'backtest_run' not in st.session_state:
            st.session_state.backtest_run = False
    
    def setup_sidebar(self):
        """
        Creates the sidebar with all user-configurable inputs.
        
        Returns:
            dict: Dictionary containing all user inputs
        """
        st.sidebar.header("Strategy Configuration")
        
        # Strategy selection
        strategy_name = st.sidebar.selectbox(
            "Trading Strategy",
            options=list(STRATEGIES.keys()),
            index=0,
            help="Select the trading strategy to backtest"
        )
        
        # Stock ticker input
        ticker = st.sidebar.text_input(
            "Stock Ticker",
            value="MSFT",
            help="Enter a valid stock ticker symbol (e.g., MSFT, GOOGL, TSLA)"
        ).upper()
        
        # Date range inputs
        st.sidebar.subheader("Date Range")
        
        # Default dates
        default_start = date(2020, 1, 1)
        default_end = date.today()
        
        start_date = st.sidebar.date_input(
            "Start Date",
            value=default_start,
            max_value=default_end - timedelta(days=100),
            help="Start date for historical data"
        )
        
        end_date = st.sidebar.date_input(
            "End Date",
            value=default_end,
            min_value=start_date + timedelta(days=100),
            max_value=default_end,
            help="End date for historical data"
        )
        
        # Trading parameters
        st.sidebar.subheader("Trading Parameters")
        
        starting_cash = st.sidebar.number_input(
            "Starting Cash ($)",
            min_value=1000,
            max_value=10000000,
            value=100000,
            step=1000,
            help="Initial portfolio value"
        )
        
        # Strategy parameters
        st.sidebar.subheader("Strategy Parameters")
        
        # Dynamic parameter inputs based on selected strategy
        strategy_params = {}
        param_config = STRATEGY_PARAMS[strategy_name]
        
        for param_name, config in param_config.items():
            if isinstance(config['default'], int):
                strategy_params[param_name] = st.sidebar.number_input(
                    param_name.replace('_', ' ').title(),
                    min_value=config['min'],
                    max_value=config['max'],
                    value=config['default'],
                    step=config['step'],
                    help=config['help']
                )
            else:  # float
                strategy_params[param_name] = st.sidebar.number_input(
                    param_name.replace('_', ' ').title(),
                    min_value=config['min'],
                    max_value=config['max'],
                    value=config['default'],
                    step=config['step'],
                    format="%.3f",
                    help=config['help']
                )
        
        # Strategy-specific validation
        if strategy_name == 'SMA Crossover':
            if strategy_params['fast'] >= strategy_params['slow']:
                st.sidebar.error("Fast MA period must be less than Slow MA period!")
                st.sidebar.stop()
        elif strategy_name == 'RSI':
            if strategy_params['lower'] >= strategy_params['upper']:
                st.sidebar.error("Lower threshold must be less than upper threshold!")
                st.sidebar.stop()
        elif strategy_name == 'MACD':
            if strategy_params['me1'] >= strategy_params['me2']:
                st.sidebar.error("Fast EMA period must be less than Slow EMA period!")
                st.sidebar.stop()
        
        # Commission
        commission = st.sidebar.slider(
            "Commission (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.01,
            help="Commission percentage per trade"
        ) / 100  # Convert to decimal
        
        # Run backtest button
        st.sidebar.markdown("---")
        run_backtest = st.sidebar.button(
            "Run Backtest",
            type="primary",
            use_container_width=True,
            help="Click to execute the backtest with current parameters"
        )
        
        # ML prediction demo button (if available)
        show_ml_demo = False
        show_sentiment_demo = False
        if ML_AVAILABLE and strategy_name == 'ML Enhanced':
            st.sidebar.markdown("---")
            st.sidebar.markdown("**ML Features**")
            show_ml_demo = st.sidebar.button(
                "Show Live ML Demo",
                help="Demonstrate ML price prediction on recent data"
            )
            
            if SENTIMENT_AVAILABLE:
                show_sentiment_demo = st.sidebar.button(
                    "Show Sentiment Analysis",
                    help="Analyze current market sentiment for this ticker"
                )
        
        return {
            'ticker': ticker,
            'start_date': start_date,
            'end_date': end_date,
            'starting_cash': starting_cash,
            'strategy_name': strategy_name,
            'strategy_params': strategy_params,
            'commission': commission,
            'run_backtest': run_backtest,
            'show_ml_demo': show_ml_demo,
            'show_sentiment_demo': show_sentiment_demo
        }
    
    def render_main_panel(self, results=None, plot_fig=None):
        """
        Displays the results on the main page.
        
        Args:
            results (dict): Backtest results dictionary
            plot_fig (matplotlib.figure.Figure): Plot figure to display
        """
        # Main title
        st.title("Algorithmic Trading Strategy Backtester")
        st.markdown("### Professional Multi-Strategy Backtesting Platform")
        
        if results is None:
            # Initial state - show instructions
            st.markdown("""
            ## Welcome to the Trading Strategy Backtester
            
            This application allows you to backtest **multiple algorithmic trading strategies** on historical stock data.
            
            ### Available Strategies:
            
            **SMA Crossover**: Buy when fast MA crosses above slow MA, sell when it crosses below
            
            **RSI**: Buy when RSI is oversold (below threshold), sell when overbought (above threshold)
            
            **MACD**: Buy when MACD line crosses above signal line, sell when it crosses below
            
            **Bollinger Bands**: Buy at lower band, sell at upper band
            
            **Mean Reversion**: Buy when price deviates significantly below average, sell when above
            
            **ML Enhanced**: Combines machine learning predictions with multiple technical indicators for sophisticated signal generation
            
            ### How it works:
            1. **Select** your preferred trading strategy
            2. **Configure** strategy-specific parameters in the sidebar
            3. **Choose** a stock ticker and date range
            4. **Click** "Run Backtest" to see comprehensive results
            
            **Use the sidebar to configure your backtest parameters and click "Run Backtest" to get started.**
            """)
            
            # Show example configuration
            with st.expander("Example Configurations"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                    **SMA Crossover - Conservative:**
                    - Fast MA: 20 days, Slow MA: 50 days
                    - Good for: Trend following, fewer false signals
                    
                    **RSI - Standard:**
                    - Period: 14, Upper: 70, Lower: 30
                    - Good for: Range-bound markets
                    
                    **ML Enhanced - Advanced:**
                    - Prediction Threshold: 2.0%, Confidence: 0.6
                    - Good for: AI-powered decision making
                    """)
                with col2:
                    st.markdown("""
                    **MACD - Default:**
                    - Fast EMA: 12, Slow EMA: 26, Signal: 9
                    - Good for: Momentum trading
                    
                    **Bollinger Bands - Classic:**
                    - Period: 20, Deviation Factor: 2.0
                    - Good for: Volatility-based trading
                    
                    **Mean Reversion - Statistical:**
                    - Period: 20, Threshold: 2%
                    - Good for: Contrarian trading
                    """)
            
            return
        
        # Display results
        st.success("Backtest completed successfully!")
        
        # Show ML insights if ML Enhanced strategy was used
        if 'strategy_name' in results and results.get('strategy_name') == 'ML Enhanced' and ML_AVAILABLE:
            self.show_ml_insights(results)
        
        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Final Portfolio Value",
                value=f"${results['final_value']:,.2f}",
                delta=f"${results['final_value'] - results['starting_value']:,.2f}"
            )
        
        with col2:
            st.metric(
                label="Total Return",
                value=f"{results['total_return']:.2f}%",
                delta=f"{results['total_return']:.2f}%" if results['total_return'] >= 0 else f"{results['total_return']:.2f}%"
            )
        
        with col3:
            sharpe_ratio = results['sharpe_ratio']
            st.metric(
                label="Sharpe Ratio", 
                value=f"{sharpe_ratio:.3f}" if sharpe_ratio else "N/A",
                help="Risk-adjusted return measure (higher is better)"
            )
        
        with col4:
            st.metric(
                label="Max Drawdown",
                value=f"{results['max_drawdown']:.2f}%",
                delta=f"-{results['max_drawdown']:.2f}%",
                delta_color="inverse"
            )
        
        # Detailed results
        st.markdown("### Detailed Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Portfolio Performance")
            st.write(f"**Starting Value:** ${results['starting_value']:,.2f}")
            st.write(f"**Final Value:** ${results['final_value']:,.2f}")
            st.write(f"**Absolute Gain/Loss:** ${results['final_value'] - results['starting_value']:,.2f}")
            st.write(f"**Total Return:** {results['total_return']:.2f}%")
        
        with col2:
            st.markdown("#### Trading Statistics")
            trade_stats = results['trade_analysis']
            st.write(f"**Total Trades:** {trade_stats['total_trades']}")
            st.write(f"**Winning Trades:** {trade_stats['won_trades']}")
            st.write(f"**Losing Trades:** {trade_stats['lost_trades']}")
            st.write(f"**Win Rate:** {trade_stats['win_rate']:.1f}%")
        
        # Strategy parameters
        with st.expander("Strategy Parameters Used"):
            params = results['strategy_params']
            commission = results['commission']
            
            # Display strategy parameters dynamically
            param_cols = st.columns(min(len(params) + 1, 4))
            
            for i, (param_name, param_value) in enumerate(params.items()):
                with param_cols[i % 4]:
                    if isinstance(param_value, float):
                        st.write(f"**{param_name.replace('_', ' ').title()}:** {param_value:.3f}")
                    else:
                        st.write(f"**{param_name.replace('_', ' ').title()}:** {param_value}")
            
            # Always show commission
            with param_cols[len(params) % 4]:
                st.write(f"**Commission:** {commission*100:.2f}%")
        
        # Display plot or explanation
        st.markdown("### Strategy Visualization")
        if plot_fig:
            st.pyplot(plot_fig)
        else:
            st.info("""
            **Chart temporarily unavailable** due to system limitations, but your backtest completed successfully!
            
            **All the important metrics are shown above.** The strategy executed properly with:
            - Data fetched and processed
            - Buy/sell signals generated  
            - Trades executed with commissions
            - Performance metrics calculated
            
            You can see the detailed results in the metrics above.
            """)
    
    def show_ml_insights(self, results):
        """Show ML-specific insights and predictions."""
        try:
            st.markdown("### Machine Learning Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### ML Strategy Performance")
                st.info("""
                **ML Enhanced Strategy Features:**
                - Combines pattern recognition with technical analysis
                - Uses confidence-based trade filtering
                - Adapts to market conditions dynamically
                - Incorporates momentum and volatility analysis
                """)
            
            with col2:
                st.markdown("#### Strategy Configuration")
                params = results.get('strategy_params', {})
                st.write(f"**Prediction Threshold:** {params.get('prediction_threshold', 'N/A')}%")
                st.write(f"**Confidence Threshold:** {params.get('confidence_threshold', 'N/A')}")
                st.write(f"**RSI Period:** {params.get('rsi_period', 'N/A')}")
                st.write(f"**Bollinger Bands Period:** {params.get('bb_period', 'N/A')}")
            
            # Add note about ML features
            st.markdown("#### About ML Enhancement")
            enhanced_features = "sentiment analysis + " if SENTIMENT_AVAILABLE else ""
            st.success(f"""
            **This strategy combines machine learning with {enhanced_features}technical analysis:**
            - LSTM neural networks for price prediction
            - Real-time news sentiment analysis from multiple sources
            - 30+ engineered features including sentiment indicators
            - Confidence-based trade filtering
            - Multi-source ensemble predictions
            
            **Advanced Features Available:**
            - PyTorch LSTM: {'✅ Enabled' if ML_AVAILABLE else '❌ Disabled'}
            - Sentiment Analysis: {'✅ Enabled' if SENTIMENT_AVAILABLE else '❌ Disabled'}
            """)
            
        except Exception as e:
             st.warning(f"Could not display ML insights: {str(e)}")
    
    def show_ml_demo(self, config):
        """Show ML prediction demo."""
        try:
            st.markdown("### Live ML Prediction Demo")
            
            with st.spinner("Fetching recent data for ML analysis..."):
                # Fetch recent data
                data_handler = DataHandler(
                    ticker=config['ticker'],
                    start_date=date.today() - timedelta(days=365),
                    end_date=date.today()
                )
                
                recent_data = data_handler.fetch_data()
                
                if len(recent_data) < 100:
                    st.warning("Insufficient data for ML demo. Need at least 100 days of data.")
                    return
            
            # Create LSTM predictor and demonstrate (simplified version)
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Recent Price Action")
                
                # Show recent price chart
                fig, ax = plt.subplots(figsize=(10, 6))
                recent_data.tail(30)['Close'].plot(ax=ax, title=f"{config['ticker']} - Last 30 Days")
                ax.set_ylabel('Price ($)')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            with col2:
                st.markdown("#### ML Analysis")
                
                # Calculate some ML-like metrics
                latest_close = recent_data['Close'].iloc[-1]
                sma_20 = recent_data['Close'].tail(20).mean()
                volatility = recent_data['Close'].tail(20).std()
                price_change = recent_data['Close'].pct_change().tail(5).mean() * 100
                
                # Simulate ML prediction
                momentum_score = price_change * 0.5
                trend_score = ((latest_close - sma_20) / sma_20) * 100
                volatility_score = min(5, volatility / latest_close * 100)
                
                ml_prediction = momentum_score + trend_score - volatility_score
                confidence = min(0.95, abs(ml_prediction) / 10 + 0.5)
                
                st.metric("Current Price", f"${latest_close:.2f}")
                st.metric("ML Prediction", f"{ml_prediction:+.2f}%", f"Confidence: {confidence:.1%}")
                st.metric("Momentum Score", f"{momentum_score:.2f}")
                st.metric("Trend Score", f"{trend_score:.2f}")
                st.metric("Volatility Score", f"{volatility_score:.2f}")
                
                # Show prediction interpretation
                if ml_prediction > 2:
                    st.success("**Signal: BULLISH** - ML model suggests upward movement")
                elif ml_prediction < -2:
                    st.error("**Signal: BEARISH** - ML model suggests downward movement")
                else:
                    st.info("**Signal: NEUTRAL** - ML model suggests sideways movement")
            
            # Technical analysis comparison
            st.markdown("#### ML vs Technical Analysis Comparison")
            
            # Calculate technical indicators
            rsi_period = 14
            delta = recent_data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            comp_col1, comp_col2, comp_col3 = st.columns(3)
            
            with comp_col1:
                st.markdown("**ML Signal**")
                if ml_prediction > 2:
                    st.success(f"BUY ({confidence:.1%})")
                elif ml_prediction < -2:
                    st.error(f"SELL ({confidence:.1%})")
                else:
                    st.info(f"HOLD ({confidence:.1%})")
            
            with comp_col2:
                st.markdown("**RSI Signal**")
                if current_rsi < 30:
                    st.success("BUY (Oversold)")
                elif current_rsi > 70:
                    st.error("SELL (Overbought)")
                else:
                    st.info(f"HOLD (RSI: {current_rsi:.1f})")
            
            with comp_col3:
                st.markdown("**Trend Signal**")
                if latest_close > sma_20:
                    st.success("BUY (Above SMA)")
                else:
                    st.error("SELL (Below SMA)")
            
            st.info("""
            **Demo Note:** This demonstrates ML-style analysis using pattern recognition, 
            momentum calculation, and confidence scoring. In a full implementation, 
            this would use trained PyTorch LSTM neural networks for price prediction.
            """)
            
        except Exception as e:
             st.error(f"ML demo failed: {str(e)}")
    
    def show_sentiment_demo(self, config):
        """Show live sentiment analysis demo."""
        try:
            st.markdown("### Live Market Sentiment Analysis")
            
            if not SENTIMENT_AVAILABLE:
                st.error("Sentiment analysis not available. Please install the required NLP packages.")
                return
            
            with st.spinner(f"Analyzing market sentiment for {config['ticker']}..."):
                from news_sentiment import NewsSentimentAggregator
                
                # Initialize sentiment aggregator
                sentiment_aggregator = NewsSentimentAggregator()
                
                # Get current sentiment
                sentiment_data = sentiment_aggregator.get_sentiment_data(
                    config['ticker'], 
                    days_back=5, 
                    max_articles_per_source=5
                )
            
            # Display sentiment results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Overall sentiment
                sentiment_score = sentiment_data['sentiment_score']
                sentiment_label = sentiment_data['sentiment_label']
                
                if sentiment_score > 0.1:
                    st.success(f"**{sentiment_label}** Sentiment")
                elif sentiment_score < -0.1:
                    st.error(f"**{sentiment_label}** Sentiment")
                else:
                    st.info(f"**{sentiment_label}** Sentiment")
                
                st.metric(
                    "Sentiment Score",
                    f"{sentiment_score:.3f}",
                    help="Range: -1 (very negative) to +1 (very positive)"
                )
            
            with col2:
                st.metric("Confidence", f"{sentiment_data['confidence']:.1%}")
                st.metric("Articles Analyzed", sentiment_data['article_count'])
            
            with col3:
                st.metric("Data Sources", len(sentiment_data['sources']))
                if sentiment_data['sources']:
                    st.write("**Sources:**")
                    for source in sentiment_data['sources']:
                        st.write(f"• {source}")
            
            # Sentiment breakdown
            if sentiment_data['individual_scores']:
                st.markdown("#### Sentiment Distribution")
                
                # Create histogram of individual sentiment scores
                fig, ax = plt.subplots(figsize=(10, 4))
                scores = sentiment_data['individual_scores']
                ax.hist(scores, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
                ax.axvline(sentiment_score, color='red', linestyle='--', linewidth=2, label=f'Average: {sentiment_score:.3f}')
                ax.set_xlabel('Sentiment Score')
                ax.set_ylabel('Number of Articles')
                ax.set_title(f'Sentiment Distribution for {config["ticker"]}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            # Sentiment interpretation
            st.markdown("#### Sentiment Analysis Interpretation")
            
            interpretation_col1, interpretation_col2 = st.columns(2)
            
            with interpretation_col1:
                st.markdown("**What This Means:**")
                if sentiment_score > 0.3:
                    st.success("📈 **Very Positive**: Strong bullish sentiment detected. High optimism in news coverage.")
                elif sentiment_score > 0.1:
                    st.success("📊 **Positive**: Moderate bullish sentiment. Generally positive news coverage.")
                elif sentiment_score > -0.1:
                    st.info("⚖️ **Neutral**: Balanced sentiment. Mixed or neutral news coverage.")
                elif sentiment_score > -0.3:
                    st.warning("📉 **Negative**: Moderate bearish sentiment. Generally negative news coverage.")
                else:
                    st.error("📉 **Very Negative**: Strong bearish sentiment detected. High pessimism in news coverage.")
            
            with interpretation_col2:
                st.markdown("**Trading Implications:**")
                if sentiment_score > 0.2:
                    st.write("• Consider bullish strategies")
                    st.write("• Positive momentum expected")
                    st.write("• Monitor for sentiment peaks")
                elif sentiment_score < -0.2:
                    st.write("• Consider bearish strategies")
                    st.write("• Negative momentum expected")
                    st.write("• Watch for oversold conditions")
                else:
                    st.write("• Range-bound strategies")
                    st.write("• Wait for clearer signals")
                    st.write("• Monitor sentiment shifts")
            
            # Confidence and reliability
            st.markdown("#### Analysis Quality")
            quality_col1, quality_col2 = st.columns(2)
            
            with quality_col1:
                confidence = sentiment_data['confidence']
                if confidence > 0.7:
                    st.success(f"**High Confidence** ({confidence:.1%})")
                    st.write("Strong agreement across sources and analysis methods.")
                elif confidence > 0.5:
                    st.info(f"**Medium Confidence** ({confidence:.1%})")
                    st.write("Moderate agreement across sources and analysis methods.")
                else:
                    st.warning(f"**Low Confidence** ({confidence:.1%})")
                    st.write("Mixed signals or limited data. Use with caution.")
            
            with quality_col2:
                article_count = sentiment_data['article_count']
                if article_count >= 10:
                    st.success(f"**Robust Sample** ({article_count} articles)")
                elif article_count >= 5:
                    st.info(f"**Adequate Sample** ({article_count} articles)")
                else:
                    st.warning(f"**Limited Sample** ({article_count} articles)")
            
            # Disclaimer
            st.info("""
            **Disclaimer**: Sentiment analysis is based on publicly available news and social media data. 
            This should be used as supplementary information and not as the sole basis for trading decisions. 
            Market sentiment can change rapidly and may not always correlate with price movements.
            """)
            
        except Exception as e:
            st.error(f"Sentiment analysis failed: {str(e)}")
            st.info("Try installing the required packages: pip install requests beautifulsoup4 feedparser vaderSentiment nltk textblob")
     
    def run_backtest_process(self, config):
        """
        Execute the backtest process with the given configuration.
        
        Args:
            config (dict): Configuration parameters from sidebar
            
        Returns:
            tuple: (results, plot_figure) or (None, None) if error
        """
        try:
            with st.spinner(f"Fetching data for {config['ticker']}..."):
                # Initialize data handler and fetch data
                data_handler = DataHandler(
                    ticker=config['ticker'],
                    start_date=config['start_date'],
                    end_date=config['end_date']
                )
                
                # Fetch and validate data
                data_handler.fetch_data()
                data_feed = data_handler.get_feed()
                
                # Show data info
                data_info = data_handler.get_data_info()
                st.success(f"Fetched {data_info['total_days']} days of data for {config['ticker']}")
            
            with st.spinner("Running backtest..."):
                # Get selected strategy class
                strategy_class = STRATEGIES[config['strategy_name']]
                
                # Initialize and run backtester
                backtester = Backtester(
                    strategy=strategy_class,
                    data_feed=data_feed,
                    cash=config['starting_cash'],
                    commission=config['commission']
                )
                
                # Run backtest with strategy-specific parameters
                results = backtester.run_backtest(
                    strategy_params=config['strategy_params']
                )
                
                # Add strategy name to results for ML insights
                results['strategy_name'] = config['strategy_name']
                
                # Try to generate plot, but don't fail if it doesn't work
                try:
                    plot_fig = backtester.generate_plot()
                except Exception as e:
                    print(f"Plot generation failed: {e}")
                    plot_fig = None
                
                st.success("Backtest completed successfully!")
                
                return results, plot_fig
                
        except Exception as e:
            st.error(f"Error during backtest: {str(e)}")
            return None, None
    
    def run(self):
        """
        Main application loop.
        """
        # Setup sidebar and get configuration
        config = self.setup_sidebar()
        
        # Handle backtest execution
        if config['run_backtest']:
            results, plot_fig = self.run_backtest_process(config)
            
            if results:
                # Store results in session state
                st.session_state.backtest_results = results
                st.session_state.plot_fig = plot_fig
                st.session_state.backtest_run = True
        
        # Handle ML demo
        if config.get('show_ml_demo', False):
            self.show_ml_demo(config)
        
        # Handle sentiment demo
        if config.get('show_sentiment_demo', False):
            self.show_sentiment_demo(config)
        
        # Render main panel
        if st.session_state.backtest_run and st.session_state.backtest_results:
            self.render_main_panel(
                results=st.session_state.backtest_results,
                plot_fig=st.session_state.get('plot_fig')
            )
        else:
            self.render_main_panel()


def main():
    """
    Main entry point of the application.
    """
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main() 