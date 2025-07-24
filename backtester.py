import backtrader as bt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for threading safety
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import warnings
warnings.filterwarnings('ignore')


class Backtester:
    """
    Manages the backtrader engine (Cerebro) and the entire backtesting process.
    """
    
    def __init__(self, strategy, data_feed, cash=100000.0, commission=0.001):
        """
        Initialize the backtester with strategy, data feed, and broker configuration.
        
        Args:
            strategy: The strategy class to use for backtesting
            data_feed: Backtrader data feed object
            cash (float): Starting cash amount
            commission (float): Commission rate for trades
        """
        self.strategy = strategy
        self.data_feed = data_feed
        self.cash = cash
        self.commission = commission
        self.cerebro = None
        self.results = None
        
    def run_backtest(self, strategy_params=None):
        """
        Main method to run the backtest.
        
        Args:
            strategy_params (dict): Dictionary of strategy-specific parameters
            
        Returns:
            dict: Backtest results containing final value, metrics, and cerebro object
        """
        try:
            # Initialize Cerebro engine
            self.cerebro = bt.Cerebro()
            
            # Add data feed
            self.cerebro.adddata(self.data_feed)
            
            # Add strategy with parameters
            if strategy_params:
                self.cerebro.addstrategy(self.strategy, **strategy_params)
            else:
                self.cerebro.addstrategy(self.strategy)
            
            # Set broker configuration
            self.cerebro.broker.setcash(self.cash)
            self.cerebro.broker.setcommission(commission=self.commission)
            
            # Add a sizer - how many shares to buy
            self.cerebro.addsizer(bt.sizers.FixedSize, stake=100)
            
            # Add analyzers for performance metrics
            self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
            self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
            self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
            
            # Record starting value
            starting_value = self.cerebro.broker.getvalue()
            print(f'Starting Portfolio Value: ${starting_value:,.2f}')
            
            # Run the backtest
            self.results = self.cerebro.run()
            
            # Get final value
            final_value = self.cerebro.broker.getvalue()
            print(f'Final Portfolio Value: ${final_value:,.2f}')
            
            # Extract analyzer results
            strategy_results = self.results[0]
            
            # Get analyzer data
            sharpe_ratio = self._get_sharpe_ratio(strategy_results)
            max_drawdown = self._get_max_drawdown(strategy_results)
            total_return = self._get_total_return(strategy_results, starting_value, final_value)
            trade_analysis = self._get_trade_analysis(strategy_results)
            
            # Prepare results dictionary
            backtest_results = {
                'starting_value': starting_value,
                'final_value': final_value,
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'trade_analysis': trade_analysis,
                'cerebro': self.cerebro,
                'strategy_params': strategy_params or {},
                'commission': self.commission
            }
            
            return backtest_results
            
        except Exception as e:
            raise Exception(f"Error running backtest: {str(e)}")
    
    def _get_sharpe_ratio(self, strategy_results):
        """Extract Sharpe ratio from analyzer results."""
        try:
            sharpe_analyzer = strategy_results.analyzers.sharpe
            sharpe_ratio = sharpe_analyzer.get_analysis().get('sharperatio', 0)
            return sharpe_ratio if sharpe_ratio else 0
        except:
            return 0
    
    def _get_max_drawdown(self, strategy_results):
        """Extract maximum drawdown from analyzer results."""
        try:
            drawdown_analyzer = strategy_results.analyzers.drawdown
            drawdown_data = drawdown_analyzer.get_analysis()
            max_drawdown = drawdown_data.get('max', {}).get('drawdown', 0)
            return max_drawdown
        except:
            return 0
    
    def _get_total_return(self, strategy_results, starting_value, final_value):
        """Calculate total return percentage."""
        try:
            returns_analyzer = strategy_results.analyzers.returns
            returns_data = returns_analyzer.get_analysis()
            total_return = returns_data.get('rtot', 0) * 100  # Convert to percentage
            
            # Fallback calculation if analyzer doesn't provide the data
            if total_return == 0:
                total_return = ((final_value - starting_value) / starting_value) * 100
            
            return total_return
        except:
            # Fallback calculation
            return ((final_value - starting_value) / starting_value) * 100
    
    def _get_trade_analysis(self, strategy_results):
        """Extract trade analysis from analyzer results."""
        try:
            trade_analyzer = strategy_results.analyzers.trades
            trade_data = trade_analyzer.get_analysis()
            
            total_trades = trade_data.get('total', {}).get('total', 0)
            won_trades = trade_data.get('won', {}).get('total', 0)
            lost_trades = trade_data.get('lost', {}).get('total', 0)
            
            win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
            
            return {
                'total_trades': total_trades,
                'won_trades': won_trades,
                'lost_trades': lost_trades,
                'win_rate': win_rate
            }
        except:
            return {
                'total_trades': 0,
                'won_trades': 0,
                'lost_trades': 0,
                'win_rate': 0
            }
    
    def generate_plot(self, figsize=(12, 8)):
        """
        Generate a matplotlib figure of the backtest results.
        
        Args:
            figsize (tuple): Figure size for the plot
            
        Returns:
            matplotlib.figure.Figure: The plot figure or None if failed
        """
        if self.cerebro is None:
            raise Exception("No backtest results to plot. Run backtest first.")
        
        # Skip plotting entirely on macOS to avoid threading issues
        import platform
        if platform.system() == 'Darwin':  # macOS
            print("Skipping plot generation on macOS to avoid threading issues")
            return None
        
        try:
            # Force matplotlib to use Agg backend and disable interactive mode
            plt.ioff()  # Turn off interactive mode
            
            # Try to create the plot with error handling
            figs = self.cerebro.plot(
                style='candlestick',
                figsize=figsize,
                volume=False,
                iplot=False,
                returnfig=True
            )
            
            if figs and len(figs) > 0 and len(figs[0]) > 0:
                fig = figs[0][0]
                return fig
            else:
                raise Exception("No figure returned from cerebro.plot()")
            
        except Exception as e:
            # If cerebro.plot() fails, return None instead of trying fallback
            print(f"Warning: Plot generation failed ({str(e)[:50]}...), skipping chart")
            return None
    
    def _create_summary_plot(self, figsize=(12, 8)):
        """
        Create a summary plot when cerebro.plot() fails.
        
        Args:
            figsize (tuple): Figure size for the plot
            
        Returns:
            matplotlib.figure.Figure: Summary plot figure
        """
        try:
            # Get data from the cerebro object
            strategy = self.results[0] if self.results else None
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                          gridspec_kw={'height_ratios': [3, 1]})
            
            # Plot 1: Portfolio value over time
            if hasattr(strategy, 'broker') and strategy.broker:
                # Try to get portfolio values
                ax1.set_title('Backtest Results - Portfolio Performance', fontsize=14, fontweight='bold')
                ax1.text(0.5, 0.5, 
                        f'Backtest Completed Successfully!\n\n'
                        f'Starting Value: ${self.cash:,.2f}\n'
                        f'Final Value: ${strategy.broker.getvalue():,.2f}\n'
                        f'Return: {((strategy.broker.getvalue() - self.cash) / self.cash * 100):.2f}%',
                        ha='center', va='center', transform=ax1.transAxes, 
                        fontsize=12, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.7))
            else:
                ax1.text(0.5, 0.5, 'Backtest Completed Successfully!\nDetailed chart unavailable.', 
                        ha='center', va='center', transform=ax1.transAxes, fontsize=12)
            
            ax1.set_ylabel('Portfolio Value ($)')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Strategy information
            if strategy:
                try:
                    data_length = len(self.data_feed) if hasattr(self, 'data_feed') else "Unknown"
                except:
                    data_length = "Unknown"
                    
                strategy_info = (
                    f"Strategy: SMA Crossover\n"
                    f"Commission: {self.commission*100:.2f}%\n"
                    f"Data Period: {data_length} days"
                )
            else:
                strategy_info = "Strategy: SMA Crossover\nStatus: Completed"
                
            ax2.text(0.5, 0.5, strategy_info,
                    ha='center', va='center', transform=ax2.transAxes, 
                    fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
            ax2.set_title('Strategy Details', fontsize=12)
            ax2.axis('off')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            # Ultimate fallback - simple text plot
            print(f"Warning: Summary plot also failed: {str(e)}")
            
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 
                   '✅ Backtest Completed Successfully!\n\n'
                   '📊 Check the metrics above for results\n'
                   '⚠️  Detailed chart unavailable due to system limitations', 
                   ha='center', va='center', transform=ax.transAxes, 
                   fontsize=14, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))
            ax.set_title('Algorithmic Trading Backtest - Results Summary', fontsize=16, fontweight='bold')
            ax.axis('off')
            
            return fig 