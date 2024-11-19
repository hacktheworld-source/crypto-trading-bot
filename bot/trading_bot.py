import os
import time
import threading
import logging
import pandas as pd
from datetime import datetime, timedelta
import json
from coinbase.rest import RESTClient
from typing import Dict, Any, List, Union, Optional
from decimal import Decimal
from .position import Position

# Set up logging
logging.basicConfig(
    filename='trading_bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class TradingBot:
    def __init__(self):
        try:
            logging.info("Starting bot initialization...")
            
            if 'COINBASE_API_KEY' not in os.environ:
                raise Exception("COINBASE_API_KEY not found in environment variables")
            if 'COINBASE_API_SECRET' not in os.environ:
                raise Exception("COINBASE_API_SECRET not found in environment variables")
            
            self.client = RESTClient(
                api_key=os.environ['COINBASE_API_KEY'].strip(),
                api_secret=os.environ['COINBASE_API_SECRET'].strip()
            )
            logging.info("Coinbase client initialized successfully")
            
            self.watched_coins: set = set()
            self.trading_interval: int = 300
            self.rsi_period: int = 14
            self.rsi_overbought: float = 70.0
            self.rsi_oversold: float = 30.0
            self.trading_active: bool = False
            self.trade_amount: float = 100.0
            self.trade_history: List[Dict[str, Any]] = []
            self.positions: Dict[str, Position] = {}  # Track active positions
            self.position_history: List[Dict[str, Any]] = []  # Track closed positions
            self.stop_loss_percentage = 5.0  # Default 5% stop loss
            self.take_profit_percentage = 10.0  # Default 10% take profit
            self.max_position_size = 1000.0  # Maximum USD in any single position
            self.load_config()
            
        except Exception as e:
            logging.error(f"Failed to initialize trading bot: {str(e)}")
            raise Exception(f"Bot initialization failed: {str(e)}")
        
    def start_trading_loop(self):
        if not self.trading_active:
            self.trading_active = True
            thread = threading.Thread(target=self._trading_loop)
            thread.daemon = True
            thread.start()
            logging.info("Trading loop started")
            return "Trading bot started successfully"
        return "Trading bot is already running"
        
    def stop_trading_loop(self):
        if self.trading_active:
            self.trading_active = False
            logging.info("Trading loop stopped")
            return "Trading bot stopped successfully"
        return "Trading bot is already stopped"
        
    def _trading_loop(self):
        while self.trading_active:
            try:
                for coin in self.watched_coins:
                    try:
                        # Check risk management first
                        self._check_risk_management(coin)
                        # Then check for new trades
                        self._check_and_trade(coin)
                    except Exception as e:
                        logging.error(f"Error processing {coin}: {str(e)}")
                time.sleep(self.trading_interval)
            except Exception as e:
                logging.error(f"Error in trading loop: {str(e)}")
            
    def _check_and_trade(self, symbol):
        try:
            rsi = self.calculate_rsi(symbol)
            
            if rsi <= self.rsi_oversold and self._should_trade(symbol, 'BUY'):
                self._place_buy_order(symbol)
            elif rsi >= self.rsi_overbought and self._should_trade(symbol, 'SELL'):
                self._place_sell_order(symbol)
            
        except Exception as e:
            logging.error(f"Error checking and trading {symbol}: {str(e)}")
            raise
    
    def calculate_rsi(self, symbol: str) -> float:
        try:
            end = datetime.now()
            start = end - timedelta(days=30)
            prices = self._get_historical_prices(symbol, start, end)
            
            logging.info(f"Calculating RSI for {symbol} with {len(prices)} data points")
            
            if len(prices) < self.rsi_period * 2:  # Need at least 2x RSI period for accuracy
                raise Exception(f"Not enough data points for accurate RSI calculation")
            
            # Calculate price changes
            delta = prices.diff()
            
            # Log some key statistics for verification
            logging.info(f"Price range: ${prices.min():.2f} - ${prices.max():.2f}")
            logging.info(f"Average daily change: ${delta.abs().mean():.2f}")
            
            # Separate gains and losses
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            
            # Calculate exponential moving averages
            avg_gain = gains.ewm(span=self.rsi_period, adjust=False).mean()
            avg_loss = losses.ewm(span=self.rsi_period, adjust=False).mean()
            
            # Calculate RS and RSI
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            # Get the latest RSI value
            current_rsi = float(rsi.iloc[-1])
            
            # Log intermediate values for debugging
            logging.info(f"Latest price: {prices.iloc[-1]:.2f}")
            logging.info(f"Latest gain EMA: {avg_gain.iloc[-1]:.4f}")
            logging.info(f"Latest loss EMA: {avg_loss.iloc[-1]:.4f}")
            logging.info(f"Calculated RSI: {current_rsi:.2f}")
            
            return current_rsi
            
        except Exception as e:
            logging.error(f"Error calculating RSI for {symbol}: {str(e)}")
            raise
    
    def _get_historical_prices(self, symbol: str, start: datetime, end: datetime) -> pd.Series:
        try:
            product_id = f"{symbol}-USD"
            
            # Get current time and calculate start time
            end_time = datetime.now()
            start_time = end_time - timedelta(days=90)  # Get 90 days for better trend analysis
            
            logging.info(f"Fetching candles for {symbol} from {start_time} to {end_time}")
            
            try:
                # Convert to Unix timestamps
                start_unix = int(start_time.timestamp())
                end_unix = int(end_time.timestamp())
                
                # Get daily candles
                response = self.client.get_candles(
                    product_id=product_id,
                    start=start_unix,
                    end=end_unix,
                    granularity="ONE_DAY"  # Daily candles
                )
                
                # Convert response to list and check if we have data
                candles = response.candles if hasattr(response, 'candles') else []
                if not candles:
                    raise Exception(f"No candle data received for {symbol}")
                
                # Convert candles to pandas Series
                prices = pd.Series(
                    [float(candle.close) for candle in reversed(candles)],
                    index=[datetime.fromtimestamp(float(candle.start)) for candle in reversed(candles)]
                )
                
                logging.info(f"Fetched {len(candles)} candles for {symbol}")
                return prices
                    
            except Exception as e:
                logging.error(f"Error fetching candle batch: {str(e)}")
                raise
            
        except Exception as e:
            logging.error(f"Error fetching historical prices for {symbol}: {str(e)}")
            raise
            
    def _place_buy_order(self, symbol: str) -> None:
        try:
            product_id = f"{symbol}-USD"
            current_price = float(self.client.get_product(product_id).price)
            quantity = self.trade_amount / current_price
            
            self.client.create_order(
                product_id=product_id,
                side='BUY',
                order_configuration={
                    'market_market_ioc': {
                        'quote_size': str(self.trade_amount)
                    }
                },
                client_order_id=str(int(time.time()))
            )
            
            # Create new position
            self.positions[symbol] = Position(
                symbol=symbol,
                entry_price=current_price,
                quantity=quantity,
                entry_time=datetime.now()
            )
            
            self.trade_history.append({
                'timestamp': datetime.now(),
                'action': 'BUY',
                'symbol': symbol,
                'amount_usd': self.trade_amount
            })
            logging.info(f"Buy order placed for {symbol}: ${self.trade_amount}")
            
        except Exception as e:
            logging.error(f"Error placing buy order for {symbol}: {str(e)}")
            raise
            
    def _place_sell_order(self, symbol: str) -> None:
        try:
            # Get current position
            position = self.positions.get(symbol)
            if not position:
                logging.warning(f"No position found for {symbol}, cannot sell")
                return
                
            product_id = f"{symbol}-USD"
            current_price = float(self.client.get_product(product_id).price)
            
            self.client.create_order(
                product_id=product_id,
                side='SELL',
                order_configuration={
                    'market_market_ioc': {
                        'quote_size': str(self.trade_amount)
                    }
                },
                client_order_id=str(int(time.time()))
            )
            
            # Calculate final profit
            profit_info = position.calculate_profit(current_price)
            
            # Add to position history
            self.position_history.append({
                'symbol': symbol,
                'entry_price': position.entry_price,
                'exit_price': current_price,
                'quantity': position.quantity,
                'entry_time': position.entry_time,
                'exit_time': datetime.now(),
                'profit_usd': profit_info['profit_usd'],
                'profit_percentage': profit_info['profit_percentage'],
                'max_profit_percentage': profit_info['highest_profit_percentage'],
                'max_drawdown': profit_info['drawdown_percentage']
            })
            
            # Remove the position
            del self.positions[symbol]
            
            self.trade_history.append({
                'timestamp': datetime.now(),
                'action': 'SELL',
                'symbol': symbol,
                'amount_usd': self.trade_amount
            })
            logging.info(f"Sell order placed for {symbol}: ${self.trade_amount}")
            
        except Exception as e:
            logging.error(f"Error placing sell order for {symbol}: {str(e)}")
            raise
            
    def set_trade_amount(self, amount):
        try:
            amount = float(amount)
            if amount <= 0:
                return False
            self.trade_amount = amount
            logging.info(f"Trade amount set to ${amount}")
            return True
        except ValueError:
            return False
            
    def set_rsi_thresholds(self, oversold, overbought):
        try:
            oversold = float(oversold)
            overbought = float(overbought)
            if 0 <= oversold <= overbought <= 100:
                self.rsi_oversold = oversold
                self.rsi_overbought = overbought
                logging.info(f"RSI thresholds updated: oversold={oversold}, overbought={overbought}")
                return True
            return False
        except ValueError:
            return False
            
    def set_trading_interval(self, minutes):
        try:
            minutes = int(minutes)
            if minutes < 1:
                return False
            self.trading_interval = minutes * 60
            logging.info(f"Trading interval set to {minutes} minutes")
            return True
        except ValueError:
            return False
            
    def get_trade_history(self):
        return self.trade_history
        
    def add_coin(self, symbol):
        try:
            # Verify the coin exists by attempting to get its price
            product_id = f"{symbol}-USD"
            self.client.get_product(product_id)
            self.watched_coins.add(symbol)
            logging.info(f"Added {symbol} to watchlist")
            return True
        except Exception as e:
            logging.error(f"Failed to add {symbol}: {str(e)}")
            return False
        
    def remove_coin(self, symbol):
        if symbol in self.watched_coins:
            self.watched_coins.remove(symbol)
            return True
        return False 

    def _check_balance(self, symbol, action='BUY'):
        try:
            accounts = self.client.get_accounts()
            
            if action == 'BUY':
                # Check USD balance
                usd_account = next((acc for acc in accounts.data if acc.currency == 'USD'), None)
                if not usd_account:
                    return False
                return float(usd_account.available_balance.value) >= self.trade_amount
            else:
                # Check crypto balance
                crypto_account = next((acc for acc in accounts.data if acc.currency == symbol), None)
                if not crypto_account:
                    return False
                    
                # Get current price using get_product instead of get_spot_price
                product = self.client.get_product(f"{symbol}-USD")
                current_price = float(product.price)
                return float(crypto_account.available_balance.value) * current_price >= self.trade_amount
                
        except Exception as e:
            logging.error(f"Error checking balance: {str(e)}")
            return False 

    def save_config(self):
        config = {
            'watched_coins': list(self.watched_coins),
            'trading_interval': self.trading_interval,
            'rsi_period': self.rsi_period,
            'rsi_overbought': self.rsi_overbought,
            'rsi_oversold': self.rsi_oversold,
            'trade_amount': self.trade_amount
        }
        try:
            with open('bot_config.json', 'w') as f:
                json.dump(config, f)
            logging.info("Configuration saved successfully")
        except Exception as e:
            logging.error(f"Error saving configuration: {str(e)}")
            
    def load_config(self):
        try:
            with open('bot_config.json', 'r') as f:
                config = json.load(f)
                self.watched_coins = set(config['watched_coins'])
                self.trading_interval = config['trading_interval']
                self.rsi_period = config['rsi_period']
                self.rsi_overbought = config['rsi_overbought']
                self.rsi_oversold = config['rsi_oversold']
                self.trade_amount = config['trade_amount']
            logging.info("Configuration loaded successfully")
        except FileNotFoundError:
            logging.info("No configuration file found, using defaults")
        except Exception as e:
            logging.error(f"Error loading configuration: {str(e)}")

    def test_api_connection(self):
        try:
            # Test authentication by getting BTC price
            btc_product = self.client.get_product('BTC-USD')
            price = float(btc_product.price)
            logging.info(f"Successfully fetched BTC price: ${price}")
            return price
            
        except Exception as e:
            logging.error(f"API test failed: {str(e)}")
            raise Exception(f"API test failed: {str(e)}")

    def get_account_balance(self) -> Dict[str, Union[Dict[str, float], float]]:
        try:
            # Get accounts from response
            accounts_response = self.client.get_accounts()
            balances = {}
            total_usd_value = 0.0

            # Debug logging
            logging.info("Raw accounts response received")
            
            # Access accounts through the accounts attribute
            if hasattr(accounts_response, 'accounts'):
                for account in accounts_response.accounts:
                    # Debug log each account
                    logging.info(f"Processing account: {account.__dict__}")
                    
                    # Check for balance in both available_balance and hold
                    balance_value = 0.0
                    
                    # Handle available balance
                    if hasattr(account, 'available_balance') and isinstance(account.available_balance, dict):
                        balance_value = float(account.available_balance.get('value', 0))
                    
                    # Add hold balance if it exists
                    if hasattr(account, 'hold') and isinstance(account.hold, dict):
                        balance_value += float(account.hold.get('value', 0))
                    
                    if balance_value > 0:
                        symbol = account.currency
                        balance = balance_value
                        
                        if symbol == 'USD':
                            usd_value = balance
                            logging.info(f"Found USD balance: ${usd_value}")
                        else:
                            try:
                                product = self.client.get_product(f"{symbol}-USD")
                                price = float(product.price)
                                usd_value = balance * price
                                logging.info(f"Calculated {symbol} value: ${usd_value}")
                            except Exception as e:
                                logging.warning(f"Could not get price for {symbol}: {str(e)}")
                                continue
                        
                        if usd_value > 0:
                            balances[symbol] = {
                                'balance': balance,
                                'usd_value': usd_value
                            }
                            total_usd_value += usd_value
                            logging.info(f"Added {symbol} balance: {balance} (${usd_value:.2f})")

            logging.info(f"Total portfolio value: ${total_usd_value:.2f}")
            return {
                'balances': balances,
                'total_usd_value': total_usd_value
            }
        except Exception as e:
            logging.error(f"Error getting account balance: {str(e)}")
            return {'balances': {}, 'total_usd_value': 0.0}

    def get_current_price(self, symbol: str) -> float:
        try:
            product_id = f"{symbol}-USD"
            product = self.client.get_product(product_id)
            price = float(product.price)
            logging.info(f"Current price for {symbol}: ${price}")
            return price
        except Exception as e:
            logging.error(f"Error getting price for {symbol}: {str(e)}")
            raise

    def analyze_volume(self, symbol: str) -> Dict[str, Any]:
        """Analyzes trading volume to confirm price trends"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=90)  # 90 days for good volume baseline
            
            # Get candles with volume data
            response = self.client.get_candles(
                product_id=f"{symbol}-USD",
                start=int(start_time.timestamp()),
                end=int(end_time.timestamp()),
                granularity="ONE_DAY"
            )
            
            candles = response.candles if hasattr(response, 'candles') else []
            if not candles:
                raise Exception("No candle data received")
            
            # Extract volume and price data
            volumes = [float(candle.volume) for candle in candles]
            prices = [float(candle.close) for candle in candles]
            
            # Calculate average volume
            avg_volume = sum(volumes) / len(volumes)
            current_volume = volumes[0]  # Most recent volume
            
            # Calculate price change
            price_change = ((prices[0] - prices[1]) / prices[1]) * 100
            
            # Determine if volume confirms trend
            volume_ratio = current_volume / avg_volume
            
            analysis = {
                'current_volume': current_volume,
                'average_volume': avg_volume,
                'volume_ratio': volume_ratio,
                'price_change': price_change,
                'trend_strength': 'strong' if volume_ratio > 1.5 else 'moderate' if volume_ratio > 1.0 else 'weak',
                'confirms_trend': volume_ratio > 1.0 and abs(price_change) > 1.0
            }
            
            logging.info(f"Volume analysis for {symbol}: {analysis}")
            return analysis
            
        except Exception as e:
            logging.error(f"Error analyzing volume for {symbol}: {str(e)}")
            raise

    def _should_trade(self, symbol: str, action: str) -> bool:
        """Determines if a trade should be executed based on multiple indicators"""
        try:
            # Get RSI
            rsi = self.calculate_rsi(symbol)
            
            # Get volume analysis
            volume_data = self.analyze_volume(symbol)
            
            # Get MA analysis
            ma_data = self.calculate_moving_averages(symbol)
            
            # Check if volume confirms trend
            volume_confirms = volume_data['confirms_trend']
            
            if action == 'BUY':
                should_trade = (
                    rsi <= self.rsi_oversold and
                    volume_confirms and
                    volume_data['trend_strength'] != 'weak' and
                    (ma_data['sma_cross_bullish'] or ma_data['ema_cross_bullish']) and
                    ma_data['trend'] in ['Weak Uptrend', 'Strong Uptrend']
                )
            else:  # SELL
                should_trade = (
                    rsi >= self.rsi_overbought and
                    volume_confirms and
                    volume_data['trend_strength'] != 'weak' and
                    (ma_data['sma_cross_bearish'] or ma_data['ema_cross_bearish']) and
                    ma_data['trend'] == 'Downtrend'
                )
            
            logging.info(f"Trade decision for {symbol} {action}: {should_trade}")
            logging.info(f"RSI: {rsi}, Volume: {volume_data['trend_strength']}, Trend: {ma_data['trend']}")
            
            return should_trade
            
        except Exception as e:
            logging.error(f"Error in trade decision for {symbol}: {str(e)}")
            return False

    def get_position_info(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get information about current positions and holdings"""
        try:
            positions_info = {}
            
            # Get account balances
            accounts_response = self.client.get_accounts()
            if hasattr(accounts_response, 'accounts'):
                for account in accounts_response.accounts:
                    # Skip USD and empty balances
                    if (account.currency == 'USD' or 
                        not hasattr(account, 'available_balance') or 
                        float(account.available_balance.get('value', 0)) <= 0):
                        continue
                    
                    symbol = account.currency
                    balance = float(account.available_balance.get('value', 0))
                    
                    try:
                        # Get current price
                        current_price = float(self.client.get_product(f"{symbol}-USD").price)
                        
                        # If we have a tracked position, use its data
                        if symbol in self.positions:
                            position = self.positions[symbol]
                            position.update_price(current_price)
                            profit_info = position.calculate_profit(current_price)
                            
                            positions_info[symbol] = {
                                'symbol': symbol,
                                'entry_price': position.entry_price,
                                'current_price': current_price,
                                'quantity': position.quantity,
                                'entry_time': position.entry_time,
                                'is_bot_position': True,
                                **profit_info
                            }
                        else:
                            # For holdings not tracked by bot, use current price as entry price
                            positions_info[symbol] = {
                                'symbol': symbol,
                                'entry_price': current_price,  # We don't know the actual entry price
                                'current_price': current_price,
                                'quantity': balance,
                                'entry_time': None,  # We don't know when it was acquired
                                'is_bot_position': False,
                                'profit_usd': 0,  # Can't calculate profit without entry price
                                'profit_percentage': 0,
                                'highest_profit_percentage': 0,
                                'drawdown_percentage': 0
                            }
                    
                    except Exception as e:
                        logging.warning(f"Could not process position for {symbol}: {str(e)}")
                        continue
                    
            return positions_info
                
        except Exception as e:
            logging.error(f"Error getting position info: {str(e)}")
            return {}

    def calculate_moving_averages(self, symbol: str) -> Dict[str, Any]:
        """Calculate various moving averages and their signals"""
        try:
            # Get 90 days of price data for reliable MA calculation
            end = datetime.now()
            start = end - timedelta(days=90)
            prices = self._get_historical_prices(symbol, start, end)
            
            # Calculate different MAs
            sma_20 = prices.rolling(window=20).mean()  # 20-day SMA
            sma_50 = prices.rolling(window=50).mean()  # 50-day SMA
            ema_12 = prices.ewm(span=12, adjust=False).mean()  # 12-day EMA
            ema_26 = prices.ewm(span=26, adjust=False).mean()  # 26-day EMA
            
            # Get latest values
            current_price = prices.iloc[-1]
            sma_20_current = sma_20.iloc[-1]
            sma_50_current = sma_50.iloc[-1]
            ema_12_current = ema_12.iloc[-1]
            ema_26_current = ema_26.iloc[-1]
            
            # Check for golden/death crosses (SMA)
            sma_cross_bullish = (sma_20.iloc[-2] <= sma_50.iloc[-2] and 
                               sma_20_current > sma_50_current)
            sma_cross_bearish = (sma_20.iloc[-2] >= sma_50.iloc[-2] and 
                               sma_20_current < sma_50_current)
            
            # Check for EMA crosses
            ema_cross_bullish = (ema_12.iloc[-2] <= ema_26.iloc[-2] and 
                               ema_12_current > ema_26_current)
            ema_cross_bearish = (ema_12.iloc[-2] >= ema_26.iloc[-2] and 
                               ema_12_current < ema_26_current)
            
            # Determine trend based on price position relative to MAs
            above_sma_20 = current_price > sma_20_current
            above_sma_50 = current_price > sma_50_current
            
            if above_sma_20 and above_sma_50:
                trend = "Strong Uptrend"
            elif above_sma_20:
                trend = "Weak Uptrend"
            elif above_sma_50:
                trend = "Mixed Trend"
            else:
                trend = "Downtrend"
                
            analysis = {
                'current_price': current_price,
                'sma_20': sma_20_current,
                'sma_50': sma_50_current,
                'ema_12': ema_12_current,
                'ema_26': ema_26_current,
                'sma_cross_bullish': sma_cross_bullish,
                'sma_cross_bearish': sma_cross_bearish,
                'ema_cross_bullish': ema_cross_bullish,
                'ema_cross_bearish': ema_cross_bearish,
                'trend': trend
            }
            
            logging.info(f"MA analysis for {symbol}: {analysis}")
            return analysis
            
        except Exception as e:
            logging.error(f"Error calculating MAs for {symbol}: {str(e)}")
            raise

    def get_performance_stats(self) -> Dict[str, Any]:
        """Calculate overall trading performance statistics"""
        try:
            stats = {
                'total_trades': len(self.trade_history),
                'active_positions': len(self.positions),
                'closed_positions': len(self.position_history),
                'total_profit_usd': 0.0,
                'win_rate': 0.0,
                'average_profit': 0.0,
                'best_trade': None,
                'worst_trade': None,
                'average_hold_time': timedelta(0)
            }
            
            if self.position_history:
                # Calculate profits
                profits = [pos['profit_usd'] for pos in self.position_history]
                winning_trades = len([p for p in profits if p > 0])
                
                stats.update({
                    'total_profit_usd': sum(profits),
                    'win_rate': (winning_trades / len(profits)) * 100,
                    'average_profit': sum(profits) / len(profits),
                    'best_trade': max(self.position_history, key=lambda x: x['profit_percentage']),
                    'worst_trade': min(self.position_history, key=lambda x: x['profit_percentage']),
                    'average_hold_time': sum(
                        [(pos['exit_time'] - pos['entry_time']) for pos in self.position_history],
                        timedelta(0)
                    ) / len(self.position_history)
                })
                
            return stats
            
        except Exception as e:
            logging.error(f"Error calculating performance stats: {str(e)}")
            return {}

    def set_risk_parameters(self, stop_loss: float, take_profit: float, max_position: float) -> bool:
        """Set risk management parameters"""
        try:
            if 0 < stop_loss < take_profit and max_position > 0:
                self.stop_loss_percentage = stop_loss
                self.take_profit_percentage = take_profit
                self.max_position_size = max_position
                logging.info(f"Risk parameters updated: SL={stop_loss}%, TP={take_profit}%, Max=${max_position}")
                return True
            return False
        except ValueError:
            return False

    def _check_risk_management(self, symbol: str) -> None:
        """Check if any positions need to be closed based on risk management"""
        try:
            position = self.positions.get(symbol)
            if not position:
                return

            current_price = float(self.client.get_product(f"{symbol}-USD").price)
            profit_info = position.calculate_profit(current_price)
            
            # Check stop loss
            if profit_info['profit_percentage'] <= -self.stop_loss_percentage:
                logging.info(f"Stop loss triggered for {symbol} at {profit_info['profit_percentage']}%")
                self._place_sell_order(symbol)
            
            # Check take profit
            elif profit_info['profit_percentage'] >= self.take_profit_percentage:
                logging.info(f"Take profit triggered for {symbol} at {profit_info['profit_percentage']}%")
                self._place_sell_order(symbol)
            
        except Exception as e:
            logging.error(f"Error in risk management for {symbol}: {str(e)}")