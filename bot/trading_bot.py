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
import asyncio

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
            
            # Paper trading attributes
            self.paper_trading = not self.trading_active  # Paper trade when real trading is off
            self.paper_balance = 1000.0  # Start with $1000 paper money
            self.paper_positions: Dict[str, Position] = {}
            self.paper_trade_history: List[Dict[str, Any]] = []
            self.paper_portfolio_value = self.paper_balance
            
            self.discord_channel = None  # Will be set when bot starts
            
        except Exception as e:
            logging.error(f"Failed to initialize trading bot: {str(e)}")
            raise Exception(f"Bot initialization failed: {str(e)}")
        
    def start_trading_loop(self, paper: bool = True):
        """Start the trading loop in either paper or real mode"""
        try:
            if paper:
                if self.trading_active:
                    return "Real trading is already active"
                self.paper_trading = True
                mode = "Paper"
            else:
                if self.paper_trading:
                    return "Paper trading is already active"
                self.trading_active = True
                mode = "Real"
            
            # Create and start the trading loop task
            loop = asyncio.get_event_loop()
            loop.create_task(self._trading_loop())
            
            logging.info(f"{mode} trading loop started")
            return f"{mode} trading bot started successfully"
            
        except Exception as e:
            logging.error(f"Failed to start trading loop: {str(e)}")
            self.trading_active = False
            self.paper_trading = False
            return f"Error starting trading bot: {str(e)}"
        
    def stop_trading_loop(self):
        """Stop the trading loop"""
        was_active = self.trading_active
        self.trading_active = False
        self.paper_trading = False  # Stop paper trading too
        logging.info("Trading loop stopped")
        return "Trading bot stopped successfully" if was_active else "Trading bot is already stopped"
        
    async def _trading_loop(self):
        """Trading loop that runs for both real and paper trading"""
        logging.info("Trading loop started")
        
        while self.trading_active or self.paper_trading:
            try:
                self._ensure_positions_watched()
                logging.info(f"Trading loop iteration - Active: {self.trading_active}, Paper: {self.paper_trading}")
                
                # Send interval update for all watched coins
                if self.watched_coins:
                    logging.info(f"Processing {len(self.watched_coins)} watched coins")
                    message = "🔄 Trading Analysis Update\n\n"
                    
                    for symbol in self.watched_coins:
                        try:
                            logging.info(f"Analyzing {symbol}")
                            # Get all analysis
                            current_price = float(self.client.get_product(f"{symbol}-USD").price)
                            rsi = self.calculate_rsi(symbol)
                            volume_data = self.analyze_volume(symbol)
                            ma_data = self.calculate_moving_averages(symbol)
                            sentiment = self.analyze_market_sentiment(symbol)
                            
                            # Check positions
                            has_real_position = symbol in self.positions
                            has_paper_position = symbol in self.paper_positions
                            
                            # Determine action based on analysis
                            action = "HOLD"
                            reason = []
                            
                            # SELL conditions - check positions first
                            if (self.trading_active and has_real_position) or (self.paper_trading and has_paper_position):
                                if rsi >= self.rsi_overbought:
                                    action = "SELL"
                                    reason.append(f"RSI overbought ({rsi:.2f})")
                                elif ma_data['trend'] == 'Strong Downtrend':
                                    action = "SELL"
                                    reason.append("Strong downtrend detected")
                                elif volume_data['volume_ratio'] > 2.0 and sentiment['overall_sentiment'] == 'Bearish':
                                    action = "SELL"
                                    reason.append("High volume bearish movement")
                            
                            # BUY conditions - only if we don't have a position
                            elif not (has_real_position or has_paper_position):
                                if rsi <= self.rsi_oversold:
                                    action = "BUY"
                                    reason.append(f"RSI oversold ({rsi:.2f})")
                                elif ma_data['trend'] == 'Strong Uptrend' and volume_data['volume_ratio'] > 1.5:
                                    action = "BUY"
                                    reason.append("Strong uptrend with volume confirmation")
                            
                            # Add analysis to message
                            message += f"📊 {symbol} Analysis:\n"
                            message += f"Price: ${current_price:,.2f}\n"
                            message += f"RSI: {rsi:.2f}\n"
                            message += f"Volume: {volume_data['volume_ratio']:.2f}x average\n"
                            message += f"Trend: {ma_data['trend']}\n"
                            message += f"Sentiment: {sentiment['overall_sentiment']}\n"
                            message += f"Position: {'Yes' if (has_real_position or has_paper_position) else 'No'}\n"
                            message += f"Decision: {action}\n"
                            if reason:
                                message += f"Reason: {', '.join(reason)}\n"
                            message += "\n"
                            
                            # Execute trades if conditions are met
                            if self.paper_trading and action != "HOLD":
                                if action == "BUY" and self._should_trade(symbol, 'BUY'):
                                    await self._simulate_buy_order(symbol)
                                elif action == "SELL" and has_paper_position and self._should_trade(symbol, 'SELL'):
                                    await self._simulate_sell_order(symbol)
                        
                        except Exception as e:
                            error_msg = f"Error analyzing {symbol}: {str(e)}"
                            logging.error(error_msg)
                            message += f"❌ {error_msg}\n\n"
                    
                    # Send the update
                    logging.info("Attempting to send interval update notification")
                    try:
                        await self.send_notification(message, is_update=True)
                        logging.info("Successfully sent interval update")
                    except Exception as e:
                        logging.error(f"Failed to send notification: {str(e)}")
                else:
                    logging.warning("No coins in watchlist")
                
                # Wait for next interval
                logging.info(f"Waiting {self.trading_interval} seconds until next update")
                await asyncio.sleep(self.trading_interval)
                
            except Exception as e:
                logging.error(f"Critical error in trading loop: {str(e)}")
                # Wait a minute before retrying if there's an error
                await asyncio.sleep(60)
    
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
        
    def add_coin(self, symbol: str) -> bool:
        """Add a coin to watchlist"""
        try:
            # Validate the symbol
            self.client.get_product(f"{symbol}-USD")
            self.watched_coins.add(symbol)
            logging.info(f"Added {symbol} to watchlist")
            return True
        except Exception as e:
            logging.error(f"Failed to add {symbol}: {str(e)}")
            return False

    def remove_coin(self, symbol: str) -> bool:
        """Remove a coin from watchlist if not in any positions"""
        if symbol in self.positions or symbol in self.paper_positions:
            logging.warning(f"Cannot remove {symbol} - active position exists")
            return False
        
        if symbol in self.watched_coins:
            self.watched_coins.remove(symbol)
            logging.info(f"Removed {symbol} from watchlist")
            return True
        return False

    def _ensure_positions_watched(self):
        """Ensure all positions are in watchlist"""
        for symbol in self.positions.keys():
            self.watched_coins.add(symbol)
        for symbol in self.paper_positions.keys():
            self.watched_coins.add(symbol)

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

    def _calculate_position_size(self, symbol: str) -> float:
        """Smarter position sizing based on risk"""
        try:
            available_funds = float(self.get_account_balance()['balances'].get('USD', {}).get('balance', 0))
            
            # Never risk more than 2% of total portfolio on any single trade
            portfolio_value = self.get_account_balance()['total_usd_value']
            max_risk_amount = portfolio_value * 0.02
            
            # Calculate position size based on stop loss
            current_price = float(self.client.get_product(f"{symbol}-USD").price)
            risk_per_share = current_price * (self.stop_loss_percentage / 100)
            
            # Position size that risks the max risk amount
            position_size = min(
                max_risk_amount / risk_per_share * current_price,
                available_funds,
                self.max_position_size
            )
            
            # Ensure minimum trade size
            return max(5.0, position_size) if position_size >= 5.0 else 0
            
        except Exception as e:
            logging.error(f"Error calculating position size: {str(e)}")
            return 0

    def _check_market_conditions(self, symbol: str) -> Dict[str, Any]:
        """Analyze current market conditions for trading"""
        try:
            # Get sentiment analysis
            sentiment = self.analyze_market_sentiment(symbol)
            
            # Get volatility (using price range as simple measure)
            prices = self._get_historical_prices(symbol, 
                                              start=datetime.now() - timedelta(days=7),
                                              end=datetime.now())
            price_range = (prices.max() - prices.min()) / prices.min() * 100
            
            # Check if market is too volatile
            is_volatile = price_range > 10  # Consider >10% range as volatile
            
            # Check trading hours (some hours historically have more volatility)
            current_hour = datetime.now().hour
            is_high_activity = 13 <= current_hour <= 21  # 9 AM - 5 PM EST
            
            # Get overall market trend (using BTC as proxy)
            if symbol != 'BTC':
                btc_sentiment = self.analyze_market_sentiment('BTC')
                market_aligned = (
                    (sentiment['overall_sentiment'] == btc_sentiment['overall_sentiment']) or
                    (sentiment['sentiment_score'] * btc_sentiment['sentiment_score'] > 0)
                )
            else:
                market_aligned = True
            
            conditions = {
                'sentiment': sentiment['overall_sentiment'],
                'sentiment_score': sentiment['sentiment_score'],
                'is_volatile': is_volatile,
                'price_range_7d': price_range,
                'is_high_activity': is_high_activity,
                'market_aligned': market_aligned,
                'suitable_for_trading': (
                    not is_volatile and
                    (is_high_activity or abs(sentiment['sentiment_score']) > 50) and
                    market_aligned
                )
            }
            
            logging.info(f"Market conditions for {symbol}: {conditions}")
            return conditions
            
        except Exception as e:
            logging.error(f"Error checking market conditions: {str(e)}")
            raise

    def _should_trade(self, symbol: str, action: str) -> bool:
        """Enhanced trade confirmation with multiple timeframes"""
        try:
            # Get current price and volume data
            current_price = float(self.client.get_product(f"{symbol}-USD").price)
            volume_data = self.analyze_volume(symbol)
            
            # Check if we have enough funds
            position_size = self._calculate_position_size(symbol)
            if position_size < 5.0:  # Minimum trade size
                return False
            
            if action == 'BUY':
                # Price must be above 20 EMA on shorter timeframe for uptrend confirmation
                ma_data = self.calculate_moving_averages(symbol)
                price_above_ema = current_price > ma_data['ema_20']
                
                # Volume should be increasing
                volume_confirming = volume_data['volume_ratio'] > 1.2
                
                # Check if we're buying near recent support
                recent_low = min(self._get_historical_prices(symbol, 
                    datetime.now() - timedelta(days=7), 
                    datetime.now()
                ))
                not_chasing = current_price < (recent_low * 1.1)  # Within 10% of recent low
                
                return price_above_ema and volume_confirming and not_chasing
                
            else:  # SELL
                # Don't sell if we're near support levels
                recent_low = min(self._get_historical_prices(symbol, 
                    datetime.now() - timedelta(days=7), 
                    datetime.now()
                ))
                near_support = current_price < (recent_low * 1.05)
                
                # Volume should confirm downtrend
                volume_confirming = volume_data['volume_ratio'] > 1.2
                
                return not near_support and volume_confirming
                
        except Exception as e:
            logging.error(f"Error in trade decision: {str(e)}")
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
        """Enhanced risk management with trailing stops"""
        try:
            position = self.positions.get(symbol)
            if not position:
                return
            
            current_price = float(self.client.get_product(f"{symbol}-USD").price)
            profit_info = position.calculate_profit(current_price)
            
            # Update trailing stop as profit increases
            if profit_info['profit_percentage'] > 5:  # Above 5% profit
                # Move stop loss to break even
                adjusted_stop = max(
                    position.entry_price,  # Don't go below entry price
                    current_price * (1 - self.stop_loss_percentage/200)  # Tighter stop
                )
            else:
                adjusted_stop = position.entry_price * (1 - self.stop_loss_percentage/100)
            
            # Check stops
            if current_price <= adjusted_stop:
                logging.info(f"Stop loss triggered for {symbol} at {profit_info['profit_percentage']}%")
                self._place_sell_order(symbol)
            
            # Take partial profits at targets
            elif profit_info['profit_percentage'] >= self.take_profit_percentage:
                # Sell half the position
                original_quantity = position.quantity
                self.trade_amount = (original_quantity * current_price) / 2
                logging.info(f"Taking partial profits for {symbol} at {profit_info['profit_percentage']}%")
                self._place_sell_order(symbol)
            
        except Exception as e:
            logging.error(f"Error in risk management: {str(e)}")

    def analyze_market_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Analyze overall market sentiment using multiple indicators"""
        try:
            # Get data for different timeframes
            end = datetime.now()
            start_long = end - timedelta(days=90)   # 90 days for long-term
            start_medium = end - timedelta(days=30)  # 30 days for medium-term
            start_short = end - timedelta(days=7)    # 7 days for short-term
            
            # Get price data for different timeframes
            prices_long = self._get_historical_prices(symbol, start_long, end)
            prices_medium = prices_long[prices_long.index >= start_medium]
            prices_short = prices_long[prices_long.index >= start_short]
            
            # Calculate price changes
            price_change_long = ((prices_long.iloc[-1] - prices_long.iloc[0]) / prices_long.iloc[0]) * 100
            price_change_medium = ((prices_medium.iloc[-1] - prices_medium.iloc[0]) / prices_medium.iloc[0]) * 100
            price_change_short = ((prices_short.iloc[-1] - prices_short.iloc[0]) / prices_short.iloc[0]) * 100
            
            # Get technical indicators
            ma_data = self.calculate_moving_averages(symbol)
            volume_data = self.analyze_volume(symbol)
            rsi = self.calculate_rsi(symbol)
            
            # Determine momentum
            momentum_signals = {
                'short_term': 'bullish' if price_change_short > 0 else 'bearish',
                'medium_term': 'bullish' if price_change_medium > 0 else 'bearish',
                'long_term': 'bullish' if price_change_long > 0 else 'bearish'
            }
            
            # Calculate strength of trend
            trend_strength = sum([
                1 if momentum_signals['short_term'] == 'bullish' else -1,
                1 if momentum_signals['medium_term'] == 'bullish' else -1,
                1 if momentum_signals['long_term'] == 'bullish' else -1
            ])
            
            # Analyze volume trend
            volume_trend = 'bullish' if volume_data['volume_ratio'] > 1.0 else 'bearish'
            
            # Combine all signals
            bullish_signals = sum([
                1 if ma_data['trend'] in ['Strong Uptrend', 'Weak Uptrend'] else 0,
                1 if volume_trend == 'bullish' else 0,
                1 if rsi < 50 else 0,  # RSI below 50 suggests room for growth
                1 if trend_strength > 0 else 0
            ])
            
            # Calculate overall sentiment score (-100 to 100)
            sentiment_score = (bullish_signals / 4) * 100 - 50
            
            analysis = {
                'sentiment_score': sentiment_score,
                'overall_sentiment': 'Strong Buy' if sentiment_score > 75 else
                                   'Buy' if sentiment_score > 25 else
                                   'Neutral' if sentiment_score > -25 else
                                   'Sell' if sentiment_score > -75 else
                                   'Strong Sell',
                'price_changes': {
                    'short_term': price_change_short,
                    'medium_term': price_change_medium,
                    'long_term': price_change_long
                },
                'momentum': momentum_signals,
                'trend_strength': trend_strength,
                'volume_trend': volume_trend,
                'technical_indicators': {
                    'ma_trend': ma_data['trend'],
                    'rsi': rsi,
                    'volume_ratio': volume_data['volume_ratio']
                }
            }
            
            logging.info(f"Market sentiment analysis for {symbol}: {analysis}")
            return analysis
            
        except Exception as e:
            logging.error(f"Error analyzing market sentiment for {symbol}: {str(e)}")
            raise

    def _simulate_buy_order(self, symbol: str) -> None:
        """Simulate a buy order with paper trading"""
        try:
            product_id = f"{symbol}-USD"
            current_price = float(self.client.get_product(product_id).price)
            
            # Calculate fees (0.6% Coinbase fee)
            fee = self.trade_amount * 0.006
            actual_trade_amount = self.trade_amount - fee
            
            # Check if we have enough paper balance
            if self.paper_balance < self.trade_amount:
                logging.warning(f"Insufficient paper balance for {symbol} buy")
                return
            
            quantity = actual_trade_amount / current_price
            
            # Create new paper position
            self.paper_positions[symbol] = Position(
                symbol=symbol,
                entry_price=current_price,
                quantity=quantity,
                entry_time=datetime.now(),
                is_paper=True
            )
            
            # Update paper balance
            self.paper_balance -= self.trade_amount
            
            # Record trade
            self.paper_trade_history.append({
                'timestamp': datetime.now(),
                'action': 'BUY',
                'symbol': symbol,
                'amount_usd': self.trade_amount,
                'price': current_price,
                'quantity': quantity,
                'fees': fee,
                'is_paper': True
            })
            
            logging.info(f"Paper buy order placed for {symbol}: ${self.trade_amount}")
            
        except Exception as e:
            logging.error(f"Error simulating buy order for {symbol}: {str(e)}")
            raise

    def _simulate_sell_order(self, symbol: str) -> None:
        """Simulate a sell order with paper trading"""
        try:
            position = self.paper_positions.get(symbol)
            if not position:
                logging.warning(f"No paper position found for {symbol}, cannot sell")
                return
            
            product_id = f"{symbol}-USD"
            current_price = float(self.client.get_product(product_id).price)
            
            # Calculate total value and fees
            total_value = position.quantity * current_price
            fee = total_value * 0.006
            actual_value = total_value - fee
            
            # Calculate profit/loss
            profit_info = position.calculate_profit(current_price)
            
            # Update paper balance
            self.paper_balance += actual_value
            
            # Record trade
            self.paper_trade_history.append({
                'timestamp': datetime.now(),
                'action': 'SELL',
                'symbol': symbol,
                'amount_usd': total_value,
                'price': current_price,
                'quantity': position.quantity,
                'fees': fee,
                'profit': profit_info['profit_usd'],
                'profit_percentage': profit_info['profit_percentage'],
                'is_paper': True
            })
            
            # Remove position
            del self.paper_positions[symbol]
            
            logging.info(f"Paper sell order placed for {symbol}: ${total_value:.2f} (Profit: ${profit_info['profit_usd']:.2f})")
            
        except Exception as e:
            logging.error(f"Error simulating sell order for {symbol}: {str(e)}")
            raise

    def get_paper_balance(self) -> Dict[str, float]:
        """Get paper trading account balance"""
        try:
            total_value = self.paper_balance
            
            # Add value of all paper positions
            for symbol, position in self.paper_positions.items():
                try:
                    current_price = float(self.client.get_product(f"{symbol}-USD").price)
                    position_value = position.quantity * current_price
                    total_value += position_value
                except Exception as e:
                    logging.error(f"Error calculating paper position value for {symbol}: {str(e)}")
                
            return {
                'cash_balance': self.paper_balance,
                'total_value': total_value
            }
        except Exception as e:
            logging.error(f"Error getting paper balance: {str(e)}")
            return {'cash_balance': 0.0, 'total_value': 0.0}

    def reset_paper_trading(self, initial_balance: float = 1000.0) -> None:
        """Reset paper trading with new balance"""
        self.paper_balance = initial_balance
        self.paper_positions.clear()
        self.paper_trade_history.clear()
        logging.info(f"Paper trading reset with ${initial_balance} balance")

    def set_discord_channel(self, channel):
        """Set the Discord channel for notifications"""
        self.discord_channel = channel
        logging.info(f"Discord notifications channel set")

    async def send_notification(self, message: str, is_update: bool = False):
        """Send a notification to Discord channel if available"""
        if not self.discord_channel:
            # Just log the message if no channel is set
            logging.info(f"Notification (no channel): {message}")
            return
        
        try:
            # Format based on type
            if is_update:
                formatted_message = f"📊 Trading Update:\n```{message}```"
            else:
                formatted_message = f"🔔 Alert:\n```{message}```"
            
            await self.discord_channel.send(formatted_message)
            logging.info(f"Notification sent: {message}")
        except Exception as e:
            logging.error(f"Error sending notification: {str(e)}")
            self.discord_channel = None  # Reset channel if we can't send messages

    async def send_trade_notification(self, action: str, symbol: str, price: float, quantity: float, 
                                    is_paper: bool = False, profit_info: Dict[str, float] = None):
        """Send a trade notification"""
        try:
            trade_type = "Paper" if is_paper else "Real"
            message = f"{trade_type} Trade: {action} {symbol}\n"
            message += f"Price: ${price:,.2f}\n"
            message += f"Quantity: {quantity:.8f}\n"
            message += f"Total: ${(price * quantity):,.2f}"
            
            if profit_info and action == 'SELL':
                message += f"\nProfit: ${profit_info['profit_usd']:+,.2f} ({profit_info['profit_percentage']:+.2f}%)"
                message += f"\nFees Paid: ${profit_info['fees_paid']:.2f}"
            
            await self.send_notification(message)
        except Exception as e:
            logging.error(f"Error sending trade notification: {str(e)}")

    async def send_interval_update(self):
        """Send periodic update of all watched coins"""
        try:
            if not self.watched_coins:
                return
            
            message = "Periodic Trading Update\n\n"
            
            for symbol in self.watched_coins:
                try:
                    # Get all analysis
                    current_price = float(self.client.get_product(f"{symbol}-USD").price)
                    rsi = self.calculate_rsi(symbol)
                    volume_data = self.analyze_volume(symbol)
                    ma_data = self.calculate_moving_averages(symbol)
                    sentiment = self.analyze_market_sentiment(symbol)
                    
                    # Determine action
                    action = "HOLD"
                    if rsi <= self.rsi_oversold and self._should_trade(symbol, 'BUY'):
                        action = "BUY SIGNAL"
                    elif rsi >= self.rsi_overbought and self._should_trade(symbol, 'SELL'):
                        action = "SELL SIGNAL"
                    
                    # Add to message
                    message += f"{symbol}:\n"
                    message += f"Price: ${current_price:,.2f}\n"
                    message += f"RSI: {rsi:.2f}\n"
                    message += f"Volume: {volume_data['volume_ratio']:.2f}x average\n"
                    message += f"Trend: {ma_data['trend']}\n"
                    message += f"Sentiment: {sentiment['overall_sentiment']}\n"
                    message += f"Action: {action}\n\n"
                    
                except Exception as e:
                    message += f"{symbol}: Error analyzing - {str(e)}\n\n"
            
            await self.send_notification(message, is_update=True)
        except Exception as e:
            logging.error(f"Error sending interval update: {str(e)}")

    async def send_alert(self, symbol: str, alert_type: str, details: str):
        """Send an alert notification"""
        try:
            message = f"Alert for {symbol}\n"
            message += f"Type: {alert_type}\n"
            message += f"Details: {details}"
            await self.send_notification(message)
        except Exception as e:
            logging.error(f"Error sending alert: {str(e)}")