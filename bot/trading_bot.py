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
import discord

# Set up logging
logging.basicConfig(
    filename='trading_bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class TradingBotError(Exception):
    """Base exception class for TradingBot errors"""
    pass

class APIError(TradingBotError):
    """Raised when there's an error with API calls"""
    pass

class DataError(TradingBotError):
    """Raised when there's an error with data processing"""
    pass

class SignalError(TradingBotError):
    """Raised when there's an error calculating trading signals"""
    pass

class TradeError(TradingBotError):
    """Raised when there's an error executing trades"""
    pass

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
        """Enhanced trading loop with proper execution and logging"""
        await self.async_log("Trading loop started", context={
            'mode': 'Paper' if self.paper_trading else 'Real',
            'watched_coins': list(self.watched_coins),
            'trading_interval': f"{self.trading_interval//60} minutes"
        })
        
        last_update_hour = None
        
        while self.trading_active or self.paper_trading:
            try:
                # Add recovery delay for serious errors
                recovery_delay = 60  # 1 minute
                
                try:
                    current_hour = datetime.now().hour
                    
                    # Log iteration start with context
                    await self.async_log(
                        "Starting trading iteration",
                        context={
                            'mode': 'Paper' if self.paper_trading else 'Real',
                            'active_coins': list(self.watched_coins),
                            'current_hour': f"{current_hour:02d}:00",
                            'paper_positions': len(self.paper_positions),
                            'real_positions': len(self.positions)
                        }
                    )
                    
                    for symbol in self.watched_coins:
                        try:
                            await self.async_log(f"Processing {symbol}...")
                            
                            # Get current analysis
                            current_price = float(self.client.get_product(f"{symbol}-USD").price)
                            position = self.paper_positions.get(symbol) if self.paper_trading else self.positions.get(symbol)
                            
                            # Log current state
                            await self.async_log(
                                f"{symbol} Status:\n"
                                f"Current Price: ${current_price:,.2f}\n"
                                f"Has Position: {position is not None}\n"
                                f"Trading Mode: {'Paper' if self.paper_trading else 'Real'}"
                            )
                            
                            # Get trade signal and log it
                            signal = self._calculate_trade_signal(symbol)
                            await self.async_log(
                                f"{symbol} Signal Analysis:\n"
                                f"Action: {signal['action']}\n"
                                f"Score: {signal['score']}\n"
                                f"Components:\n" +
                                '\n'.join(f"- {k}: {v}" for k, v in signal['signals'].items())
                            )
                            
                            # Update position tracking if we have one
                            if position:
                                position.update_price(current_price)
                                profit_info = position.calculate_profit(current_price)
                                
                                await self.async_log(
                                    f"{symbol} Position Update:\n"
                                    f"Entry Price: ${position.entry_price:,.2f}\n"
                                    f"Current P/L: {profit_info['profit_percentage']:+.2f}%\n"
                                    f"Max Profit: {profit_info['highest_profit_percentage']:+.2f}%\n"
                                    f"Max Drawdown: {profit_info['drawdown_percentage']:+.2f}%"
                                )
                                
                                # Check if we should sell
                                if self._should_trade(symbol, 'SELL'):
                                    await self.async_log(f"SELL signal confirmed for {symbol}")
                                    
                                    if self.paper_trading:
                                        await self._simulate_sell_order(symbol)
                                        await self.send_trade_notification(
                                            'SELL', symbol, current_price, 
                                            position.quantity, is_paper=True,
                                            profit_info=profit_info
                                        )
                                    else:
                                        self._place_sell_order(symbol)
                                    continue
                        
                            # Check if we should buy
                            elif self._should_trade(symbol, 'BUY'):
                                position_size = self._calculate_position_size(symbol)
                                await self.async_log(
                                    f"BUY signal confirmed for {symbol}\n"
                                    f"Calculated position size: ${position_size:,.2f}"
                                )
                                
                                if position_size < 5.0:
                                    await self.async_log(f"Position size too small for {symbol}, skipping")
                                    continue
                                    
                                if self.paper_trading:
                                    await self._simulate_buy_order(symbol)
                                    await self.send_trade_notification(
                                        'BUY', symbol, current_price,
                                        position_size / current_price, is_paper=True
                                    )
                                else:
                                    self._place_buy_order(symbol)
                    
                        except Exception as e:
                            await self.async_log(f"Error processing {symbol}: {str(e)}", level="error")
                            continue
                        
                    # Send periodic update if hour has changed
                    if current_hour != last_update_hour:
                        await self.async_log("Preparing hourly update...")
                        await self.send_interval_update()
                        last_update_hour = current_hour
                        
                    # Log end of iteration
                    await self.async_log(
                        "Trading iteration completed\n"
                        f"Paper Positions: {len(self.paper_positions)}\n"
                        f"Real Positions: {len(self.positions)}\n"
                        f"Next update in {self.trading_interval} seconds"
                    )
                    
                    # Normal delay between iterations
                    await asyncio.sleep(self.trading_interval)
                    
                except APIError as e:
                    await self.async_log(f"API Error in trading loop: {str(e)}", level="error")
                    await asyncio.sleep(recovery_delay)
                    continue
                    
                except DataError as e:
                    await self.async_log(f"Data Error in trading loop: {str(e)}", level="error")
                    await asyncio.sleep(recovery_delay)
                    continue
                    
                except Exception as e:
                    await self.async_log(f"Unexpected error in trading loop: {str(e)}", level="error")
                    await asyncio.sleep(recovery_delay * 2)  # Longer delay for unexpected errors
                    continue
                    
            except Exception as e:
                await self.async_log(f"Critical error in trading loop: {str(e)}", level="error")
                await asyncio.sleep(recovery_delay * 3)  # Even longer delay for critical errors
    
    def _check_and_trade(self, symbol):
        try:
            rsi = self.calculate_rsi(symbol)
            
            if rsi <= self.rsi_oversold and self._should_trade(symbol, 'BUY'):
                self._place_buy_order(symbol)
            elif rsi >= self.rsi_overbought and self._should_trade(symbol, 'SELL'):
                self._place_sell_order(symbol)
            
        except Exception as e:
            self.log(f"Error checking and trading {symbol}: {str(e)}", level="error")
            raise
    
    def calculate_rsi(self, symbol: str) -> float:
        try:
            end = datetime.now()
            start = end - timedelta(days=30)
            prices = self._get_historical_prices(symbol, start, end)
            
            self.log(f"Calculating RSI for {symbol} with {len(prices)} data points")
            
            if len(prices) < self.rsi_period * 2:  # Need at least 2x RSI period for accuracy
                raise Exception(f"Not enough data points for accurate RSI calculation")
            
            # Calculate price changes
            delta = prices.diff()
            
            # Log some key statistics for verification
            self.log(f"Price range: ${prices.min():.2f} - ${prices.max():.2f}")
            self.log(f"Average daily change: ${delta.abs().mean():.2f}")
            
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
            self.log(f"Latest price: {prices.iloc[-1]:.2f}")
            self.log(f"Latest gain EMA: {avg_gain.iloc[-1]:.4f}")
            self.log(f"Latest loss EMA: {avg_loss.iloc[-1]:.4f}")
            self.log(f"Calculated RSI: {current_rsi:.2f}")
            
            return current_rsi
            
        except Exception as e:
            self.log(f"Error calculating RSI for {symbol}: {str(e)}", level="error")
            raise
    
    def _get_historical_prices(self, symbol: str, start: datetime, end: datetime) -> pd.Series:
        """Get historical prices with rate limiting protection"""
        try:
            # Add rate limiting protection
            if hasattr(self, '_last_api_call'):
                time_since_last_call = time.time() - self._last_api_call
                if time_since_last_call < 0.1:  # Max 10 calls per second
                    time.sleep(0.1 - time_since_last_call)
            
            # Convert to Unix timestamps
            start_unix = int(start.timestamp())
            end_unix = int(end.timestamp())
            
            # Get candles using correct method name
            response = self.client.get_candles(
                product_id=f"{symbol}-USD",
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
            
            self._last_api_call = time.time()
            self.log(f"Fetched {len(candles)} candles for {symbol}")
            return prices
            
        except Exception as e:
            self.log(f"Error fetching historical prices: {str(e)}", level="error")
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
                client_order_id=str(int(time.time())))
            
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
                client_order_id=str(int(time.time())))
            
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
        """Load configuration from file or use defaults"""
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
                self.rsi_period = config.get('rsi_period', 14)
                self.rsi_overbought = config.get('rsi_overbought', 70.0)
                self.rsi_oversold = config.get('rsi_oversold', 30.0)
                self.trading_interval = config.get('trading_interval', 300)
                self.trade_amount = config.get('trade_amount', 100.0)
                self.stop_loss_percentage = config.get('stop_loss_percentage', 5.0)
                self.take_profit_percentage = config.get('take_profit_percentage', 10.0)
                self.max_position_size = config.get('max_position_size', 1000.0)
        except FileNotFoundError:
            self.log("No config file found, using defaults", level="warning")

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
            
            # Create conditions dict before using it
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
            
            self.log(f"Market conditions for {symbol}: {conditions}")
            return conditions
            
        except Exception as e:
            self.log(f"Error checking market conditions: {str(e)}", level="error")
            raise

    def _calculate_trade_signal(self, symbol: str) -> Dict[str, Any]:
        """Enhanced trade signal calculation with comprehensive error handling"""
        try:
            # Get current price
            current_price = float(self.client.get_product(f"{symbol}-USD").price)
            
            # Get technical analysis data
            ma_data = self.calculate_moving_averages(symbol)
            ma_data['symbol'] = symbol  # Add symbol to ma_data
            ma_data['current_price'] = current_price  # Add current price to ma_data
            
            volume_data = self.analyze_volume(symbol)
            sentiment = self.analyze_market_sentiment(symbol)
            market_conditions = self._check_market_conditions(symbol)

            # Log raw data for debugging
            self.log(f"Raw signal data for {symbol}:", context={
                'ma_data': ma_data,
                'volume_data': volume_data, 
                'sentiment': sentiment,
                'market_conditions': market_conditions
            })

            # Validate data
            if not all(k in ma_data for k in ['trend', 'sma_50', 'sma_200']):
                raise DataError("Missing required MA data")
            if not all(k in volume_data for k in ['volume_ratio', 'confirms_trend']):
                raise DataError("Missing required volume data")
            if not all(k in sentiment for k in ['sentiment_score', 'overall_sentiment', 'momentum']):
                raise DataError("Missing required sentiment data")

            # Calculate signal components
            signals = self._calculate_signal_components(
                ma_data, volume_data, sentiment, market_conditions
            )

            return signals

        except Exception as e:
            self.log(f"Error calculating trade signal for {symbol}: {str(e)}", level="error")
            return self._get_fallback_signal(symbol, str(e))

    def _get_fallback_signal(self, symbol: str, error_type: str) -> Dict[str, Any]:
        """Get a safe fallback signal when errors occur"""
        return {
            'symbol': symbol,
            'score': 0,
            'signals': {
                'trend': 0,
                'momentum': 0,
                'volume': 0,
                'risk': 0
            },
            'action': 'HOLD',
            'timestamp': datetime.now(),
            'error': error_type
        }

    def _should_trade(self, symbol: str, action: str) -> bool:
        """Determine if we should trade based on the weighted signal system"""
        try:
            # Calculate position size first
            position_size = self._calculate_position_size(symbol)
            if position_size < 5.0:  # Minimum trade size
                return False

            # Get trade signal
            signal = self._calculate_trade_signal(symbol)
            if not signal:
                return False

            if action == 'BUY':
                return signal['action'] in ['BUY', 'STRONG_BUY']
            elif action == 'SELL':
                return signal['action'] in ['SELL', 'STRONG_SELL']

            return False

        except Exception as e:
            self.log(f"Error in trade decision: {str(e)}", level="error")
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
            # Get 200+ days of price data for reliable MA calculation
            end = datetime.now()
            start = end - timedelta(days=250)  # Extra days for proper 200MA calculation
            prices = self._get_historical_prices(symbol, start, end)
            
            # Calculate different MAs
            sma_20 = prices.rolling(window=20).mean()  # 20-day SMA
            sma_50 = prices.rolling(window=50).mean()  # 50-day SMA
            sma_200 = prices.rolling(window=200).mean()  # 200-day SMA
            ema_12 = prices.ewm(span=12, adjust=False).mean()  # 12-day EMA
            ema_26 = prices.ewm(span=26, adjust=False).mean()  # 26-day EMA
            
            # Get latest values
            current_price = prices.iloc[-1]
            sma_20_current = sma_20.iloc[-1]
            sma_50_current = sma_50.iloc[-1]
            sma_200_current = sma_200.iloc[-1]
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
            above_sma_200 = current_price > sma_200_current
            
            if above_sma_20 and above_sma_50 and above_sma_200:
                trend = "Strong Uptrend"
            elif above_sma_20 and above_sma_50:
                trend = "Moderate Uptrend"
            elif above_sma_20:
                trend = "Weak Uptrend"
            elif not (above_sma_20 or above_sma_50 or above_sma_200):
                trend = "Strong Downtrend"
            else:
                trend = "Mixed Trend"
                
            # Add EMA-20
            ema_20 = prices.ewm(span=20, adjust=False).mean()  # 20-day EMA
            
            return {
                'current_price': current_price,
                'sma_20': sma_20_current,
                'sma_50': sma_50_current,
                'sma_200': sma_200_current,
                'ema_12': ema_12_current,
                'ema_20': ema_20.iloc[-1],  # Add EMA-20
                'ema_26': ema_26_current,
                'sma_cross_bullish': sma_cross_bullish,
                'sma_cross_bearish': sma_cross_bearish,
                'ema_cross_bullish': ema_cross_bullish,
                'ema_cross_bearish': ema_cross_bearish,
                'trend': trend
            }
            
        except Exception as e:
            self.log(f"Error calculating MAs for {symbol}: {str(e)}", level="error")
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
                self.log(f"Stop loss triggered for {symbol} at {profit_info['profit_percentage']}%")
                self._place_sell_order(symbol)
            
            # Take partial profits at targets
            elif profit_info['profit_percentage'] >= self.take_profit_percentage:
                # Sell half the position
                original_quantity = position.quantity
                self.trade_amount = (original_quantity * current_price) / 2
                self.log(f"Taking partial profits for {symbol} at {profit_info['profit_percentage']}%")
                self._place_sell_order(symbol)
            
        except Exception as e:
            self.log(f"Error in risk management: {str(e)}", level="error")

    def analyze_market_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Analyze overall market sentiment using multiple indicators"""
        try:
            # Get data for different timeframes
            end = datetime.now()
            start_long = end - timedelta(days=90)   
            start_medium = end - timedelta(days=30)  
            start_short = end - timedelta(days=7)    
            
            # Get price data for different timeframes with validation
            prices_long = self._get_historical_prices(symbol, start_long, end)
            if len(prices_long) < 2:
                raise DataError(f"Insufficient historical data for {symbol}")
                
            self.log(f"Got {len(prices_long)} price points for {symbol}")
            
            prices_medium = prices_long[prices_long.index >= start_medium]
            prices_short = prices_long[prices_long.index >= start_short]
            
            # Validate we have enough data points
            if len(prices_short) < 2 or len(prices_medium) < 2:
                raise DataError(f"Insufficient data points for sentiment calculation")
                
            # Calculate and log price changes
            price_changes = {
                'long': ((prices_long.iloc[-1] - prices_long.iloc[0]) / prices_long.iloc[0]) * 100,
                'medium': ((prices_medium.iloc[-1] - prices_medium.iloc[0]) / prices_medium.iloc[0]) * 100,
                'short': ((prices_short.iloc[-1] - prices_short.iloc[0]) / prices_short.iloc[0]) * 100
            }
            
            self.log(f"Price changes for {symbol}:", context=price_changes)
            
            return {
                'sentiment_score': price_changes['short'],
                'overall_sentiment': 'Bullish' if price_changes['short'] > 0 else 'Bearish',
                'momentum': {
                    'short_term': 'bullish' if price_changes['short'] > 0 else 'bearish',
                    'medium_term': 'bullish' if price_changes['medium'] > 0 else 'bearish',
                    'long_term': 'bullish' if price_changes['long'] > 0 else 'bearish'
                },
                'price_changes': {
                    'short_term': price_changes['short'],
                    'medium_term': price_changes['medium'],
                    'long_term': price_changes['long']
                }
            }
                
        except Exception as e:
            self.log(f"Error in sentiment analysis for {symbol}: {str(e)}", level="error")
            # Return a neutral sentiment rather than raising
            return {
                'sentiment_score': 0,
                'overall_sentiment': 'Neutral',
                'momentum': {
                    'short_term': 'neutral',
                    'medium_term': 'neutral',
                    'long_term': 'neutral'
                },
                'price_changes': {
                    'short_term': 0,
                    'medium_term': 0,
                    'long_term': 0
                }
            }

    async def _simulate_buy_order(self, symbol: str) -> None:
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
            
            # Create new paper position - Pass self as trading_bot
            self.paper_positions[symbol] = Position(
                symbol=symbol,
                entry_price=current_price,
                quantity=quantity,
                entry_time=datetime.now(),
                is_paper=True,
                trading_bot=self  # Add this line
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
            
            await self.async_log(
                f"Paper buy order executed for {symbol}",
                context={
                    'price': f"${current_price:,.2f}",
                    'quantity': f"{quantity:.8f}",
                    'total_value': f"${self.trade_amount:,.2f}",
                    'fees': f"${fee:.2f}",
                    'remaining_balance': f"${self.paper_balance:,.2f}",
                    'position_count': len(self.paper_positions)
                }
            )
            
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
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                if not self.discord_channel:
                    logging.info(f"Notification (no channel): {message}")
                    return
                
                # Format based on type
                if is_update:
                    formatted_message = f"📊 Trading Update:\n```{message}```"
                else:
                    formatted_message = f"🔔 Alert:\n```{message}```"
                
                await self.discord_channel.send(formatted_message)
                logging.info(f"Notification sent: {message}")
                return
                
            except discord.errors.HTTPException as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    continue
                logging.error(f"HTTP Error sending notification: {str(e)}")
                
            except Exception as e:
                logging.error(f"Error sending notification: {str(e)}")
                self.discord_channel = None  # Reset channel if we can't send messages
                break

    async def send_trade_notification(self, action: str, symbol: str, price: float, 
                                    quantity: float, is_paper: bool = False, 
                                    profit_info: Dict[str, float] = None):
        """Enhanced trade notification with market context"""
        try:
            # Create rich embed
            color = discord.Color.green() if action == 'BUY' else \
                    discord.Color.red() if action == 'SELL' else \
                    discord.Color.blue()
            
            embed = discord.Embed(
                title=f"{'📈' if action == 'BUY' else '📉'} {action} {symbol}",
                timestamp=datetime.now(),
                color=color
            )
            
            # Add trade details
            embed.add_field(
                name="Trade Details",
                value=f"```\n"
                      f"Price: ${price:,.2f}\n"
                      f"Quantity: {quantity:.8f}\n"
                      f"Total: ${(price * quantity):,.2f}\n"
                      f"Type: {'Paper' if is_paper else 'Real'}"
                      f"```",
                inline=False
            )
            
            # Add profit info for sells
            if profit_info and action == 'SELL':
                embed.add_field(
                    name="Trade Result",
                    value=f"```\n"
                          f"Profit: ${profit_info['profit_usd']:+,.2f}\n"
                          f"Return: {profit_info['profit_percentage']:+.2f}%\n"
                          f"Fees: ${profit_info['fees_paid']:.2f}"
                          f"```",
                    inline=False
                )
            
            # Add market context
            signal = self._calculate_trade_signal(symbol)
            embed.add_field(
                name="Market Context",
                value=f"```\n"
                      f"Signal Score: {signal['score']:.2f}\n"
                      f"Trend: {signal['signals']['trend']}\n"
                      f"Volume: {signal['signals']['volume']}"
                      f"```",
                inline=False
            )
            
            if self.discord_channel:
                await self.discord_channel.send(embed=embed)
                
        except Exception as e:
            self.log(f"Error sending trade notification: {str(e)}", level="error")

    async def send_interval_update(self):
        """Enhanced periodic update with individual coin notifications"""
        try:
            if not self.watched_coins:
                await self.async_log("No coins in watchlist for update")
                return
            
            await self.async_log("Starting periodic trading update...")
            
            # Send summary header
            header = (
                "🔄 Trading Update Started\n"
                f"Mode: {'Paper' if self.paper_trading else 'Real'} Trading\n"
                f"Analyzing {len(self.watched_coins)} coins..."
            )
            await self.send_notification(header, is_update=True)
            
            for symbol in self.watched_coins:
                try:
                    await self.async_log(f"Analyzing {symbol}...")
                    
                    # Get comprehensive analysis with error handling
                    try:
                        current_price = float(self.client.get_product(f"{symbol}-USD").price)
                        signal = self._calculate_trade_signal(symbol)
                        position = self.paper_positions.get(symbol) if self.paper_trading else self.positions.get(symbol)
                        
                        # Create embed for this coin
                        embed = discord.Embed(
                            title=f"📊 {symbol} Analysis",
                            timestamp=datetime.now(),
                            color=discord.Color.green() if signal['action'] in ['BUY', 'STRONG_BUY'] else
                                  discord.Color.red() if signal['action'] in ['SELL', 'STRONG_SELL'] else
                                  discord.Color.blue()
                        )
                        
                        # Add price and signal info
                        embed.add_field(
                            name="💰 Price",
                            value=f"${current_price:,.2f}",
                            inline=True
                        )
                        embed.add_field(
                            name="📈 Signal",
                            value=signal['action'],
                            inline=True
                        )
                        embed.add_field(
                            name="📊 Score",
                            value=f"{signal['score']:.2f}",
                            inline=True
                        )
                        
                        # Add signal components
                        components = "\n".join([
                            f"• 📈 Trend:     {signal['signals']['trend']:>6.1f}",
                            f"• 🔄 Momentum:  {signal['signals']['momentum']:>6.1f}",
                            f"• 📊 Volume:    {signal['signals']['volume']:>6.1f}",
                            f"• ⚠️ Risk:      {signal['signals']['risk']:>6.1f}"
                        ])
                        embed.add_field(
                            name="📋 Signal Components",
                            value=f"```{components}```",
                            inline=False
                        )
                        
                        # Add position info if exists
                        if position:
                            profit_info = position.calculate_profit(current_price)
                            position_info = (
                                f"Entry Price: ${position.entry_price:,.2f}\n"
                                f"P/L: {profit_info['profit_percentage']:+.2f}%\n"
                                f"Max Profit: {profit_info['highest_profit_percentage']:+.2f}%\n"
                                f"Max Drawdown: {profit_info['drawdown_percentage']:+.2f}%"
                            )
                            embed.add_field(
                                name="💼 Position Status",
                                value=f"```{position_info}```",
                                inline=False
                            )
                        
                        # Send individual coin analysis
                        if self.discord_channel:
                            await self.discord_channel.send(embed=embed)
                        await self.async_log(f"Analysis sent for {symbol}")
                        
                        # Small delay between messages to prevent rate limiting
                        await asyncio.sleep(1)
                        
                    except Exception as analysis_error:
                        await self.async_log(f"Error in analysis for {symbol}: {str(analysis_error)}", level="error")
                        continue
                    
                except Exception as e:
                    await self.async_log(f"Error analyzing {symbol}: {str(e)}", level="error")
                    continue
            
            # Send summary footer
            footer = (
                f"{'='*30}\n"
                f"📈 Active Positions: {len(self.paper_positions) if self.paper_trading else len(self.positions)}\n"
                f"💰 Mode: {'Paper' if self.paper_trading else 'Real'} Trading\n"
                f"⏰ Next Update: {self.trading_interval//60} minutes"
            )
            await self.send_notification(footer, is_update=True)
            await self.async_log("Periodic update completed")
            
        except Exception as e:
            await self.async_log(f"Error sending interval update: {str(e)}", level="error")

    async def send_alert(self, symbol: str, alert_type: str, details: str):
        """Send an alert notification"""
        try:
            message = f"Alert for {symbol}\n"
            message += f"Type: {alert_type}\n"
            message += f"Details: {details}"
            await self.send_notification(message)
        except Exception as e:
            logging.error(f"Error sending alert: {str(e)}")

    async def sync_positions(self):
        """Sync local position tracking with Coinbase positions"""
        try:
            if self.paper_trading:
                return
            
            await self.async_log("Syncing positions with Coinbase...")
            
            # Get all accounts with balances
            accounts = self.client.get_accounts()
            
            for account in accounts.accounts:
                if account.currency != 'USD' and float(account.available_balance.value) > 0:
                    symbol = account.currency
                    await self.async_log(f"Found active position in {symbol}")
                    
                    try:
                        # Get current price
                        current_price = float(self.client.get_product(f"{symbol}-USD").price)
                        quantity = float(account.available_balance.value)
                        
                        # Get order history to find entry price
                        orders = self.client.get_orders(product_id=f"{symbol}-USD")
                        buy_orders = [o for o in orders if o.side == 'BUY' and o.status == 'FILLED']
                        
                        if buy_orders:
                            # Use most recent buy order as entry
                            entry_order = buy_orders[0]
                            entry_price = float(entry_order.average_filled_price)
                            entry_time = datetime.fromtimestamp(float(entry_order.created_time))
                            
                            # Create position object
                            self.positions[symbol] = Position(
                                symbol=symbol,
                                entry_price=entry_price,
                                quantity=quantity,
                                entry_time=entry_time,
                                is_paper=False
                            )
                            
                            await self.async_log(f"Restored position tracking for {symbol}:")
                            await self.async_log(f"Entry Price: ${entry_price}")
                            await self.async_log(f"Quantity: {quantity}")
                            await self.async_log(f"Current Value: ${quantity * current_price}")
                            
                    except Exception as e:
                        await self.async_log(f"Error syncing position for {symbol}: {str(e)}", level="error")
                    
        except Exception as e:
            await self.async_log(f"Error syncing positions: {str(e)}", level="error")

    async def post_init(self):
        """Perform post-initialization tasks"""
        await self.async_log("Starting bot initialization...")
        await self.async_log("Coinbase client initialized successfully")
        await self.sync_positions()

    async def async_log(self, message: str, level: str = "info", context: Dict[str, Any] = None, error: Exception = None) -> None:
        try:
            if hasattr(self, 'logs_channel') and self.logs_channel:
                embed = discord.Embed(
                    timestamp=datetime.now(),
                    color=discord.Color.red() if level == "error" else
                          discord.Color.yellow() if level == "warning" else
                          discord.Color.blue()
                )
                
                if level == "error" and error:
                    embed.title = f"🔴 ERROR: {error.__class__.__name__}"
                    # Add error details
                    embed.add_field(
                        name="Error Details",
                        value=f"```python\n{str(error)}\n```",
                        inline=False
                    )
                    # Add stack trace if available
                    if hasattr(error, '__traceback__'):
                        import traceback
                        trace = ''.join(traceback.format_tb(error.__traceback__))
                        if len(trace) > 1000:
                            trace = trace[:997] + "..."
                        embed.add_field(
                            name="Stack Trace",
                            value=f"```python\n{trace}\n```",
                            inline=False
                        )
                else:
                    embed.title = "⚠️ WARNING" if level == "warning" else "ℹ️ INFO"
                
                # Add main message with better formatting
                if len(message) > 1024:
                    parts = [message[i:i+1024] for i in range(0, len(message), 1024)]
                    for i, part in enumerate(parts):
                        embed.add_field(
                            name=f"Details (Part {i+1})" if i > 0 else "Details",
                            value=f"```yaml\n{part}\n```",
                            inline=False
                        )
                else:
                    embed.description = f"```yaml\n{message}\n```"
                
                # Add context with better organization
                if context:
                    # Group context items by category
                    categories = {
                        'Market Data': ['price', 'volume', 'trend'],
                        'Analysis': ['rsi', 'macd', 'signal'],
                        'Trading': ['position', 'balance', 'mode'],
                        'System': ['timestamp', 'iteration', 'status']
                    }
                    
                    for category, fields in categories.items():
                        category_items = {k: v for k, v in context.items() if any(f in k.lower() for f in fields)}
                        if category_items:
                            embed.add_field(
                                name=category,
                                value=f"```yaml\n" + "\n".join(f"{k}: {v}" for k, v in category_items.items()) + "\n```",
                                inline=True
                            )
                
                await self.logs_channel.send(embed=embed)
                
        except Exception as e:
            logging.error(f"Failed to send log: {str(e)}")

    def set_logs_channel(self, channel):
        """Set the Discord channel for logs"""
        self.logs_channel = channel
        logging.info(f"Discord logs channel set")

    def log(self, message: str, level: str = "info", context: Dict[str, Any] = None) -> None:
        """Synchronous logging to file and console"""
        # Log the main message
        if level == "error":
            logging.error(message)
        elif level == "warning":
            logging.warning(message)
        else:
            logging.info(message)
        
        # Log context if provided
        if context:
            for key, value in context.items():
                if level == "error":
                    logging.error(f"{key}: {value}")
                elif level == "warning":
                    logging.warning(f"{key}: {value}")
                else:
                    logging.info(f"{key}: {value}")

    def _calculate_macd(self, symbol: str) -> Dict[str, float]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        try:
            # Get historical prices
            end = datetime.now()
            start = end - timedelta(days=60)  # Need more data for accurate MACD
            prices = self._get_historical_prices(symbol, start, end)
            
            # Calculate EMAs
            ema_12 = prices.ewm(span=12, adjust=False).mean()
            ema_26 = prices.ewm(span=26, adjust=False).mean()
            
            # Calculate MACD line and signal line
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            
            # Calculate histogram
            histogram = macd_line - signal_line
            
            return {
                'line': float(macd_line.iloc[-1]),
                'signal': float(signal_line.iloc[-1]),
                'histogram': float(histogram.iloc[-1]),
                'prev_histogram': float(histogram.iloc[-2]),
                'trend': 'bullish' if histogram.iloc[-1] > histogram.iloc[-2] else 'bearish'
            }
        except Exception as e:
            self.log(f"Error calculating MACD for {symbol}: {str(e)}", level="error")
            raise

    def _calculate_stochastic(self, symbol: str) -> Dict[str, float]:
        """Calculate Stochastic Oscillator"""
        try:
            # Get historical prices
            end = datetime.now()
            start = end - timedelta(days=30)
            prices = self._get_historical_prices(symbol, start, end)
            
            # Calculate 14-period high and low
            period = 14
            low_14 = prices.rolling(window=period).min()
            high_14 = prices.rolling(window=period).max()
            
            # Calculate %K
            k_percent = 100 * ((prices - low_14) / (high_14 - low_14))
            
            # Calculate %D (3-period SMA of %K)
            d_percent = k_percent.rolling(window=3).mean()
            
            return {
                'k': float(k_percent.iloc[-1]),
                'd': float(d_percent.iloc[-1]),
                'trend': 'oversold' if k_percent.iloc[-1] < 20 else 
                        'overbought' if k_percent.iloc[-1] > 80 else 
                        'neutral'
            }
        except Exception as e:
            self.log(f"Error calculating Stochastic for {symbol}: {str(e)}", level="error")
            raise

    def _calculate_bollinger_bands(self, symbol: str) -> Dict[str, float]:
        """Calculate Bollinger Bands"""
        try:
            # Get historical prices
            end = datetime.now()
            start = end - timedelta(days=30)
            prices = self._get_historical_prices(symbol, start, end)
            
            # Calculate 20-period SMA and standard deviation
            period = 20
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            
            # Calculate bands
            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)
            
            # Calculate bandwidth and %B
            bandwidth = (upper_band - lower_band) / sma * 100
            percent_b = (prices - lower_band) / (upper_band - lower_band)
            
            return {
                'upper': float(upper_band.iloc[-1]),
                'middle': float(sma.iloc[-1]),
                'lower': float(lower_band.iloc[-1]),
                'bandwidth': float(bandwidth.iloc[-1]),
                'percent_b': float(percent_b.iloc[-1]),
                'is_squeeze': bandwidth.iloc[-1] < bandwidth.rolling(window=20).mean().iloc[-1]
            }
        except Exception as e:
            self.log(f"Error calculating Bollinger Bands for {symbol}: {str(e)}", level="error")
            raise

    def _find_support_resistance(self, symbol: str) -> Dict[str, float]:
        """Find support and resistance levels using price action"""
        try:
            # Get historical prices
            end = datetime.now()
            start = end - timedelta(days=90)  # 90 days for reliable levels
            prices = self._get_historical_prices(symbol, start, end)
            
            # Function to identify pivot points
            def find_pivots(data: pd.Series, window: int = 5) -> tuple:
                highs = []
                lows = []
                for i in range(window, len(data) - window):
                    if all(data[i] > data[i-j] for j in range(1, window+1)) and \
                       all(data[i] > data[i+j] for j in range(1, window+1)):
                        highs.append(data[i])
                    if all(data[i] < data[i-j] for j in range(1, window+1)) and \
                       all(data[i] < data[i+j] for j in range(1, window+1)):
                        lows.append(data[i])
                return highs, lows

            # Find pivot points
            pivot_highs, pivot_lows = find_pivots(prices)
            current_price = float(self.client.get_product(f"{symbol}-USD").price)
            
            # Find nearest support and resistance
            supports = [p for p in pivot_lows if p < current_price]
            resistances = [p for p in pivot_highs if p > current_price]
            
            nearest_support = max(supports) if supports else current_price * 0.85
            nearest_resistance = min(resistances) if resistances else current_price * 1.15
            
            return {
                'nearest_support': nearest_support,
                'nearest_resistance': nearest_resistance,
                'all_supports': sorted(supports, reverse=True)[:3],  # Top 3 support levels
                'all_resistances': sorted(resistances)[:3],  # Top 3 resistance levels
                'support_strength': len([p for p in pivot_lows if abs(p - nearest_support) / nearest_support < 0.02])
            }
        except Exception as e:
            self.log(f"Error finding support/resistance for {symbol}: {str(e)}", level="error")
            raise

    def _calculate_btc_correlation(self, symbol: str) -> float:
        """Calculate correlation with BTC"""
        try:
            # Get historical prices for both assets
            end = datetime.now()
            start = end - timedelta(days=30)
            
            symbol_prices = self._get_historical_prices(symbol, start, end)
            btc_prices = self._get_historical_prices('BTC', start, end)
            
            # Calculate returns
            symbol_returns = symbol_prices.pct_change().dropna()
            btc_returns = btc_prices.pct_change().dropna()
            
            # Calculate correlation
            correlation = symbol_returns.corr(btc_returns)
            
            return float(correlation)
        except Exception as e:
            self.log(f"Error calculating BTC correlation for {symbol}: {str(e)}", level="error")
            return 0.0  # Default to no correlation on error

    def _calculate_signal_components(self, ma_data: Dict[str, Any], volume_data: Dict[str, Any], 
                                   sentiment: Dict[str, Any], market_conditions: Dict[str, Any]) -> Dict[str, float]:
        """Calculate individual signal components for trading decision"""
        try:
            # Get current price
            current_price = ma_data.get('current_price', 0)
            
            # Initialize signals dictionary
            signals = {
                'trend': 0,
                'momentum': 0, 
                'volume': 0,
                'risk': 0
            }
            
            # Calculate trend signal (-30 to +30)
            if ma_data['trend'] == 'Strong Uptrend':
                signals['trend'] = 30
            elif ma_data['trend'] == 'Weak Uptrend':
                signals['trend'] = 15
            elif ma_data['trend'] == 'Strong Downtrend':
                signals['trend'] = -30
            elif ma_data['trend'] == 'Weak Downtrend':
                signals['trend'] = -15
                
            # Calculate momentum signal (-25 to +25)
            momentum_score = 0
            if sentiment['momentum']['short_term'] == 'bullish':
                momentum_score += 10
            elif sentiment['momentum']['short_term'] == 'bearish':
                momentum_score -= 10
                
            if sentiment['momentum']['medium_term'] == 'bullish':
                momentum_score += 8
            elif sentiment['momentum']['medium_term'] == 'bearish':
                momentum_score -= 8
                
            if sentiment['momentum']['long_term'] == 'bullish':
                momentum_score += 7
            elif sentiment['momentum']['long_term'] == 'bearish':
                momentum_score -= 7
                
            signals['momentum'] = min(max(momentum_score, -25), 25)
            
            # Calculate volume signal (-20 to +20)
            volume_score = 0
            if volume_data['volume_ratio'] > 1.5:
                volume_score = 20
            elif volume_data['volume_ratio'] > 1.2:
                volume_score = 15
            elif volume_data['volume_ratio'] > 1.0:
                volume_score = 10
            elif volume_data['volume_ratio'] < 0.5:
                volume_score = -20
            elif volume_data['volume_ratio'] < 0.8:
                volume_score = -15
            
            if volume_data['confirms_trend']:
                volume_score *= 1.2
            else:
                volume_score *= 0.8
                
            signals['volume'] = min(max(volume_score, -20), 20)
            
            # Calculate risk signal (-10 to +10)
            risk_score = 0
            if market_conditions['is_volatile']:
                risk_score -= 5
            if market_conditions['is_high_activity']:
                risk_score -= 3
            if market_conditions['market_aligned']:
                risk_score += 5
            if market_conditions['suitable_for_trading']:
                risk_score += 3
                
            signals['risk'] = min(max(risk_score, -10), 10)
            
            # Calculate total score (-85 to +85)
            total_score = sum(signals.values())
            
            # Log the calculation results
            self.log(f"Signal components calculated: {signals}")
            
            # Define thresholds
            buy_threshold = 35
            sell_threshold = -35
            
            return {
                'symbol': ma_data['symbol'],
                'score': total_score,
                'signals': signals,
                'action': 'STRONG_BUY' if total_score >= 70 else
                         'BUY' if total_score >= buy_threshold else
                         'STRONG_SELL' if total_score <= -70 else
                         'SELL' if total_score <= sell_threshold else
                         'HOLD',
                'timestamp': datetime.now(),
                'price': current_price
            }
            
        except Exception as e:
            self.log(f"Error calculating signal components: {str(e)}", level="error")
            raise

    def _format_signal_response(self, symbol: str, signals: Dict[str, float], current_price: float) -> Dict[str, Any]:
        """Format the final trading signal response"""
        total_score = sum(signals.values())
        
        # Dynamic thresholds based on market conditions
        buy_threshold = 35
        sell_threshold = -35
        
        return {
            'symbol': symbol,
            'score': total_score,
            'signals': signals,
            'action': 'STRONG_BUY' if total_score >= 70 else
                     'BUY' if total_score >= buy_threshold else
                     'STRONG_SELL' if total_score <= -70 else
                     'SELL' if total_score <= sell_threshold else
                     'HOLD',
            'timestamp': datetime.now(),
            'price': current_price
        }