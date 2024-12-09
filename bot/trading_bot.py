import os
import time
import threading
import logging
import pandas as pd
from datetime import datetime, timedelta
import json
from coinbase.rest import RESTClient
from typing import Dict, Any, List, Union, Optional, Literal, Callable
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

class TradingError(Exception):
    """Base trading error with type classification"""
    TYPES = {
        'API': 'API call failed',
        'DATA': 'Data processing error',
        'SIGNAL': 'Signal calculation error',
        'TRADE': 'Trade execution error',
        'VALIDATION': 'Validation error'
    }
    
    def __init__(self, message: str, error_type: str = None):
        self.error_type = error_type
        self.timestamp = datetime.now()
        super().__init__(f"[{error_type}] {message}" if error_type else message)

class PriceManager:
    """Centralized price data management"""
    def __init__(self, client: RESTClient, cache_size: int, cache_ttl: int, rate_limit: float, 
                 log_callback: Optional[Callable] = None):
        self.client = client
        self.log = log_callback or (lambda msg, **kwargs: None)
        self._cache = {}
        self._cache_lock = asyncio.Lock()
        self._cache_size = cache_size
        self._cache_ttl = cache_ttl
        self._rate_limit = rate_limit
        self._last_api_call = 0

    async def _get_cached_price(self, symbol: str, days: int = 1) -> pd.Series:
        """Get price from cache or fetch from API"""
        current_time = time.time()
        cache_key = f"{symbol}_{days}"
        
        # Check cache
        if cache_key in self._cache:
            cached_data, cache_time = self._cache[cache_key]
            if current_time - cache_time < self._cache_ttl:
                return cached_data
                
        # Rate limiting
        if current_time - self._last_api_call < self._rate_limit:
            await asyncio.sleep(self._rate_limit - (current_time - self._last_api_call))
            
        try:
            # Get candles synchronously
            end = datetime.now()
            start = end - timedelta(days=days)
            
            response = self.client.get_candles(
                product_id=f"{symbol}-USD",
                start=int(start.timestamp()),
                end=int(end.timestamp()),
                granularity="ONE_DAY"
            )
            
            # Convert to pandas Series
            prices = pd.Series(
                [float(candle.close) for candle in reversed(response.candles)],
                index=[datetime.fromtimestamp(float(candle.start)) for candle in reversed(response.candles)]
            )
            
            # Update cache
            self._cache[cache_key] = (prices, current_time)
            self._last_api_call = current_time
            
            # Clean cache if needed
            if len(self._cache) > self._cache_size:
                oldest_key = min(self._cache.items(), key=lambda x: x[1][1])[0]
                del self._cache[oldest_key]
            
            return prices
            
        except Exception as e:
            self.log(f"Failed to fetch price data: {str(e)}", level="error")
            raise TradingError(f"Failed to fetch price data: {str(e)}", error_type='API')

    async def get_price(self, symbol: str, days: int = 1) -> pd.Series:
        """Thread-safe price retrieval with caching"""
        async with self._cache_lock:
            return await self._get_cached_price(symbol, days)

    async def close(self):
        """Cleanup resources"""
        async with self._cache_lock:
            self._cache.clear()
            self._last_api_call = 0

    async def get_current_price(self, symbol: str) -> float:
        """Get latest price for symbol"""
        prices = await self.get_price(symbol, days=1)
        return float(prices.iloc[-1])

    async def get_historical_prices(self, symbol: str, start: datetime, end: datetime) -> pd.Series:
        """Get historical prices for a date range"""
        days = (end - start).days + 1
        prices = await self.get_price(symbol, days=days)
        return prices[(prices.index >= start) & (prices.index <= end)]

    async def get_cached_price_data(self, symbol: str, days: int = 30) -> pd.Series:
        """
        Get cached price data asynchronously.
        
        Args:
            symbol: The cryptocurrency symbol
            days: Number of days of historical data
            
        Returns:
            pd.Series: Price data series
            
        Raises:
            TradingError: If data cannot be fetched
        """
        try:
            return await self.get_price(symbol, days=days)
        except Exception as e:
            self.log(f"Error getting cached price data: {str(e)}", level="error")
            raise TradingError(f"Failed to get cached price data: {str(e)}", error_type='DATA')

class TradingConfig:
    def __init__(self) -> None:
        self.load_from_env()
    
    def load_from_env(self) -> None:
        """Load configuration from environment variables"""
        self.interval = int(os.getenv('TRADING_INTERVAL', '300'))
        self.trade_amount = float(os.getenv('TRADE_AMOUNT', '100.0'))
        self.max_position_size = float(os.getenv('MAX_POSITION_SIZE', '1000.0'))
        self.trailing_stop = {
            'enabled': bool(os.getenv('TRAILING_STOP_ENABLED', 'True')),
            'percentage': float(os.getenv('TRAILING_STOP_PERCENTAGE', '5.0')),
            'activation': float(os.getenv('TRAILING_STOP_ACTIVATION', '3.0'))
        }
        # ... other config items ...

    def update(self, **kwargs):
        """Update config values with validation"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logging.info(f"Updated config {key}={value}")

class TradingBot:
    FEE_RATE = 0.006  # 0.6% Coinbase fee
    CONFIG_FILE = 'bot_config.json'  # Single source of truth
    
    def __init__(self):
        try:
            logging.info("Starting bot initialization...")
            
            # 1. First define basic attributes
            self.discord_channel = None
            self.logs_channel = None
            self._paper_trading_active = False
            self._trading_active = False
            self.positions = {}
            self.paper_positions = {}
            self.watched_coins = set()
            self.trade_history = []
            self.paper_trade_history = []
            self.paper_balance = 1000.0  # Default paper balance
            
            # Technical analysis parameters
            self.rsi_period = 14
            self.rsi_overbought = 70
            self.rsi_oversold = 30
            self.trading_interval = 300  # 5 minutes
            self.stop_loss_percentage = float(os.getenv('STOP_LOSS_PERCENTAGE', 5.0))
            self.take_profit_percentage = float(os.getenv('TAKE_PROFIT_PERCENTAGE', 10.0))
            self.max_position_size = float(os.getenv('MAX_POSITION_SIZE', 1000.0))
            self.partial_tp_percentage = float(os.getenv('PARTIAL_TP_PERCENTAGE', 7.0))
            self.partial_tp_size = float(os.getenv('PARTIAL_TP_SIZE', 0.5))
            self.trailing_stop_percentage = 2.0
            self.trailing_stop_enabled = True
            self.trailing_stop_activation = 3.0
            
            # 2. Define the log method before using it
            self._log_message_sync = lambda msg, level="info", **kwargs: logging.log(
                getattr(logging, level.upper()), msg
            )
            
            # 3. Check environment variables
            if 'COINBASE_API_KEY' not in os.environ:
                raise Exception("COINBASE_API_KEY not found in environment variables")
            if 'COINBASE_API_SECRET' not in os.environ:
                raise Exception("COINBASE_API_SECRET not found in environment variables")
            
            # 4. Create REST client
            self.client = RESTClient(
                api_key=os.environ['COINBASE_API_KEY'].strip(),
                api_secret=os.environ['COINBASE_API_SECRET'].strip()
            )
            
            # 5. Initialize PriceManager with synchronous logging
            self.price_manager = PriceManager(
                client=self.client,
                cache_size=100,
                cache_ttl=300,
                rate_limit=0.1,
                log_callback=self._log_message_sync  # Use sync logging during init
            )
            
            # 6. Now define the async log method for future use
            self.log = self._log_message
            
            # Add signal generator
            self.signal_generator = {
                'generate_signal': self._calculate_trade_signal
            }
            
            logging.info("Bot initialization completed successfully")
            
        except Exception as e:
            logging.error(f"Bot initialization failed: {str(e)}")
            raise Exception(f"Bot initialization failed: {str(e)}")

    async def post_init(self) -> None:
        """
        Perform post-initialization tasks after Discord connection is established.
        This includes loading saved configuration and initializing trading components.
        """
        try:
            logging.info("Starting post-initialization...")
            
            # Load saved configuration
            self.load_config()
            
            # Initialize watched coins set if empty
            if not hasattr(self, 'watched_coins'):
                self.watched_coins = set()
            
            # Add default coins to watch
            default_coins = {'BTC', 'ETH'}
            for coin in default_coins:
                if await self.add_coin(coin):
                    logging.info(f"Added default coin {coin} to watchlist")
            
            # Ensure all positions are in watchlist
            self._ensure_positions_watched()
            
            # Test API connection
            try:
                await self.test_api_connection()
                logging.info("API connection test successful")
            except Exception as e:
                logging.error(f"API connection test failed: {str(e)}")
            
            # Initialize price cache for watched coins
            for symbol in self.watched_coins:
                try:
                    await self.price_manager.get_cached_price_data(symbol)
                    logging.info(f"Initialized price cache for {symbol}")
                except Exception as e:
                    logging.error(f"Failed to initialize price cache for {symbol}: {str(e)}")
            
            # Start heartbeat task
            asyncio.create_task(self._heartbeat())
            
            logging.info("Post-initialization completed successfully")
            
        except Exception as e:
            error_msg = f"Post-initialization failed: {str(e)}"
            logging.error(error_msg)
            if hasattr(self, 'logs_channel'):
                await self.send_notification(error_msg)
            raise

    async def start_trading(self, mode: str = None) -> str:
        """
        Start trading in either paper or real mode.
        
        Args:
            mode: Either 'paper' or 'real'. If None, returns usage instructions.
            
        Returns:
            str: Status message about trading start
        """
        try:
            if not mode:
                return "Please specify trading mode: !start paper or !start real"
            
            mode = mode.lower()
            if mode not in ['paper', 'real']:
                return "Invalid mode. Use: !start paper or !start real"
            
            # Check if any trading is already active
            if self.trading_active or self.paper_trading_active:
                current_mode = "paper" if self.paper_trading_active else "real"
                return f"{current_mode.capitalize()} trading is already active. Stop it first with !stop"
            
            # Start requested mode
            if mode == 'paper':
                self._paper_trading_active = True
                self.paper_balance = 1000.0  # Reset to default
                await self.log("Paper trading started")
            else:  # real
                self._trading_active = True
                await self.log("Real trading started")
            
            # Start trading loop
            asyncio.create_task(self._trading_loop())
            
            return f"{mode.capitalize()} trading started successfully"
            
        except Exception as e:
            await self.log(f"Error starting {mode} trading: {str(e)}", level="error")
            self._paper_trading_active = False
            self._trading_active = False
            return f"Failed to start {mode} trading: {str(e)}"

    async def stop_trading(self) -> str:
        """Stop any active trading."""
        try:
            was_active = self.trading_active or self.paper_trading_active
            mode = "paper" if self.paper_trading_active else "real" if self.trading_active else None
            
            self._trading_active = False
            self._paper_trading_active = False
            
            if was_active:
                await self.log(f"{mode.capitalize()} trading stopped")
                return f"{mode.capitalize()} trading stopped successfully"
            return "No active trading to stop"
            
        except Exception as e:
            await self.log(f"Error stopping trading: {str(e)}", level="error")
            return f"Error stopping trading: {str(e)}"

    async def _trading_loop(self):
        """Main trading loop using position and signal managers"""
        while self.trading_active or self.paper_trading_active:
            try:
                for symbol in self.watched_coins:
                    try:
                        await self.log(f"Analyzing {symbol}...", level="info")
                        
                        # Generate signal with detailed logging
                        signal = await self._calculate_trade_signal(symbol)
                        await self.log(f"Signal for {symbol}:", context={
                            'action': signal['action'],
                            'score': signal['score'],
                            'signals': signal['signals']
                        })
                        
                        if signal['action'] != 'HOLD':
                            # Execute trade with better error handling
                            if self.paper_trading_active:
                                try:
                                    success = await self._simulate_trade(symbol, signal['action'], f"Signal: {signal['score']:.2f}")
                                    if success:
                                        await self.log(f"Paper trade executed for {symbol}: {signal['action']}", level="info")
                                    else:
                                        await self.log(f"Paper trade skipped for {symbol}: conditions not met", level="info")
                                except Exception as trade_error:
                                    await self.log(f"Paper trade failed for {symbol}: {str(trade_error)}", level="error")
                                    continue
                            else:
                                # Similar handling for real trades...
                                pass
                        else:
                            await self.log(f"No action needed for {symbol} (Score: {signal['score']:.2f})", level="info")
                        
                        # Update position metrics with error handling
                        position = self.paper_positions.get(symbol) if self.paper_trading_active else self.positions.get(symbol)
                        if position:
                            try:
                                current_price = float(self.client.get_product(f"{symbol}-USD").price)
                                await position.update_metrics(current_price)
                            except Exception as e:
                                await self.log(f"Error updating metrics for {symbol}: {str(e)}", level="error")
                
                    except Exception as coin_error:
                        await self.log(f"Error processing {symbol}: {str(coin_error)}", level="error")
                        continue
                
                # Sleep for configured interval
                await asyncio.sleep(self.trading_interval)
                
            except Exception as e:
                await self.log(f"Trading loop error: {str(e)}", level="error")
                await asyncio.sleep(10)  # Sleep on error to prevent rapid retries
    
    async def _check_and_trade(self, symbol):
        try:
            if await self._should_trade(symbol, 'BUY'):
                await self._execute_trade(symbol, 'BUY')
            elif await self._should_trade(symbol, 'SELL'):
                await self._execute_trade(symbol, 'SELL')
            
        except Exception as e:
            await self.log(f"Error checking and trading {symbol}: {str(e)}", level="error")
            raise
    
    async def calculate_rsi(self, symbol: str) -> float:
        try:
            # Use PriceManager instead of direct API calls
            prices = await self.price_manager.get_price(symbol, days=30)
            
            # Calculate RSI
            delta = prices.diff()
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            
            avg_gain = gains.ewm(span=self.rsi_period, adjust=False).mean()
            avg_loss = losses.ewm(span=self.rsi_period, adjust=False).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi.iloc[-1])
        except Exception as e:
            await self.log(f"Error calculating RSI for {symbol}: {str(e)}", level="error")
            raise

    async def _place_buy_order(self, symbol: str) -> None:
        """Execute a buy order with proper async handling"""
        try:
            if not self._validate_real_trade(symbol, 'BUY', self.trade_amount):
                raise TradingError(f"Trade validation failed for {symbol}")
            
            product_id = f"{symbol}-USD"
            # Synchronous price fetch is fine here
            current_price = float(self.client.get_product(product_id).price)
            
            # Place the order synchronously (Coinbase client method is sync)
            order = self.client.create_order(
                product_id=product_id,
                side='BUY',
                order_configuration={
                    'market_market_ioc': {
                        'quote_size': str(self.trade_amount)
                    }
                }
            )
            
            # Get actual filled data
            filled_quantity = float(order.filled_size)
            filled_price = float(order.average_filled_price)
            
            # Create position using actual filled data
            self.positions[symbol] = Position(
                trading_bot=self,
                symbol=symbol,
                entry_price=filled_price,
                quantity=filled_quantity,
                entry_time=datetime.now(),
                is_paper=False
            )
            
            # Record trade
            self.trade_history.append({
                'timestamp': datetime.now(),
                'action': 'BUY',
                'symbol': symbol,
                'amount_usd': self.trade_amount,
                'price': filled_price,
                'quantity': filled_quantity,
                'order_id': order.order_id
            })
            
            await self.log(f"Buy order filled for {symbol}: {filled_quantity} @ ${filled_price}")
            
        except Exception as e:
            await self.log(f"Error placing buy order for {symbol}: {str(e)}", level="error")
            raise
            
    async def _place_sell_order(self, symbol: str, partial: bool = False) -> None:
        """Execute a sell order with proper async handling"""
        try:
            position = self.positions.get(symbol)
            if not position:
                await self.log(f"No position found for {symbol}, cannot sell", level="warning")
                return
                
            if not self._validate_real_trade(symbol, 'SELL', position.quantity):
                raise TradingError(f"Trade validation failed for {symbol}")
            
            product_id = f"{symbol}-USD"
            current_price = float(self.client.get_product(product_id).price)
            
            # Calculate sell quantity
            sell_quantity = position.quantity * (self.partial_tp_size if partial else 1.0)
            
            # Place the order synchronously
            order = self.client.create_order(
                product_id=product_id,
                side='SELL',
                order_configuration={
                    'market_market_ioc': {
                        'base_size': str(sell_quantity)
                    }
                }
            )
            
            # Process order results
            filled_price = float(order.average_filled_price)
            filled_quantity = float(order.filled_size)
            profit_info = position.calculate_profit(filled_price)
            
            # Update position
            if not partial:
                await self._record_closed_position(position, filled_price, profit_info)
                del self.positions[symbol]
            else:
                position.quantity -= filled_quantity
                position.partial_exit_taken = True
            
            # Record trade
            await self._record_trade(symbol, 'SELL', filled_quantity, filled_price, profit_info, partial)
            
            await self.log(
                f"{'Partial' if partial else 'Full'} sell order filled for {symbol}: {filled_quantity} @ ${filled_price}"
            )
            
        except Exception as e:
            await self.log(f"Error placing sell order for {symbol}: {str(e)}", level="error")
            raise
            
    def get_trade_history(self):
        return self.trade_history
        
    async def add_coin(self, symbol: str) -> bool:
        """
        Add a coin to watchlist with validation.
        
        Args:
            symbol: The cryptocurrency symbol to add (e.g., 'BTC')
            
        Returns:
            bool: True if added successfully, False otherwise
        """
        try:
            symbol = symbol.upper()
            
            # Check if already in watchlist
            if symbol in self.watched_coins:
                await self.log(f"{symbol} already in watchlist", level="warning")
                return True
            
            # Validate symbol with Coinbase API
            try:
                product = self.client.get_product(f"{symbol}-USD")
                if not product:
                    raise ValueError(f"Invalid symbol: {symbol}")
            except Exception as e:
                await self.log(f"Failed to validate {symbol}: {str(e)}", level="error")
                return False
            
            # Initialize price cache for new symbol
            try:
                await self.price_manager.get_cached_price_data(symbol)
            except Exception as e:
                await self.log(f"Failed to initialize price cache for {symbol}: {str(e)}", level="error")
                return False
            
            # Add to watchlist
            self.watched_coins.add(symbol)
            await self.log(f"Added {symbol} to watchlist")
            
            # Save updated configuration
            self.save_config()
            
            return True
            
        except Exception as e:
            await self.log(f"Error adding {symbol}: {str(e)}", level="error")
            return False

    async def remove_coin(self, symbol: str) -> bool:
        """
        Remove a coin from watchlist if not in any positions.
        
        Args:
            symbol: The cryptocurrency symbol to remove
            
        Returns:
            bool: True if removed successfully, False otherwise
        """
        try:
            symbol = symbol.upper()
            
            # Check paper positions
            if hasattr(self, 'paper_positions') and symbol in self.paper_positions:
                await self.log(f"Cannot remove {symbol} - active paper position exists", level="warning")
                return False
                
            # Check real positions
            if hasattr(self, 'positions') and symbol in self.positions:
                await self.log(f"Cannot remove {symbol} - active position exists", level="warning")
                return False
            
            # Remove from watchlist if exists
            if symbol in self.watched_coins:
                self.watched_coins.remove(symbol)
                await self.log(f"Removed {symbol} from watchlist")
                return True
                
            await self.log(f"{symbol} not in watchlist", level="warning")
            return False
            
        except Exception as e:
            await self.log(f"Error removing {symbol}: {str(e)}", level="error")
            return False

    def _ensure_positions_watched(self):
        """Ensure all positions are in watchlist"""
        for symbol in self.positions.keys():
            self.watched_coins.add(symbol)
        for symbol in self.paper_positions.keys():
            self.watched_coins.add(symbol)

    async def _check_balance(self, symbol: str, action: str = 'BUY') -> bool:
        """Check if sufficient balance exists for trade."""
        try:
            if self.paper_trading_active:
                if action == 'BUY':
                    return self.paper_balance > 0
                else:
                    return symbol in self.paper_positions
                    
            # Real trading balance check
            accounts = self.client.get_accounts()
            
            if action == 'BUY':
                usd_account = next((acc for acc in accounts.data if acc.currency == 'USD'), None)
                return bool(usd_account and float(usd_account.available_balance.value) > 0)
            else:
                crypto_account = next((acc for acc in accounts.data if acc.currency == symbol), None)
                return bool(crypto_account and float(crypto_account.available_balance.value) > 0)
                
        except Exception as e:
            await self.log(f"Error checking balance: {str(e)}", level="error")
            return False

    def save_config(self) -> None:
        """Save current configuration to file with error handling"""
        try:
            config = {
                'watched_coins': list(self.watched_coins),
                'trading_interval': self.trading_interval,
                'rsi_period': self.rsi_period,
                'rsi_overbought': self.rsi_overbought,
                'rsi_oversold': self.rsi_oversold,
                'trade_amount': self.trade_amount,
                'stop_loss_percentage': self.stop_loss_percentage,
                'take_profit_percentage': self.take_profit_percentage,
                'max_position_size': self.max_position_size,
                'partial_tp_percentage': self.partial_tp_percentage,
                'partial_tp_size': self.partial_tp_size,
                'trailing_stop_percentage': self.trailing_stop_percentage,
                'trailing_stop_enabled': self.trailing_stop_enabled,
                'trailing_stop_activation': self.trailing_stop_activation
            }
            
            with open(self.CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=4)
            
            logging.info("Configuration saved successfully")
            
        except Exception as e:
            logging.error(f"Error saving configuration: {str(e)}")

    def load_config(self) -> None:
        """Load configuration from file or use defaults"""
        try:
            if not os.path.exists(self.CONFIG_FILE):
                logging.warning("No config file found, using defaults")
                return
            
            with open(self.CONFIG_FILE, 'r') as f:
                config = json.load(f)
            
            # Load watched coins
            self.watched_coins = set(config.get('watched_coins', []))
            
            # Load trading parameters
            self.trading_interval = config.get('trading_interval', 300)
            self.trade_amount = config.get('trade_amount', 100.0)
            self.max_position_size = config.get('max_position_size', 1000.0)
            
            # Load technical analysis parameters
            self.rsi_period = config.get('rsi_period', 14)
            self.rsi_overbought = config.get('rsi_overbought', 70)
            self.rsi_oversold = config.get('rsi_oversold', 30)
            
            # Load risk management parameters
            self.stop_loss_percentage = config.get('stop_loss_percentage', 5.0)
            self.take_profit_percentage = config.get('take_profit_percentage', 10.0)
            self.partial_tp_percentage = config.get('partial_tp_percentage', 7.0)
            self.partial_tp_size = config.get('partial_tp_size', 0.5)
            
            # Load trailing stop parameters
            self.trailing_stop_percentage = config.get('trailing_stop_percentage', 2.0)
            self.trailing_stop_enabled = config.get('trailing_stop_enabled', True)
            self.trailing_stop_activation = config.get('trailing_stop_activation', 3.0)
            
            logging.info("Configuration loaded successfully")
            
        except Exception as e:
            logging.error(f"Error loading configuration: {str(e)}")
            logging.warning("Using default configuration")

    async def test_api_connection(self):
        """Test API connection by fetching BTC price"""
        try:
            # Get product synchronously since client methods are not async
            product = self.client.get_product('BTC-USD')
            price = float(product.price)
            logging.info(f"Successfully fetched BTC price: ${price}")
            return price
        except Exception as e:
            logging.error(f"API test failed: {str(e)}")
            raise

    async def get_account_balance(self) -> Dict[str, Union[Dict[str, float], float]]:
        """Get account balances"""
        try:
            # Get accounts synchronously
            accounts_response = self.client.get_accounts()
            balances = {}
            total_usd_value = 0.0
            
            if hasattr(accounts_response, 'accounts'):
                for account in accounts_response.accounts:
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
                                # Get price synchronously
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

    async def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        try:
            # Get product synchronously
            product = self.client.get_product(f"{symbol}-USD")
            return float(product.price)
        except Exception as e:
            self.log(f"Error getting price for {symbol}: {str(e)}", level="error")
            raise

    async def analyze_volume(self, symbol: str) -> Dict[str, Any]:
        """Analyzes trading volume to confirm price trends"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=90)
            
            response = await self.client.get_candles(
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

    async def _calculate_position_size(self, symbol: str) -> float:
        """Calculate position size based on available funds and risk."""
        try:
            # Get available funds
            if self.paper_trading_active:
                available_funds = self.paper_balance
            else:
                account = await self.get_account_balance()
                available_funds = float(account['balances'].get('USD', {}).get('balance', 0))

            if available_funds <= 0:
                return 0

            # Get current price
            current_price = await self.price_manager.get_current_price(symbol)
            
            # Calculate risk score (0-1)
            signal = await self._calculate_trade_signal(symbol)
            risk_score = min(max((signal['score'] / 100), 0), 1)  # Normalize to 0-1
            
            # Base position size on risk score (1-10% of available funds)
            position_percentage = 0.01 + (risk_score * 0.09)  # 1-10% based on risk
            position_size = available_funds * position_percentage
            
            # Ensure minimum order size (0.1% of available funds)
            min_order = available_funds * 0.001
            if position_size < min_order:
                return 0
            
            # Calculate quantity
            quantity = position_size / current_price
            
            # Log calculation
            await self.log(
                f"Position size calculation for {symbol}:",
                context={
                    'available_funds': available_funds,
                    'risk_score': risk_score,
                    'position_percentage': position_percentage,
                    'position_size': position_size,
                    'quantity': quantity
                }
            )
            
            return quantity
            
        except Exception as e:
            await self.log(f"Error calculating position size: {str(e)}", level="error")
            return 0

    async def check_market_conditions(self, symbol: str) -> Dict[str, Any]:
        """Check comprehensive market conditions"""
        try:
            # Get price data
            prices = await self.price_manager.get_cached_price_data(symbol, days=7)
            
            # Calculate volatility
            price_range = ((prices.max() - prices.min()) / prices.min()) * 100
            is_volatile = price_range > 10  # Define volatility threshold
            
            await self.log(f"Market conditions for {symbol}:", context={
                'price_range': f"{price_range:.1f}%",
                'is_volatile': is_volatile,
                'threshold': "10%"
            })
            
            # Check trading hours
            current_hour = datetime.now().hour
            is_high_activity = 13 <= current_hour <= 21  # 9 AM - 5 PM EST
            
            # Get market alignment
            btc_correlation = await self._calculate_btc_correlation(symbol) if symbol != 'BTC' else 1.0
            
            conditions = {
                'is_volatile': is_volatile,
                'price_range_7d': price_range,
                'is_high_activity': is_high_activity,
                'market_aligned': btc_correlation > 0.5,
                'suitable_for_trading': (
                    not is_volatile and
                    is_high_activity and
                    btc_correlation > 0.5
                )
            }
            
            await self.log(f"Trading conditions for {symbol}:", context=conditions)
            return conditions
            
        except Exception as e:
            await self.log(f"Error checking market conditions: {str(e)}", level="error")
            raise TradingError(f"Failed to check market conditions: {str(e)}", error_type='DATA')

    def _calculate_trade_signal(self, symbol: str) -> Dict[str, Any]:
        """Main entry point for all trading signal calculations"""
        try:
            # Get current price and technical indicators
            current_price = float(self.client.get_product(f"{symbol}-USD").price)
            technical_data = self._calculate_technical_indicators(symbol)
            
            # Get market analysis data
            market_data = self._analyze_market_data(symbol)
            
            # Calculate final signal
            signal = self._calculate_signal_components(
                technical_data=technical_data,
                market_data=market_data,
                current_price=current_price,
                symbol=symbol
            )
            
            # Log comprehensive analysis
            self.log(f"Trade signal calculated for {symbol}:", context={
                'price': current_price,
                'action': signal['action'],
                'score': signal['score'],
                'components': signal['signals']
            })
            
            return signal
            
        except Exception as e:
            self.log(f"Error calculating trade signal for {symbol}: {str(e)}", level="error")
            return self._get_fallback_signal(symbol, str(e))

    async def _calculate_technical_indicators(self, symbol: str) -> Dict[str, Any]:
        try:
            prices = await self.price_manager.get_price(symbol, days=250)
            current_price = float(prices.iloc[-1])
            
            # Calculate RSI
            delta = prices.diff()
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            avg_gain = gains.ewm(span=self.rsi_period, adjust=False).mean()
            avg_loss = losses.ewm(span=self.rsi_period, adjust=False).mean()
            rs = avg_gain / avg_loss
            rsi = float(100 - (100 / (1 + rs)).iloc[-1])
            
            # Calculate Moving Averages
            sma_20 = prices.rolling(window=20).mean()
            sma_50 = prices.rolling(window=50).mean()
            sma_200 = prices.rolling(window=200).mean()
            
            # Calculate Bollinger Bands
            bb_sma = prices.rolling(window=20).mean()
            bb_std = prices.rolling(window=20).std()
            
            # Get current values
            current_price = prices.iloc[-1]
            
            return {
                'price': current_price,
                'rsi': float(rsi),
                'moving_averages': {
                    'sma_20': float(sma_20.iloc[-1]),
                    'sma_50': float(sma_50.iloc[-1]),
                    'sma_200': float(sma_200.iloc[-1])
                },
                'bollinger_bands': {
                    'middle': float(bb_sma.iloc[-1]),
                    'upper': float(bb_sma.iloc[-1] + (bb_std.iloc[-1] * 2)),
                    'lower': float(bb_sma.iloc[-1] - (bb_std.iloc[-1] * 2)),
                    'bandwidth': float((bb_std.iloc[-1] * 4 / bb_sma.iloc[-1]) * 100)
                },
                'trend': self._determine_trend(
                    current_price,
                    sma_20.iloc[-1],
                    sma_50.iloc[-1],
                    sma_200.iloc[-1])
            }
        except Exception as e:
            await self.log(f"Technical analysis error: {str(e)}", level="error")
            raise

    def _analyze_market_data(self, symbol: str) -> Dict[str, Any]:
        """Analyze market conditions and sentiment"""
        try:
            sentiment = self.analyze_market_sentiment(symbol)
            market_conditions = self._check_market_conditions(symbol)
            volume_data = self.analyze_volume(symbol)
            
            return {
                'sentiment': sentiment,
                'market_conditions': market_conditions,
                'volume': volume_data
            }
        except Exception as e:
            self.log(f"Error analyzing market data: {str(e)}", level="error")
            raise

    def _calculate_signal_components(self, technical_data: Dict[str, Any], 
                               market_data: Dict[str, Any],
                               current_price: float,
                               symbol: str) -> Dict[str, Any]:
        """Calculate final signal with standardized scoring"""
        try:
            # Component weights
            WEIGHTS = {
                'trend': 0.4,
                'momentum': 0.3,
                'volume': 0.2,
                'risk': 0.1
            }
            
            # Signal thresholds
            THRESHOLDS = {
                'strong_buy': 20,
                'buy': 15,
                'strong_sell': -20,
                'sell': -15
            }
            
            # Calculate individual scores
            scores = {
                'trend': self._calculate_trend_score(technical_data),
                'momentum': self._calculate_momentum_score(market_data['sentiment'], technical_data),
                'volume': self._calculate_volume_score(market_data['volume'], market_data['market_conditions']),
                'risk': self._calculate_risk_score(market_data['market_conditions'], market_data['sentiment']) 
            }
            
            # Calculate weighted final score
            final_score = sum(score * WEIGHTS[component] for component, score in scores.items())
            
            # Determine action based on scores and thresholds
            action = self._determine_signal_action(final_score, scores, THRESHOLDS)
            
            return {
                'symbol': symbol,
                'price': current_price,
                'score': final_score,
                'signals': scores,
                'action': action,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.log(f"Error calculating signal components: {str(e)}", level="error")
            return self._get_fallback_signal(symbol, str(e))

    def _determine_signal_action(self, final_score: float, 
                           component_scores: Dict[str, float],
                           thresholds: Dict[str, float]) -> str:
        """Determine trading action based on scores and thresholds"""
        if (final_score >= thresholds['strong_buy'] and 
            component_scores['trend'] > 15 and 
            component_scores['volume'] > 5):
            return 'STRONG_BUY'
        elif (final_score >= thresholds['buy'] and 
              component_scores['trend'] > 10):
            return 'BUY'
        elif (final_score <= thresholds['strong_sell'] and 
              component_scores['trend'] < -15 and 
              component_scores['volume'] < -5):
            return 'STRONG_SELL'
        elif (final_score <= thresholds['sell'] and 
              component_scores['trend'] < -10):
            return 'SELL'
        return 'HOLD'

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
        try:
            signal = self._calculate_trade_signal(symbol)
            market_conditions = self._check_market_conditions(symbol)
            
            # Add position check
            has_position = symbol in (self.positions if not self.paper_trading_active else self.paper_positions)
            
            # Define thresholds as class constants
            TREND_THRESHOLD = 15
            MOMENTUM_THRESHOLD = 0  # Changed from 5
            VOLUME_THRESHOLD = 0    # Changed from 3
            RISK_THRESHOLD = -10
            SCORE_THRESHOLD = 15
            
            if action == 'BUY' and not has_position:
                # Log decision factors
                self.log(f"Trade decision factors for {symbol}:", context={
                    'action': signal['action'],
                    'trend_score': signal['signals']['trend'],
                    'momentum_score': signal['signals']['momentum'],
                    'volume_score': signal['signals']['volume'],
                    'risk_score': signal['signals']['risk'],
                    'total_score': signal['score'],
                    'market_suitable': market_conditions['suitable_for_trading']
                })
                
                return (
                    signal['action'] in ['BUY', 'STRONG_BUY'] and
                    signal['signals']['trend'] > TREND_THRESHOLD and
                    signal['signals']['momentum'] > MOMENTUM_THRESHOLD and
                    signal['signals']['volume'] > VOLUME_THRESHOLD and
                    signal['signals']['risk'] > RISK_THRESHOLD and
                    signal['score'] > SCORE_THRESHOLD and
                    market_conditions['suitable_for_trading']
                )
            elif action == 'SELL':
                return (
                    signal['action'] in ['SELL', 'STRONG_SELL'] or
                    signal['signals']['risk'] < -15 or
                    (signal['signals']['trend'] < -10 and signal['signals']['momentum'] < -10) or
                    not market_conditions['suitable_for_trading']
            
            return False
            
        except Exception as e:
            self.log(f"Trade validation error: {str(e)}", level="error")
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

    async def calculate_moving_averages(self, symbol: str) -> Dict[str, Any]:
        """Calculate moving averages with comprehensive data"""
        try:
            # Get price data using price manager
            prices = await self.price_manager.get_cached_price_data(symbol, days=250)
            current_price = float(prices.iloc[-1])
            
            # Calculate moving averages
            sma_20 = prices.rolling(window=20).mean().iloc[-1]
            sma_50 = prices.rolling(window=50).mean().iloc[-1]
            sma_200 = prices.rolling(window=200).mean().iloc[-1]
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'sma_20': float(sma_20),
                'sma_50': float(sma_50),
                'sma_200': float(sma_200),
                'trend': self._determine_trend(current_price, sma_20, sma_50, sma_200)
            }
        except Exception as e:
            await self.log(f"Error calculating MAs for {symbol}: {str(e)}", level="error")
            raise TradingError(f"Failed to calculate moving averages: {str(e)}", error_type='DATA')

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
                        timedelta(0)) / len(self.position_history)
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

    async def _check_risk_management(self, symbol: str) -> None:
        try:
            position = self.positions.get(symbol) or self.paper_positions.get(symbol)
            if not position:
                return
            
            current_price = float(self.client.get_product(f"{symbol}-USD").price)
            profit_info = position.calculate_profit(current_price)
            
            # Check stop loss
            if current_price <= position.stop_loss_price:
                if await self._execute_trade(symbol, 'SELL', reason="Stop Loss"):
                    await self.async_log(f"Stop loss executed for {symbol}")
                return
            
            # Check trailing stop
            if position.should_trigger_trailing_stop(current_price):
                if await self._execute_trade(symbol, 'SELL', reason="Trailing Stop"):
                    await self.async_log(f"Trailing stop executed for {symbol}")
                return
            
            # Check take profit levels
            if not position.partial_exit_taken and profit_info['profit_percentage'] >= self.take_profit_percentage:
                # Record partial exit
                original_quantity = position.quantity
                exit_quantity = original_quantity * self.partial_tp_size
                
                # Execute partial exit
                if self.paper_trading_active:
                    await self._simulate_sell_order(symbol, partial=True)
                else:
                    self._place_sell_order(symbol, partial=True)
                
                # Record the exit
                position.record_exit(
                    exit_price=current_price,
                    exit_quantity=exit_quantity,
                    reason="Partial Take Profit"
                )
                return
                
            # Check full take profit
            if profit_info['profit_percentage'] >= self.take_profit_percentage:
                if await self._execute_trade(symbol, 'SELL', reason="Take Profit"):
                    await self.async_log(f"Take profit executed for {symbol}")
                return
            
        except Exception as e:
            self.log(f"Error in risk management for {symbol}: {str(e)}", level="error")

    async def analyze_market_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Analyze overall market sentiment using multiple indicators"""
        try:
            # Get data for different timeframes
            end = datetime.now()
            start_long = end - timedelta(days=90)   
            start_medium = end - timedelta(days=30)  
            start_short = end - timedelta(days=7)    
            
            # Get price data for different timeframes with validation
            prices_long = await self.price_manager.get_historical_prices(symbol, start_long, end)
            if len(prices_long) < 2:
                raise TradingError(f"Insufficient historical data for {symbol}")
                
            await self.log(f"Got {len(prices_long)} price points for {symbol}")
            
            prices_medium = prices_long[prices_long.index >= start_medium]
            prices_short = prices_long[prices_long.index >= start_short]
            
            # Validate we have enough data points
            if len(prices_short) < 2 or len(prices_medium) < 2:
                raise TradingError(f"Insufficient data points for sentiment calculation")
                
            # Calculate and log price changes
            price_changes = {
                'long': ((prices_long.iloc[-1] - prices_long.iloc[0]) / prices_long.iloc[0]) * 100,
                'medium': ((prices_medium.iloc[-1] - prices_medium.iloc[0]) / prices_medium.iloc[0]) * 100,
                'short': ((prices_short.iloc[-1] - prices_short.iloc[0]) / prices_short.iloc[0]) * 100
            }
            
            await self.log(f"Price changes for {symbol}:", context=price_changes)
            
            return {
                'sentiment_score': price_changes['short'],
                'overall_sentiment': 'Bullish' if price_changes['short'] > 0 else 'Bearish',
                'momentum': {
                    'short_term': 'bullish' if price_changes['short'] > 0 else 'bearish',
                    'medium_term': 'bullish' if price_changes['medium'] > 0 else 'bearish',
                    'long_term': 'bullish' if price_changes['long'] > 0 else 'bearish'
                },
                'price_changes': price_changes
            }
                
        except Exception as e:
            await self.log(f"Error analyzing market sentiment for {symbol}: {str(e)}", level="error")
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
            fee = self.trade_amount * self.FEE_RATE
            actual_trade_amount = self.trade_amount - fee
            
            # Check if we have enough paper balance with buffer for fees
            if self.paper_balance < self.trade_amount * 1.01:  # Add 1% buffer
                await self.async_log(f"Insufficient paper balance for {symbol} buy: ${self.paper_balance:.2f} < ${self.trade_amount:.2f}")
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
            raise TradingError(f"Error simulating buy order for {symbol}: {str(e)}")

    async def _simulate_sell_order(self, symbol: str, partial: bool = False) -> None:
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
                'action': 'PARTIAL_SELL' if partial else 'SELL',
                'symbol': symbol,
                'amount_usd': total_value,
                'price': current_price,
                'quantity': position.quantity,
                'fees': fee,
                'profit': profit_info['profit_usd'],
                'profit_percentage': profit_info['profit_percentage'],
                'is_paper': True,
                'is_partial': partial
            })
            
            # For partial sells, update position instead of removing it
            if partial:
                position.quantity *= (1 - self.partial_tp_size)
                position.partial_exit_taken = True
                logging.info(f"Partial paper sell for {symbol}: ${total_value:.2f} (Remaining: {position.quantity:.8f})")
            else:
                # Remove position for full sells
                del self.paper_positions[symbol]
                logging.info(f"Full paper sell for {symbol}: ${total_value:.2f}")
            
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

    def set_discord_channel(self, channel) -> None:
        """
        Set the Discord channel for notifications.
        
        Args:
            channel: Discord channel for trading notifications
        """
        try:
            self.discord_channel = channel
            logging.info("Discord notifications channel set")
        except Exception as e:
            logging.error(f"Error setting Discord channel: {str(e)}")

    def set_logs_channel(self, channel) -> None:
        """
        Set the Discord channel for logging messages.
        
        Args:
            channel: Discord channel for log messages
        """
        try:
            self.logs_channel = channel
            logging.info("Discord logs channel set")
        except Exception as e:
            logging.error(f"Error setting logs channel: {str(e)}")

    async def send_notification(self, message: str, is_update: bool = False) -> None:
        """
        Send a notification to the Discord channel.
        
        Args:
            message: The message to send
            is_update: Whether this is a status update (affects formatting)
        """
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

    async def send_log(self, message: str, level: str = "info") -> None:
        """
        Send a log message to the logs channel.
        
        Args:
            message: The message to log
            level: Log level (info, warning, error)
        """
        try:
            if not self.logs_channel:
                return
            
            # Add emoji based on level
            prefix = "🔴" if level == "error" else "⚠️" if level == "warning" else "ℹ️"
            formatted_message = f"{prefix} {message}"
            
            await self.logs_channel.send(formatted_message)
            
        except Exception as e:
            logging.error(f"Error sending log message: {str(e)}")
            # Don't reset channel here to keep trying

    async def _log_message(self, message: str, level: str = "info", context: Dict = None) -> None:
        """
        Log a message with optional context data.
        
        Args:
            message: The main log message
            level: Log level (info, warning, error)
            context: Optional dictionary of context data to display
        """
        try:
            # Log to file
            log_func = getattr(logging, level.lower())
            log_func(message)
            
            # Format context data if present
            if context:
                formatted_context = "\n".join([
                    f"  • {key}: {self._format_context_value(value)}"
                    for key, value in context.items()
                ])
                message = f"{message}\n{formatted_context}"
            
            # Send to Discord if channel is set
            if hasattr(self, 'logs_channel'):
                prefix = {
                    "info": "ℹ️",
                    "warning": "⚠️",
                    "error": "❌"
                }.get(level.lower(), "ℹ️")
                
                await self.logs_channel.send(f"{prefix} {message}")
                
        except Exception as e:
            logging.error(f"Error in logging: {str(e)}")

    def _format_context_value(self, value: Any) -> str:
        """Format context values for display"""
        if isinstance(value, dict):
            return "\n    " + "\n    ".join([
                f"- {k}: {self._format_context_value(v)}"
                for k, v in value.items()
            ])
        elif isinstance(value, (list, tuple)):
            return "\n    " + "\n    ".join([
                f"- {self._format_context_value(item)}"
                for item in value
            ])
        elif isinstance(value, float):
            return f"{value:.2f}"
        else:
            return str(value)

    async def _heartbeat(self) -> None:
        """
        Periodic heartbeat to check system health and maintain connections.
        """
        while True:
            try:
                # Check API connection
                await self.test_api_connection()
                
                # Update price caches
                for symbol in self.watched_coins:
                    await self.price_manager.get_cached_price_data(symbol)
                
                # Log heartbeat
                logging.debug("Heartbeat check completed successfully")
                
                # Wait for next interval
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logging.error(f"Heartbeat check failed: {str(e)}")
                await asyncio.sleep(60)  # Shorter retry interval on failure

    async def calculate_bollinger_bands(self, symbol: str, period: int = 20, std_dev: float = 2.0) -> Dict[str, float]:
        """Calculate Bollinger Bands for a symbol."""
        try:
            # Get historical prices
            prices = await self.price_manager.get_cached_price_data(symbol, days=30)
            if len(prices) < period:
                raise TradingError("Insufficient price data for BB calculation", "DATA")
            
            # Calculate middle band (SMA)
            middle_band = prices.rolling(window=period).mean()
            
            # Calculate standard deviation
            std = prices.rolling(window=period).std()
            
            # Calculate upper and lower bands
            upper_band = middle_band + (std * std_dev)
            lower_band = middle_band - (std * std_dev)
            
            # Calculate bandwidth
            bandwidth = ((upper_band - lower_band) / middle_band) * 100
            
            return {
                'upper': float(upper_band.iloc[-1]),
                'middle': float(middle_band.iloc[-1]),
                'lower': float(lower_band.iloc[-1]),
                'bandwidth': float(bandwidth.iloc[-1])
            }
        except Exception as e:
            logging.error(f"BB calculation error for {symbol}: {str(e)}")
            raise TradingError(f"Failed to calculate Bollinger Bands: {str(e)}", "DATA")

    async def _determine_trend(self, current_price: float, sma_20: float, sma_50: float, sma_200: float) -> str:
        """
        Determine market trend based on moving averages.
        
        Args:
            current_price: Current asset price
            sma_20: 20-period simple moving average
            sma_50: 50-period simple moving average
            sma_200: 200-period simple moving average
        
        Returns:
            str: 'bullish', 'bearish', or 'neutral'
        """
        try:
            # Short-term trend
            short_term = 'bullish' if current_price > sma_20 else 'bearish'
            
            # Medium-term trend
            medium_term = 'bullish' if sma_20 > sma_50 else 'bearish'
            
            # Long-term trend
            long_term = 'bullish' if sma_50 > sma_200 else 'bearish'
            
            # Weight the trends
            if short_term == medium_term == long_term:
                return short_term
            elif short_term == medium_term:
                return short_term
            elif medium_term == long_term:
                return medium_term
            return 'neutral'
            
        except Exception as e:
            await self.log(f"Error determining trend: {str(e)}", level="error")
            return 'neutral'

    @property
    def trading_active(self) -> bool:
        """Check if real trading is active."""
        return getattr(self, '_trading_active', False)

    @property
    def paper_trading_active(self) -> bool:
        """Check if paper trading is active."""
        return getattr(self, '_paper_trading_active', False)

    async def get_status(self) -> str:
        """Get current trading bot status."""
        try:
            watched = len(self.watched_coins)
            real_positions = len(getattr(self, 'positions', {}))
            paper_positions = len(getattr(self, 'paper_positions', {}))
            
            # Get balances with error handling
            try:
                real_balance = await self.get_account_balance()
            except Exception as e:
                logging.warning(f"Failed to get real balance: {str(e)}")
                real_balance = {'balances': {}, 'total_usd_value': 0.0}
            
            try:
                paper_balance = self.get_paper_balance()
            except Exception as e:
                logging.warning(f"Failed to get paper balance: {str(e)}")
                paper_balance = {'cash_balance': 0, 'total_value': 0}
            
            status = "Bot Status:\n```"
            
            # Trading Status
            status += "\n🤖 Mode:\n"
            status += f"  • Real Trading: {'✅' if self.trading_active else '❌'}\n"
            status += f"  • Paper Trading: {'✅' if self.paper_trading_active else '❌'}\n"
            
            # Positions
            status += "\n📊 Positions:\n"
            status += f"  • Real: {real_positions}\n"
            status += f"  • Paper: {paper_positions}\n"
            status += f"  • Watching: {watched} coins\n"
            
            # Real Account Balances
            if real_balance['balances']:
                status += "\n💰 Real Account Balances:\n"
                for symbol, data in real_balance['balances'].items():
                    status += f"  • {symbol}: {data['balance']:.8f} (${data['usd_value']:.2f})\n"
                status += f"  • Total Real Portfolio Value: ${real_balance['total_usd_value']:.2f}\n"
            
            # Paper Trading Account
            status += "\n📝 Paper Trading Account:\n"
            status += f"  • Paper Cash: ${paper_balance['cash_balance']:.2f}\n"
            status += f"  • Paper Portfolio Value: ${paper_balance['total_value']:.2f}\n"
            
            # Watched Coins with Current Prices
            status += "\n🔍 Watched Coins:\n"
            for coin in sorted(self.watched_coins):
                try:
                    product = self.client.get_product(f"{coin}-USD")
                    current_price = float(product.price)
                    rsi = await self.calculate_rsi(coin)
                    status += f"  • {coin}: ${current_price:,.2f} (RSI: {rsi:.1f})\n"
                except Exception as e:
                    status += f"  • {coin}: Error fetching data\n"
            
            # Position Sizing Strategy
            status += "\n📊 Position Sizing:\n"
            status += f"  • Strategy: Dynamic (1-10% of portfolio)\n"
            status += f"  • Min Size: 1% of available funds\n"
            status += f"  • Max Size: 10% of available funds\n"
            status += f"  • Scaling: Based on signal strength\n"
            
            status += "```"
            return status
            
        except Exception as e:
            await self.log(f"Error getting status: {str(e)}", level="error")
            raise TradingError(f"Failed to get status: {str(e)}", "DATA")

    async def get_ma_analysis(self, symbol: str) -> str:
        """Get moving average analysis for a symbol."""
        try:
            symbol = symbol.upper()
            # Get enough data for 200-day MA
            prices = await self.price_manager.get_cached_price_data(symbol, days=250)
            
            # Ensure we have enough data
            if len(prices) < 200:
                return f"Insufficient historical data for {symbol} (need 200 days, got {len(prices)} days)"
            
            current_price = float(prices.iloc[-1])
            
            # Calculate SMAs
            sma_20 = float(prices.rolling(window=20).mean().iloc[-1])
            sma_50 = float(prices.rolling(window=50).mean().iloc[-1])
            sma_200 = float(prices.rolling(window=200).mean().iloc[-1])
            
            # Get trend
            trend = await self._determine_trend(current_price, sma_20, sma_50, sma_200)
            
            # Format trend string
            trend_str = trend[0].upper() + trend[1:]  # Capitalize first letter
            
            # Add trend emoji
            trend_emoji = "🟢" if trend == 'bullish' else "🔴" if trend == 'bearish' else "⚪"
            
            return f"Moving Average Analysis for {symbol}:\n```" \
                   f"📊 Price Levels:\n" \
                   f"  • Current: ${current_price:,.2f}\n" \
                   f"  • SMA 20: ${sma_20:,.2f}\n" \
                   f"  • SMA 50: ${sma_50:,.2f}\n" \
                   f"  • SMA 200: ${sma_200:,.2f}\n\n" \
                   f"📈 Trend Analysis:\n" \
                   f"  • Status: {trend_emoji} {trend_str}\n" \
                   f"  • Above 20 MA: {'Yes ✅' if current_price > sma_20 else 'No ❌'}\n" \
                   f"  • Above 50 MA: {'Yes ✅' if current_price > sma_50 else 'No ❌'}\n" \
                   f"  • Above 200 MA: {'Yes ✅' if current_price > sma_200 else 'No ❌'}```"
        except Exception as e:
            await self.log(f"Error getting MA analysis: {str(e)}", level="error")
            raise TradingError(f"Failed to get MA analysis: {str(e)}", "DATA")

    async def _calculate_trade_signal(self, symbol: str) -> Dict[str, Any]:
        """Calculate trading signal based on multiple indicators."""
        try:
            # Get current price and technical indicators
            current_price = await self.price_manager.get_current_price(symbol)
            prices = await self.price_manager.get_cached_price_data(symbol, days=30)
            
            # Calculate indicators
            rsi = await self.calculate_rsi(symbol)
            bb_data = await self.calculate_bollinger_bands(symbol)
            sentiment = await self.analyze_market_sentiment(symbol)
            conditions = await self.check_market_conditions(symbol)
            
            # Calculate scores
            rsi_score = 50 - rsi  # Higher when oversold, lower when overbought
            bb_score = ((current_price - bb_data['lower']) / (bb_data['upper'] - bb_data['lower']) - 0.5) * -100
            sentiment_score = sentiment['sentiment_score']
            
            # Combine scores with detailed logging
            total_score = (rsi_score * 0.3) + (bb_score * 0.3) + (sentiment_score * 0.4)
            
            # Create comprehensive signal data
            signal_data = {
                'indicators': {
                    'rsi': {'value': rsi, 'score': rsi_score, 'weight': 0.3},
                    'bollinger': {
                        'score': bb_score, 
                        'weight': 0.3,
                        'bands': bb_data
                    },
                    'sentiment': {
                        'score': sentiment_score, 
                        'weight': 0.4,
                        'analysis': sentiment
                    }
                },
                'market_conditions': conditions,
                'current_price': current_price,
                'total_score': total_score
            }
            
            # Log detailed analysis
            await self.log(f"Signal analysis for {symbol}:", context=signal_data)
            
            # Determine action
            action = 'HOLD'
            if total_score > 20 and conditions['suitable_for_trading']:
                action = 'BUY'
            elif total_score < -20:
                action = 'SELL'
            
            # Log final decision
            decision_data = {
                'action': action,
                'total_score': total_score,
                'thresholds': {'buy': 20, 'sell': -20},
                'market_suitable': conditions['suitable_for_trading']
            }
            await self.log(f"Trading decision for {symbol}:", context=decision_data)
            
            # Return complete signal
            return {
                'action': action,
                'score': total_score,
                'signals': {
                    'rsi': rsi_score,
                    'bb': bb_score,
                    'sentiment': sentiment_score,
                    'conditions': conditions
                },
                'analysis': signal_data
            }
            
        except Exception as e:
            await self.log(f"Error calculating trade signal: {str(e)}", level="error")
            return {
                'action': 'HOLD',
                'score': 0,
                'signals': {},
                'analysis': {}
            }

    async def _simulate_trade(self, symbol: str, action: str, reason: str) -> bool:
        """Execute a simulated trade for paper trading."""
        try:
            if action == 'BUY':
                position_size = await self._calculate_position_size(symbol)
                if position_size > 0:
                    await self._simulate_buy_order(symbol, position_size)
                    await self.log(f"Paper buy executed for {symbol}: ${position_size:.2f}", level="info")
                    return True
            elif action == 'SELL':
                if symbol in self.paper_positions:
                    await self._simulate_sell_order(symbol)
                    await self.log(f"Paper sell executed for {symbol}", level="info")
                    return True
            return False
        except Exception as e:
            await self.log(f"Error simulating trade: {str(e)}", level="error")
            return False

    async def _execute_trade(self, symbol: str, action: str, reason: str) -> bool:
        """Execute a real trade."""
        try:
            if action == 'BUY':
                position_size = await self._calculate_position_size(symbol)
                if position_size > 0:
                    await self._place_buy_order(symbol, position_size)
                    await self.log(f"Buy order executed for {symbol}: ${position_size:.2f}", level="info")
                    return True
            elif action == 'SELL':
                if symbol in self.positions:
                    await self._place_sell_order(symbol)
                    await self.log(f"Sell order executed for {symbol}", level="info")
                    return True
            return False
        except Exception as e:
            await self.log(f"Error executing trade: {str(e)}", level="error")
            return False

    async def _calculate_btc_correlation(self, symbol: str) -> float:
        """
        Calculate correlation between symbol and BTC prices.
        
        Args:
            symbol: The cryptocurrency symbol to compare with BTC
            
        Returns:
            float: Correlation coefficient (-1 to 1)
        """
        try:
            if symbol == 'BTC':
                return 1.0
            
            # Get price data for both assets
            btc_prices = await self.price_manager.get_cached_price_data('BTC', days=30)
            symbol_prices = await self.price_manager.get_cached_price_data(symbol, days=30)
            
            # Calculate daily returns
            btc_returns = btc_prices.pct_change().dropna()
            symbol_returns = symbol_prices.pct_change().dropna()
            
            # Calculate correlation
            correlation = btc_returns.corr(symbol_returns)
            return float(correlation) if not pd.isna(correlation) else 0.0
            
        except Exception as e:
            await self.log(f"Error calculating BTC correlation for {symbol}: {str(e)}", level="error")
            return 0.0  # Default to no correlation on error