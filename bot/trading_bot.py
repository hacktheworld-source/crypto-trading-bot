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
            # Get candles from API
            end = datetime.now()
            start = end - timedelta(days=days)
            
            response = await self.client.get_candles(
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
            
            if 'COINBASE_API_KEY' not in os.environ:
                raise Exception("COINBASE_API_KEY not found in environment variables")
            if 'COINBASE_API_SECRET' not in os.environ:
                raise Exception("COINBASE_API_SECRET not found in environment variables")
            
            # First create client
            self.client = RESTClient(
                api_key=os.environ['COINBASE_API_KEY'].strip(),
                api_secret=os.environ['COINBASE_API_SECRET'].strip()
            )
            
            # Then create PriceManager with client and logging
            self.price_manager = PriceManager(
                client=self.client,
                cache_size=100,
                cache_ttl=300,
                rate_limit=0.1,
                log_callback=self.log
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
            self.stop_loss_percentage = 5.0  # Default 5% stop loss
            self.take_profit_percentage = 10.0  # Default 10% take profit
            self.partial_tp_percentage = 7.0  # Take partial profits at 7%
            self.partial_tp_size = 0.5  # Sell 50% at first take profit
            self.max_position_size = 1000.0  # Maximum USD in any single position
            
            # Add configurable trailing stop
            self.trailing_stop_percentage = 5.0  # Default 5%
            self.trailing_stop_enabled = True
            self.trailing_stop_activation = 3.0  # Only activate after 3% profit
            
            self.load_config()
            
            # Paper trading attributes
            self.paper_trading = not self.trading_active
            self.paper_balance = float(os.getenv('PAPER_BALANCE', '1000.0'))
            self.paper_positions = {}  # Add this
            self.paper_trade_history = []  # Add this
            
            self.discord_channel = None  # Will be set when bot starts
            
            # Pass fee rate to Position
            self.fee_rate = self.FEE_RATE
            
            # Add to existing init
            self._price_cache = {}
            self._cache_lock = asyncio.Lock()
            self._cache_max_size = 100  # Maximum number of symbols to cache
            self._cache_cleanup_threshold = 80  # Clean when we hit 80% capacity
            
            # Initialize managers
            self.position_manager = PositionManager(self)
            self.signal_generator = SignalGenerator(self)
            self.trade_executor = TradeExecutor(self)
            
            # Configuration
            self.config = TradingConfig()
            
            # Trading state
            self.positions: Dict[str, Position] = {}  # Add this
            
        except Exception as e:
            logging.error(f"Failed to initialize trading bot: {str(e)}")
            raise Exception(f"Bot initialization failed: {str(e)}")
        
    async def start_trading_loop(self, paper: bool = True) -> str:
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
            
            # Start both trading loop and heartbeat
            asyncio.create_task(self._trading_loop())
            asyncio.create_task(self._heartbeat())
            
            await self.log(f"{mode} trading loop started")
            return f"{mode} trading bot started successfully"
            
        except Exception as e:
            await self.log(f"Failed to start trading loop: {str(e)}", level="error")
            self.trading_active = False
            self.paper_trading = False
            return f"Error starting trading bot: {str(e)}"
        
    async def stop_trading_loop(self) -> str:
        """Stop the trading loop"""
        was_active = self.trading_active or self.paper_trading
        self.trading_active = False
        self.paper_trading = False  # Stop paper trading too
        await self.log("Trading loop stopped")
        return "Trading bot stopped successfully" if was_active else "Trading bot is already stopped"
        
    async def _trading_loop(self):
        """Main trading loop using position and signal managers"""
        while self.trading_active or self.paper_trading:  # Check both modes
            try:
                for symbol in self.watched_coins:
                    # Generate signal using signal manager
                    signal = await self.signal_generator.generate_signal(symbol)
                    
                    if signal['action'] != 'HOLD':
                        # Execute trade using trade executor
                        success = await self.trade_executor.execute_trade(
                            symbol=symbol,
                            action=signal['action'],
                            reason=f"Signal: {signal['score']:.2f}"
                        )
                        
                        if not success:
                            await self.log(f"Trade execution failed for {symbol}", level="error")
                            continue
                        
                    # Update position metrics
                    if symbol in self.position_manager.positions:
                        await self.position_manager.update_positions()
                        
                # Sleep for configured interval
                await asyncio.sleep(self.config.interval)
                
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
            # Convert symbol to uppercase
            symbol = symbol.upper()
            
            # Get historical candles synchronously
            end = datetime.now()
            start = end - timedelta(days=30)
            
            candles = self.client.get_product_candles(
                product_id=f"{symbol}-USD",
                granularity='ONE_DAY',
                start=start,
                end=end
            )
            
            # Convert candles to prices
            prices = pd.Series([float(candle.close) for candle in reversed(candles)])
            
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
        """Save current configuration to file"""
        config = {
            'watched_coins': list(self.watched_coins),
            'trading_interval': self.trading_interval,
            'rsi_period': self.rsi_period,
            'rsi_overbought': self.rsi_overbought,
            'rsi_oversold': self.rsi_oversold,
            'trade_amount': self.trade_amount,
            'stop_loss_percentage': self.stop_loss_percentage,  # Add this
            'take_profit_percentage': self.take_profit_percentage,
            'partial_tp_percentage': self.partial_tp_percentage,
            'partial_tp_size': self.partial_tp_size,
            'trailing_stop_percentage': self.trailing_stop_percentage,
            'trailing_stop_enabled': self.trailing_stop_enabled,
            'trailing_stop_activation': self.trailing_stop_activation
        }
        try:
            with open(self.CONFIG_FILE, 'w') as f:
                json.dump(config, f)
            logging.info("Configuration saved successfully")
        except Exception as e:
            logging.error(f"Error saving configuration: {str(e)}")
            
    def load_config(self):
        """Load configuration from file or use defaults"""
        try:
            with open(self.CONFIG_FILE, 'r') as f:
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
            # Use synchronous logging here since we're in a sync method
            logging.warning("No config file found, using defaults")

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

    def _calculate_position_size(self, symbol: str) -> float:
        try:
            # For paper trading, we should use paper balance instead of real balance
            if self.paper_trading:
                available_funds = self.paper_balance
                portfolio_value = self.get_paper_balance()['total_value']
            else:
                available_funds = float(self.get_account_balance()['balances'].get('USD', {}).get('balance', 0))
                portfolio_value = self.get_account_balance()['total_usd_value']
            
            # Never risk more than 2% of total portfolio on any single trade
            max_risk_amount = portfolio_value * 0.02
            
            # Calculate position size based on stop loss
            current_price = float(self.client.get_product(f"{symbol}-USD").price)
            risk_per_share = current_price * (self.stop_loss_percentage / 100)
            
            # Position size that risks the max risk amount
            position_size = min(
                max_risk_amount / risk_per_share * current_price,
                available_funds * 0.95,  # Leave 5% buffer for fees
                self.max_position_size
            )
            
            # Log calculation details
            self.log(f"Position size calculation for {symbol}:", context={
                'available_funds': available_funds,
                'portfolio_value': portfolio_value,
                'max_risk_amount': max_risk_amount,
                'current_price': current_price,
                'risk_per_share': risk_per_share,
                'calculated_size': position_size
            })
            
            # Ensure minimum trade size
            return max(5.0, position_size) if position_size >= 5.0 else 0
            
        except Exception as e:
            self.log(f"Error calculating position size: {str(e)}", level="error")
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
                    (sentiment['sentiment_score'] * btc_sentiment['sentiment_score'] > 0))
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
            rsi = float(100 - (100 / (1 + rs)).iloc[-1]) # Don't forget the parenthesis
            
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
                    sma_200.iloc[-1]
                )
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
            has_position = symbol in (self.positions if not self.paper_trading else self.paper_positions)
            
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
                )  # Added missing parenthesis
            
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

    def calculate_moving_averages(self, symbol: str) -> Dict[str, Any]:
        try:
            # Get enough historical data for calculations
            end = datetime.now()
            start = end - timedelta(days=250)  # Need enough data for 200MA
            
            # Get candles with OHLCV data
            response = self.client.get_candles(
                product_id=f"{symbol}-USD",
                start=int(start.timestamp()),
                end=int(end.timestamp()),
                granularity="ONE_DAY"
            )
            
            # Convert candles to pandas Series
            prices = pd.Series(
                [float(candle.close) for candle in reversed(response.candles)],
                index=[datetime.fromtimestamp(float(candle.start)) for candle in reversed(response.candles)]
            )
            
            # Calculate moving averages
            sma_20 = prices.rolling(window=20).mean()
            sma_50 = prices.rolling(window=50).mean()
            sma_200 = prices.rolling(window=200).mean()
            ema_12 = prices.ewm(span=12, adjust=False).mean()
            ema_26 = prices.ewm(span=26, adjust=False).mean()
            
            # Get latest values
            current_price = prices.iloc[-1]
            
            # Store recent candles for trend analysis
            recent_candles = [
                {
                    'open': float(candle.open),
                    'high': float(candle.high),
                    'low': float(candle.low),
                    'close': float(candle.close),
                    'volume': float(candle.volume)
                }
                for candle in response.candles[-10:]  # Keep last 10 candles
            ]
            
            return {
                'current_price': current_price,
                'sma_20': sma_20.iloc[-1],
                'sma_50': sma_50.iloc[-1],
                'sma_200': sma_200.iloc[-1],
                'ema_12': ema_12.iloc[-1],
                'ema_26': ema_26.iloc[-1],
                'candles': recent_candles,
                'trend': self._determine_trend(current_price, 
                                             sma_20.iloc[-1],
                                             sma_50.iloc[-1],
                                             sma_200.iloc[-1])
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
            
            # Update position with current price
            position.update_price(current_price)
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
                if self.paper_trading:
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
                
            self.log(f"Got {len(prices_long)} price points for {symbol}")
            
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
            
            self.log(f"Price changes for {symbol}:", context=price_changes)
            
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
            self.log(f"Error analyzing market sentiment for {symbol}: {str(e)}", level="error")
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
            raise

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
                                    profit_info: Dict[str, float] = None,
                                    reason: str = None):
        """Enhanced trade notification with market context"""
        try:
            color = discord.Color.green() if action == 'BUY' else \
                    discord.Color.red() if action == 'SELL' else \
                    discord.Color.blue()
            
            # Add reason to title if provided
            title = f"{'' if action == 'BUY' else '📉'} {action} {symbol}"
            if reason:
                title += f" ({reason})"
            
            if is_paper:
                title = "📝 PAPER " + title
            
            embed = discord.Embed(
                title=title,
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
                      f"💰 Price: ${price:,.2f}\n"
                      f"📈 Signal: {signal['action']}\n"
                      f"📊 Score: {signal['score']:.2f}\n"
                      f"������ Signal Components\n"
                      f"• 📈 Trend (0.4x):    {signal['signals']['trend']:.1f}\n"
                      f"• 🔄 Momentum (0.3x): {signal['signals']['momentum']:.1f}\n"
                      f"• 📊 Volume (0.2x):   {signal['signals']['volume']:.1f}\n"
                      f"• ⚠️ Risk (0.1x):     {signal['signals']['risk']:.1f}"
                      f"```",
                inline=False
            )
            
            if self.discord_channel:
                await self.discord_channel.send(embed=embed)
                
        except Exception as e:
            self.log(f"Error sending trade notification: {str(e)}", level="error")

    def _format_signal_text(self, symbol: str, signal: Dict[str, Any]) -> str:
        """Format signal data into readable text"""
        try:
            # Format basic info
            text = (
                f"💰 {symbol} Analysis\n"
                f"💰 Price: ${signal['price']:,.2f}\n"
                f"📈 Signal: {signal['action']}\n"
                f"📊 Score: {signal['score']:.2f}\n"
                f"📋 Signal Components\n"
                f"• 📈 Trend (0.4x):    {signal['signals']['trend']:>6.1f}\n"
                f"• 🔄 Momentum (0.3x): {signal['signals']['momentum']:>6.1f}\n"
                f"• 📊 Volume (0.2x):   {signal['signals']['volume']:>6.1f}\n"
                f"• ⚠️ Risk (0.1x):     {signal['signals']['risk']:>6.1f}\n"
            )
            
            # Add position info if exists
            position = self.paper_positions.get(symbol) if self.paper_trading else self.positions.get(symbol)
            if position:
                profit_info = position.calculate_profit(signal['price'])
                text += (
                    f"\n💼 Position Status\n"
                    f"Entry Price: ${position.entry_price:.2f}\n"
                    f"P/L: {profit_info['profit_percentage']:+.2f}%\n"
                    f"Max Profit: {profit_info['highest_profit_percentage']:+.2f}%\n"
                    f"Max Drawdown: {profit_info['drawdown_percentage']:+.2f}%\n"
                )
            
            text += "\n"  # Add spacing between coins
            return text
            
        except Exception as e:
            self.log(f"Error formatting signal text: {str(e)}", level="error")
            return f"Error formatting {symbol} signal: {str(e)}\n"

    async def send_interval_update(self):
        """Send periodic trading update to Discord"""
        try:
            update_text = "🔄 Trading Update Started\n"
            update_text += f"Mode: {'Paper' if self.paper_trading else 'Real'} Trading\n"
            update_text += f"Analyzing {len(self.watched_coins)} coins...\n\n"
            
            # Split the message into chunks to avoid rate limits
            chunk_size = 1500  # Discord has a 2000 char limit, we use less to be safe
            chunks = []
            current_chunk = ""
            
            for symbol in self.watched_coins:
                signal = self._calculate_trade_signal(symbol)
                signal_text = self._format_signal_text(symbol, signal)
                
                # If adding this signal would exceed chunk size, start new chunk
                if len(current_chunk) + len(signal_text) > chunk_size:
                    chunks.append(current_chunk)
                    current_chunk = signal_text
                else:
                    current_chunk += signal_text
                
                # Add delay between API calls
                await asyncio.sleep(0.5)
            
            # Add final chunk
            if current_chunk:
                chunks.append(current_chunk)
            
            # Send chunks with delay between each
            for i, chunk in enumerate(chunks):
                await self.send_notification(
                    f"Part {i+1}/{len(chunks)}:\n{chunk}", 
                    is_update=True
                )
                await asyncio.sleep(2)  # 2 second delay between messages to avoid rate limits
            
            # Send summary footer
            footer = (
                f"{'='*30}\n"
                f"📈 Active Positions: {len(self.paper_positions) if self.paper_trading else len(self.positions)}\n"
                f"💰 Mode: {'Paper' if self.paper_trading else 'Real'} Trading\n"
                f"⏰ Next Update: {self.trading_interval//60} minutes"
            )
            await asyncio.sleep(1)  # Wait before sending footer
            await self.send_notification(footer, is_update=True)
            
        except Exception as e:
            self.log(f"Error sending interval update: {str(e)}", level="error")

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
        """Async initialization steps after bot creation"""
        try:
            # Verify API connection
            self.client.get_accounts()
            await self.log("API connection verified")  # This is async but we're in a sync context
            
            # Initialize price cache
            for symbol in self.watched_coins:
                await self.price_manager.get_price(symbol)
                
            # Start heartbeat
            asyncio.create_task(self._heartbeat())
                
        except Exception as e:
            await self.log(f"Post-initialization failed: {str(e)}", level="error")
            raise

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

    async def log(self, message: str, level: str = "info", context: Dict[str, Any] = None) -> None:
        """
        Log message to both file and Discord if available.
        
        Args:
            message: The message to log
            level: Log level (info, warning, error)
            context: Optional context dictionary
        """
        # Always log to file first
        if level == "error":
            logging.error(message)
        elif level == "warning":
            logging.warning(message)
        else:
            logging.info(message)
        
        # Log context if provided
        if context:
            for key, value in context.items():
                logging.info(f"{key}: {value}")
        
        # Only try Discord logging if channel is set
        if hasattr(self, 'logs_channel') and self.logs_channel:
            try:
                await self.async_log(message, level, context)
            except Exception as e:
                logging.error(f"Failed to send log to Discord: {str(e)}")

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

    def _calculate_bollinger_bands(self, symbol: str) -> Dict[str, Any]:
        """Calculate Bollinger Bands using cached price data"""
        try:
            prices = self._get_cached_price_data(symbol)
            sma = prices.rolling(window=20).mean()
            std = prices.rolling(window=20).std()
            
            current_sma = float(sma.iloc[-1])
            current_std = float(std.iloc[-1])
            
            return {
                'middle': current_sma,
                'upper': current_sma + (current_std * 2),  # Removed extra parenthesis
                'lower': current_sma - (current_std * 2),  # Removed extra parenthesis
                'bandwidth': float((current_std * 4 / current_sma) * 100)
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
                'support_strength': len([p for p in pivot_lows if abs(p - nearest_support) / nearest_support < 0.02])}
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

    def _calculate_trend_score(self, ma_data: Dict[str, Any]) -> float:
        # Add Bollinger Band integration
        bb_data = self._calculate_bollinger_bands(ma_data['symbol'])
        support_resistance = self._find_support_resistance(ma_data['symbol'])
        
        try:
            score = 0
            current_price = ma_data['current_price']
            
            # Price vs MA scoring (max ±20)
            if current_price > ma_data['sma_20']:
                score += 5
            if current_price > ma_data['sma_50']:
                score += 7
            if current_price > ma_data['sma_200']:
                score += 8
                
            # Trend strength (max ±10)
            trend = ma_data.get('trend', 'Mixed Trend')
            if trend == 'Strong Uptrend':
                score += 10
            elif trend == 'Moderate Uptrend':
                score += 5
            elif trend == 'Strong Downtrend':
                score -= 10
            elif trend == 'Moderate Downtrend':
                score -= 5
                
            # Add Bollinger Band influence (max ±5)
            price_location = (current_price - bb_data['lower']) / (bb_data['upper'] - bb_data['lower'])
            if price_location > 0.8:
                score -= 3  # Overbought
            elif price_location < 0.2:
                score += 3  # Oversold
                
            # Add squeeze detection
            bandwidth = bb_data['bandwidth']
            if bandwidth < 10:  # Low bandwidth indicates potential squeeze
                score += 2  # Potential breakout
                
            # Support/Resistance influence (max ±5)
            if current_price > support_resistance['nearest_resistance']:
                score += 5  # Breakout
            elif current_price < support_resistance['nearest_support']:
                score -= 5  # Breakdown
                
            return min(max(score, -30), 30)  # Cap at ±30
            
        except Exception as e:
            self.log(f"Error calculating trend score: {str(e)}", level="error")
            return 0

    def _calculate_volume_score(self, volume_data: Dict[str, Any], market_conditions: Dict[str, Any]) -> float:
        """Calculate volume score with reduced weights"""
        try:
            base_score = 0
            volume_ratio = volume_data['volume_ratio']
            
            # Reduced scoring
            if volume_ratio > 2.0:
                base_score = 20  # Reduced from 40
            elif volume_ratio > 1.5:
                base_score = 15  # Reduced from 30
            elif volume_ratio > 1.2:
                base_score = 10  # Reduced from 20
            elif volume_ratio > 1.0:
                base_score = 5   # Reduced from 10
            elif volume_ratio < 0.5:
                base_score = -15 # Changed from -30
            elif volume_ratio < 0.8:
                base_score = -10 # Changed from -20
                
            # Apply modifiers
            if volume_data['confirms_trend']:
                base_score *= 1.1
            if market_conditions['is_high_activity']:
                base_score *= 0.9
                
            # Cap at ±20 (reduced from ±40)
            return min(max(base_score, -20), 20)
            
        except Exception as e:
            self.log(f"Error calculating volume score: {str(e)}", level="error")
            return 0

    def _calculate_momentum_score(self, sentiment: Dict[str, Any], ma_data: Dict[str, Any]) -> float:
        """Calculate momentum score with standardized weights"""
        try:
            score = 0
            
            # Price momentum (max ±15 from timeframes)
            if sentiment['momentum']['short_term'] == 'bullish':
                score += 7  # Short term highest weight
            elif sentiment['momentum']['short_term'] == 'bearish':
                score -= 7
                
            if sentiment['momentum']['medium_term'] == 'bullish':
                score += 5  # Medium term middle weight
            elif sentiment['momentum']['medium_term'] == 'bearish':
                score -= 5
                
            if sentiment['momentum']['long_term'] == 'bullish':
                score += 3  # Long term lowest weight
            elif sentiment['momentum']['long_term'] == 'bearish':
                score -= 3
                
            # RSI influence (max ±10)
            if 'rsi' in ma_data:
                rsi = ma_data['rsi']
                if rsi > 70:
                    score -= 10  # Overbought
                elif rsi > 60:
                    score -= 5   # Approaching overbought
                elif rsi < 30:
                    score += 10  # Oversold
                elif rsi < 40:
                    score += 5   # Approaching oversold
                
            # Cap at ±25 to align with other components
            return min(max(score, -25), 25)
            
        except Exception as e:
            self.log(f"Error calculating momentum score: {str(e)}", level="error")
            return 0

    def _calculate_risk_score(self, market_conditions: Dict[str, Any], sentiment: Dict[str, Any]) -> float:
        """Calculate risk score with standardized weights"""
        try:
            score = 0
            
            # Volatility impact (max ±6)
            volatility = market_conditions['price_range_7d']
            if volatility > 20:
                score -= 6
            elif volatility > 15:
                score -= 4
            elif volatility > 10:
                score -= 2
            elif volatility < 5:
                score += 2
                
            # Market alignment (max ±4)
            if market_conditions['market_aligned']:
                btc_correlation = market_conditions.get('btc_correlation', 0)
                score += min(abs(btc_correlation) * 4, 4)
            else:
                score -= 2
            
            # Trading conditions (max ±3)
            if market_conditions['suitable_for_trading']:
                score += 3
            if market_conditions['is_high_activity']:
                score -= 2
                
            # Sentiment alignment (max ±2)
            sentiment_score = sentiment.get('sentiment_score', 0)
            if abs(sentiment_score) > 50:
                score += 2 if sentiment_score > 0 else -2
                
            # Cap at ±15 to align with other components
            return min(max(score, -15), 15)
            
        except Exception as e:
            self.log(f"Error calculating risk score: {str(e)}", level="error")
            return 0

    def _determine_trend(self, current_price: float, sma20: float, sma50: float, sma200: float) -> str:
        """Determine trend based on price vs moving averages"""
        if current_price > sma20 and current_price > sma50 and current_price > sma200:
            return "Strong Uptrend"
        elif current_price > sma50 and current_price > sma200:
            return "Moderate Uptrend"
        elif current_price < sma20 and current_price < sma50 and current_price < sma200:
            return "Strong Downtrend"
        elif current_price < sma50 and current_price < sma200:
            return "Moderate Downtrend"
        else:
            return "Mixed Trend"

    async def _execute_exit(self, symbol: str, reason: str, is_paper: bool) -> None:
        """Centralized exit execution"""
        try:
            current_price = float(self.client.get_product(f"{symbol}-USD").price)  # Add this line
            
            if is_paper:
                await self._simulate_sell_order(symbol)
            else:
                self._place_sell_order(symbol)
                
            position = self.paper_positions.get(symbol) if is_paper else self.positions.get(symbol)
            if position:
                profit_info = position.calculate_profit(current_price)
                await self.send_trade_notification(
                    'SELL', symbol, current_price,
                    position.quantity, is_paper=is_paper,
                    profit_info=profit_info,
                    reason=reason
                )
        except Exception as e:
            self.log(f"Error executing exit for {symbol}: {str(e)}", level="error")
            raise  # Add this to propagate errors to calling functions

    async def _execute_trade(self, symbol: str, action: str, reason: str = None) -> bool:
        """Centralized trade execution with price caching"""
        try:
            # Get price data from cache if possible
            prices = self._get_cached_price_data(symbol, days=1)
            current_price = float(prices.iloc[-1])
            
            if action == 'BUY':
                position_size = self._calculate_position_size(symbol, current_price)
                if position_size < 5.0:
                    self.log(f"Position size too small for {symbol}: ${position_size:.2f}")
                    return False
                
                original_trade_amount = self.trade_amount
                self.trade_amount = position_size
                
                try:
                    if self.paper_trading:
                        await self._simulate_buy_order(symbol, current_price)
                    else:
                        self._place_buy_order(symbol, current_price)
                    
                    await self.send_trade_notification(
                        'BUY', symbol, current_price,
                        position_size / current_price,
                        is_paper=self.paper_trading,
                        reason=reason
                    )
                    return True
                
                finally:
                    self.trade_amount = original_trade_amount
            
            elif action == 'SELL':
                position = self.paper_positions.get(symbol) if self.paper_trading else self.positions.get(symbol)
                if not position:
                    self.log(f"No position found for {symbol}, cannot sell")
                    return False
                
                profit_info = position.calculate_profit(current_price)
                
                if self.paper_trading:
                    await self._simulate_sell_order(symbol, current_price)
                else:
                    self._place_sell_order(symbol, current_price)
                
                await self.send_trade_notification(
                    'SELL', symbol, current_price,
                    position.quantity,
                    is_paper=self.paper_trading,
                    profit_info=profit_info,
                    reason=reason
                )
                return True
            
            return False
            
        except Exception as e:
            self.log(f"Error executing {action} trade for {symbol}: {str(e)}", level="error")
            return False

    async def _analyze_technical_indicators(self, symbol: str) -> Dict[str, Any]:
        """Unified technical analysis using cached data"""
        try:
            # Use PriceManager instead of direct methods
            prices = await self.price_manager.get_price(symbol)
            current_price = float(prices.iloc[-1])
            
            # Calculate RSI
            delta = prices.diff()
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            avg_gain = gains.ewm(span=self.rsi_period, adjust=False).mean()
            avg_loss = losses.ewm(span=self.rsi_period, adjust=False).mean()
            rs = avg_gain / avg_loss
            rsi = float(100 - (100 / (1 + rs)).iloc[-1])  # Fixed missing parenthesis
            
            # Calculate Moving Averages
            sma_20 = prices.rolling(window=20).mean()
            sma_50 = prices.rolling(window=50).mean()
            sma_200 = prices.rolling(window=200).mean()
            
            # Calculate Bollinger Bands
            bb_sma = prices.rolling(window=20).mean()
            bb_std = prices.rolling(window=20).std()
            bb_data = {
                'middle': float(bb_sma.iloc[-1]),
                'upper': float(bb_sma.iloc[-1] + (bb_std.iloc[-1] * 2)),
                'lower': float(bb_sma.iloc[-1] - (bb_std.iloc[-1] * 2)),
                'bandwidth': float((bb_std.iloc[-1] * 4 / bb_sma.iloc[-1]) * 100)
            }
            
            # Calculate Support/Resistance
            sr_levels = self._calculate_support_resistance_levels(prices, current_price)
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'moving_averages': {
                    'sma_20': float(sma_20.iloc[-1]),
                    'sma_50': float(sma_50.iloc[-1]),
                    'sma_200': float(sma_200.iloc[-1])
                },
                'bollinger_bands': bb_data,
                'support_resistance': sr_levels,
                'rsi': rsi
            }
        except Exception as e:
            self.log(f"Error in technical analysis for {symbol}: {str(e)}", level="error")
            raise

    def _calculate_support_resistance_levels(self, prices: pd.Series, current_price: float) -> Dict[str, Any]:
        """Calculate support/resistance using provided price data"""
        try:
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

            pivot_highs, pivot_lows = find_pivots(prices)
            
            supports = [p for p in pivot_lows if p < current_price]
            resistances = [p for p in pivot_highs if p > current_price]
            
            nearest_support = max(supports) if supports else current_price * 0.85
            nearest_resistance = min(resistances) if resistances else current_price * 1.15
            
            return {
                'nearest_support': nearest_support,
                'nearest_resistance': nearest_resistance,
                'all_supports': sorted(supports, reverse=True)[:3],
                'all_resistances': sorted(resistances)[:3],
                'support_strength': len([p for p in pivot_lows if abs(p - nearest_support) / nearest_support < 0.02])
            }
            
        except Exception as e:
            self.log(f"Error calculating support/resistance levels: {str(e)}", level="error")
            raise

    def _cleanup_cache(self):
        """Remove least recently used items from cache"""
        if len(self._price_cache) <= self._cache_max_size / 2:
            return
        
        # Sort by last access time
        sorted_cache = sorted(
            self._price_cache.items(),
            key=lambda x: x[1]['last_access']
        )
        
        # Remove oldest items until we're at 50% capacity
        items_to_remove = len(sorted_cache) - (self._cache_max_size // 2)
        for i in range(items_to_remove):
            del self._price_cache[sorted_cache[i][0]]

    async def cleanup(self):
        """Ensure proper cleanup of resources"""
        try:
            # Stop any active trading
            if self.trading_active or self.paper_trading:
                await self.stop_trading_loop()
                
            # Close price manager
            if hasattr(self, 'price_manager'):
                await self.price_manager.close()
                
            # Close client connection
            if hasattr(self, 'client'):
                # Note: RESTClient may not have async close
                self.client = None
                
            # Clear caches
            if hasattr(self, '_price_cache'):
                self._price_cache.clear()
                
        except Exception as e:
            await self.log(f"Error during cleanup: {str(e)}", level="error")

    def _validate_symbol(self, symbol: str) -> bool:
        """Validate cryptocurrency symbol format and existence."""
        try:
            if not isinstance(symbol, str):
                self.sync_log(f"Invalid symbol type: {type(symbol)}", level="warning")
                return False
                
            symbol = symbol.upper()
            if not symbol.isalnum():
                self.sync_log(f"Invalid symbol format: {symbol}", level="warning")
                return False
                
            # Check if symbol exists on Coinbase
            try:
                self.client.get_product(f"{symbol}-USD")
                return True
            except Exception as e:
                self.sync_log(f"Symbol validation failed for {symbol}: {str(e)}", level="warning")
                return False
                
        except Exception as e:
            self.sync_log(f"Symbol validation error: {str(e)}", level="error")
            return False

    async def add_coin(self, symbol: str) -> bool:
        """Add a coin to watchlist with validation"""
        try:
            symbol = symbol.upper()
            if not self._validate_symbol(symbol):
                return False
            self.watched_coins.add(symbol)
            logging.info(f"Added {symbol} to watchlist")
            return True
        except Exception as e:
            logging.error(f"Failed to add {symbol}: {str(e)}")
            return False

    def sync_log(self, message: str, level: str = "info") -> None:
        """
        Synchronous logging for non-async contexts.
        
        Args:
            message: The message to log
            level: Log level (info, warning, error)
        """
        if level == "error":
            logging.error(message)
        elif level == "warning":
            logging.warning(message)
        else:
            logging.info(message)

    async def _heartbeat(self):
        """Periodic heartbeat to ensure bot is running"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self.log("Bot heartbeat", level="info")
            except Exception as e:
                logging.error(f"Heartbeat error: {str(e)}")
                await asyncio.sleep(5)  # Wait before retry

    async def _record_closed_position(self, position: Position, exit_price: float, profit_info: Dict[str, Any]) -> None:
        """Record a closed position in history"""
        self.position_history.append({
            'symbol': position.symbol,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'quantity': position.original_quantity,
            'entry_time': position.entry_time,
            'exit_time': datetime.now(),
            'profit_usd': profit_info['profit_usd'],
            'profit_percentage': profit_info['profit_percentage'],
            'max_profit_percentage': profit_info['highest_profit_percentage'],
            'max_drawdown': profit_info['drawdown_percentage'],
            'partial_exits_taken': position.partial_exit_taken
        })

    async def _record_trade(self, symbol: str, action: str, quantity: float, price: float, 
                        profit_info: Optional[Dict[str, Any]] = None, partial: bool = False) -> None:
        """Record a trade in history"""
        trade_record = {
            'timestamp': datetime.now(),
            'action': 'PARTIAL_SELL' if partial else action,
            'symbol': symbol,
            'amount_usd': quantity * price,
            'price': price,
            'quantity': quantity
        }
        
        if profit_info and action == 'SELL':
            trade_record.update({
                'profit': profit_info['profit_usd'],
                'profit_percentage': profit_info['profit_percentage']
            })
        
        self.trade_history.append(trade_record)

class PositionManager:
    """Centralized position management"""
    def __init__(self, trading_bot):
        self.trading_bot = trading_bot
        self.positions = {}
        self.position_lock = asyncio.Lock()
        
    async def open_position(self, symbol: str, entry_price: float, quantity: float) -> Position:
        async with self.position_lock:
            position = Position(
                self.trading_bot, 
                symbol, 
                entry_price, 
                quantity,
                datetime.now(),
                self.trading_bot.paper_trading
            )
            self.positions[symbol] = position
            return position

    async def close_position(self, symbol: str, exit_price: float, reason: str) -> bool:
        """Close a position and record its history"""
        async with self.position_lock:
            if symbol not in self.positions:
                return False
            position = self.positions[symbol]
            profit_info = position.calculate_profit(exit_price)
            self.history.append({
                'symbol': symbol,
                'entry_price': position.entry_price,
                'exit_price': exit_price,
                'quantity': position.quantity,
                'profit': profit_info,
                'reason': reason,
                'timestamp': datetime.now()
            })
            del self.positions[symbol]
            return True

    async def update_positions(self) -> None:
        """Update all position metrics and check risk management"""
        async with self.position_lock:
            for symbol, position in list(self.positions.items()):
                try:
                    # Get current price synchronously since we're using the client directly
                    current_price = float(self.trading_bot.client.get_product(f"{symbol}-USD").price)
                    
                    # Update position metrics
                    await position.update_metrics(current_price)
                    
                    # Calculate profit info for checks
                    profit_info = position.calculate_profit(current_price)
                    
                    # Check stop loss
                    if profit_info['profit_percentage'] <= -self.trading_bot.stop_loss_percentage:
                        await self.trading_bot.log(f"Stop loss triggered for {symbol}")
                        await self.trading_bot.trade_executor.execute_trade(
                            symbol=symbol,
                            action='SELL',
                            reason="Stop Loss"
                        )
                        continue
                    
                    # Check trailing stop
                    if position.should_trigger_trailing_stop(current_price):
                        await self.trading_bot.log(f"Trailing stop triggered for {symbol}")
                        await self.trading_bot.trade_executor.execute_trade(
                            symbol=symbol,
                            action='SELL',
                            reason="Trailing Stop"
                        )
                        continue
                        
                    # Check take profit
                    if profit_info['profit_percentage'] >= self.trading_bot.take_profit_percentage:
                        await self.trading_bot.log(f"Take profit triggered for {symbol}")
                        await self.trading_bot.trade_executor.execute_trade(
                            symbol=symbol,
                            action='SELL',
                            reason="Take Profit"
                        )
                        
                except Exception as e:
                    await self.trading_bot.log(f"Error updating position for {symbol}: {str(e)}", level="error")

    async def check_stops(self) -> None:
        """Check and execute stop orders"""
        async with self.position_lock:
            for symbol, position in list(self.positions.items()):
                try:
                    current_price = await self.trading_bot.price_manager.get_price(symbol)
                    
                    # Check stop conditions
                    if (position.should_trigger_trailing_stop(current_price) or
                       position.should_trigger_stop_loss(current_price)):
                        await self.trading_bot.trade_executor.execute_trade(
                            symbol=symbol,
                            action='SELL',
                            reason="Stop Triggered"
                        )
                        
                except Exception as e:
                    await self.trading_bot.log(f"Error checking stops for {symbol}: {str(e)}", level="error")

class SignalGenerator:
    """Centralized signal generation"""
    def __init__(self, trading_bot):
        self.trading_bot = trading_bot
        
    async def generate_signal(self, symbol: str) -> Dict[str, Any]:
        """Generate trading signal with technical and market analysis"""
        try:
            prices = await self.trading_bot.price_manager.get_price(symbol)
            technical_data = await self._analyze_technical(prices)
            market_data = await self._analyze_market(symbol)
            return await self._calculate_final_signal(technical_data, market_data)
        except Exception as e:
            await self.trading_bot.log(f"Signal generation error: {str(e)}", level="error")
            return self._get_fallback_signal(symbol, str(e))

    async def _analyze_technical(self, prices: pd.Series) -> Dict[str, Any]:
        """Analyze technical indicators"""
        try:
            # Calculate RSI
            delta = prices.diff()
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            avg_gain = gains.ewm(span=14, adjust=False).mean()
            avg_loss = losses.ewm(span=14, adjust=False).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
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
            await self.trading_bot.log(f"Technical analysis error: {str(e)}", level="error")
            raise

    async def _analyze_market(self, symbol: str) -> Dict[str, Any]:
        """Analyze market conditions with proper async handling"""
        try:
            # Get volume data - already async
            volume_data = await self.trading_bot.analyze_volume(symbol)
            
            # Get market sentiment - already async
            sentiment = await self.trading_bot.analyze_market_sentiment(symbol)
            
            # Get market conditions - needs to be async
            conditions = await self.trading_bot._check_market_conditions(symbol)
            
            # Get correlation with BTC - needs to be async
            correlation = await self._calculate_btc_correlation(symbol)
            
            return {
                'volume': volume_data,
                'sentiment': sentiment,
                'conditions': conditions,
                'btc_correlation': correlation
            }
            
        except Exception as e:
            await self.trading_bot.log(f"Market analysis error: {str(e)}", level="error")
            raise

    async def _calculate_final_signal(self, technical: Dict, market: Dict) -> Dict[str, Any]:
        """Calculate final trading signal"""
        try:
            # Calculate component scores
            trend_score = self._calculate_trend_score(technical)
            momentum_score = self._calculate_momentum_score(market['sentiment'], technical)
            volume_score = self._calculate_volume_score(market['volume'], market['conditions'])
            risk_score = self._calculate_risk_score(market['conditions'], market['sentiment'])
            
            # Calculate weighted final score
            weights = {
                'trend': 0.4,
                'momentum': 0.3,
                'volume': 0.2,
                'risk': 0.1
            }
            
            scores = {
                'trend': trend_score,
                'momentum': momentum_score,
                'volume': volume_score,
                'risk': risk_score
            }
            
            final_score = sum(score * weights[component] for component, score in scores.items())
            
            # Determine action
            action = self._determine_signal_action(final_score, scores)
            
            return {
                'action': action,
                'score': final_score,
                'signals': scores,
                'technical': technical,
                'market': market,
                'timestamp': datetime.now()
            }
        except Exception as e:
            await self.trading_bot.log(f"Signal calculation error: {str(e)}", level="error")
            raise

class TradeExecutor:
    """Centralized trade execution"""
    def __init__(self, trading_bot):
        self.trading_bot = trading_bot
        self.execution_lock = asyncio.Lock()
        self.min_position_size = 5.0

    async def execute_trade(
        self, 
        symbol: str, 
        action: Literal['BUY', 'SELL'], 
        reason: Optional[str] = None
    ) -> bool:
        async with self.execution_lock:
            try:
                current_price = await self.trading_bot.price_manager.get_current_price(symbol)
                if action == 'BUY':
                    return await self._execute_buy(symbol, current_price, reason)
                elif action == 'SELL':
                    return await self._execute_sell(symbol, current_price, reason)
                return False
            except Exception as e:
                await self.trading_bot.log(f"Trade execution error: {str(e)}", level="error")
                return False

    async def _execute_buy(self, symbol: str, price: float, reason: str) -> bool:
        """Execute buy order with position management"""
        try:
            # Calculate position size
            position_size = self.trading_bot._calculate_position_size(symbol)
            if position_size < self.min_position_size:
                await self.trading_bot.log(f"Position size too small for {symbol}: ${position_size:.2f}")
                return False

            if self.trading_bot.paper_trading:
                # Paper trading execution
                if self.trading_bot.paper_balance < position_size:
                    await self.trading_bot.log(f"Insufficient paper balance for {symbol}")
                    return False
                
                # Calculate quantity after fees
                fee = position_size * self.trading_bot.FEE_RATE
                actual_position_size = position_size - fee
                quantity = actual_position_size / price
                
                # Create paper position
                position = await self.trading_bot.position_manager.open_position(
                    symbol=symbol,
                    entry_price=price,
                    quantity=quantity
                )
                
                # Update paper balance
                self.trading_bot.paper_balance -= position_size
                
                await self.trading_bot.log(
                    f"Paper buy executed for {symbol}",
                    context={
                        'price': price,
                        'quantity': quantity,
                        'position_size': position_size,
                        'fees': fee
                    }
                )
                return True
                
            else:
                # Real trading execution
                if not self.trading_bot._check_balance(symbol, 'BUY'):
                    await self.trading_bot.log(f"Insufficient balance for {symbol}")
                    return False
                
                # Place market buy order
                order = self.trading_bot.client.create_order(
                    product_id=f"{symbol}-USD",
                    side='BUY',
                    order_configuration={
                        'market_market_ioc': {
                            'quote_size': str(position_size)
                        }
                    }
                )
                
                # Create real position from filled order
                position = await self.trading_bot.position_manager.open_position(
                    symbol=symbol,
                    entry_price=float(order.average_filled_price),
                    quantity=float(order.filled_size)
                )
                
                await self.trading_bot.log(
                    f"Real buy executed for {symbol}",
                    context={
                        'price': float(order.average_filled_price),
                        'quantity': float(order.filled_size),
                        'position_size': position_size,
                        'order_id': order.order_id
                    }
                )
                return True
                
        except Exception as e:
            await self.trading_bot.log(f"Buy execution error: {str(e)}", level="error")
            return False

    async def _execute_sell(self, symbol: str, price: float, reason: str) -> bool:
        """Execute sell order with position management"""
        try:
            position = self.trading_bot.position_manager.positions.get(symbol)
            if not position:
                await self.trading_bot.log(f"No position found for {symbol}")
                return False

            if self.trading_bot.paper_trading:
                # Calculate paper trade results
                profit_info = position.calculate_profit(price)
                
                # Update paper balance
                sell_value = position.quantity * price
                fee = sell_value * self.trading_bot.FEE_RATE
                self.trading_bot.paper_balance += (sell_value - fee)
                
                # Close position
                await self.trading_bot.position_manager.close_position(
                    symbol=symbol,
                    exit_price=price,
                    reason=reason
                )
                
                await self.trading_bot.log(
                    f"Paper sell executed for {symbol}",
                    context={
                        'price': price,
                        'quantity': position.quantity,
                        'profit': profit_info,
                        'fees': fee
                    }
                )
                return True
                
            else:
                # Real trading execution
                if not self.trading_bot._check_balance(symbol, 'SELL'):
                    await self.trading_bot.log(f"Insufficient balance for {symbol}")
                    return False
                
                # Place market sell order
                order = self.trading_bot.client.create_order(
                    product_id=f"{symbol}-USD",
                    side='SELL',
                    order_configuration={
                        'market_market_ioc': {
                            'base_size': str(position.quantity)
                        }
                    }
                )
                
                # Close position
                await self.trading_bot.position_manager.close_position(
                    symbol=symbol,
                    exit_price=float(order.average_filled_price),
                    reason=reason
                )
                
                await self.trading_bot.log(
                    f"Real sell executed for {symbol}",
                    context={
                        'price': float(order.average_filled_price),
                        'quantity': float(order.filled_size),
                        'order_id': order.order_id
                    }
                )
                return True
                
        except Exception as e:
            await self.trading_bot.log(f"Sell execution error: {str(e)}", level="error")
            return False

    @property
    def is_active(self) -> bool:
        """Check if either paper or real trading is active"""
        return self.trading_bot.trading_active or self.trading_bot.paper_trading

    async def _validate_trade(self, symbol: str, action: str, amount: float) -> bool:
        """Validate trade parameters before execution"""
        try:
            # Check symbol validity
            if not self.trading_bot._validate_symbol(symbol):
                await self.trading_bot.log(f"Invalid symbol: {symbol}", level="error")
                return False
                
            # Check trading is active
            if not self.is_active:
                await self.trading_bot.log("Trading is not active", level="error")
                return False
                
            # Check sufficient balance
            if action == 'BUY':
                if self.trading_bot.paper_trading:
                    if self.trading_bot.paper_balance < amount:
                        await self.trading_bot.log("Insufficient paper balance", level="error")
                        return False
                else:
                    balance = await self.trading_bot.get_account_balance()
                    if balance['balances'].get('USD', {}).get('balance', 0) < amount:
                        await self.trading_bot.log("Insufficient USD balance", level="error")
                        return False
                        
            return True
            
        except Exception as e:
            await self.trading_bot.log(f"Trade validation error: {str(e)}", level="error")
            return False