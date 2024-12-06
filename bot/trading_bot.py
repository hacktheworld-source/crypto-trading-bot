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
            self.paper_trading = True
            self.is_trading = False
            self.positions = {}
            self.paper_positions = {}
            self.watched_symbols = set()
            self.trade_history = []
            self.paper_trade_history = []
            self.paper_balance = 1000.0  # Default paper balance
            
            # Technical analysis parameters
            self.rsi_period = 14
            self.rsi_overbought = 70
            self.rsi_oversold = 30
            self.trading_interval = 300  # 5 minutes
            self.trade_amount = 100.0
            self.stop_loss_percentage = 5.0
            self.take_profit_percentage = 10.0
            self.max_position_size = 1000.0
            self.partial_tp_percentage = 5.0
            self.partial_tp_size = 0.5
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
            
            logging.info("Bot initialization completed successfully")
            
        except Exception as e:
            logging.error(f"Bot initialization failed: {str(e)}")
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

    async def check_market_conditions(self, symbol: str) -> Dict[str, Any]:
        """Check comprehensive market conditions"""
        try:
            # Get price data
            prices = await self.price_manager.get_cached_price_data(symbol, days=7)
            
            # Calculate volatility
            price_range = ((prices.max() - prices.min()) / prices.min()) * 100
            is_volatile = price_range > 10
            
            # Check trading hours
            current_hour = datetime.now().hour
            is_high_activity = 13 <= current_hour <= 21  # 9 AM - 5 PM EST
            
            # Get market alignment
            btc_correlation = await self._calculate_btc_correlation(symbol) if symbol != 'BTC' else 1.0
            
            return {
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
                    name="Profit Information",
                    value=f"```\n"
                          f"Profit: ${profit_info['profit_usd']:,.2f}\n"
                          f"Profit Percentage: {profit_info['profit_percentage']:.2f}%\n"
                          f"```",
                    inline=False
                )
            
            await self.discord_channel.send(embed=embed)
            
        except Exception as e:
            logging.error(f"Error sending trade notification: {str(e)}")

    async def calculate_bollinger_bands(self, symbol: str) -> Dict[str, Any]:
        """Calculate Bollinger Bands using cached price data"""
        try:
            prices = await self.price_manager.get_cached_price_data(symbol, days=20)
            sma = prices.rolling(window=20).mean()
            std = prices.rolling(window=20).std()
            
            current_sma = float(sma.iloc[-1])
            current_std = float(std.iloc[-1])
            
            return {
                'middle': current_sma,
                'upper': current_sma + (current_std * 2),
                'lower': current_sma - (current_std * 2),
                'bandwidth': float((current_std * 4 / current_sma) * 100)
            }
        except Exception as e:
            await self.log(f"Error calculating Bollinger Bands: {str(e)}", level="error")
            raise TradingError(f"Failed to calculate Bollinger Bands: {str(e)}", error_type='DATA')

    async def _log_message(self, message: str, level: str = "info", **kwargs):
        """Internal logging method"""
        if level == "error":
            logging.error(message)
        elif level == "warning":
            logging.warning(message)
        else:
            logging.info(message)

    def _determine_trend(self, current_price: float, sma_20: float, sma_50: float, sma_200: float) -> str:
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
            # Short-term trend (current price vs SMAs)
            short_term = (
                'bullish' if current_price > sma_20 else
                'bearish' if current_price < sma_20 else
                'neutral'
            )
            
            # Medium-term trend (SMA20 vs SMA50)
            medium_term = (
                'bullish' if sma_20 > sma_50 else
                'bearish' if sma_20 < sma_50 else
                'neutral'
            )
            
            # Long-term trend (SMA50 vs SMA200)
            long_term = (
                'bullish' if sma_50 > sma_200 else
                'bearish' if sma_50 < sma_200 else
                'neutral'
            )
            
            # Weight the trends (short term most important)
            if short_term == medium_term == long_term:
                return short_term
            elif short_term == medium_term:
                return short_term
            elif medium_term == long_term:
                return medium_term
            else:
                return 'neutral'  # When trends conflict
                
        except Exception as e:
            self.log(f"Error determining trend: {str(e)}", level="error")
            return 'neutral'

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
            
            # Align the timestamps
            common_dates = btc_prices.index.intersection(symbol_prices.index)
            if len(common_dates) < 2:
                return 0.0
                
            btc_returns = btc_prices[common_dates].pct_change().dropna()
            symbol_returns = symbol_prices[common_dates].pct_change().dropna()
            
            # Calculate correlation
            correlation = btc_returns.corr(symbol_returns)
            return float(correlation)
            
        except Exception as e:
            await self.log(f"Error calculating correlation: {str(e)}", level="error")
            return 0.0

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