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
from .position import Position, PartialExit, ScaleLevel
import asyncio
import discord
import numpy as np
from bot.technical_analysis import TechnicalAnalyzer
from bot.risk_manager import RiskManager
from bot.data_manager import DataManager
from bot.exceptions import TradingError
from bot.config import TradingConfig
from bot.constants import TradingConstants

# Set up logging
logging.basicConfig(
    filename='trading_bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class MessageFormatter:
    """Handles consistent message formatting for Discord channels."""
    
    @staticmethod
    def format_notification(message: str, category: str = "info") -> str:
        """Format notification messages with emojis and proper styling."""
        emojis = {
            "success": "✅",
            "error": "🚨",
            "warning": "⚠️",
            "info": "ℹ️",
            "trade": "📊",
            "profit": "💰",
            "loss": "📉",
            "alert": "🔔"
        }
        
        return f"{emojis.get(category, 'ℹ️')} {message}"

    @staticmethod
    def format_trade_alert(symbol: str, action: str, price: float, quantity: float) -> str:
        """Format trade execution alerts."""
        emoji = "🟢" if action.lower() == "buy" else "🔴"
        return (
            f"{emoji} Trade Alert: {action.upper()} {symbol}\n"
            f"```\n"
            f"Price: ${price:,.2f}\n"
            f"Size:  {quantity:.8f} {symbol}\n"
            f"Time:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            "```"
        )

    @staticmethod
    def format_position_update(position: 'Position') -> str:
        """Format position status updates."""
        profit_emoji = "📈" if position.unrealized_pnl > 0 else "📉"
        return (
            f"{profit_emoji} Position Update: {position.symbol}\n"
            f"```\n"
            f"Entry:     ${position.entry_price:,.2f}\n"
            f"Current:   ${position.current_price:,.2f}\n"
            f"P/L:       {position.unrealized_pnl:+,.2f} USD ({(position.current_price/position.entry_price - 1)*100:+.2f}%)\n"
            f"Size:      {position.remaining_quantity:.8f} {position.symbol}\n"
            f"Stop:      ${position.stop_loss:,.2f}\n"
            f"Target:    ${position.take_profit:,.2f}\n"
            "```"
        )

    @staticmethod
    def format_error(error: str) -> str:
        """Format error messages with detailed information."""
        # Split error message if it contains multiple lines
        error_lines = error.split('\n')
        main_error = error_lines[0]
        
        # Format the error message
        formatted_lines = [
            "🚨 Error Alert",
            "```diff",
            f"- Error: {main_error}"
        ]
        
        # Add additional error details if present
        if len(error_lines) > 1:
            formatted_lines.extend([
                "# Additional Details:",
                *[f"# {line}" for line in error_lines[1:]]
            ])
            
        formatted_lines.append("```")
        return "\n".join(formatted_lines)

    @staticmethod
    def format_risk_alert(message: str, level: str = "warning") -> str:
        """Format risk management alerts."""
        emoji = "🚨" if level == "critical" else "⚠️"
        return (
            f"{emoji} Risk Alert\n"
            "```yaml\n"
            f"Level: {level.upper()}\n"
            f"Alert: {message}\n"
            "```"
        )

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

class RateLimiter:
    """Rate limiter for API calls"""
    
    def __init__(self, max_requests: int, time_window: float):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Wait if necessary and acquire rate limit slot"""
        async with self._lock:
            now = time.time()
            
            # Remove old requests
            self.requests = [req for req in self.requests if now - req < self.time_window]
            
            # Wait if at limit
            if len(self.requests) >= self.max_requests:
                wait_time = self.requests[0] + self.time_window - now
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                self.requests = self.requests[1:]
            
            # Add new request
            self.requests.append(now)

class TradingBot:
    def __init__(self, client: RESTClient, config: TradingConfig):
        """Initialize the trading bot with client and config"""
        self.client = client
        self.config = config
        self.start_time = datetime.now()  # Add start time tracking
        
        # Initialize components
        self.data_manager = DataManager(self)
        self.price_manager = PriceManager(
            client=client,
            cache_size=config.CACHE_SIZE,
            cache_ttl=config.CACHE_TTL['1d'],
            rate_limit=config.RATE_LIMIT,
            log_callback=self.log
        )
        
        # Initialize other components
        self.fee_rate = self.config.EXCHANGE_FEE
        
        # Initialize analyzers that depend on data_manager
        self.technical_analyzer = TechnicalAnalyzer(self)
        self.risk_manager = RiskManager(self)
        
        # Core state
        self.positions = {}
        self.trading_active = False
        self.paper_trading = True
        self.watched_symbols = set()
        self.last_trade_time = None  # Track last trade timestamp

        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.position_history: List[Dict[str, Any]] = []
        self.closed_positions: List[Dict[str, Any]] = []

        # Trading state
        self.trailing_stop_enabled = config.TRAILING_STOP_ENABLED
        self.position_scaling_enabled = True  # Can be made configurable
        self.auto_risk_enabled = True  # Can be made configurable

        self.message_formatter = MessageFormatter()

        # Add rate limiter
        self.rate_limiter = RateLimiter(
            max_requests=TradingConstants.MAX_REQUESTS_PER_SECOND,
            time_window=1.0
        )

    def get_uptime(self) -> str:
        """Calculate and format the bot's uptime."""
        uptime = datetime.now() - self.start_time
        days = uptime.days
        hours, remainder = divmod(uptime.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        parts.append(f"{seconds}s")
        
        return " ".join(parts)

    async def analyze_symbol(self, symbol: str) -> Dict[str, Any]:
        """Analyze trading symbol"""
        try:
            # Get technical analysis
            analysis = await self.technical_analyzer.analyze(symbol)
            
            # Get current price
            current_price = await self.data_manager.get_current_price(symbol)
            
            return {
                'symbol': symbol,
                'price': current_price,
                'analysis': analysis,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            await self.log(f"Symbol analysis error: {str(e)}", level="error")
            raise TradingError(f"Failed to analyze symbol: {str(e)}", "ANALYSIS")

    async def execute_scale_in(self, symbol: str, quantity: float, price: float) -> bool:
        """
        Execute a scale-in order for an existing position.
        
        Args:
            symbol: Trading pair symbol
            quantity: Amount to add to position
            price: Target price for scale-in
            
        Returns:
            bool: True if scale-in was successful
        """
        try:
            position = self.positions.get(symbol)
            if not position:
                await self.log(f"No position found for {symbol}", level="error")
                return False
                
            # Check risk limits
            if not await self.risk_manager.can_increase_position(symbol, quantity):
                await self.log(f"Risk limits prevent scaling into {symbol}", level="warning")
                return False
                
            # Execute order
            if self.paper_trading:
                success = await self._execute_paper_order(symbol, quantity, price, "buy")
            else:
                success = await self._execute_live_order(symbol, quantity, price, "buy")
                
            if success:
                await self.log(f"Successfully scaled into {symbol} position", level="info")
                return True
                
            return False
            
        except Exception as e:
            await self.log(f"Scale-in execution error: {str(e)}", level="error")
            return False

    async def execute_partial_exit(self, symbol: str, quantity: float, price: float) -> bool:
        """
        Execute a partial position exit.
        
        Args:
            symbol: Trading pair symbol
            quantity: Amount to reduce position by
            price: Target exit price
            
        Returns:
            bool: True if partial exit was successful
        """
        try:
            position = self.positions.get(symbol)
            if not position:
                await self.log(f"No position found for {symbol}", level="error")
                return False
                
            if quantity > position.remaining_quantity:
                await self.log(f"Invalid exit quantity for {symbol}", level="error")
                return False
                
            # Execute order
            if self.paper_trading:
                success = await self._execute_paper_order(symbol, quantity, price, "sell")
            else:
                success = await self._execute_live_order(symbol, quantity, price, "sell")
                
            if success:
                # Calculate profit for this exit
                exit_profit = (price - position.entry_price) * quantity
                
                # Record partial exit
                partial_exit = PartialExit(
                    price=price,
                    quantity=quantity,
                    profit=exit_profit,
                    timestamp=datetime.now()
                )
                position.partial_exits.append(partial_exit)
                position.remaining_quantity -= quantity
                
                await self.log(f"Partial exit executed for {symbol}", level="info")
                return True
                
            return False
            
        except Exception as e:
            await self.log(f"Partial exit error: {str(e)}", level="error")
            return False

    async def update_position_stops(self, symbol: str, 
                                  stop_loss: Optional[float] = None,
                                  take_profit: Optional[float] = None,
                                  trailing_stop: Optional[float] = None) -> bool:
        """
        Update position risk levels.
        
        Args:
            symbol: Trading pair symbol
            stop_loss: New stop loss price
            take_profit: New take profit price
            trailing_stop: New trailing stop distance (%)
        """
        try:
            position = self.positions.get(symbol)
            if not position:
                await self.log(f"No position found for {symbol}", level="error")
                return False
                
            if stop_loss:
                position.stop_loss = stop_loss
                
            if take_profit:
                position.take_profit = take_profit
                
            if trailing_stop is not None:
                position.trailing_stop_distance = trailing_stop / 100
                position.trailing_stop_enabled = trailing_stop > 0
                
            await self.log(f"Updated stops for {symbol} position", level="info")
            return True
            
        except Exception as e:
            await self.log(f"Stop update error: {str(e)}", level="error")
            return False

    async def close_position(self, symbol: str, price: float) -> bool:
        """Close entire position and record history"""
        try:
            position = self.positions.get(symbol)
            if not position:
                return False
                
            # Execute full position close
            success = await self.execute_partial_exit(symbol, position.remaining_quantity, price)
            
            if success:
                # Record position history
                history_entry = {
                    'symbol': symbol,
                    'entry_price': position.entry_price,
                    'exit_price': price,
                    'initial_quantity': position.initial_quantity,
                    'partial_exits': [exit.__dict__ for exit in position.partial_exits],
                    'scale_levels': [level.__dict__ for level in position.scale_levels],
                    'entry_time': position.entry_time,
                    'exit_time': datetime.now(),
                    'total_profit': sum(exit.profit for exit in position.partial_exits) + 
                                  (price - position.entry_price) * position.remaining_quantity
                }
                
                self.position_history.append(history_entry)
                del self.positions[symbol]
                
                await self.log(f"Position closed for {symbol}", level="info")
                return True
                
            return False
            
        except Exception as e:
            await self.log(f"Position close error: {str(e)}", level="error")
            return False

    async def trading_loop(self):
        """Main trading loop - runs every TRADING_INTERVAL seconds"""
        while self.trading_active:
            try:
                # 1. Update Global State
                await self.update_account_state()
                await self.update_positions()
                
                # 2. Risk Checks
                await self.risk_manager.check_portfolio_health()
                
                # 3. Position Management
                for symbol, position in self.positions.items():
                    await self._manage_position(symbol)
                
                # 4. Entry Analysis
                for symbol in self.watched_symbols:
                    if await self._should_enter_position(symbol):
                        await self._execute_entry(symbol)
                        
            except Exception as e:
                await self.log(f"Trading loop error: {str(e)}", level="error")
                
            finally:
                await asyncio.sleep(self.config.TRADING_INTERVAL)

    async def _manage_position(self, symbol: str) -> None:
        """Comprehensive position management"""
        position = self.positions[symbol]
        current_price = await self.data_manager.get_current_price(symbol)
        
        # 1. Update metrics
        await position.update_metrics(current_price)
        
        # 2. Check exit conditions
        if await self._should_exit_position(position):
            await self.close_position(symbol, current_price)
            return
            
        # 3. Check partial exit conditions
        if await self._should_take_partial_profit(position):
            await self._execute_partial_exit(position)
            
        # 4. Update stops
        await self._update_position_stops(position)

    async def update_account_state(self) -> None:
        """Update global account state and metrics"""
        try:
            # Update account balance
            if self.paper_trading:
                self.account_value = await self._calculate_paper_account_value()
            else:
                self.account_value = await self._get_live_account_value()
            
            # Update daily metrics
            self.daily_pnl = await self._calculate_daily_pnl()
            self.total_exposure = await self._calculate_total_exposure()
            
            # Log state update
            await self.log(
                f"Account State Updated - Value: ${self.account_value:.2f}, "
                f"Daily P/L: ${self.daily_pnl:.2f}, "
                f"Exposure: {self.total_exposure*100:.1f}%",
                level="debug"
            )
            
        except Exception as e:
            await self.log(f"Account state update error: {str(e)}", level="error")
            raise TradingError(f"Failed to update account state: {str(e)}", "STATE")

    async def update_positions(self) -> None:
        """Update all position metrics and states"""
        try:
            for symbol, position in self.positions.items():
                current_price = await self.data_manager.get_current_price(symbol)
                await position.update_metrics(current_price)
                
        except Exception as e:
            await self.log(f"Position update error: {str(e)}", level="error")
            raise TradingError(f"Failed to update positions: {str(e)}", "STATE")

    async def _should_enter_position(self, symbol: str) -> bool:
        """Enhanced position entry logic with position count consideration"""
        try:
            current_positions = len(self.positions)
            
            # Basic entry conditions
            analysis = await self.technical_analyzer.analyze_trend(symbol)
            current_price = await self.data_manager.get_current_price(symbol)
            
            # Adjust entry criteria based on position count
            if current_positions >= self.risk_manager.target_positions:
                # Require stronger signals above target
                if not (analysis['trend']['aligned'] and 
                       analysis['trend']['daily'] > 0.5 and  # Stronger trend required
                       analysis['volume_confirmed']):
                    return False
            else:
                # Normal entry criteria
                if not (analysis['trend']['aligned'] and 
                       analysis['trend']['daily'] > 0 and
                       analysis['volume_confirmed']):
                    return False
            
            # Check risk management
            if not await self.risk_manager.can_open_position(symbol):
                return False
            
            # Check price conditions
            price_checks = await self._check_price_conditions(symbol, current_price)
            if not price_checks['valid']:
                return False
            
            return True
            
        except Exception as e:
            await self.log(f"Entry analysis error: {str(e)}", level="error")
            return False

    async def _check_price_conditions(self, symbol: str, current_price: float) -> Dict[str, Any]:
        """Check price position relative to key levels"""
        try:
            daily_data = await self.data_manager.get_price_data(symbol, '1d')
            
            # Get key levels
            resistance = await self.technical_analyzer._get_key_levels(symbol)
            daily_low = daily_data['low'].iloc[-1]
            
            # Calculate conditions
            near_resistance = any(
                abs(current_price - level) / level < 0.02  # 2% buffer
                for level in resistance['resistance_levels']
            )
            
            up_from_low = (current_price - daily_low) / daily_low
            
            # Get RSI
            rsi = await self.technical_analyzer.calculate_rsi(daily_data['close'])
            
            return {
                'valid': not near_resistance 
                        and up_from_low < 0.7  # Not more than 70% up from daily low
                        and (
                            (40 <= rsi <= 60)  # Trending
                            or (rsi < 30)  # Oversold
                        ),
                'details': {
                    'near_resistance': near_resistance,
                    'up_from_low': up_from_low,
                    'rsi': rsi
                }
            }
            
        except Exception as e:
            await self.log(f"Price condition check error: {str(e)}", level="error")
            raise TradingError(f"Failed to check price conditions: {str(e)}", "ANALYSIS")

    async def _should_exit_position(self, position: Position) -> bool:
        """
        Check if position should be exited based on:
        1. Stop loss/take profit levels
        2. Trailing stop
        3. Trend reversal
        4. Risk management
        """
        try:
            current_price = await self.data_manager.get_current_price(position.symbol)
            
            # 1. Check stop levels
            if await position.should_exit(current_price):
                return True
            
            # 2. Check trend reversal with multiple timeframes
            signals = await self.technical_analyzer.get_signals(position.symbol)
            
            # Exit on strong trend reversal
            trend_reversal = (
                signals['trend']['daily'] < -0.5 and position.side == 'long'
                or signals['trend']['daily'] > 0.5 and position.side == 'short'
            )
            
            if trend_reversal and position.unrealized_pnl > 0:
                await self.log(f"Trend reversal exit signal for {position.symbol}", level="info")
                return True
            
            # 3. Check risk metrics
            risk_check = await self.risk_manager.check_position_risk(position)
            if not risk_check['acceptable']:
                await self.log(
                    f"Risk-based exit for {position.symbol}: {risk_check['reason']}", 
                    level="info"
                )
                return True
            
            return False
            
        except Exception as e:
            await self.log(f"Exit analysis error: {str(e)}", level="error")
            return False

    async def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        try:
            total_trades = len(self.position_history)
            if total_trades == 0:
                return {
                    'total_trades': 0,
                    'win_rate': 0.0,
                    'avg_profit': 0.0,
                    'max_drawdown': 0.0,
                    'sharpe_ratio': 0.0,
                    'active_positions': len(self.positions),
                    'closed_positions': 0
                }
            
            # Calculate win rate
            winning_trades = sum(1 for pos in self.position_history if pos['total_profit'] > 0)
            win_rate = winning_trades / total_trades
            
            # Calculate average profit
            total_profit = sum(pos['total_profit'] for pos in self.position_history)
            avg_profit = total_profit / total_trades
            
            # Calculate max drawdown
            max_drawdown = await self._calculate_max_drawdown()
            
            # Calculate Sharpe ratio
            returns = [pos['total_profit'] / pos['entry_value'] for pos in self.position_history]
            if returns:
                avg_return = sum(returns) / len(returns)
                std_dev = np.std(returns) if len(returns) > 1 else 0
                sharpe = (avg_return / std_dev) * np.sqrt(252) if std_dev > 0 else 0
            else:
                sharpe = 0
            
            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe,
                'active_positions': len(self.positions),
                'closed_positions': len(self.position_history)
            }
            
        except Exception as e:
            await self.log(f"Performance metrics calculation error: {str(e)}", level="error")
            return None

    async def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from position history"""
        try:
            if not self.position_history:
                return 0.0
                
            # Calculate running balance
            balance = self.config.PAPER_BALANCE if self.paper_trading else self.initial_balance
            peak_balance = balance
            max_drawdown = 0.0
            
            for pos in sorted(self.position_history, key=lambda x: x['entry_time']):
                balance += pos['total_profit']
                peak_balance = max(peak_balance, balance)
                drawdown = (peak_balance - balance) / peak_balance
                max_drawdown = max(max_drawdown, drawdown)
                
            return float(max_drawdown)
            
        except Exception as e:
            await self.log(f"Max drawdown calculation error: {str(e)}", level="error")
            return 0.0

    async def _calculate_daily_pnl(self) -> float:
        """Calculate daily profit/loss"""
        try:
            today = datetime.now().date()
            daily_pnl = 0.0
            
            # Add closed position P/L
            for pos in self.position_history:
                if pos['exit_time'].date() == today:
                    daily_pnl += pos['total_profit']
            
            # Add unrealized P/L from open positions
            for pos in self.positions.values():
                daily_pnl += pos.unrealized_pnl
                
            return daily_pnl
            
        except Exception as e:
            await self.log(f"Daily P/L calculation error: {str(e)}", level="error")
            return 0.0

    async def _calculate_total_exposure(self) -> float:
        """Calculate total portfolio exposure"""
        try:
            total_value = await self._calculate_paper_account_value() if self.paper_trading else await self._get_live_account_value()
            
            if not total_value:
                return 0.0
                
            position_value = sum(
                pos.current_price * pos.remaining_quantity 
                for pos in self.positions.values()
            )
            
            return position_value / total_value
            
        except Exception as e:
            await self.log(f"Exposure calculation error: {str(e)}", level="error")
            return 0.0

    async def cleanup_expired_data(self):
        """Clean up old data and maintain system health"""
        try:
            # Clean up position history (keep last 1000 trades)
            if len(self.position_history) > 1000:
                self.position_history = self.position_history[-1000:]
            
            # Clean data manager cache
            await self.data_manager.cleanup_cache()
            
            # Log cleanup
            await self.log("System cleanup completed", level="debug")
            
        except Exception as e:
            await self.log(f"Cleanup error: {str(e)}", level="error")

    async def log_trading_metrics(self):
        """Log current trading metrics"""
        try:
            metrics = await self.calculate_performance_metrics()
            
            await self.log(
                f"Trading Metrics:\n"
                f"Win Rate: {metrics['win_rate']*100:.1f}%\n"
                f"Avg Profit: ${metrics['avg_profit']:.2f}\n"
                f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
                f"Max Drawdown: {metrics['max_drawdown']*100:.1f}%",
                level="info"
            )
            
        except Exception as e:
            await self.log(f"Metrics logging error: {str(e)}", level="error")

    async def _handle_connection_error(self):
        """Handle network disconnection"""
        await self.log("Connection lost, attempting reconnect...", level="error")
        # Implement retry logic

    async def _send_position_updates(self):
        """Send position status to Discord"""
        for symbol, position in self.positions.items():
            await self.send_notification(f"Position Update: {position.get_status()}")

    async def post_init(self) -> None:
        """
        Post-initialization setup that runs after Discord bot is ready.
        Handles channel setup and initial state configuration.
        """
        try:
            # Initialize Discord channels
            self.notification_channel = None
            self.logs_channel = None
            
            # Set initial trading state
            self.trading_active = False
            self.paper_trading = True
            self.watched_symbols = set()
            
            # Initialize data structures
            self.positions = {}
            self.position_history = []
            self.closed_positions = []
            
            # Log initialization
            await self.log("Trading bot post-initialization complete", level="info")
            
        except Exception as e:
            logging.error(f"Post-initialization error: {str(e)}")
            raise TradingError(f"Failed to complete post-initialization: {str(e)}", "INIT")

    def set_discord_channel(self, channel) -> None:
        """Set the Discord notification channel"""
        self.notification_channel = channel

    def set_logs_channel(self, channel) -> None:
        """Set the Discord logs channel"""
        self.logs_channel = channel

    async def send_notification(self, message: str, category: str = "info") -> None:
        """Send a formatted notification to the Discord channel."""
        try:
            if self.notification_channel:
                formatted_message = self.message_formatter.format_notification(message, category)
                await self.notification_channel.send(formatted_message)
        except Exception as e:
            logging.error(f"Failed to send notification: {str(e)}")

    async def log(self, message: str, level: str = "info") -> None:
        """Log a message and optionally send to Discord logs channel."""
        try:
            # Standard logging
            log_func = getattr(logging, level.lower())
            log_func(message)
            
            # Discord logging with prettier formatting
            if self.logs_channel and level in ["error", "warning", "info"]:
                if level == "error":
                    formatted_message = self.message_formatter.format_error(message)
                elif level == "warning":
                    formatted_message = self.message_formatter.format_risk_alert(message, "warning")
                else:
                    formatted_message = self.message_formatter.format_notification(message, "info")
                    
                await self.logs_channel.send(formatted_message)
                
        except Exception as e:
            logging.error(f"Logging error: {str(e)}")

    async def get_status(self) -> str:
        """Get current bot status and configuration"""
        try:
            status = "🟢 Active" if self.trading_active else "🔴 Inactive"
            mode = "📝 Paper Trading" if self.paper_trading else "💵 Live Trading"
            
            active_positions = len(self.positions)
            watched_coins = len(self.watched_symbols)
            
            # Format response with target positions
            return (
                f"Trading Bot Status:\n```"
                f"Status: {status}\n"
                f"Mode: {mode}\n"
                f"Active Positions: {active_positions}/{self.config.RISK_MAX_POSITIONS} "
                f"(Target: {TradingConstants.TARGET_POSITIONS})\n"
                f"Watched Coins: {watched_coins}\n"
                f"\nRisk Settings:\n"
                f"• Risk per Trade: {self.config.RISK_PER_TRADE*100:.1f}%\n"
                f"• Max Drawdown: {self.config.RISK_MAX_DRAWDOWN*100:.1f}%\n"
                f"• Stop Loss: {self.config.STOP_LOSS_PERCENTAGE:.1f}%\n"
                f"• Take Profit: {self.config.TAKE_PROFIT_PERCENTAGE:.1f}%"
                "```"
            )
            
        except Exception as e:
            await self.log(f"Error getting status: {str(e)}", level="error")
            return "❌ Error getting bot status"

    async def add_coin(self, symbol: str) -> bool:
        """
        Add a coin to the watchlist.
        
        Args:
            symbol: Base symbol (e.g., 'BTC', 'ETH')
            
        Returns:
            bool: True if successfully added
        """
        try:
            # Convert to uppercase and add USD pair
            symbol = symbol.upper()
            trading_pair = f"{symbol}-USD"
            
            # Validate symbol exists
            try:
                await self.data_manager.get_current_price(symbol)
            except Exception as e:
                await self.log(f"Invalid symbol {symbol}: {str(e)}", level="error")
                return False
            
            # Add to watchlist
            self.watched_symbols.add(symbol)
            await self.log(f"Added {symbol} to watchlist", level="info")
            return True
            
        except Exception as e:
            await self.log(f"Error adding {symbol}: {str(e)}", level="error")
            return False

    async def remove_coin(self, symbol: str) -> bool:
        """
        Remove a coin from the watchlist.
        
        Args:
            symbol: Base symbol (e.g., 'BTC', 'ETH')
            
        Returns:
            bool: True if successfully removed
        """
        try:
            symbol = symbol.upper()
            if symbol in self.watched_symbols:
                self.watched_symbols.remove(symbol)
                await self.log(f"Removed {symbol} from watchlist", level="info")
                return True
            return False
            
        except Exception as e:
            await self.log(f"Error removing {symbol}: {str(e)}", level="error")
            return False

    async def test_api_connection(self) -> float:
        """
        Test API connection by fetching BTC price.
        
        Returns:
            float: Current BTC price if successful
            
        Raises:
            TradingError: If API test fails
        """
        try:
            # Test authentication
            if not self.client:
                raise TradingError("API client not initialized", "API")
            
            # Test price fetch
            btc_price = await self.data_manager.get_current_price('BTC')
            
            # Log successful test
            await self.log("API connection test successful", level="info")
            
            return btc_price
            
        except Exception as e:
            await self.log(f"API test failed: {str(e)}", level="error")
            raise TradingError(f"API connection test failed: {str(e)}", "API")

    async def get_account_balance(self) -> str:
        """Get current account balance with portfolio breakdown."""
        try:
            if self.paper_trading:
                paper_value = await self._calculate_paper_account_value()
                return f"Paper Trading Balance: ${paper_value:.2f}"
            else:
                try:
                    portfolio = await self._get_live_account_value()
                    
                    # Build response string
                    response = [
                        "Portfolio Breakdown:",
                        f"Cash Balance: ${float(portfolio['cash_balance']):.2f}"  # Ensure float conversion
                    ]
                    
                    # Add crypto holdings if any exist
                    if portfolio['crypto_holdings']:
                        response.append("\nCrypto Holdings:")
                        for holding in portfolio['crypto_holdings']:
                            # Ensure float conversion for all numeric values
                            amount = float(holding['amount'])
                            usd_value = float(holding['usd_value'])
                            response.append(
                                f"{holding['currency']}: {amount:.8f} "
                                f"(${usd_value:.2f})"
                            )
                    
                    # Add total value
                    response.append(f"\nTotal Portfolio Value: ${float(portfolio['total_value']):.2f}")
                    
                    return "\n".join(response)
                    
                except KeyError as ke:
                    error_msg = f"Missing data in portfolio response: {str(ke)}"
                    await self.log(error_msg, level="error")
                    raise TradingError("Portfolio data structure error", {"details": error_msg})
                except ValueError as ve:
                    error_msg = f"Invalid numeric value in portfolio data: {str(ve)}"
                    await self.log(error_msg, level="error")
                    raise TradingError("Portfolio value conversion error", {"details": error_msg})
                
        except TradingError as te:
            # Re-raise TradingError with more context
            raise te
        except Exception as e:
            error_msg = (
                f"Failed to get account balance: {str(e)}\n"
                f"Error type: {type(e).__name__}"
            )
            await self.log(error_msg, level="error")
            raise TradingError("Account balance retrieval failed", {"error": error_msg})

    async def _calculate_paper_account_value(self) -> float:
        """Calculate paper trading account value."""
        try:
            # Start with base paper balance
            total_value = self.config.PAPER_BALANCE
            
            # Add unrealized P/L from open positions
            for position in self.positions.values():
                total_value += position.unrealized_pnl
                
            return total_value
            
        except Exception as e:
            await self.log(f"Paper account value calculation error: {str(e)}", level="error")
            return 0.0

    async def _get_live_account_value(self) -> Dict[str, Any]:
        """Get live trading account value from exchange."""
        try:
            # Get account balance using the Coinbase API
            accounts = self.client.get_accounts()
            
            # Initialize portfolio breakdown
            portfolio = {
                'cash_balance': 0.0,
                'crypto_holdings': [],
                'total_value': 0.0
            }
            
            # Process all accounts
            for acc in accounts.accounts:
                if acc.available_balance and 'value' in acc.available_balance:
                    # Convert string balance to float before processing
                    balance = float(acc.available_balance['value'])
                    if balance > 0:  # Only process accounts with non-zero balance
                        if acc.currency == 'USD':
                            portfolio['cash_balance'] = balance
                            portfolio['total_value'] += balance
                        elif acc.type == 'ACCOUNT_TYPE_CRYPTO':
                            # Get current price for crypto
                            try:
                                current_price = await self.data_manager.get_current_price(acc.currency)
                                usd_value = balance * current_price
                                portfolio['crypto_holdings'].append({
                                    'currency': acc.currency,
                                    'amount': balance,
                                    'usd_value': usd_value
                                })
                                portfolio['total_value'] += usd_value
                            except Exception as e:
                                await self.log(f"Error getting price for {acc.currency}: {str(e)}", level="error")
            
            return portfolio
            
        except Exception as e:
            await self.log(f"Live account value fetch error: {str(e)}", level="error")
            raise TradingError("Failed to get live account value", {"error": str(e)})

    async def get_daily_pnl(self) -> float:
        """Get daily profit/loss."""
        try:
            return await self._calculate_daily_pnl()
        except Exception as e:
            await self.log(f"Failed to get daily PnL: {str(e)}", level="error")
            return 0.0

    async def get_total_exposure(self) -> float:
        """Get total portfolio exposure as a percentage."""
        try:
            return await self._calculate_total_exposure()
        except Exception as e:
            await self.log(f"Failed to get total exposure: {str(e)}", level="error")
            return 0.0

    async def get_trading_stats(self) -> Dict[str, Any]:
        """Get comprehensive trading statistics."""
        try:
            total_trades = len(self.position_history)
            if total_trades == 0:
                return {
                    'total_trades': 0,
                    'win_rate': 0.0,
                    'avg_profit': 0.0,
                    'max_drawdown': 0.0,
                    'sharpe_ratio': 0.0,
                    'active_positions': len(self.positions),
                    'closed_positions': 0
                }
            
            # Calculate win rate
            winning_trades = sum(1 for pos in self.position_history if pos['total_profit'] > 0)
            win_rate = winning_trades / total_trades
            
            # Calculate average profit
            total_profit = sum(pos['total_profit'] for pos in self.position_history)
            avg_profit = total_profit / total_trades
            
            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'max_drawdown': await self._calculate_max_drawdown(),
                'sharpe_ratio': await self._calculate_sharpe_ratio(),
                'active_positions': len(self.positions),
                'closed_positions': len(self.position_history)
            }
            
        except Exception as e:
            await self.log(f"Failed to get trading stats: {str(e)}", level="error")
            raise TradingError("Failed to get trading statistics", {"error": str(e)})

    async def _calculate_trade_signal(self, symbol: str) -> Dict[str, Any]:
        """Calculate comprehensive trade signal with component scores."""
        try:
            # Get technical analysis
            analysis = await self.technical_analyzer.get_signals(symbol)
            current_price = await self.data_manager.get_current_price(symbol)
            
            # Component signals (normalized to -1 to 1 scale)
            trend_signal = analysis['trend']['daily'] * 0.4  # 40% weight
            momentum_signal = analysis['signals']['daily']['momentum'] * 0.3  # 30% weight
            volume_signal = analysis['volume_confirmed'] * 0.2  # 20% weight
            
            # Risk component based on volatility and market conditions
            risk_score = await self._calculate_risk_score(symbol)
            risk_signal = risk_score * 0.1  # 10% weight
            
            # Combine signals
            total_score = (
                trend_signal +
                momentum_signal +
                volume_signal +
                risk_signal
            )
            
            return {
                'symbol': symbol,
                'price': current_price,
                'action': 'buy' if total_score > 0.2 else 'sell' if total_score < -0.2 else 'hold',
                'score': total_score,
                'signals': {
                    'trend': trend_signal,
                    'momentum': momentum_signal,
                    'volume': volume_signal,
                    'risk': risk_signal
                }
            }
            
        except Exception as e:
            await self.log(f"Signal calculation error: {str(e)}", level="error")
            raise TradingError(f"Failed to calculate trade signal: {str(e)}", "ANALYSIS")

    async def analyze_market_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Analyze market sentiment using multiple indicators."""
        try:
            # Get technical analysis data
            analysis = await self.technical_analyzer.get_signals(symbol)
            
            # Calculate sentiment score (-10 to 10 scale)
            sentiment_score = (
                analysis['trend']['daily'] * 4 +  # Trend weight
                analysis['signals']['daily']['momentum'] * 3 +  # Momentum weight
                analysis['volume_confirmed'] * 2 +  # Volume weight
                (1 if analysis['trend']['aligned'] else -1)  # Timeframe alignment
            )
            
            # Determine momentum across timeframes
            momentum = {
                'short_term': 'bullish' if analysis['signals']['hourly']['momentum'] > 0 else 'bearish',
                'medium_term': 'bullish' if analysis['signals']['daily']['momentum'] > 0 else 'bearish',
                'long_term': 'bullish' if analysis['trend']['daily'] > 0 else 'bearish'
            }
            
            return {
                'overall_sentiment': 'bullish' if sentiment_score > 0 else 'bearish',
                'sentiment_score': sentiment_score,
                'momentum': momentum,
                'strength': abs(sentiment_score) / 10,  # Normalized to 0-1
                'timeframes_aligned': analysis['trend']['aligned']
            }
            
        except Exception as e:
            await self.log(f"Sentiment analysis error: {str(e)}", level="error")
            raise TradingError(f"Failed to analyze sentiment: {str(e)}", "ANALYSIS")

    async def _calculate_risk_score(self, symbol: str) -> float:
        """Calculate risk score for a symbol (-1 to 1 scale)."""
        try:
            # Get market conditions
            conditions = await self.technical_analyzer.check_market_conditions(symbol)
            
            # Risk factors
            volatility_penalty = -0.3 if conditions['is_volatile'] else 0
            market_bonus = 0.2 if conditions['market_aligned'] else 0
            activity_score = 0.2 if conditions['is_high_activity'] else -0.1
            
            # Combine risk factors
            risk_score = (
                volatility_penalty +
                market_bonus +
                activity_score
            )
            
            # Normalize to -1 to 1 range
            return max(min(risk_score, 1.0), -1.0)
            
        except Exception as e:
            await self.log(f"Risk score calculation error: {str(e)}", level="error")
            return 0.0  # Neutral score on error

    async def cleanup(self) -> None:
        """Cleanup resources and prepare for shutdown."""
        try:
            # Stop trading
            self.trading_active = False
            
            # Close all positions if in paper trading
            if self.paper_trading:
                for symbol in list(self.positions.keys()):
                    current_price = await self.data_manager.get_current_price(symbol)
                    await self.close_position(symbol, current_price)
            
            # Cleanup data manager
            await self.data_manager._clean_cache()
            
            # Close any open resources
            if hasattr(self, 'client'):
                self.client = None
            
            await self.log("Trading bot cleanup completed", level="info")
            
        except Exception as e:
            await self.log(f"Cleanup error: {str(e)}", level="error")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.post_init()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self.cleanup()
        
        if exc_type is not None:
            await self.log(f"Error during exit: {str(exc_val)}", level="error")
            return False  # Re-raise the exception
        return True

    async def execute_entry(self, symbol: str, quantity: float, signals: Dict[str, Any]) -> bool:
        """Execute entry order with proper notifications."""
        try:
            success = await self._execute_order(symbol, quantity, "buy")
            if success:
                price = await self.data_manager.get_current_price(symbol)
                await self.send_notification(
                    self.message_formatter.format_trade_alert(symbol, "buy", price, quantity),
                    category="trade"
                )
            return success
        except Exception as e:
            await self.log(f"Entry execution error: {str(e)}", level="error")
            return False

    async def execute_exit(self, symbol: str, quantity: float) -> bool:
        """Execute exit order with proper notifications."""
        try:
            success = await self._execute_order(symbol, quantity, "sell")
            if success:
                price = await self.data_manager.get_current_price(symbol)
                await self.send_notification(
                    self.message_formatter.format_trade_alert(symbol, "sell", price, quantity),
                    category="trade"
                )
            return success
        except Exception as e:
            await self.log(f"Exit execution error: {str(e)}", level="error")
            return False

    async def update_position_metrics(self, position: 'Position') -> None:
        """Update and notify position metrics."""
        try:
            await position.update_metrics(await self.data_manager.get_current_price(position.symbol))
            
            # Send position update if significant change
            if abs(position.unrealized_pnl_change) > 0.02:  # 2% change threshold
                await self.send_notification(
                    self.message_formatter.format_position_update(position),
                    category="trade"
                )
        except Exception as e:
            await self.log(f"Position metrics update error: {str(e)}", level="error")