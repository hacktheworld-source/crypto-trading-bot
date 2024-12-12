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
from bot.technical_analysis import TechnicalAnalyzer  # Add this import
from bot.risk_manager import RiskManager  # Add this import
from bot.data_manager import DataManager  # Add this import
from bot.exceptions import TradingError
from bot.config import TradingConfig as Config  # Rename to avoid conflict
import numpy as np

# Set up logging
logging.basicConfig(
    filename='trading_bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
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

class TradingBot:
    def __init__(self):
        self.config = Config()
        self.fee_rate = self.config.EXCHANGE_FEE
        
        # Initialize Coinbase client first
        self.client = RESTClient(
            api_key=self.config.COINBASE_API_KEY,
            api_secret=self.config.COINBASE_API_SECRET
        )
        
        # Core components only
        self.data_manager = DataManager(self)
        self.technical_analyzer = TechnicalAnalyzer(self)
        self.risk_manager = RiskManager(self)
        self.positions = {}
        
        # Essential state tracking
        self.trading_active = False
        self.paper_trading = True
        self.watched_symbols = set()

        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.position_history: List[Dict[str, Any]] = []
        self.closed_positions: List[Dict[str, Any]] = []

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
        """
        Determine if we should enter a position based on comprehensive analysis.
        
        Implements the entry rules from documentation:
        1. Daily trend aligned
        2. Hourly confirmation
        3. Risk checks pass
        4. Volume confirmation
        5. Price position checks
        """
        try:
            # Get technical analysis
            analysis = await self.technical_analyzer.analyze_trend(symbol)
            current_price = await self.data_manager.get_current_price(symbol)
            
            # 1. Check trend alignment
            if not analysis['trend']['aligned']:
                return False
                
            # 2. Check if daily trend is positive
            if analysis['trend']['daily'] <= 0:
                return False
                
            # 3. Risk checks
            if not await self.risk_manager.can_open_position(symbol):
                return False
                
            # 4. Volume confirmation
            if not analysis['volume_confirmed']:
                return False
                
            # 5. Price position checks
            price_checks = await self._check_price_conditions(symbol, current_price)
            if not price_checks['valid']:
                return False
                
            # All conditions met
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
        4. Time-based exits
        """
        try:
            current_price = await self.data_manager.get_current_price(position.symbol)
            
            # 1. Check stop levels
            if await position.should_exit(current_price):
                return True
            
            # 2. Check trend reversal
            analysis = await self.technical_analyzer.analyze_trend(position.symbol)
            if analysis['trend']['daily'] < 0 and position.unrealized_pnl > 0:
                await self.log(f"Trend reversal exit signal for {position.symbol}", level="info")
                return True
            
            # 3. Check holding time
            max_hold_days = self.config.MAX_POSITION_HOLD_DAYS
            if (datetime.now() - position.entry_time).days > max_hold_days:
                await self.log(f"Time-based exit for {position.symbol}", level="info")
                return True
            
            return False
            
        except Exception as e:
            await self.log(f"Exit analysis error: {str(e)}", level="error")
            return False

    async def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive trading performance metrics"""
        try:
            closed_positions = self.position_history[-100:]  # Last 100 trades
            
            if not closed_positions:
                return {}
                
            wins = sum(1 for p in closed_positions if p['total_profit'] > 0)
            total_trades = len(closed_positions)
            
            return {
                'win_rate': wins / total_trades if total_trades > 0 else 0,
                'avg_profit': sum(p['total_profit'] for p in closed_positions) / total_trades,
                'sharpe_ratio': await self._calculate_sharpe_ratio(),
                'max_drawdown': await self._calculate_max_drawdown(),
                'total_trades': total_trades
            }
            
        except Exception as e:
            await self.log(f"Metrics calculation error: {str(e)}", level="error")
            return {}

    async def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio based on position history"""
        try:
            if not self.position_history:
                return 0.0
                
            # Get daily returns
            daily_returns = []
            for pos in self.position_history[-100:]:  # Last 100 trades
                days_held = (pos['exit_time'] - pos['entry_time']).days or 1
                daily_return = (pos['total_profit'] / (pos['entry_price'] * pos['initial_quantity'])) / days_held
                daily_returns.append(daily_return)
                
            if not daily_returns:
                return 0.0
                
            # Calculate Sharpe ratio
            returns_array = np.array(daily_returns)
            avg_return = np.mean(returns_array)
            std_dev = np.std(returns_array)
            
            # Annualize (assuming daily returns)
            sharpe = (avg_return * 252) / (std_dev * np.sqrt(252))
            
            return float(sharpe)
            
        except Exception as e:
            await self.log(f"Sharpe ratio calculation error: {str(e)}", level="error")
            return 0.0

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