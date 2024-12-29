from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
from coinbase.rest import RESTClient
from bot.exceptions import TradingError, DataError
from bot.constants import TimeFrame, TradingConstants
import time

class DataManager:
    """
    Centralized data management system for cryptocurrency trading.
    
    Features:
    - Multi-timeframe support
    - Efficient caching with TTL
    - Rate limiting
    - Data validation
    - Asynchronous operations
    """
    
    def __init__(self, trading_bot):
        self.trading_bot = trading_bot
        self.client = trading_bot.client
        
        # Initialize cache with separate TTLs for different timeframes
        self._cache: Dict[str, tuple[pd.DataFrame, float]] = {}
        self._cache_lock = asyncio.Lock()
        
        # Timeframe configurations with optimized settings
        self.timeframes = {
            TimeFrame.HOUR_1: {
                'days': 14,              # 14 days * 24 hours = 336 candles
                'granularity': 'ONE_HOUR',
                'min_periods': 24,       # Minimum 1 day of hourly data
                'cache_ttl': 300         # 5 minutes cache
            },
            TimeFrame.DAY_1: {
                'days': 100,             # 100 days of daily data
                'granularity': 'ONE_DAY',
                'min_periods': 20,       # Minimum 20 days
                'cache_ttl': 3600        # 1 hour cache
            },
            TimeFrame.DAY_30: {
                'days': 300,             # 300 days for reliable monthly analysis
                'granularity': 'ONE_DAY',
                'min_periods': 90,       # Minimum 3 months
                'cache_ttl': 14400       # 4 hours cache
            }
        }
        
        # Rate limiting
        self.rate_limit = 0.1  # seconds between requests
        self.last_request = 0
        
        # API lock for thread safety
        self.api_lock = asyncio.Lock()

    def _format_product_id(self, symbol: str) -> str:
        """
        Format symbol into valid Coinbase product ID.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC', 'ETH', 'BTC-USD')
            
        Returns:
            Formatted product ID (e.g., 'BTC-USD')
            
        Raises:
            DataError: If symbol is invalid
        """
        try:
            if not symbol:
                raise DataError("Symbol cannot be empty")
                
            # Clean and standardize the symbol
            symbol = symbol.upper().strip()
            
            # Remove USD suffix if present to standardize
            if symbol.endswith('-USD'):
                symbol = symbol[:-4]
            
            # Validate the cleaned symbol
            if not symbol.isalnum():
                raise DataError(f"Invalid symbol format: {symbol}")
            
            # Return standardized format
            return f"{symbol}-USD"
            
        except Exception as e:
            if isinstance(e, DataError):
                raise
            raise DataError(f"Invalid symbol: {str(e)}")

    async def get_price_data(
        self, 
        symbol: str, 
        timeframe: TimeFrame,
        force_fetch: bool = False
    ) -> pd.DataFrame:
        """
        Get price data for specified timeframe with optimized caching.
        
        Args:
            symbol: Trading pair symbol
            timeframe: TimeFrame enum value
            force_fetch: Force fetch new data ignoring cache
            
        Returns:
            DataFrame with OHLCV data
            
        Raises:
            DataError: If data cannot be fetched or is invalid
        """
        try:
            # Validate inputs
            if not isinstance(timeframe, TimeFrame):
                raise DataError(f"Invalid timeframe type: {type(timeframe)}")
            
            symbol = self._format_product_id(symbol)
            cache_key = f"{symbol}_{timeframe.name}"
            
            timeframe_config = self.timeframes.get(timeframe)
            if not timeframe_config:
                raise DataError(f"Unsupported timeframe: {timeframe}")
            
            async with self._cache_lock:
                # Check cache first if not forcing fetch
                if not force_fetch and cache_key in self._cache:
                    data, timestamp = self._cache[cache_key]
                    if (time.time() - timestamp) < timeframe_config['cache_ttl']:
                        return data
                
                # Acquire rate limit
                await self.trading_bot.rate_limiter.acquire()
                
                # Fetch new data
                data = await self._fetch_price_data(symbol, timeframe)
                
                # Validate data
                if data is None or data.empty:
                    raise DataError(f"No data returned for {symbol}")
                
                if len(data) < timeframe_config['min_periods']:
                    raise DataError(
                        f"Insufficient data points for {symbol}: "
                        f"got {len(data)}, need {timeframe_config['min_periods']}"
                    )
                
                # Update cache
                self._cache[cache_key] = (data, time.time())
                
                # Clean cache if needed
                if len(self._cache) > TradingConstants.CACHE_SIZE:
                    await self._clean_cache()
                
                return data
                
        except Exception as e:
            error_msg = f"Failed to get price data for {symbol}: {str(e)}"
            await self.trading_bot.log(error_msg, level="error")
            raise DataError(error_msg)
            
    async def _fetch_price_data(
        self, 
        symbol: str, 
        timeframe: TimeFrame
    ) -> pd.DataFrame:
        """Fetch historical price data with optimized handling."""
        try:
            # Get timeframe configuration
            timeframe_config = self.timeframes.get(timeframe)
            if not timeframe_config:
                raise DataError(f"Unsupported timeframe: {timeframe}")
            
            # Calculate time range
            end = datetime.now()
            days = timeframe_config['days']
            start = end - timedelta(days=days)
            
            # Log request details
            await self.trading_bot.log(
                f"Fetching {timeframe.name} data for {symbol} "
                f"(granularity: {timeframe_config['granularity']}, days: {days})",
                level="debug"
            )
            
            # Get candles from Coinbase with retry logic
            max_retries = 3
            retry_delay = 1.0
            
            for attempt in range(max_retries):
                try:
                    response = self.client.get_candles(
                        product_id=symbol,
                        start=int(start.timestamp()),
                        end=int(end.timestamp()),
                        granularity=timeframe_config['granularity']
                    )
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(retry_delay * (attempt + 1))
            
            if not response or not response.candles:
                raise DataError(f"No data returned for {symbol}")
            
            # Process candle data with validation
            candles_data = []
            for candle in response.candles:
                try:
                    candle_time = datetime.fromtimestamp(float(candle.start))
                    
                    # Skip invalid candles
                    if any(pd.isna(x) or pd.isinf(x) for x in [
                        float(candle.open),
                        float(candle.high),
                        float(candle.low),
                        float(candle.close),
                        float(candle.volume)
                    ]):
                        continue
                        
                    candles_data.append({
                        'timestamp': candle_time,
                        'open': float(candle.open),
                        'high': float(candle.high),
                        'low': float(candle.low),
                        'close': float(candle.close),
                        'volume': float(candle.volume)
                    })
                except (ValueError, AttributeError) as e:
                    await self.trading_bot.log(
                        f"Error processing candle: {str(e)}", 
                        level="error"
                    )
                    continue
            
            if not candles_data:
                raise DataError(f"Failed to process any candle data for {symbol}")
            
            # Create DataFrame and validate
            df = pd.DataFrame(candles_data)
            df = df.sort_values('timestamp')
            df.set_index('timestamp', inplace=True)
            
            # Remove duplicates and handle missing data
            df = df[~df.index.duplicated(keep='last')]
            df = df.resample(self._get_resample_rule(timeframe)).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).fillna(method='ffill')
            
            # Validate final dataset
            if len(df) < timeframe_config['min_periods']:
                raise DataError(
                    f"Insufficient data points for {symbol}: "
                    f"got {len(df)}, need {timeframe_config['min_periods']}"
                )
            
            return df
            
        except Exception as e:
            if isinstance(e, DataError):
                raise
            await self.trading_bot.log(f"Data fetch error: {str(e)}", level="error")
            raise DataError(f"Failed to fetch price data: {str(e)}")
            
    def _get_resample_rule(self, timeframe: TimeFrame) -> str:
        """Get pandas resample rule for timeframe."""
        rules = {
            TimeFrame.HOUR_1: '1H',
            TimeFrame.DAY_1: '1D',
            TimeFrame.DAY_30: '1D'  # We'll handle 30-day resampling separately
        }
        return rules.get(timeframe, '1D')
            
    async def _clean_cache(self) -> None:
        """Remove expired cache entries with timeframe-specific TTLs."""
        try:
            current_time = time.time()
            expired_keys = []
            
            for key, (_, timestamp) in self._cache.items():
                timeframe = TimeFrame[key.split('_')[1]]
                timeframe_config = self.timeframes.get(timeframe)
                if timeframe_config:
                    ttl = timeframe_config['cache_ttl']
                    if current_time - timestamp > ttl:
                        expired_keys.append(key)
            
            for key in expired_keys:
                del self._cache[key]
                
        except Exception as e:
            await self.trading_bot.log(f"Error cleaning cache: {str(e)}", level="error")

    async def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        try:
            product_id = self._format_product_id(symbol)
            product = self.client.get_product(product_id)
            return float(product.price)
        except Exception as e:
            await self.trading_bot.log(f"Error getting current price: {str(e)}", level="error")
            raise TradingError(f"Failed to get current price: {str(e)}", "DATA")

    async def get_order_book(self, symbol: str, level: int = 2) -> Dict[str, Any]:
        """Get order book data."""
        try:
            product_id = self._format_product_id(symbol)
            async with self.api_lock:
                # Rate limiting
                now = time.time()
                if now - self.last_request < self.rate_limit:
                    await asyncio.sleep(self.rate_limit - (now - self.last_request))
                
                order_book = self.client.get_product_book(
                    product_id=product_id,
                    level=level
                )
                self.last_request = time.time()
            
            return {
                'bids': [[float(p), float(s)] for p, s, _ in order_book.bids],
                'asks': [[float(p), float(s)] for p, s, _ in order_book.asks],
                'sequence': order_book.sequence,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            await self.trading_bot.log(f"Error getting order book: {str(e)}", level="error")
            raise TradingError(f"Failed to get order book: {str(e)}", "DATA")

    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get current ticker data for a symbol."""
        try:
            product_id = self._format_product_id(symbol)
            product = self.client.get_product(product_id)
            
            # Get 24h stats for volume using consistent granularity mapping
            candles = self.client.get_candles(
                product_id=product_id,
                start=int((datetime.now() - timedelta(days=1)).timestamp()),
                end=int(datetime.now().timestamp()),
                granularity=self.timeframes[TimeFrame.HOUR_1]['granularity']  # Use consistent mapping
            )
            
            volume_24h = sum(float(candle.volume) for candle in candles.candles)
            
            return {
                'price': float(product.price),
                'volume': volume_24h,
                'time': datetime.now()
            }
        except Exception as e:
            await self.trading_bot.log(f"Error getting ticker: {str(e)}", level="error")
            raise TradingError(f"Failed to get ticker: {str(e)}", "DATA")

    async def get_trades(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent trades for a symbol.
        
        Args:
            symbol: Trading pair symbol
            limit: Maximum number of trades to return
            
        Returns:
            List of recent trades
        """
        try:
            product_id = self._format_product_id(symbol)
            trades = self.client.get_trades(product_id=product_id, limit=limit)
            
            return [{
                'time': datetime.fromisoformat(trade.time.replace('Z', '+00:00')),
                'price': float(trade.price),
                'size': float(trade.size),
                'side': trade.side
            } for trade in trades]
        except Exception as e:
            await self.trading_bot.log(f"Error getting trades: {str(e)}", level="error")
            raise TradingError(f"Failed to get trades: {str(e)}", "DATA")

    async def get_volume_profile(self, symbol: str, timeframe: str = '1d') -> Dict[str, Any]:
        """
        Calculate volume profile for a symbol.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Time interval for analysis
            
        Returns:
            Dict containing volume profile data
        """
        try:
            symbol = self._format_product_id(symbol)
            df = await self.get_price_data(symbol, timeframe)
            
            # Calculate price levels
            price_range = df['high'].max() - df['low'].min()
            num_levels = 50
            level_size = price_range / num_levels
            
            # Create volume profile
            levels = []
            current_price = df['low'].min()
            
            for _ in range(num_levels):
                mask = (df['low'] <= current_price + level_size) & (df['high'] >= current_price)
                volume = df.loc[mask, 'volume'].sum()
                
                if volume > 0:
                    levels.append({
                        'price': current_price + (level_size / 2),
                        'volume': float(volume),
                        'trades': len(df[mask])
                    })
                
                current_price += level_size
            
            return {
                'levels': levels,
                'total_volume': float(df['volume'].sum()),
                'period_start': df.index[0].isoformat(),
                'period_end': df.index[-1].isoformat()
            }
            
        except Exception as e:
            await self.trading_bot.log(f"Error calculating volume profile: {str(e)}", level="error")
            raise TradingError(f"Failed to calculate volume profile: {str(e)}", "DATA")

    async def get_market_depth(self, symbol: str) -> Dict[str, Any]:
        """
        Get detailed market depth analysis.
        
        Returns:
            Dict containing:
            - Bid/ask spread
            - Order book imbalance
            - Liquidity analysis
        """
        try:
            order_book = await self.get_order_book(symbol, level=2)
            
            # Calculate bid/ask spread
            best_bid = order_book['bids'][0][0]
            best_ask = order_book['asks'][0][0]
            spread = best_ask - best_bid
            spread_percentage = (spread / best_bid) * 100
            
            # Calculate order book imbalance
            bid_volume = sum(bid[1] for bid in order_book['bids'][:10])
            ask_volume = sum(ask[1] for ask in order_book['asks'][:10])
            total_volume = bid_volume + ask_volume
            imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0
            
            return {
                'spread': spread,
                'spread_percentage': spread_percentage,
                'imbalance': imbalance,
                'bid_volume': bid_volume,
                'ask_volume': ask_volume,
                'best_bid': best_bid,
                'best_ask': best_ask
            }
            
        except Exception as e:
            await self.trading_bot.log(f"Error getting market depth: {str(e)}", level="error")
            raise TradingError(f"Failed to get market depth: {str(e)}", "DATA")