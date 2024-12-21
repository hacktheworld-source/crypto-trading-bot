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
        
        # Initialize cache
        self._cache: Dict[str, tuple[pd.DataFrame, float]] = {}
        self._cache_lock = asyncio.Lock()
        
        # Timeframe configurations with correct Coinbase API granularity values
        self.timeframes = {
            TimeFrame.HOUR_1: {'days': 14, 'granularity': 'ONE_HOUR'},    # 14 days * 24 hours = 336 candles
            TimeFrame.DAY_1: {'days': 200, 'granularity': 'ONE_DAY'}      # Daily candles, well under limit
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
        periods: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get price data for specified timeframe with caching.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC', 'ETH')
            timeframe: TimeFrame enum value
            periods: Optional number of periods to fetch
            
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
            
            async with self._cache_lock:
                # Check cache first
                if cache_key in self._cache:
                    data, timestamp = self._cache[cache_key]
                    ttl = TradingConstants.CACHE_TTL.get(timeframe, 3600)
                    if (time.time() - timestamp) < ttl:
                        return data
                
                # Acquire rate limit
                await self.trading_bot.rate_limiter.acquire()
                
                # Fetch new data
                data = await self._fetch_price_data(symbol, timeframe, periods)
                
                # Validate data
                if data is None or data.empty:
                    raise DataError(f"No data returned for {symbol}")
                
                if len(data) < TradingConstants.MIN_DATA_POINTS:
                    raise DataError(
                        f"Insufficient data points for {symbol}: "
                        f"got {len(data)}, need {TradingConstants.MIN_DATA_POINTS}"
                    )
                
                # Update cache
                self._cache[cache_key] = (data, time.time())
                
                # Clean cache if needed
                if len(self._cache) > TradingConstants.CACHE_SIZE:
                    oldest_key = min(self._cache.items(), key=lambda x: x[1][1])[0]
                    del self._cache[oldest_key]
                
                return data
                
        except Exception as e:
            error_msg = f"Failed to get price data for {symbol}: {str(e)}"
            await self.trading_bot.log(error_msg, level="error")
            raise DataError(error_msg)

    async def _fetch_price_data(
        self, 
        symbol: str, 
        timeframe: TimeFrame,
        periods: Optional[int] = None
    ) -> pd.DataFrame:
        """Fetch historical price data from exchange."""
        try:
            # Get timeframe configuration
            timeframe_config = self.timeframes.get(timeframe)
            if not timeframe_config:
                raise DataError(f"Unsupported timeframe: {timeframe.name}")
            
            # Calculate time range
            end = datetime.now()
            days = timeframe_config['days']  # Use configured days directly
            start = end - timedelta(days=days)
            
            # Log request details
            await self.trading_bot.log(
                f"Fetching {timeframe.name} data for {symbol} "
                f"(granularity: {timeframe_config['granularity']}, days: {days})",
                level="debug"
            )
            
            # Get candles from Coinbase
            response = self.client.get_candles(
                product_id=symbol,
                start=int(start.timestamp()),
                end=int(end.timestamp()),
                granularity=timeframe_config['granularity']
            )
            
            if not response or not response.candles:
                raise DataError(f"No data returned for {symbol}")
            
            # Process candle data
            candles_data = []
            for candle in response.candles:
                try:
                    candles_data.append({
                        'timestamp': datetime.fromtimestamp(float(candle.start)),
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
            
            # Validate number of candles
            if len(df) > 350:
                await self.trading_bot.log(
                    f"Warning: Got {len(df)} candles, trimming to latest 350",
                    level="warning"
                )
                df = df.iloc[-350:]  # Keep the most recent 350 candles if we somehow get more
            
            if len(df) < 15:  # Minimum required for most indicators
                raise DataError(f"Insufficient data points for {symbol}: {len(df)}")
            
            return df
            
        except Exception as e:
            if isinstance(e, DataError):
                raise
            await self.trading_bot.log(f"Data fetch error: {str(e)}", level="error")
            raise DataError(f"Failed to fetch price data: {str(e)}")

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

    def _clean_cache(self) -> None:
        """Remove expired cache entries."""
        try:
            current_time = time.time()
            expired_keys = []
            
            for key, (_, timestamp) in self.cache.items():
                timeframe = key.split('_')[1]
                if current_time - timestamp > self.cache_ttl[timeframe]:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache[key]
                
        except Exception as e:
            self.trading_bot.log(f"Error cleaning cache: {str(e)}", level="error")