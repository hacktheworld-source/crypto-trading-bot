from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
from coinbase.rest import RESTClient
from bot.exceptions import TradingError, DataError
from bot.constants import TimeFrame
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
            TimeFrame.HOUR_1: {'days': 14, 'granularity': 'ONE_HOUR'},    # Short-term analysis
            TimeFrame.DAY_1: {'days': 90, 'granularity': 'ONE_DAY'}      # Long-term trend
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
        """Get price data for specified timeframe with caching"""
        try:
            symbol = self._format_product_id(symbol)
            cache_key = f"{symbol}_{timeframe.value}"
            
            async with self._cache_lock:
                # Check cache first
                if cache_key in self._cache:
                    data, timestamp = self._cache[cache_key]
                    if (time.time() - timestamp) < self.trading_bot.config.CACHE_TTL[timeframe.value]:
                        return data
                    
                # Fetch new data
                data = await self._fetch_price_data(symbol, timeframe, periods)
                
                # Update cache
                self._cache[cache_key] = (data, time.time())
                
                return data
                
        except Exception as e:
            await self.trading_bot.log(f"Price data fetch error: {str(e)}", level="error")
            raise DataError(f"Failed to get price data: {str(e)}")

    async def _fetch_price_data(
        self, 
        symbol: str, 
        timeframe: TimeFrame,
        periods: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch historical price data from exchange.
        
        Args:
            symbol: Trading pair symbol (must be in format 'BTC-USD')
            timeframe: TimeFrame enum value
            periods: Optional number of periods to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            async with self.api_lock:
                # Rate limiting
                now = time.time()
                if now - self.last_request < self.rate_limit:
                    await asyncio.sleep(self.rate_limit - (now - self.last_request))
                
                # Get time range based on timeframe
                end = datetime.now()
                timeframe_config = self.timeframes.get(timeframe)
                if not timeframe_config:
                    raise DataError(f"Unsupported timeframe: {timeframe}")
                
                days = timeframe_config['days']
                start = end - timedelta(days=days)
                
                # Get candles from Coinbase
                response = self.client.get_candles(
                    product_id=symbol,  # Already formatted by get_price_data
                    start=int(start.timestamp()),
                    end=int(end.timestamp()),
                    granularity=timeframe_config['granularity']
                )
                
                if not response.candles:
                    raise DataError(f"No data returned for {symbol}")
                
                # Convert to DataFrame
                df = pd.DataFrame([
                    {
                        'timestamp': datetime.fromtimestamp(float(candle.start)),
                        'open': float(candle.open),
                        'high': float(candle.high),
                        'low': float(candle.low),
                        'close': float(candle.close),
                        'volume': float(candle.volume)
                    }
                    for candle in response.candles
                ])
                
                # Sort by timestamp and set index
                df = df.sort_values('timestamp')
                df.set_index('timestamp', inplace=True)
                
                # Validate data
                if len(df) < 2:
                    raise DataError(f"Insufficient data points for {symbol}")
                
                self.last_request = time.time()
                return df
                
        except Exception as e:
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