from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from bot.exceptions import TradingError
from bot.constants import TimeFrame
import asyncio

class TechnicalAnalyzer:
    """Core technical analysis engine"""
    
    def __init__(self, trading_bot):
        self.trading_bot = trading_bot
        self.config = trading_bot.config
        self.data_manager = trading_bot.data_manager
        
        # Use config timeframes
        self.timeframes = self.config.TIMEFRAMES
        self.rsi_period = self.config.RSI_PERIOD
        self.rsi_overbought = self.config.RSI_OVERBOUGHT
        self.rsi_oversold = self.config.RSI_OVERSOLD
        
        # Essential indicators only
        self.settings = {
            'ma_fast': 20,
            'ma_slow': 50,
            'rsi_period': 14,
            'volume_ma': 20
        }

    async def log(self, message: str, level: str = "info") -> None:
        """Forward logging to trading bot"""
        await self.trading_bot.log(message, level)

    async def get_signals(self, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive trading signals across timeframes.
        Returns confidence-weighted signals based on multiple indicators.
        """
        try:
            # Get data for different timeframes
            daily_data = await self.data_manager.get_price_data(symbol, TimeFrame.DAY_1)
            hourly_data = await self.data_manager.get_price_data(symbol, TimeFrame.HOUR_1)
            
            # 1. Daily Analysis (60% weight)
            daily_signals = await self._analyze_timeframe(daily_data, TimeFrame.DAY_1)
            
            # 2. Hourly Analysis (40% weight)
            hourly_signals = await self._analyze_timeframe(hourly_data, TimeFrame.HOUR_1)
            
            # 3. Volume Analysis
            volume_confirmed = await self._check_volume_confirmation(symbol)
            
            # 4. Support/Resistance
            key_levels = await self._get_key_levels(symbol)
            
            # Combine signals with weights
            confidence = (
                daily_signals['strength'] * 0.6 +
                hourly_signals['strength'] * 0.4
            ) * (1 if volume_confirmed else 0.5)
            
            return {
                'trend': {
                    'daily': daily_signals['trend'],
                    'hourly': hourly_signals['trend'],
                    'aligned': daily_signals['trend'] == hourly_signals['trend']
                },
                'volume_confirmed': volume_confirmed,
                'key_levels': key_levels,
                'confidence': confidence,
                'signals': {
                    'daily': daily_signals,
                    'hourly': hourly_signals
                }
            }
            
        except Exception as e:
            await self.log(f"Signal generation error: {str(e)}", level="error")
            raise TradingError(f"Failed to generate signals: {str(e)}", "ANALYSIS")

    async def _analyze_timeframe(self, data: pd.DataFrame, weight: float) -> Dict[str, Any]:
        """Analyze single timeframe and return weighted signals"""
        try:
            # Calculate core indicators
            ma_fast = data['close'].rolling(self.settings['ma_fast']).mean()
            ma_slow = data['close'].rolling(self.settings['ma_slow']).mean()
            rsi = await self.calculate_rsi(data['close'])
            volume = data['volume'].rolling(self.settings['volume_ma']).mean()
            
            # Generate signals
            trend_signal = 1 if ma_fast.iloc[-1] > ma_slow.iloc[-1] else -1
            trend_strength = abs(ma_fast.iloc[-1] - ma_slow.iloc[-1]) / ma_slow.iloc[-1]
            
            momentum_signal = 1 if rsi > 50 else -1
            volume_signal = 1 if data['volume'].iloc[-1] > volume.iloc[-1] else -1
            
            # Combine weighted signals
            signal = {
                'trend': trend_signal * trend_strength * weight,
                'momentum': momentum_signal * weight,
                'volume': volume_signal * weight,
                'indicators': {
                    'rsi': float(rsi.iloc[-1]),
                    'ma_fast': float(ma_fast.iloc[-1]),
                    'ma_slow': float(ma_slow.iloc[-1]),
                    'volume_ratio': float(data['volume'].iloc[-1] / volume.iloc[-1])
                }
            }
            
            return signal
            
        except Exception as e:
            await self.log(f"Timeframe analysis error: {str(e)}", level="error")
            raise TradingError(f"Timeframe analysis failed: {str(e)}", "ANALYSIS")

    def _combine_signals(self, signals: Dict[str, Dict]) -> Dict[str, Any]:
        """Combine signals from all timeframes into one decision"""
        total_signal = 0
        signal_count = 0
        
        for timeframe, signal in signals.items():
            total_signal += signal['trend'] + signal['momentum'] + signal['volume']
            signal_count += 3
        
        average_signal = total_signal / signal_count
        
        return {
            'action': 'buy' if average_signal > 0.2 else 'sell' if average_signal < -0.2 else 'hold',
            'confidence': abs(average_signal),
            'value': average_signal
        }

    async def analyze_trend(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze trend using current price data.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dict containing trend analysis
        """
        try:
            # Get current price data
            data = await self.data_manager.get_price_data(symbol, TimeFrame.HOUR_1)
            current_price = float(data['close'].iloc[-1])
            
            # Get ticker for volume info
            ticker = await self.data_manager.get_ticker(symbol)
            volume_confirmed = float(ticker['volume']) > 0  # Simple volume check
            
            # For now, use simple trend determination
            # Future enhancement: Implement proper trend analysis when historical data is available
            return {
                'trend': {
                    'daily': 1,  # Placeholder
                    'hourly': 1,  # Placeholder
                    'aligned': True
                },
                'strength': 0.5,  # Placeholder
                'volume_confirmed': volume_confirmed,
                'description': 'Current Price Only'
            }
            
        except Exception as e:
            await self.log(f"Trend analysis failed: {str(e)}", level="error")
            raise TradingError(f"Trend analysis failed: {str(e)}", "ANALYSIS")

    async def identify_key_levels(self, symbol: str) -> Dict[str, List[float]]:
        """Identify key support and resistance levels"""
        try:
            daily_data = await self.data_manager.get_price_data(symbol, TimeFrame.DAY_1)
            
            # Multiple methods for robustness
            pivot_levels = self._calculate_pivot_points(daily_data)
            volume_levels = await self.analyze_volume_profile(symbol)
            swing_levels = self._identify_swing_levels(daily_data)
            
            return {
                'support': self._consolidate_levels(
                    pivot_levels['support'],
                    volume_levels['significant_levels'],
                    swing_levels['support']
                ),
                'resistance': self._consolidate_levels(
                    pivot_levels['resistance'],
                    volume_levels['significant_levels'],
                    swing_levels['resistance']
                )
            }
            
        except Exception as e:
            await self.log(f"Key levels analysis error: {str(e)}", level="error")
            raise TradingError(f"Failed to identify key levels: {str(e)}", "ANALYSIS")

    async def analyze_volume_profile(self, symbol: str) -> Dict[str, Any]:
        """Enhanced volume analysis with price levels"""
        try:
            data = await self.data_manager.get_price_data(symbol, TimeFrame.HOUR_1)
            
            # Calculate volume-weighted price levels
            volume_profile = self._calculate_volume_profile(data)
            
            # Identify high-volume nodes
            significant_levels = self._find_volume_nodes(volume_profile)
            
            # Volume trend analysis
            volume_trend = self._analyze_volume_trend(data)
            
            return {
                'volume_profile': volume_profile,
                'significant_levels': significant_levels,
                'volume_trend': volume_trend,
                'is_volume_confirmed': volume_trend['trend'] == 'increasing'
            }
            
        except Exception as e:
            await self.log(f"Volume analysis error: {str(e)}", level="error")
            raise TradingError(f"Failed to analyze volume: {str(e)}", "ANALYSIS")

    def _calculate_volume_profile(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volume profile and price levels"""
        try:
            price_range = data['high'].max() - data['low'].min()
            num_levels = 50
            level_size = price_range / num_levels
            
            volume_levels = {}
            current_price = data['low'].min()
            
            for _ in range(num_levels):
                mask = (data['low'] <= current_price + level_size) & (data['high'] >= current_price)
                volume = data.loc[mask, 'volume'].sum()
                
                if volume > 0:
                    volume_levels[current_price + (level_size / 2)] = {
                        'volume': float(volume),
                        'trades': len(data[mask])
                    }
                
                current_price += level_size
                
            return volume_levels
            
        except Exception as e:
            self.trading_bot.log(f"Volume profile calculation error: {str(e)}", level="error")
            return {}

    def _find_volume_nodes(self, volume_profile: Dict[str, Any]) -> List[float]:
        """Identify significant volume nodes"""
        try:
            if not volume_profile:
                return []
                
            # Calculate volume threshold
            volumes = [level['volume'] for level in volume_profile.values()]
            avg_volume = sum(volumes) / len(volumes)
            volume_std = np.std(volumes)
            threshold = avg_volume + volume_std
            
            # Find significant levels
            significant_levels = [
                price for price, data in volume_profile.items()
                if data['volume'] > threshold
            ]
            
            return sorted(significant_levels)
            
        except Exception as e:
            self.trading_bot.log(f"Volume node analysis error: {str(e)}", level="error")
            return []

    def _analyze_volume_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume trend and characteristics"""
        try:
            volume_ma = data['volume'].rolling(self.settings['volume_ma']).mean()
            current_volume = data['volume'].iloc[-1]
            avg_volume = volume_ma.iloc[-1]
            
            # Calculate trend
            volume_change = (current_volume - avg_volume) / avg_volume
            
            return {
                'trend': 'increasing' if volume_change > 0.1 else 'decreasing' if volume_change < -0.1 else 'neutral',
                'strength': abs(volume_change),
                'current_ratio': current_volume / avg_volume,
                'is_above_average': current_volume > avg_volume
            }
            
        except Exception as e:
            self.trading_bot.log(f"Volume trend analysis error: {str(e)}", level="error")
            return {
                'trend': 'neutral',
                'strength': 0,
                'current_ratio': 1,
                'is_above_average': False
            }

    def _calculate_pivot_points(self, data: pd.DataFrame) -> Dict[str, List[float]]:
        """Calculate pivot points and support/resistance levels"""
        try:
            # Get most recent data point
            high = data['high'].iloc[-1]
            low = data['low'].iloc[-1]
            close = data['close'].iloc[-1]
            
            # Calculate pivot point
            pivot = (high + low + close) / 3
            
            # Calculate support and resistance levels
            r1 = 2 * pivot - low
            r2 = pivot + (high - low)
            s1 = 2 * pivot - high
            s2 = pivot - (high - low)
            
            return {
                'pivot': float(pivot),
                'resistance': [float(r1), float(r2)],
                'support': [float(s1), float(s2)]
            }
            
        except Exception as e:
            self.trading_bot.log(f"Pivot point calculation error: {str(e)}", level="error")
            return {'pivot': 0, 'resistance': [], 'support': []}

    def _identify_swing_levels(self, data: pd.DataFrame, window: int = 20) -> Dict[str, List[float]]:
        """Identify swing highs and lows"""
        try:
            highs = []
            lows = []
            
            for i in range(window, len(data) - window):
                # Check for swing high
                if all(data['high'].iloc[i] > data['high'].iloc[i-j] for j in range(1, window+1)) and \
                   all(data['high'].iloc[i] > data['high'].iloc[i+j] for j in range(1, window+1)):
                    highs.append(float(data['high'].iloc[i]))
                
                # Check for swing low
                if all(data['low'].iloc[i] < data['low'].iloc[i-j] for j in range(1, window+1)) and \
                   all(data['low'].iloc[i] < data['low'].iloc[i+j] for j in range(1, window+1)):
                    lows.append(float(data['low'].iloc[i]))
            
            return {
                'resistance': sorted(highs[-5:]),  # Most recent 5 swing highs
                'support': sorted(lows[-5:])       # Most recent 5 swing lows
            }
            
        except Exception as e:
            self.trading_bot.log(f"Swing level identification error: {str(e)}", level="error")
            return {'resistance': [], 'support': []}

    async def calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            data: Price series data
            period: RSI period (default: 14)
            
        Returns:
            pd.Series: RSI values
        """
        try:
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        except Exception as e:
            await self.log(f"RSI calculation error: {str(e)}", level="error")
            raise TradingError(f"Failed to calculate RSI: {str(e)}", "ANALYSIS")

    async def get_full_analysis(self, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive technical analysis for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dict containing full analysis results
        """
        try:
            # Get price data
            price_data = await self.data_manager.get_price_data(symbol, TimeFrame.DAY_1)
            current_price = float(price_data['close'].iloc[-1])
            
            # Calculate price change
            price_change_24h = (
                (current_price - float(price_data['close'].iloc[-2])) /
                float(price_data['close'].iloc[-2]) * 100
            )
            
            # Calculate volatility
            returns = price_data['close'].pct_change()
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized
            
            # Get technical signals
            signals = await self.get_signals(symbol)
            rsi = await self.calculate_rsi(price_data['close'])
            
            # Calculate position size based on volatility
            base_position = 1.0
            vol_adjustment = max(0.5, 1 - (volatility / 100))  # Reduce size for high volatility
            position_size = base_position * vol_adjustment * signals['confidence']
            
            return {
                'price': current_price,
                'price_change_24h': price_change_24h,
                'volatility': volatility,
                'trend': signals['trend'],
                'rsi': float(rsi.iloc[-1]),
                'volume_confirmed': signals['volume_confirmed'],
                'strength': signals['confidence'],
                'position_size': position_size
            }
            
        except Exception as e:
            await self.log(f"Full analysis error: {str(e)}", level="error")
            raise TradingError(f"Failed to get full analysis: {str(e)}", "ANALYSIS")

    async def check_market_conditions(self, symbol: str) -> Dict[str, Any]:
        """
        Check current market conditions for trading suitability.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dict containing market condition analysis
        """
        try:
            # Get price data
            daily_data = await self.data_manager.get_price_data(symbol, TimeFrame.DAY_1)
            hourly_data = await self.data_manager.get_price_data(symbol, TimeFrame.HOUR_1)
            
            # Calculate volatility
            returns = daily_data['close'].pct_change()
            volatility = returns.std() * np.sqrt(252)
            is_volatile = volatility > 0.03  # 3% daily volatility threshold
            
            # Calculate price range
            high_7d = daily_data['high'].tail(7).max()
            low_7d = daily_data['low'].tail(7).min()
            price_range_7d = (high_7d - low_7d) / low_7d * 100
            
            # Check trading hours activity
            current_hour = datetime.now().hour
            is_high_activity = 8 <= current_hour <= 22  # Active trading hours
            
            # Check market alignment
            btc_correlation = await self._calculate_btc_correlation(symbol)
            market_aligned = (
                btc_correlation > 0.5 if symbol != 'BTC' else True
            )
            
            # Overall suitability
            suitable_for_trading = (
                not is_volatile and
                market_aligned and
                is_high_activity
            )
            
            return {
                'is_volatile': is_volatile,
                'price_range_7d': price_range_7d,
                'is_high_activity': is_high_activity,
                'market_aligned': market_aligned,
                'suitable_for_trading': suitable_for_trading,
                'volatility': volatility * 100  # Convert to percentage
            }
            
        except Exception as e:
            await self.log(f"Market conditions check error: {str(e)}", level="error")
            raise TradingError(f"Failed to check market conditions: {str(e)}", "ANALYSIS")

    async def _calculate_btc_correlation(self, symbol: str) -> float:
        """
        Calculate correlation with BTC for the given symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            float: Correlation coefficient (-1 to 1)
        """
        try:
            if symbol == 'BTC':
                return 1.0
                
            # Get price data
            symbol_data = await self.data_manager.get_price_data(symbol, TimeFrame.HOUR_1)
            btc_data = await self.data_manager.get_price_data('BTC', TimeFrame.HOUR_1)
            
            # Calculate returns
            symbol_returns = symbol_data['close'].pct_change().dropna()
            btc_returns = btc_data['close'].pct_change().dropna()
            
            # Ensure same length
            min_len = min(len(symbol_returns), len(btc_returns))
            symbol_returns = symbol_returns[-min_len:]
            btc_returns = btc_returns[-min_len:]
            
            # Calculate correlation
            correlation = symbol_returns.corr(btc_returns)
            
            return float(correlation)
            
        except Exception as e:
            await self.log(f"Correlation calculation error: {str(e)}", level="error")
            return 0.0  # Return neutral correlation on error