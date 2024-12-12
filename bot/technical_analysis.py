from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from bot.exceptions import TradingError
import asyncio

class TechnicalAnalyzer:
    """Core technical analysis engine"""
    
    def __init__(self, trading_bot):
        self.trading_bot = trading_bot
        self.config = trading_bot.config
        
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
            await self.trading_bot.log(f"Signal generation error: {str(e)}", level="error")
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
            await self.trading_bot.log(f"Timeframe analysis error: {str(e)}", level="error")
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
        """Multi-timeframe trend analysis"""
        daily = await self._analyze_timeframe(symbol, '1d')
        hourly = await self._analyze_timeframe(symbol, '1h')
        
        return {
            'trend': {
                'daily': daily['trend'],
                'hourly': hourly['trend'],
                'aligned': daily['trend'] == hourly['trend']
            },
            'strength': daily['strength'] * 0.6 + hourly['strength'] * 0.4,
            'support_resistance': await self._get_key_levels(symbol),
            'volume_confirmed': await self._check_volume_confirmation(symbol)
        }

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