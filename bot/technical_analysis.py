from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from bot.exceptions import TradingError, DataError
from bot.constants import TimeFrame
import asyncio

class TechnicalAnalyzer:
    """Core technical analysis engine"""
    
    def __init__(self, trading_bot):
        self.trading_bot = trading_bot
        self.config = trading_bot.config
        self.data_manager = trading_bot.data_manager
        
        # Timeframe-specific indicator settings
        self.indicator_settings = {
            'monthly': {
                'rsi_period': 7,          # Shorter for monthly
                'macd_fast': 6,           # Faster for monthly
                'macd_slow': 13,
                'macd_signal': 4,
                'bb_period': 10,          # Shorter for monthly
                'volume_ma': 6            # 6 months volume MA
            },
            'daily': {
                'rsi_period': 14,         # Standard daily
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'bb_period': 20,
                'volume_ma': 20
            },
            'hourly': {
                'rsi_period': 14,         # Standard hourly
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'bb_period': 20,
                'volume_ma': 24           # Full day of hourly volume
            }
        }
        
        # Use data manager's timeframes for consistency
        self.timeframes = {
            TimeFrame.DAY_30: {'weight': 0.4},  # Monthly for long-term trend
            TimeFrame.DAY_1: {'weight': 0.35},  # Daily for trend direction
            TimeFrame.HOUR_1: {'weight': 0.25}  # Hourly for entry timing
        }
        
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
        Get trading signals for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dict containing signal analysis for different timeframes
            
        Raises:
            TradingError: If signal generation fails
        """
        try:
            symbol = self.data_manager._format_product_id(symbol.upper())
            
            # Get data for different timeframes
            data_30d = await self.data_manager.get_price_data(symbol, TimeFrame.DAY_30)
            data_1d = await self.data_manager.get_price_data(symbol, TimeFrame.DAY_1)
            data_1h = await self.data_manager.get_price_data(symbol, TimeFrame.HOUR_1)
            
            if any(data is None for data in [data_30d, data_1d, data_1h]):
                raise TradingError("Failed to fetch data for signal generation", "DATA")
                
            # Calculate signals for each timeframe
            signals_30d = await self._calculate_timeframe_signals(data_30d, "monthly")
            signals_1d = await self._calculate_timeframe_signals(data_1d, "daily")
            signals_1h = await self._calculate_timeframe_signals(data_1h, "1h")
            
            # Calculate trend alignment across all timeframes
            trend_alignment = await self._calculate_trend_alignment(
                signals_30d['trend'],
                signals_1d['trend'],
                signals_1h['trend']
            )
            
            return {
                'signals': {
                    'monthly': signals_30d,
                    'daily': signals_1d,
                    '1h': signals_1h
                },
                'trend': {
                    'monthly': signals_30d['trend'],
                    'daily': signals_1d['trend'],
                    'hourly': signals_1h['trend'],
                    'aligned': trend_alignment['aligned'],
                    'strength': trend_alignment['strength']
                }
            }
            
        except Exception as e:
            await self.log(f"Signal generation error: {str(e)}", level="error")
            raise TradingError(f"Failed to generate signals: {str(e)}", "ANALYSIS")
            
    async def _calculate_timeframe_signals(self, data: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
        """
        Calculate technical signals for a specific timeframe.
        
        Args:
            data: Price data
            timeframe: Timeframe identifier
            
        Returns:
            Dict containing signal analysis
        """
        try:
            # Adjust periods based on timeframe
            rsi_period = self.rsi_period
            if timeframe == "monthly":
                rsi_period = max(7, self.rsi_period // 2)  # Shorter period for monthly data
            
            # Calculate technical indicators
            rsi = await self.calculate_rsi(data['close'], period=rsi_period)
            macd = self._calculate_macd(data)
            bb = await self.calculate_bollinger_bands(data)
            ema_short = await self._calculate_ema(data['close'], 9)
            ema_long = await self._calculate_ema(data['close'], 21)
            
            # Get latest values with proper error handling
            try:
                current_price = float(data['close'].iloc[-1])
                current_rsi = float(rsi.iloc[-1]) if rsi is not None else 50.0
                current_macd = float(macd['macd'].iloc[-1]) if macd is not None else 0.0
                current_signal = float(macd['signal'].iloc[-1]) if macd is not None else 0.0
                current_ema_short = float(ema_short.iloc[-1]) if ema_short is not None else current_price
                current_ema_long = float(ema_long.iloc[-1]) if ema_long is not None else current_price
                
                if bb is not None:
                    current_bb = {
                        'upper': float(bb['upper']),
                        'middle': float(bb['middle']),
                        'lower': float(bb['lower'])
                    }
                else:
                    current_bb = {
                        'upper': current_price * 1.02,
                        'middle': current_price,
                        'lower': current_price * 0.98
                    }
            except (IndexError, ValueError, KeyError, AttributeError) as e:
                await self.trading_bot.log(f"Error getting latest values: {str(e)}", level="error")
                raise TradingError("Failed to get latest values", "ANALYSIS")
            
            # Calculate trend with timeframe-specific adjustments
            trend_score = await self._calculate_trend_score(
                current_price,
                current_ema_short,
                current_ema_long,
                current_bb['middle'],
                current_macd,
                current_rsi,
                timeframe=timeframe
            )
            
            # Calculate momentum with timeframe adjustments
            momentum_score = self._calculate_momentum_score(
                current_rsi,
                current_macd,
                current_signal,
                data,
                timeframe=timeframe
            )
            
            # Get volume confirmation with timeframe consideration
            volume_analysis = self._analyze_volume_trend(data, timeframe=timeframe)
            volume_confirmed = volume_analysis.get('is_favorable', False)
            
            # Determine signal strength with timeframe weights
            base_strength = abs(trend_score + momentum_score) / 2
            timeframe_weight = (
                1.2 if timeframe == "monthly" else
                1.0 if timeframe == "daily" else
                0.8  # hourly
            )
            signal_strength = base_strength * timeframe_weight
            
            return {
                'trend': trend_score,
                'momentum': {
                    'value': momentum_score,
                    'rsi': current_rsi,
                    'macd': current_macd,
                    'signal': current_signal
                },
                'strength': signal_strength,
                'volume_confirmed': volume_confirmed,
                'indicators': {
                    'rsi': current_rsi,
                    'macd': {
                        'value': current_macd,
                        'signal': current_signal,
                        'histogram': current_macd - current_signal
                    },
                    'bb': current_bb,
                    'ema': {
                        'short': current_ema_short,
                        'long': current_ema_long
                    }
                }
            }
            
        except Exception as e:
            await self.trading_bot.log(f"Signal calculation error: {str(e)}", level="error")
            # Return neutral signals on error
            return {
                'trend': 0.0,
                'momentum': {
                    'value': 0.0,
                    'rsi': 50.0,
                    'macd': 0.0,
                    'signal': 0.0
                },
                'strength': 0.0,
                'volume_confirmed': False,
                'indicators': {
                    'rsi': 50.0,
                    'macd': {'value': 0.0, 'signal': 0.0, 'histogram': 0.0},
                    'bb': {
                        'upper': float(data['close'].iloc[-1]) * 1.02,
                        'middle': float(data['close'].iloc[-1]),
                        'lower': float(data['close'].iloc[-1]) * 0.98
                    },
                    'ema': {
                        'short': float(data['close'].iloc[-1]),
                        'long': float(data['close'].iloc[-1])
                    }
                }
            }
        
    async def _calculate_trend_score(self, price: float, ema_short: float, 
                             ema_long: float, bb_middle: float,
                             macd: float, rsi: float,
                             timeframe: str = "daily") -> float:
        """
        Calculate trend score based on multiple indicators.
        
        Returns:
            Float between -1 and 1 indicating trend strength and direction
        """
        try:
            # Calculate price momentum relative to EMAs
            ema_short_dist = (price - ema_short) / ema_short
            ema_long_dist = (price - ema_long) / ema_long
            
            # More granular EMA scoring
            ema_score = (
                1.0 if price > ema_short > ema_long and ema_short_dist > 0.002 else
                -1.0 if price < ema_short < ema_long and ema_short_dist < -0.002 else
                0.75 if price > ema_short > ema_long else
                -0.75 if price < ema_short < ema_long else
                0.5 if price > ema_long and ema_long_dist > 0.001 else
                -0.5 if price < ema_long and ema_long_dist < -0.001 else
                0.25 if price > ema_long else
                -0.25 if price < ema_long else
                0.0
            )
            
            # BB score with more realistic distance scaling
            bb_distance = abs(price - bb_middle) / bb_middle
            bb_score = (1.0 if price > bb_middle else -1.0) * min(1.0, bb_distance * 5)
            
            # MACD score with signal line consideration
            macd_abs = abs(macd)
            macd_score = (
                1.0 if macd > 0 and macd_abs > 0.002 else
                -1.0 if macd < 0 and macd_abs > 0.002 else
                0.5 if macd > 0 else
                -0.5 if macd < 0 else
                0.0
            )
            
            # RSI score with more zones for better accuracy
            rsi_score = (
                -1.0 if rsi > 75 else    # Extremely overbought
                -0.75 if rsi > 70 else   # Strongly overbought
                -0.5 if rsi > 65 else    # Moderately overbought
                0.5 if rsi < 35 else     # Moderately oversold
                0.75 if rsi < 30 else    # Strongly oversold
                1.0 if rsi < 25 else     # Extremely oversold
                0.25 if rsi > 60 else    # Slight overbought
                -0.25 if rsi < 40 else   # Slight oversold
                0.0                      # Neutral
            )
            
            # Weight and combine scores
            weighted_score = (
                ema_score * 0.35 +      # Trend following (primary)
                bb_score * 0.25 +       # Price position
                macd_score * 0.25 +     # Momentum
                rsi_score * 0.15        # Overbought/Oversold
            )
            
            # Add minimum threshold for trend signals
            if abs(weighted_score) < 0.2:
                weighted_score = 0.0  # Neutral if trend is too weak
            
            # Ensure the score is between -1 and 1
            return max(-1.0, min(1.0, weighted_score))
            
        except Exception as e:
            await self.log(f"Trend score calculation error: {str(e)}", level="error")
            return 0.0  # Neutral score on error
            
    def _calculate_momentum_score(self, rsi: float, macd: float, 
                                signal: float, data: pd.DataFrame,
                                timeframe: str = "daily") -> float:
        """
        Calculate momentum score based on multiple indicators.
        
        Returns:
            Float between -1 and 1 indicating momentum strength and direction
        """
        # RSI momentum
        rsi_score = 0
        if rsi > 70:
            rsi_score = -1
        elif rsi < 30:
            rsi_score = 1
        else:
            rsi_score = (rsi - 50) / 20  # Scaled score between -1 and 1
            
        # MACD momentum
        macd_score = 1 if macd > signal else -1 if macd < signal else 0
        
        # Price momentum (using returns)
        returns = data['close'].pct_change()
        recent_returns = returns.iloc[-3:]  # Last 3 periods
        momentum_score = np.sign(recent_returns.mean()) * min(1.0, abs(recent_returns.mean() * 100))
        
        # Combine scores with weights
        weighted_score = (
            rsi_score * 0.4 +
            macd_score * 0.3 +
            momentum_score * 0.3
        )
        
        return max(-1.0, min(1.0, weighted_score))
        
    async def _calculate_trend_alignment(self, *trends: float) -> Dict[str, Any]:
        """
        Calculate trend alignment across multiple timeframes.
        
        Args:
            *trends: Trend scores for different timeframes
            
        Returns:
            Dict containing alignment analysis
        """
        try:
            if not trends:
                return {'aligned': False, 'strength': 0.0}
            
            # All trends should be in the same direction
            all_positive = all(t > 0 for t in trends)
            all_negative = all(t < 0 for t in trends)
            aligned = all_positive or all_negative
            
            # Calculate average strength
            avg_strength = sum(abs(t) for t in trends) / len(trends)
            
            # Weight longer timeframes more heavily
            weighted_strength = 0.0
            weights = [0.5, 0.3, 0.2]  # Monthly, Daily, Hourly
            for trend, weight in zip(trends, weights):
                weighted_strength += abs(trend) * weight
            
            return {
                'aligned': aligned,
                'strength': weighted_strength,
                'direction': 'bullish' if all_positive else 'bearish' if all_negative else 'mixed'
            }
            
        except Exception as e:
            await self.log(f"Trend alignment calculation error: {str(e)}", level="error")
            return {'aligned': False, 'strength': 0.0, 'direction': 'unknown'}

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
            # Get price data
            data = await self.data_manager.get_price_data(symbol, TimeFrame.DAY_1)
            if data is None or len(data) < 2:
                raise TradingError("Insufficient price data", "ANALYSIS")
            
            # Get signals and volume analysis
            signals = await self.get_signals(symbol)
            volume_analysis = self._analyze_volume_trend(data)
            
            # Extract trend information
            trend_info = signals['trend']
            
            # Determine trend description
            if trend_info['aligned']:
                base_desc = "Strong" if abs(trend_info['strength']) > 0.5 else "Moderate"
                direction = "Uptrend" if trend_info['daily'] > 0 else "Downtrend"
                description = f"{base_desc} {direction}"
            else:
                description = "Mixed Trend"
            
            # Add volume context
            if volume_analysis['trend'] == "strongly":
                volume_context = f" with {volume_analysis['description'].lower()} volume"
                description += volume_context
            
            return {
                'trend': {
                    'daily': trend_info['daily'],
                    'hourly': trend_info['hourly'],
                    'aligned': trend_info['aligned']
                },
                'strength': float(trend_info['strength']),
                'volume': {
                    'trend': volume_analysis['description'],
                    'strength': volume_analysis['strength'],
                    'score': volume_analysis['score']
                },
                'description': description
            }
            
        except Exception as e:
            await self.log(f"Trend analysis failed: {str(e)}", level="error")
            # Return neutral values on error
            return {
                'trend': {
                    'daily': 0.0,
                    'hourly': 0.0,
                    'aligned': False
                },
                'strength': 0.0,
                'volume': {
                    'trend': "Unknown",
                    'strength': "Average",
                    'score': 0.0
                },
                'description': "Error - Unable to analyze trend"
            }

    async def identify_key_levels(self, symbol: str) -> Dict[str, List[float]]:
        """Identify key support and resistance levels"""
        try:
            daily_data = await self.data_manager.get_price_data(symbol, TimeFrame.DAY_1)
            
            # Multiple methods for robustness
            pivot_levels = await self._calculate_pivot_points(daily_data)
            volume_levels = await self.analyze_volume_profile(symbol)
            swing_levels = await self._identify_swing_levels(daily_data)
            
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
            volume_profile = await self._calculate_volume_profile(data)
            
            # Identify high-volume nodes
            significant_levels = await self._find_volume_nodes(volume_profile)
            
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

    async def _calculate_volume_profile(self, data: pd.DataFrame) -> Dict[str, Any]:
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
            await self.trading_bot.log(f"Volume profile calculation error: {str(e)}", level="error")
            return {}

    async def _find_volume_nodes(self, volume_profile: Dict[str, Any]) -> List[float]:
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
            await self.trading_bot.log(f"Volume node analysis error: {str(e)}", level="error")
            return []

    def _analyze_volume_trend(self, data: pd.DataFrame, timeframe: str = "daily") -> Dict[str, Any]:
        """
        Analyze volume trend with timeframe-specific thresholds.
        
        Args:
            data: Price and volume data
            timeframe: Analysis timeframe
            
        Returns:
            Dict containing volume analysis
        """
        try:
            settings = self.indicator_settings[timeframe]
            
            # Calculate volume metrics
            volume = data['volume']
            volume_ma = volume.rolling(window=settings['volume_ma']).mean()
            volume_std = volume.rolling(window=settings['volume_ma']).std()
            
            # Get recent volume data
            current_volume = float(volume.iloc[-1])
            recent_volumes = volume.iloc[-settings['volume_ma']:]
            
            # Calculate z-score with timeframe consideration
            current_ma = float(volume_ma.iloc[-1])
            current_std = float(volume_std.iloc[-1])
            z_score = (current_volume - current_ma) / current_std if current_std != 0 else 0
            
            # Adjust volume trend calculation for timeframe
            volume_trend = float(recent_volumes.mean() / current_ma) if current_ma != 0 else 1.0
            
            # Adjust thresholds based on timeframe
            if timeframe == "monthly":
                trend_threshold = 1.1  # 10% above average for monthly
                z_score_threshold = 1.5
            elif timeframe == "daily":
                trend_threshold = 1.2  # 20% above average for daily
                z_score_threshold = 2.0
            else:  # hourly
                trend_threshold = 1.3  # 30% above average for hourly
                z_score_threshold = 2.5
            
            # Determine volume characteristics
            is_favorable = volume_trend > trend_threshold and z_score > -z_score_threshold
            
            # Score from -1 to 1 with timeframe consideration
            score = min(1.0, max(-1.0, z_score / z_score_threshold))
            
            # Get descriptive strength
            if abs(z_score) > z_score_threshold:
                strength = "Very High" if z_score > 0 else "Very Low"
            elif abs(z_score) > z_score_threshold / 2:
                strength = "High" if z_score > 0 else "Low"
            else:
                strength = "Average"
            
            # Get trend description
            if volume_trend > trend_threshold * 1.2:
                description = "Strongly Increasing"
            elif volume_trend > trend_threshold:
                description = "Moderately Increasing"
            elif volume_trend < 1 / trend_threshold:
                description = "Strongly Decreasing"
            elif volume_trend < 1:
                description = "Moderately Decreasing"
            else:
                description = "Stable"
            
            return {
                'score': float(score),
                'strength': strength,
                'description': description,
                'is_favorable': is_favorable,
                'trend': description.split()[0].lower(),
                'z_score': float(z_score),
                'volume_trend': float(volume_trend)
            }
            
        except Exception as e:
            # Return neutral values on error
            return {
                'score': 0.0,
                'strength': "Average",
                'description': "Stable",
                'is_favorable': False,
                'trend': "stable",
                'z_score': 0.0,
                'volume_trend': 1.0
            }

    async def _calculate_pivot_points(self, data: pd.DataFrame) -> Dict[str, List[float]]:
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
            await self.trading_bot.log(f"Pivot point calculation error: {str(e)}", level="error")
            return {'pivot': 0, 'resistance': [], 'support': []}

    async def _identify_swing_levels(self, data: pd.DataFrame, window: int = 20) -> Dict[str, List[float]]:
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
            await self.trading_bot.log(f"Swing level identification error: {str(e)}", level="error")
            return {'resistance': [], 'support': []}

    async def calculate_rsi(
        self, 
        prices: pd.Series,
        period: int = None,
        timeframe: str = "daily"
    ) -> pd.Series:
        """
        Calculate RSI with timeframe-specific parameters.
        
        Args:
            prices: Price series
            period: Optional override for RSI period
            timeframe: Analysis timeframe
            
        Returns:
            RSI values as Series
        """
        try:
            settings = self.indicator_settings[timeframe]
            # Use provided period if specified, otherwise use timeframe settings
            rsi_period = period if period is not None else settings['rsi_period']
            
            # Calculate price changes
            delta = prices.diff()
            
            # Separate gains and losses
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            
            # Calculate average gains and losses
            avg_gains = gains.ewm(com=rsi_period-1, min_periods=rsi_period).mean()
            avg_losses = losses.ewm(com=rsi_period-1, min_periods=rsi_period).mean()
            
            # Calculate RS and RSI
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            if isinstance(e, TradingError):
                raise
            await self.log(f"RSI calculation error: {str(e)}", level="error")
            raise TradingError(f"Failed to calculate RSI: {str(e)}", "ANALYSIS")

    async def calculate_bollinger_bands(
        self, 
        data: pd.DataFrame,
        timeframe: str = "daily"
    ) -> Dict[str, Any]:
        """
        Calculate Bollinger Bands with timeframe-specific parameters.
        
        Args:
            data: Price data DataFrame
            timeframe: Analysis timeframe
            
        Returns:
            Dict containing upper, middle, lower bands and bandwidth
        """
        try:
            settings = self.indicator_settings[timeframe]
            period = settings['bb_period']
            
            # Calculate middle band (SMA)
            middle = data['close'].rolling(window=period).mean()
            
            # Calculate standard deviation
            std = data['close'].rolling(window=period).std()
            
            # Calculate bands
            upper = middle + (std * 2)
            lower = middle - (std * 2)
            
            # Calculate bandwidth
            bandwidth = (upper - lower) / middle
            
            return {
                'upper': float(upper.iloc[-1]),
                'middle': float(middle.iloc[-1]),
                'lower': float(lower.iloc[-1]),
                'bandwidth': float(bandwidth.iloc[-1])
            }
            
        except Exception as e:
            if isinstance(e, TradingError):
                raise
            await self.log(f"Bollinger Bands calculation error: {str(e)}", level="error")
            raise TradingError(f"Failed to calculate Bollinger Bands: {str(e)}", "ANALYSIS")

    async def get_ma_analysis(self, symbol: str) -> str:
        """
        Get moving average analysis for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            str: Formatted MA analysis
            
        Raises:
            TradingError: If calculation fails
        """
        try:
            # Get price data from data manager
            data = await self.data_manager.get_price_data(symbol.upper(), TimeFrame.DAY_1)
            
            # Use standardized validation helper
            prices = await self._validate_price_data(
                data=data,
                min_periods=self.settings['ma_slow'] + 1,
                symbol=symbol,
                indicator="Moving Average"
            )
            
            # Calculate MAs with proper minimum periods
            ma_fast = prices.rolling(window=self.settings['ma_fast'], min_periods=self.settings['ma_fast']).mean()
            ma_slow = prices.rolling(window=self.settings['ma_slow'], min_periods=self.settings['ma_slow']).mean()
            
            # Get latest values with proper error handling
            try:
                latest_values = {
                    'current_price': float(prices.iloc[-1]),
                    'fast_ma': float(ma_fast.iloc[-1]),
                    'slow_ma': float(ma_slow.iloc[-1])
                }
            except (IndexError, ValueError) as e:
                raise TradingError(f"Failed to get latest values: {str(e)}", "ANALYSIS")
            
            # Validate results
            if any(pd.isna(val) for val in latest_values.values()):
                raise TradingError("Invalid results in MA calculation", "ANALYSIS")
            
            # Calculate trend and strength
            trend = "Bullish" if latest_values['fast_ma'] > latest_values['slow_ma'] else "Bearish"
            strength = abs(latest_values['fast_ma'] - latest_values['slow_ma']) / latest_values['slow_ma'] * 100
            
            # Calculate price position
            above_fast = latest_values['current_price'] > latest_values['fast_ma']
            above_slow = latest_values['current_price'] > latest_values['slow_ma']
            
            # Add trend emoji
            trend_emoji = "🟢" if trend == "Bullish" else "🔴"
            
            # Calculate momentum
            momentum = "Strong" if strength > 5 else \
                      "Moderate" if strength > 2 else \
                      "Weak"
            
            return (
                f"Moving Average Analysis for {symbol}:\n```"
                f"📊 Price Levels:\n"
                f"  • Current Price: ${latest_values['current_price']:,.2f}\n"
                f"  • Fast MA ({self.settings['ma_fast']}): ${latest_values['fast_ma']:,.2f}\n"
                f"  • Slow MA ({self.settings['ma_slow']}): ${latest_values['slow_ma']:,.2f}\n\n"
                f"📈 Trend Analysis:\n"
                f"  • Direction: {trend} {trend_emoji}\n"
                f"  • Strength: {strength:.1f}% ({momentum})\n"
                f"  • Price > Fast MA: {'Yes ✅' if above_fast else 'No ❌'}\n"
                f"  • Price > Slow MA: {'Yes ✅' if above_slow else 'No ❌'}"
                "```"
            )
            
        except Exception as e:
            if isinstance(e, TradingError):
                raise
            await self.log(f"MA analysis error: {str(e)}", level="error")
            raise TradingError(f"Failed to get MA analysis: {str(e)}", "ANALYSIS")

    async def check_market_conditions(self, symbol: str) -> Dict[str, Any]:
        """
        Check market conditions for trading.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dict containing market condition analysis
            
        Raises:
            TradingError: If analysis fails
        """
        try:
            # Format symbol and get price data
            symbol = self.data_manager._format_product_id(symbol.upper())
            data = await self.data_manager.get_price_data(symbol, TimeFrame.DAY_1)
            
            if data is None or len(data) < 30:  # Need at least 30 days for proper analysis
                raise TradingError(
                    f"Insufficient data for market analysis. Need at least 30 days of data.",
                    "ANALYSIS"
                )
            
            # Calculate volatility (30-day)
            returns = data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Define volatility thresholds based on market type
            is_crypto = symbol.endswith('-USD')
            volatility_threshold = 0.8 if is_crypto else 0.4  # Higher threshold for crypto
            is_volatile = volatility > volatility_threshold
            
            # Calculate price ranges
            week_high = data['high'][-7:].max()
            week_low = data['low'][-7:].min()
            price_range_7d = ((week_high - week_low) / week_low) * 100 if week_low > 0 else 0
            
            # Get current price and ATR
            current_price = float(data['close'].iloc[-1])
            atr = self._calculate_atr(data, period=14)
            
            # Calculate volume analysis
            volume_ma = data['volume'].rolling(window=20).mean()
            current_volume = float(data['volume'].iloc[-1])
            volume_ma_last = float(volume_ma.iloc[-1])
            volume_ratio = current_volume / volume_ma_last if volume_ma_last > 0 else 1.0
            
            # Volume trend analysis (more sophisticated than just high/low)
            volume_trend = self._analyze_volume_trend(data)
            
            # Get market alignment
            signals = await self.get_signals(symbol)
            market_aligned = signals['trend']['aligned']
            
            # Calculate market strength
            strength = {
                'trend': signals['trend']['daily'],
                'momentum': signals['signals']['daily']['momentum'],
                'volume': volume_trend['score']  # Use volume trend score
            }
            
            # Overall market score (-1 to 1)
            market_score = (
                strength['trend'] * 0.4 +
                strength['momentum'] * 0.4 +
                strength['volume'] * 0.2
            )
            
            return {
                'volatility': {
                    'value': float(volatility),
                    'is_high': is_volatile,
                    'threshold': volatility_threshold
                },
                'price_action': {
                    'range_7d': float(price_range_7d),
                    'atr': float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0,
                    'current_price': current_price
                },
                'volume': {
                    'ratio': float(volume_ratio),
                    'trend': volume_trend['description'],
                    'strength': volume_trend['strength']
                },
                'market_alignment': {
                    'aligned': market_aligned,
                    'score': float(market_score)
                },
                'trading_summary': {
                    'suitable': not is_volatile and market_aligned and volume_trend['is_favorable'],
                    'confidence': abs(market_score),
                    'recommendation': self._get_market_recommendation(market_score, is_volatile)
                }
            }
            
        except Exception as e:
            await self.log(f"Market conditions check error: {str(e)}", level="error")
            raise TradingError(f"Failed to check market conditions: {str(e)}", "ANALYSIS")
            
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
        
    def _get_market_recommendation(self, score: float, is_volatile: bool) -> str:
        """Get market recommendation based on score and volatility"""
        if is_volatile:
            return "High Risk - Reduce Position Sizes"
        elif score > 0.6:
            return "Strong Buy Zone"
        elif score > 0.2:
            return "Accumulate"
        elif score < -0.6:
            return "Strong Sell Zone"
        elif score < -0.2:
            return "Reduce Exposure"
        else:
            return "Neutral - Monitor"

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
            data = await self.data_manager.get_price_data(symbol.upper(), TimeFrame.DAY_1)
            if data is None or len(data) < 2:
                raise TradingError("Insufficient price data", "ANALYSIS")
                
            prices = pd.to_numeric(data['close'], errors='coerce')
            current_price = float(prices.iloc[-1])
            
            # Calculate price change
            price_change_24h = ((current_price - float(prices.iloc[-2])) / float(prices.iloc[-2])) * 100
            
            # Calculate volatility
            returns = prices.pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized
            
            # Get technical signals
            signals = await self.get_signals(symbol)
            rsi = await self.calculate_rsi(prices)
            
            # Calculate position size based on volatility
            base_position = 1.0
            vol_adjustment = max(0.5, 1 - (volatility / 100))  # Reduce size for high volatility
            position_size = base_position * vol_adjustment * signals['trend']['strength']
            
            # Determine trend description
            if signals['trend']['aligned']:
                trend_desc = "Strong " + ("Uptrend" if signals['trend']['daily'] > 0 else "Downtrend")
            else:
                trend_desc = "Mixed - Daily: " + ("Up" if signals['trend']['daily'] > 0 else "Down") + \
                           ", Hourly: " + ("Up" if signals['trend']['hourly'] > 0 else "Down")
            
            # Calculate confidence score
            confidence = abs(signals['trend']['strength']) * vol_adjustment
            
            return {
                'price': current_price,
                'price_change_24h': price_change_24h,
                'volatility': volatility,
                'trend': {
                    'daily': signals['trend']['daily'],
                    'hourly': signals['trend']['hourly'],
                    'aligned': signals['trend']['aligned'],
                    'description': trend_desc
                },
                'rsi': float(rsi.iloc[-1]),
                'volume_confirmed': signals['signals']['daily'].get('volume_confirmed', False),
                'strength': signals['trend']['strength'],
                'confidence': confidence,  # Add confidence score
                'position_size': position_size
            }
            
        except Exception as e:
            await self.log(f"Full analysis error: {str(e)}", level="error")
            raise TradingError(f"Failed to get full analysis: {str(e)}", "ANALYSIS")

    def _get_coinbase_granularity(self, timeframe: TimeFrame) -> str:
        """
        Get properly formatted Coinbase granularity string.
        
        Args:
            timeframe: TimeFrame enum value
            
        Returns:
            str: Coinbase API granularity string
            
        Raises:
            ValidationError: If timeframe is invalid
        """
        granularity_map = {
            TimeFrame.HOUR_1: "ONE_HOUR",
            TimeFrame.DAY_1: "ONE_DAY",
            TimeFrame.DAY_30: "ONE_DAY"  # 30-day analysis uses daily candles
        }
        
        if timeframe not in granularity_map:
            raise ValidationError(f"Invalid timeframe: {timeframe}")
            
        return granularity_map[timeframe]
        
    async def get_price_data(self, symbol: str, timeframe: TimeFrame) -> pd.DataFrame:
        """
        Get price data with proper error handling and validation.
        
        Args:
            symbol: Trading pair symbol
            timeframe: TimeFrame enum value
            
        Returns:
            pd.DataFrame: Price data with OHLCV columns
            
        Raises:
            TradingError: If data cannot be fetched or validated
        """
        try:
            # Format symbol
            symbol = self.data_manager._format_product_id(symbol.upper())
            
            # Calculate time range based on timeframe
            end = datetime.now()
            
            # Get timeframe configuration from constants
            timeframe_config = TradingConstants.TIMEFRAMES.get(timeframe)
            if not timeframe_config:
                raise TradingError(f"Unsupported timeframe: {timeframe}", "CONFIG")
                
            days = timeframe_config['days']
            start = end - timedelta(days=days)
            
            # Use data manager to get price data
            data = await self.data_manager.get_price_data(
                symbol,
                timeframe
            )
            
            if data is None or data.empty:
                raise DataError(f"No data returned for {symbol}")
            
            # For 30-day analysis, we need to resample daily data
            if timeframe == TimeFrame.DAY_30:
                # Resample to 30-day periods
                data = self._resample_for_monthly(data)
            
            return data
            
        except Exception as e:
            await self.log(f"Failed to get price data: {str(e)}", level="error")
            raise TradingError(f"Failed to get price data: {str(e)}", "DATA")
            
    def _resample_for_monthly(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Resample daily data to 30-day periods for monthly analysis.
        
        Args:
            data: Daily price data
            
        Returns:
            pd.DataFrame: Resampled data for 30-day analysis
        """
        try:
            # Set proper datetime index if not already
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            
            # Resample to 30-day periods
            monthly = data.resample('30D').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
            
            return monthly.dropna()
            
        except Exception as e:
            self.log(f"Error resampling data: {str(e)}", level="error")
            return data  # Return original data if resampling fails

    async def _validate_price_data(
        self, 
        data: pd.DataFrame, 
        timeframe: str,
        indicator: str
    ) -> pd.DataFrame:
        """
        Validate and clean price data for indicator calculations.
        
        Args:
            data: Raw price data DataFrame
            timeframe: Analysis timeframe
            indicator: Name of indicator for error messages
            
        Returns:
            pd.DataFrame: Cleaned and validated price data
            
        Raises:
            TradingError: If data validation fails
        """
        try:
            # Check for None or empty DataFrame
            if data is None or data.empty:
                raise TradingError(
                    f"No data available for {indicator} calculation",
                    "DATA"
                )
            
            # Get timeframe settings
            settings = self.indicator_settings.get(timeframe)
            if not settings:
                raise TradingError(f"Invalid timeframe: {timeframe}", "CONFIG")
            
            # Check minimum length based on indicator requirements
            min_periods = max(
                settings['rsi_period'],
                settings['macd_slow'],
                settings['bb_period']
            ) + 1
            
            if len(data) < min_periods:
                raise TradingError(
                    f"Insufficient data for {indicator} calculation. "
                    f"Need at least {min_periods} data points, got {len(data)}.",
                    "ANALYSIS"
                )
            
            # Validate OHLCV data
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise TradingError(
                    f"Missing required columns: {', '.join(missing_columns)}",
                    "DATA"
                )
            
            # Convert to numeric and handle missing values
            for col in required_columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Check for invalid values
            invalid_rows = data[
                data[required_columns].isna().any(axis=1) |
                data[required_columns].apply(lambda x: np.isinf(x)).any(axis=1)
            ]
            
            if not invalid_rows.empty:
                await self.log(
                    f"Found {len(invalid_rows)} invalid rows in {timeframe} data",
                    level="warning"
                )
                # Remove invalid rows
                data = data.drop(invalid_rows.index)
            
            # Validate after cleaning
            if len(data) < min_periods:
                raise TradingError(
                    f"Insufficient valid data after cleaning for {indicator}. "
                    f"Need at least {min_periods} points, got {len(data)}.",
                    "ANALYSIS"
                )
            
            # Ensure proper datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                try:
                    data.index = pd.to_datetime(data.index)
                except Exception as e:
                    raise TradingError(f"Invalid datetime index: {str(e)}", "DATA")
            
            # Sort index and remove duplicates
            data = data.sort_index()
            data = data[~data.index.duplicated(keep='last')]
            
            return data
            
        except Exception as e:
            if isinstance(e, TradingError):
                raise
            await self.log(f"Data validation error: {str(e)}", level="error")
            raise TradingError(f"Failed to validate data: {str(e)}", "DATA")
            
    async def _validate_analysis_results(
        self,
        results: Dict[str, Any],
        timeframe: str,
        indicator: str
    ) -> None:
        """
        Validate analysis results for consistency and completeness.
        
        Args:
            results: Analysis results to validate
            timeframe: Analysis timeframe
            indicator: Name of indicator for error messages
            
        Raises:
            TradingError: If validation fails
        """
        try:
            # Check for required components
            required_components = {
                'trend': [-1.0, 1.0],
                'momentum': {
                    'value': [-1.0, 1.0],
                    'rsi': [0, 100],
                    'macd': None,  # No specific range
                    'signal': None
                },
                'strength': [0.0, 1.0],
                'volume_confirmed': [False, True],
                'indicators': {
                    'rsi': [0, 100],
                    'macd': {
                        'value': None,
                        'signal': None,
                        'histogram': None
                    },
                    'bb': {
                        'upper': None,
                        'middle': None,
                        'lower': None
                    },
                    'ema': {
                        'short': None,
                        'long': None
                    }
                }
            }
            
            def validate_component(value, spec):
                if spec is None:
                    return pd.notna(value)
                if isinstance(spec, list):
                    return pd.notna(value) and (
                        isinstance(value, bool) or
                        (spec[0] <= float(value) <= spec[1])
                    )
                if isinstance(spec, dict):
                    return isinstance(value, dict) and all(
                        k in value and validate_component(value[k], v)
                        for k, v in spec.items()
                    )
                return False
            
            # Validate each component
            for component, spec in required_components.items():
                if component not in results:
                    raise TradingError(
                        f"Missing required component '{component}' in {indicator} results",
                        "ANALYSIS"
                    )
                if not validate_component(results[component], spec):
                    raise TradingError(
                        f"Invalid {component} value in {indicator} results for {timeframe}",
                        "ANALYSIS"
                    )
            
        except Exception as e:
            if isinstance(e, TradingError):
                raise
            await self.log(f"Results validation error: {str(e)}", level="error")
            raise TradingError(f"Failed to validate {indicator} results: {str(e)}", "ANALYSIS")

    def _calculate_macd(
        self, 
        data: pd.DataFrame,
        timeframe: str = "daily"
    ) -> Dict[str, pd.Series]:
        """
        Calculate MACD indicator with timeframe-specific parameters.
        
        Args:
            data: Price data DataFrame
            timeframe: Analysis timeframe
            
        Returns:
            Dict containing MACD line, signal line, and histogram
        """
        try:
            # Get timeframe-specific settings
            settings = self.indicator_settings[timeframe]
            
            # Convert to numeric and handle missing values
            close = pd.to_numeric(data['close'], errors='coerce')
            
            # Calculate EMAs with timeframe-specific periods
            ema_fast = close.ewm(span=settings['macd_fast'], adjust=False).mean()
            ema_slow = close.ewm(span=settings['macd_slow'], adjust=False).mean()
            
            # Calculate MACD line
            macd_line = ema_fast - ema_slow
            
            # Calculate signal line
            signal_line = macd_line.ewm(span=settings['macd_signal'], adjust=False).mean()
            
            # Calculate histogram
            histogram = macd_line - signal_line
            
            return {
                'macd': macd_line,
                'signal': signal_line,
                'histogram': histogram
            }
            
        except Exception as e:
            self.log(f"MACD calculation error: {str(e)}", level="error")
            return {
                'macd': pd.Series(0, index=data.index),
                'signal': pd.Series(0, index=data.index),
                'histogram': pd.Series(0, index=data.index)
            }

    async def _calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average.
        
        Args:
            data: Price series
            period: EMA period
            
        Returns:
            Series containing EMA values
        """
        try:
            return data.ewm(span=period, adjust=False).mean()
        except Exception as e:
            await self.trading_bot.log(f"EMA calculation error: {str(e)}", level="error")
            return pd.Series(0, index=data.index)

    def _consolidate_levels(self, *level_lists: List[float], tolerance: float = 0.01) -> List[float]:
        """
        Consolidate multiple lists of price levels into a single list, merging nearby levels.
        
        Args:
            *level_lists: Variable number of lists containing price levels
            tolerance: Percentage difference to consider levels as the same
            
        Returns:
            List of consolidated price levels
        """
        # Combine all levels into a single list
        all_levels = []
        for levels in level_lists:
            all_levels.extend(levels)
            
        if not all_levels:
            return []
            
        # Sort levels
        all_levels = sorted(all_levels)
        
        # Consolidate nearby levels
        consolidated = []
        current_group = [all_levels[0]]
        
        for level in all_levels[1:]:
            # Check if level is within tolerance of current group average
            group_avg = sum(current_group) / len(current_group)
            if abs(level - group_avg) / group_avg <= tolerance:
                current_group.append(level)
            else:
                # Add average of current group to consolidated list
                consolidated.append(sum(current_group) / len(current_group))
                current_group = [level]
                
        # Add final group
        if current_group:
            consolidated.append(sum(current_group) / len(current_group))
            
        return consolidated

    async def get_volume_profile(self, symbol: str) -> Dict[str, Any]:
        """
        Get volume profile analysis for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dict containing volume profile analysis
            
        Raises:
            TradingError: If analysis fails
        """
        try:
            # Get hourly data for more granular analysis
            data = await self.data_manager.get_price_data(
                self.data_manager._format_product_id(symbol.upper()),
                TimeFrame.HOUR_1
            )
            
            if data is None or len(data) < 24:  # Need at least 24 hours
                raise TradingError(
                    f"Insufficient data for {symbol}. Need at least 24 hours.",
                    "ANALYSIS"
                )
            
            # Calculate price levels
            price_range = data['high'].max() - data['low'].min()
            num_levels = 50  # Number of price levels to analyze
            level_size = price_range / num_levels
            
            # Initialize volume profile
            levels = []
            total_volume = float(data['volume'].sum())
            
            # Calculate volume for each price level
            current_price = float(data['low'].min())
            for _ in range(num_levels):
                # Find trades within this price level
                mask = (data['low'] <= current_price + level_size) & (data['high'] >= current_price)
                level_volume = float(data.loc[mask, 'volume'].sum())
                
                if level_volume > 0:
                    levels.append({
                        'price': current_price + (level_size / 2),  # Mid-point of level
                        'volume': level_volume,
                        'percentage': (level_volume / total_volume) * 100
                    })
                
                current_price += level_size
            
            # Sort levels by volume
            levels = sorted(levels, key=lambda x: x['volume'], reverse=True)
            
            return {
                'total_volume': total_volume,
                'levels': levels,
                'level_count': len(levels),
                'high_volume_zones': [
                    level for level in levels
                    if level['percentage'] > 5  # More than 5% of total volume
                ]
            }
            
        except Exception as e:
            await self.log(f"Volume profile analysis error: {str(e)}", level="error")
            raise TradingError(f"Failed to analyze volume profile: {str(e)}", "ANALYSIS")