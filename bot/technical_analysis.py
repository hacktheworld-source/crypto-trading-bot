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
            timeframe: Timeframe identifier ('monthly', 'daily', 'hourly', '1h', '1d', '30d')
            
        Returns:
            Dict containing signal analysis
        """
        try:
            # Map timeframe to settings key
            timeframe_map = {
                'hourly': 'hourly',
                '1h': 'hourly',
                'daily': 'daily',
                '1d': 'daily',
                'monthly': 'monthly',
                '30d': 'monthly'
            }
            
            settings_key = timeframe_map.get(timeframe.lower(), 'daily')  # Default to daily if unknown
            settings = self.indicator_settings[settings_key]
            
            # Calculate technical indicators
            rsi = await self.calculate_rsi(data['close'], period=settings['rsi_period'])
            macd = self._calculate_macd(data, timeframe=settings_key)
            bb = await self.calculate_bollinger_bands(data, timeframe=settings_key)
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
                
                # Handle Bollinger Bands values properly
                if bb is not None and isinstance(bb, dict) and all(k in bb for k in ['upper', 'middle', 'lower']):
                    current_bb = bb  # Use BB values directly since they're already processed
                else:
                    # Default values if BB calculation fails
                    current_bb = {
                        'upper': current_price * 1.02,
                        'middle': current_price,
                        'lower': current_price * 0.98,
                        'bandwidth': 0.02
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
            
            # Get volume analysis with timeframe consideration
            volume_analysis = self._analyze_volume_trend(data, timeframe=settings_key)
            
            # Calculate volume score
            volume_score = volume_analysis  # Already returns a float between -1 and 1
            
            # Determine signal strength with timeframe weights
            base_strength = (abs(trend_score) + abs(momentum_score) + abs(volume_score)) / 3
            timeframe_weight = (
                1.2 if timeframe == "monthly" else
                1.0 if timeframe == "daily" else
                0.8  # hourly
            )
            signal_strength = base_strength * timeframe_weight
            
            return {
                'trend': trend_score,
                'momentum': momentum_score,
                'volume': volume_score,  # Add actual volume score
                'strength': signal_strength,
                'volume_confirmed': volume_score > 0,  # Convert score to confirmation
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
                'momentum': 0.0,
                'volume': 0.0,
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
            print(f"\n=== Trend Score Calculation ===")
            print(f"Timeframe: {timeframe}")
            print(f"Current price: {price:.2f}")
            print(f"EMAs - Short: {ema_short:.2f}, Long: {ema_long:.2f}")
            print(f"BB Middle: {bb_middle:.2f}")
            print(f"MACD: {macd:.4f}")
            print(f"RSI: {rsi:.2f}")
            
            # Calculate price momentum relative to EMAs
            if ema_short != 0:
                ema_short_dist = (price - ema_short) / ema_short
            else:
                ema_short_dist = 0
            
            if ema_long != 0:
                ema_long_dist = (price - ema_long) / ema_long
            else:
                ema_long_dist = 0
            
            print(f"EMA distances - Short: {ema_short_dist:.4f}, Long: {ema_long_dist:.4f}")
            
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
            print(f"EMA Score: {ema_score:.4f}")
            
            # BB score with more realistic distance scaling
            if bb_middle != 0:
                bb_distance = abs(price - bb_middle) / bb_middle
            else:
                bb_distance = 0
            bb_score = (1.0 if price > bb_middle else -1.0)
            print(f"BB Distance: {bb_distance:.4f}")
            print(f"BB Score: {bb_score:.4f}")
            
            # MACD score with signal line consideration
            macd_abs = abs(macd)
            macd_score = (
                1.0 if macd > 0 and macd_abs > 0.002 else
                -1.0 if macd < 0 and macd_abs > 0.002 else
                0.5 if macd > 0 else
                -0.5 if macd < 0 else
                0.0
            )
            print(f"MACD Score: {macd_score:.4f}")
            
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
            print(f"RSI Score: {rsi_score:.4f}")
            
            # Weight and combine scores
            weighted_score = (
                ema_score * 0.35 +      # Trend following (primary)
                bb_score * 0.25 +       # Price position
                macd_score * 0.25 +     # Momentum
                rsi_score * 0.15        # Overbought/Oversold
            )
            print(f"Initial weighted score: {weighted_score:.4f}")
            
            if abs(weighted_score) < 0.2:
                weighted_score = 0.0  # Neutral if trend is too weak
                print("Score too weak, setting to neutral (0.0)")
            
            # Ensure the score is between -1 and 1
            final_score = max(-1.0, min(1.0, weighted_score))
            print(f"Final trend score: {final_score:.4f}")
            
            return final_score
            
        except Exception as e:
            await self.log(f"Trend score calculation error: {str(e)}", level="error")
            print(f"Error type: {type(e)}")
            print(f"Error location: {e.__traceback__.tb_lineno}")
            return 0.0  # Neutral score on error
            
    def _calculate_momentum_score(self, rsi: float, macd: float, 
                                signal: float, data: pd.DataFrame,
                                timeframe: str = "daily") -> float:
        """
        Calculate momentum score with improved context awareness.
        
        Returns:
            Float between -1 and 1 indicating momentum strength and direction
        """
        try:
            # RSI momentum with more granular zones
            rsi_score = (
                -1.0 if rsi > 80 else    # Extremely overbought
                -0.75 if rsi > 70 else   # Very overbought
                -0.5 if rsi > 65 else    # Moderately overbought
                -0.25 if rsi > 60 else   # Slightly overbought
                0.25 if rsi < 40 else    # Slightly oversold
                0.5 if rsi < 35 else     # Moderately oversold
                0.75 if rsi < 30 else    # Very oversold
                1.0 if rsi < 20 else     # Extremely oversold
                (rsi - 50) / 40          # Scaled score between -0.25 and 0.25 for middle range
            )
            
            # MACD momentum with sensitivity adjustment
            macd_diff = macd - signal
            macd_score = np.clip(macd_diff / abs(signal) if abs(signal) > 0 else 0, -1, 1)
            
            # Price momentum using multiple timeframes
            returns = data['close'].pct_change().fillna(0)
            
            # Calculate short-term and medium-term momentum
            short_term = returns.iloc[-3:].mean()  # 3 periods
            medium_term = returns.iloc[-7:].mean() # 7 periods
            
            # Weight recent momentum more heavily
            momentum_score = (short_term * 0.7 + medium_term * 0.3) * 100
            momentum_score = np.clip(momentum_score, -1, 1)
            
            # Combine scores with adjusted weights based on timeframe
            if timeframe == "monthly":
                weights = [0.4, 0.3, 0.3]  # More weight on RSI for longer timeframes
            elif timeframe == "daily":
                weights = [0.3, 0.4, 0.3]  # Balanced weights
            else:  # hourly
                weights = [0.3, 0.3, 0.4]  # More weight on price momentum
            
            weighted_score = (
                rsi_score * weights[0] +
                macd_score * weights[1] +
                momentum_score * weights[2]
            )
            
            return max(-1.0, min(1.0, weighted_score))
        except Exception as e:
            print(f"Momentum score calculation error: {str(e)}")
            return 0.0  # Return neutral score on error
        
    async def _calculate_trend_alignment(self, *trends: float) -> Dict[str, Any]:
        """
        Calculate trend alignment across multiple timeframes with improved weighting.
        
        Args:
            *trends: Trend scores for different timeframes (monthly, daily, hourly)
            
        Returns:
            Dict containing alignment analysis
        """
        try:
            if not trends:
                return {'aligned': False, 'strength': 0.0, 'direction': 'unknown'}
            
            # Assign weights to different timeframes
            weights = [0.5, 0.3, 0.2]  # Monthly (50%), Daily (30%), Hourly (20%)
            weighted_trends = [t * w for t, w in zip(trends, weights)]
            
            # Calculate weighted average trend
            weighted_avg = sum(weighted_trends)
            
            # Check trend alignment with more granular thresholds
            monthly, daily, hourly = trends if len(trends) == 3 else (0, 0, 0)
            
            # Strong trend if monthly and daily align
            strong_bullish = monthly > 0.2 and daily > 0.1
            strong_bearish = monthly < -0.2 and daily < -0.1
            
            # Moderate trend if daily and hourly align
            moderate_bullish = daily > 0.1 and hourly > 0
            moderate_bearish = daily < -0.1 and hourly < 0
            
            # Determine alignment status
            if strong_bullish or strong_bearish:
                aligned = True
                strength = abs(weighted_avg)
                direction = 'bullish' if weighted_avg > 0 else 'bearish'
            elif moderate_bullish or moderate_bearish:
                aligned = True
                strength = abs(weighted_avg) * 0.8  # Slightly reduce strength for moderate trends
                direction = 'bullish' if weighted_avg > 0 else 'bearish'
            else:
                # Mixed signals - check if there's still a dominant trend
                if abs(weighted_avg) > 0.15:  # Reduced threshold for mixed signals
                    aligned = True
                    strength = abs(weighted_avg) * 0.6  # Further reduce strength for mixed trends
                    direction = 'bullish' if weighted_avg > 0 else 'bearish'
                else:
                    aligned = False
                    strength = abs(weighted_avg)
                    direction = 'mixed'
            
            return {
                'aligned': aligned,
                'strength': float(strength),
                'direction': direction,
                'components': {
                    'monthly': float(monthly),
                    'daily': float(daily),
                    'hourly': float(hourly),
                    'weighted_avg': float(weighted_avg)
                }
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
            volume_score = self._analyze_volume_trend(data)
            
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
            volume_trend = (
                'Strongly Increasing' if volume_score > 0.5 else
                'Increasing' if volume_score > 0.2 else
                'Strongly Decreasing' if volume_score < -0.5 else
                'Decreasing' if volume_score < -0.2 else
                'Neutral'
            )
            volume_strength = (
                'Strong' if abs(volume_score) > 0.5 else
                'Moderate' if abs(volume_score) > 0.2 else
                'Weak'
            )
            
            if abs(volume_score) > 0.2:
                description += f" with {volume_trend.lower()} volume"
            
            return {
                'trend': {
                    'daily': trend_info['daily'],
                    'hourly': trend_info['hourly'],
                    'aligned': trend_info['aligned']
                },
                'strength': float(trend_info['strength']),
                'volume': {
                    'trend': volume_trend,
                    'strength': volume_strength,
                    'score': float(volume_score)
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
            volume_score = self._analyze_volume_trend(data)
            
            # Convert score to descriptive format
            volume_trend = {
                'description': (
                    'Strongly Increasing' if volume_score > 0.5 else
                    'Increasing' if volume_score > 0.2 else
                    'Strongly Decreasing' if volume_score < -0.5 else
                    'Decreasing' if volume_score < -0.2 else
                    'Neutral'
                ),
                'strength': (
                    'Strong' if abs(volume_score) > 0.5 else
                    'Moderate' if abs(volume_score) > 0.2 else
                    'Weak'
                ),
                'score': float(volume_score)
            }
            
            return {
                'volume_profile': volume_profile,
                'significant_levels': significant_levels,
                'volume_trend': volume_trend,
                'is_volume_confirmed': volume_score > 0
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

    def _analyze_volume_trend(self, data: pd.DataFrame, timeframe: str = "daily") -> float:
        """
        Analyze volume trend with price correlation and volatility consideration.
        
        Args:
            data: Price and volume data
            timeframe: Analysis timeframe (hourly/daily/monthly)
            
        Returns:
            Float between -1 and 1 indicating volume trend strength
        """
        try:
            if len(data) < 2:
                print(f"Volume Analysis: Insufficient data points ({len(data)})")
                return 0.0
            
            # Calculate returns and volume changes
            returns = data['close'].pct_change().fillna(0)
            volume_changes = data['volume'].pct_change().fillna(0)
            
            print(f"\n=== Volume Analysis Debug ===")
            print(f"Timeframe: {timeframe}")
            print(f"Data points: {len(data)}")
            print(f"Recent volume changes: {volume_changes.iloc[-5:].values}")
            
            # Adjust periods based on timeframe
            if timeframe == "hourly":
                short_period = 6    # 6 hours
                medium_period = 24  # 1 day
                long_period = 72    # 3 days
            elif timeframe == "monthly":
                short_period = 3    # 3 months
                medium_period = 6   # 6 months
                long_period = 12    # 1 year
            else:  # daily
                short_period = 3    # 3 days
                medium_period = 7   # 1 week
                long_period = 20    # 1 month
            
            # Calculate timeframe-adjusted averages
            short_vol = volume_changes.iloc[-short_period:].mean() if len(volume_changes) >= short_period else volume_changes.mean()
            medium_vol = volume_changes.iloc[-medium_period:].mean() if len(volume_changes) >= medium_period else short_vol
            
            print(f"Short-term volume change (last {short_period} periods): {short_vol:.4f}")
            print(f"Medium-term volume change (last {medium_period} periods): {medium_vol:.4f}")
            
            # Calculate volume-price correlation
            correlation = returns.iloc[-medium_period:].corr(volume_changes.iloc[-medium_period:]) if len(returns) >= medium_period else returns.corr(volume_changes)
            print(f"Price-volume correlation: {correlation:.4f}")
            
            # Calculate volume trend score
            base_score = (short_vol * 0.7 + medium_vol * 0.3)
            print(f"Base score: {base_score:.4f}")
            
            # Adjust score based on correlation
            if not np.isnan(correlation):
                # Positive correlation (price up + volume up) is bullish
                # Negative correlation (price down + volume up) is bearish
                score = base_score * (correlation if abs(correlation) > 0.3 else 0.3)
                print(f"Correlation-adjusted score: {score:.4f}")
            else:
                score = base_score
                print("Using base score (correlation is NaN)")
            
            # Adjust based on absolute volume levels
            recent_vol_ratio = (
                data['volume'].iloc[-short_period:].mean() / 
                data['volume'].iloc[-long_period:-short_period].mean() 
                if len(data) >= long_period else 1.0
            )
            print(f"Recent volume ratio: {recent_vol_ratio:.4f}")
            
            # Volume surge detection
            if recent_vol_ratio > 2.0:  # Significant volume surge
                score *= 1.5
                print("Volume surge detected (score * 1.5)")
            elif recent_vol_ratio < 0.5:  # Volume dry-up
                score *= 0.5
                print("Volume dry-up detected (score * 0.5)")
            
            # Normalize and clip
            final_score = max(-1.0, min(1.0, score))
            print(f"Final volume score: {final_score:.4f}")
            
            return final_score
            
        except Exception as e:
            print(f"Volume trend analysis error: {str(e)}")
            print(f"Error type: {type(e)}")
            print(f"Error location: {e.__traceback__.tb_lineno}")
            return 0.0

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
        data: Union[pd.DataFrame, str],
        timeframe: str = "daily"
    ) -> Dict[str, Any]:
        """
        Calculate Bollinger Bands with timeframe-specific parameters.
        
        Args:
            data: Price data DataFrame or symbol string
            timeframe: Analysis timeframe ('monthly', 'daily', 'hourly')
            
        Returns:
            Dict containing upper, middle, lower bands and bandwidth
            
        Raises:
            TradingError: If calculation fails
        """
        try:
            print(f"\n=== Debug: Bollinger Bands Input ===")
            print(f"Input type: {type(data)}")
            print(f"Timeframe: {timeframe}")
            
            # Handle case where data is a symbol string
            if isinstance(data, str):
                symbol = data
                print(f"Converting symbol {symbol} to DataFrame...")
                data = await self.data_manager.get_price_data(symbol.upper(), TimeFrame.DAY_1)
                if data is None:
                    raise ValueError(f"No data available for {symbol}")
                print(f"Converted data type: {type(data)}")
                if isinstance(data, pd.DataFrame):
                    print(f"DataFrame columns: {data.columns.tolist()}")
                    print(f"DataFrame shape: {data.shape}")

            # Validate DataFrame
            if not isinstance(data, pd.DataFrame):
                raise ValueError(f"Invalid data type: {type(data)}")

            if len(data) == 0:
                raise ValueError("Empty DataFrame provided")

            # Ensure we have the 'close' column
            if 'close' not in data.columns:
                raise ValueError("Price data missing 'close' column")

            # Map timeframe to settings key
            timeframe_map = {
                'hourly': 'hourly',
                '1h': 'hourly',
                'daily': 'daily',
                '1d': 'daily',
                'monthly': 'monthly',
                '30d': 'monthly'
            }
            
            print(f"\n=== Debug: Timeframe Mapping ===")
            print(f"Input timeframe: {timeframe}")
            settings_key = timeframe_map.get(timeframe.lower())
            print(f"Mapped to settings key: {settings_key}")
            
            if not settings_key:
                raise ValueError(f"Invalid timeframe: {timeframe}")

            # Get settings
            settings = self.indicator_settings.get(settings_key)
            if not settings:
                raise ValueError(f"No settings found for timeframe: {settings_key}")

            period = settings['bb_period']
            print(f"Using BB period: {period}")
            
            # Convert to numeric and handle missing values
            print("\n=== Debug: Price Data ===")
            print(f"First few rows of close prices:\n{data['close'].head()}")
            
            close_prices = pd.to_numeric(data['close'], errors='coerce')
            close_prices = close_prices.fillna(method='ffill').fillna(method='bfill')
            
            print(f"After numeric conversion and NA handling:\n{close_prices.head()}")
            print(f"Any NaN values: {close_prices.isna().any()}")
            
            if close_prices.isna().all():
                raise ValueError("No valid price data available")
            
            # Calculate middle band (SMA)
            middle = close_prices.rolling(window=period, min_periods=1).mean()
            
            # Calculate standard deviation
            std = close_prices.rolling(window=period, min_periods=1).std()
            
            # Calculate bands
            upper = middle + (std * 2)
            lower = middle - (std * 2)
            
            # Calculate bandwidth
            bandwidth = ((upper - lower) / middle).replace([np.inf, -np.inf], 0)
            
            print("\n=== Debug: Calculated Values ===")
            print(f"Last values:")
            print(f"Close: {close_prices.iloc[-1]}")
            print(f"Upper: {upper.iloc[-1]}")
            print(f"Middle: {middle.iloc[-1]}")
            print(f"Lower: {lower.iloc[-1]}")
            print(f"Bandwidth: {bandwidth.iloc[-1]}")
            
            # Get the last values safely
            try:
                last_close = float(close_prices.iloc[-1])
                last_upper = float(upper.iloc[-1]) if not pd.isna(upper.iloc[-1]) else last_close * 1.02
                last_middle = float(middle.iloc[-1]) if not pd.isna(middle.iloc[-1]) else last_close
                last_lower = float(lower.iloc[-1]) if not pd.isna(lower.iloc[-1]) else last_close * 0.98
                last_bandwidth = float(bandwidth.iloc[-1]) if not pd.isna(bandwidth.iloc[-1]) else 0.02
                
                print("\n=== Debug: Final Values ===")
                print(f"Final results:")
                print(f"Upper: {last_upper}")
                print(f"Middle: {last_middle}")
                print(f"Lower: {last_lower}")
                print(f"Bandwidth: {last_bandwidth}")
                
            except (IndexError, ValueError) as e:
                await self.log(f"Error getting last BB values: {str(e)}", level="error")
                raise ValueError(f"Failed to get last BB values: {str(e)}")
            
            result = {
                'upper': last_upper,
                'middle': last_middle,
                'lower': last_lower,
                'bandwidth': last_bandwidth
            }
            
            print("\n=== Debug: Returning Result ===")
            print(f"Final result dictionary: {result}")
            
            return result
            
        except Exception as e:
            print(f"\n=== Debug: Error Occurred ===")
            print(f"Error type: {type(e)}")
            print(f"Error message: {str(e)}")
            
            await self.log(f"Bollinger Bands calculation error: {str(e)}", level="error")
            if isinstance(data, pd.DataFrame) and len(data) > 0 and 'close' in data.columns:
                try:
                    current_price = float(data['close'].iloc[-1])
                    fallback_result = {
                        'upper': current_price * 1.02,
                        'middle': current_price,
                        'lower': current_price * 0.98,
                        'bandwidth': 0.02
                    }
                    print(f"\n=== Debug: Using Fallback Values ===")
                    print(f"Fallback result: {fallback_result}")
                    return fallback_result
                except Exception as nested_e:
                    print(f"Fallback error: {str(nested_e)}")
                    await self.log(f"Failed to get fallback price: {str(nested_e)}", level="error")
            raise TradingError("Failed to calculate Bollinger Bands", "ANALYSIS")

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
        Check market conditions with improved error handling and edge cases.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dict containing market analysis
            
        Raises:
            TradingError: If analysis fails
        """
        try:
            # Get price data with validation
            data = await self.data_manager.get_price_data(symbol.upper(), TimeFrame.DAY_1)
            if data is None or len(data) < 30:  # Require minimum data points
                raise TradingError("Insufficient price data for market analysis", "DATA")
            
            # Convert price data to numeric
            prices = pd.to_numeric(data['close'], errors='coerce')
            volumes = pd.to_numeric(data['volume'], errors='coerce')
            
            # Handle missing values
            if prices.isna().all() or volumes.isna().all():
                raise TradingError("Invalid price or volume data", "DATA")
            
            # Get current values safely
            try:
                current_price = float(prices.iloc[-1])
                current_volume = float(volumes.iloc[-1])
            except (IndexError, ValueError) as e:
                raise TradingError(f"Failed to get current values: {str(e)}", "DATA")
            
            # Calculate volatility with error handling
            try:
                returns = prices.pct_change().dropna()
                volatility = returns.std() * np.sqrt(252)  # Annualized
                volatility_threshold = returns.std().mean() * 2  # Dynamic threshold
                is_volatile = volatility > volatility_threshold
            except Exception as e:
                self.log(f"Volatility calculation error: {str(e)}", level="warning")
                volatility = 0
                is_volatile = False
            
            # Calculate price ranges safely
            try:
                week_high = float(prices.iloc[-7:].max())
                week_low = float(prices.iloc[-7:].min())
                price_range_7d = ((week_high - week_low) / week_low) * 100 if week_low > 0 else 0
            except Exception as e:
                self.log(f"Price range calculation error: {str(e)}", level="warning")
                price_range_7d = 0
            
            # Calculate ATR
            atr = self._calculate_atr(data)
            
            # Volume analysis with error handling
            try:
                avg_volume = volumes.rolling(window=20, min_periods=1).mean()
                volume_ratio = current_volume / float(avg_volume.iloc[-1]) if not pd.isna(avg_volume.iloc[-1]) and avg_volume.iloc[-1] > 0 else 1.0
                
                volume_score = self._analyze_volume_trend(data)
                # Convert volume score to descriptive format
                volume_trend = {
                    'description': 'Strongly Increasing' if volume_score > 0.5 else
                                 'Increasing' if volume_score > 0.2 else
                                 'Strongly Decreasing' if volume_score < -0.5 else
                                 'Decreasing' if volume_score < -0.2 else
                                 'Neutral',
                    'strength': 'Strong' if abs(volume_score) > 0.5 else
                               'Moderate' if abs(volume_score) > 0.2 else
                               'Weak',
                    'is_favorable': volume_score > 0
                }
            except Exception as e:
                self.log(f"Volume analysis error: {str(e)}", level="warning")
                volume_ratio = 1.0
                volume_trend = {'description': 'Neutral', 'strength': 'Normal', 'is_favorable': True}
            
            # Market alignment check
            try:
                signals = await self.get_signals(symbol)
                trend_score = signals['trend']['daily']
                momentum_score = signals['signals']['daily']['momentum']
                
                market_score = (trend_score + momentum_score) / 2
                market_aligned = (
                    trend_score > 0 and momentum_score > 0 and volume_trend['is_favorable'] and not is_volatile
                )
            except Exception as e:
                self.log(f"Market alignment check error: {str(e)}", level="warning")
                market_score = 0
                market_aligned = False
            
            return {
                'volatility': {
                    'value': float(volatility),
                    'is_high': is_volatile,
                    'threshold': float(volatility_threshold)
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
        """
        Calculate Average True Range with proper error handling.
        
        Args:
            data: DataFrame with high, low, close prices
            period: ATR period
            
        Returns:
            Series containing ATR values
        """
        try:
            # Convert price data to numeric and handle missing values
            high = pd.to_numeric(data['high'], errors='coerce')
            low = pd.to_numeric(data['low'], errors='coerce')
            close = pd.to_numeric(data['close'], errors='coerce')
            
            # Calculate true range components
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            # Get maximum of the three components
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate ATR with proper minimum periods
            atr = tr.rolling(window=period, min_periods=1).mean()
            
            # Fill any remaining NaN values with 0
            return atr.fillna(0)
            
        except Exception as e:
            # Use synchronous logging for non-async methods
            print(f"ATR calculation error: {str(e)}")
            return pd.Series(0, index=data.index)
        
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