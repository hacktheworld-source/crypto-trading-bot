from typing import Optional, List, Union, Literal, Dict, Any, Callable
from datetime import datetime, timedelta
import asyncio
import pandas as pd
import logging
from bot.constants import TradingConstants, TimeFrame

class CommandHandler:
    """
    Handles Discord command processing for the trading bot.
    
    Attributes:
        ERROR_PREFIX (str): Prefix for error messages
        SUCCESS_PREFIX (str): Prefix for success messages
        trading_bot (TradingBot): Reference to the main trading bot instance
        commands (Dict[str, callable]): Mapping of command names to handler methods
    """
    
    ERROR_PREFIX: str = "❌ "
    SUCCESS_PREFIX: str = "✅ "
    
    def __init__(self, trading_bot: 'TradingBot') -> None:
        """
        Initialize the command handler.
        
        Args:
            trading_bot: Instance of the trading bot to handle commands for
        """
        self.trading_bot = trading_bot
        self.data_manager = trading_bot.data_manager
        self.commands = self._initialize_commands()
    
    def _initialize_commands(self) -> Dict[str, callable]:
        """
        Initialize the command mapping dictionary.
        
        Returns:
            Dict mapping command names to their handler methods
        """
        return {
            'add': self.add_coin,
            'addcoin': self.add_coin,
            'remove': self.remove_coin,
            'removecoin': self.remove_coin,
            'list': self.list_coins,
            'position': self.get_position_details,
            'metrics': self.get_position_metrics,
            'testapi': self.test_api,
            'price': self.get_price,
            'rsi': self.get_rsi,
            'ma': self.get_ma_analysis,
            'volume': self.get_volume_analysis,
            'sentiment': self.get_sentiment_analysis,
            'paper': self.handle_paper_commands,
            'start': self.start_trading,
            'stop': self.stop_trading,
            'status': self.get_status,
            'balance': self.get_balance,
            'help': self.get_help,
            'ping': self.ping,
            'version': self.version,
            'stats': self.get_stats,
            'bb': self.get_bb_analysis,
            'conditions': self.get_market_conditions,
            'commands': self.get_help,
            'risk': self.get_risk_analysis,
            'portfolio': self.get_portfolio_analysis,
            'alerts': self.get_risk_alerts,
            'limits': self.show_risk_limits,
            'performance': self.get_performance,
            'set_risk': self.set_stop_loss,
            'set_max_positions': self.set_max_positions,
            'set_risk_per_trade': self.set_risk_per_trade,
            'set_max_drawdown': self.set_max_drawdown,
            'set_daily_var': self.set_daily_var,
            'set_trailing_stop': self.set_trailing_stop,
            'set_take_profit': self.set_take_profit,
            'signals': self.get_signals
        }
    
    async def handle_command(self, command: str, args: List[str], message: Message) -> None:
        """
        Handle incoming bot commands.
        
        Args:
            command: Command name without prefix
            args: List of command arguments
            message: Original message object
            
        Raises:
            CommandError: If command handling fails
        """
        try:
            # Get symbol from args or use default
            symbol = args[0].upper() if args else self.settings['default_symbol']
            symbol = self.technical_analyzer.data_manager._format_product_id(symbol)
            
            # Handle commands
            if command == "help":
                await self._handle_help(message)
            elif command == "price":
                await self._handle_price(symbol, message)
            elif command == "signals":
                await self._handle_signals(symbol, message)
            elif command == "conditions":
                await self._handle_conditions(symbol, message)
            elif command == "rsi":
                await self._handle_rsi(symbol, message)
            elif command == "macd":
                await self._handle_macd(symbol, message)
            elif command == "bb":
                await self._handle_bb(symbol, message)
            elif command == "volume":
                await self._handle_volume(symbol, message)
            else:
                await message.reply(f"Unknown command: {command}")
                
        except Exception as e:
            error_msg = f"Command failed: {str(e)}"
            await self.log(error_msg, level="error")
            await message.reply(error_msg)
            
    async def _handle_help(self, message: Message) -> None:
        """Send help message with available commands"""
        help_text = (
            "Available Commands:\n"
            "```\n"
            "!help - Show this help message\n"
            "!price [symbol] - Get current price and 24h change\n"
            "!signals [symbol] - Get trading signals\n"
            "!conditions [symbol] - Get market conditions\n"
            "!rsi [symbol] - Get RSI indicator\n"
            "!macd [symbol] - Get MACD indicator\n"
            "!bb [symbol] - Get Bollinger Bands\n"
            "!volume [symbol] - Get volume analysis\n"
            "```\n"
            "Note: [symbol] is optional. Default: BTC-USD"
        )
        await message.reply(help_text)
        
    async def _handle_price(self, symbol: str, message: Message) -> None:
        """Handle price command"""
        try:
            # Get current price data
            data = await self.technical_analyzer.data_manager.get_price_data(
                symbol, TimeFrame.HOUR_1, limit=25
            )
            
            if data is None or len(data) < 2:
                raise CommandError("Failed to fetch price data")
                
            # Calculate price change
            current_price = float(data['close'].iloc[-1])
            prev_price = float(data['close'].iloc[-2])
            price_change = ((current_price - prev_price) / prev_price) * 100
            
            # Format response
            response = (
                f"**{symbol}**\n"
                f"Price: ${current_price:,.2f}\n"
                f"24h Change: {price_change:+.2f}%"
            )
            await message.reply(response)
            
        except Exception as e:
            raise CommandError(f"Price command failed: {str(e)}")
            
    async def _handle_signals(self, symbol: str, message: Message) -> None:
        """Handle signals command"""
        try:
            signals = await self.technical_analyzer.get_signals(symbol)
            
            # Format response
            daily = signals['signals']['daily']
            hourly = signals['signals']['1h']
            
            response = (
                f"**{symbol} Signals**\n"
                f"```\n"
                f"Daily Timeframe:\n"
                f"  Trend: {self._format_signal_strength(daily['trend'])}\n"
                f"  Momentum: {self._format_signal_strength(daily['momentum'])}\n"
                f"  RSI: {daily['indicators']['rsi']:.1f}\n"
                f"  MACD: {daily['indicators']['macd']['value']:.2f}\n\n"
                f"Hourly Timeframe:\n"
                f"  Trend: {self._format_signal_strength(hourly['trend'])}\n"
                f"  Momentum: {self._format_signal_strength(hourly['momentum'])}\n"
                f"  RSI: {hourly['indicators']['rsi']:.1f}\n"
                f"  MACD: {hourly['indicators']['macd']['value']:.2f}\n"
                f"```\n"
                f"Trend Alignment: {'✅' if signals['trend']['aligned'] else '❌'}"
            )
            await message.reply(response)
            
        except Exception as e:
            raise CommandError(f"Signals command failed: {str(e)}")
            
    async def _handle_conditions(self, symbol: str, message: Message) -> None:
        """Handle market conditions command"""
        try:
            conditions = await self.technical_analyzer.check_market_conditions(symbol)
            
            # Format response
            response = (
                f"**{symbol} Market Conditions**\n"
                f"```\n"
                f"Volatility: {conditions['volatility']['value']:.2%}\n"
                f"Price Range (7d): {conditions['price_action']['range_7d']:.2f}%\n"
                f"Volume: {conditions['volume']['trend']}\n"
                f"Market Score: {conditions['market_alignment']['score']:.2f}\n\n"
                f"Trading Summary:\n"
                f"  Suitable: {'Yes' if conditions['trading_summary']['suitable'] else 'No'}\n"
                f"  Confidence: {conditions['trading_summary']['confidence']:.2f}\n"
                f"  {conditions['trading_summary']['recommendation']}\n"
                f"```"
            )
            await message.reply(response)
            
        except Exception as e:
            raise CommandError(f"Conditions command failed: {str(e)}")
            
    async def _handle_rsi(self, symbol: str, message: Message) -> None:
        """Handle RSI command"""
        try:
            data = await self.technical_analyzer.data_manager.get_price_data(
                symbol, TimeFrame.HOUR_1
            )
            rsi = self.technical_analyzer._calculate_rsi(data)
            
            current_rsi = float(rsi.iloc[-1])
            prev_rsi = float(rsi.iloc[-2])
            rsi_change = current_rsi - prev_rsi
            
            # Get RSI zones
            zone = (
                "Overbought" if current_rsi > 70
                else "Oversold" if current_rsi < 30
                else "Neutral"
            )
            
            response = (
                f"**{symbol} RSI Analysis**\n"
                f"```\n"
                f"Current RSI: {current_rsi:.1f}\n"
                f"Change: {rsi_change:+.1f}\n"
                f"Zone: {zone}\n"
                f"```"
            )
            await message.reply(response)
            
        except Exception as e:
            raise CommandError(f"RSI command failed: {str(e)}")
            
    async def _handle_macd(self, symbol: str, message: Message) -> None:
        """Handle MACD command"""
        try:
            data = await self.technical_analyzer.data_manager.get_price_data(
                symbol, TimeFrame.HOUR_1
            )
            macd = self.technical_analyzer._calculate_macd(data)
            
            current_macd = float(macd['macd'].iloc[-1])
            current_signal = float(macd['signal'].iloc[-1])
            current_hist = float(macd['histogram'].iloc[-1])
            
            # Determine trend
            trend = (
                "Bullish" if current_macd > current_signal
                else "Bearish" if current_macd < current_signal
                else "Neutral"
            )
            
            response = (
                f"**{symbol} MACD Analysis**\n"
                f"```\n"
                f"MACD: {current_macd:.2f}\n"
                f"Signal: {current_signal:.2f}\n"
                f"Histogram: {current_hist:.2f}\n"
                f"Trend: {trend}\n"
                f"```"
            )
            await message.reply(response)
            
        except Exception as e:
            raise CommandError(f"MACD command failed: {str(e)}")
            
    async def _handle_bb(self, symbol: str, message: Message) -> None:
        """Handle Bollinger Bands command"""
        try:
            data = await self.technical_analyzer.data_manager.get_price_data(
                symbol, TimeFrame.HOUR_1
            )
            bb = self.technical_analyzer._calculate_bollinger_bands(data)
            
            current_price = float(data['close'].iloc[-1])
            upper = float(bb['upper'].iloc[-1])
            middle = float(bb['middle'].iloc[-1])
            lower = float(bb['lower'].iloc[-1])
            
            # Calculate price position
            bb_width = ((upper - lower) / middle) * 100
            position = (current_price - lower) / (upper - lower) * 100
            
            # Determine zone
            zone = (
                "Upper Band" if position > 80
                else "Lower Band" if position < 20
                else "Middle Band"
            )
            
            response = (
                f"**{symbol} Bollinger Bands**\n"
                f"```\n"
                f"Current Price: ${current_price:,.2f}\n"
                f"Upper Band: ${upper:,.2f}\n"
                f"Middle Band: ${middle:,.2f}\n"
                f"Lower Band: ${lower:,.2f}\n"
                f"Band Width: {bb_width:.2f}%\n"
                f"Position: {zone} ({position:.1f}%)\n"
                f"```"
            )
            await message.reply(response)
            
        except Exception as e:
            raise CommandError(f"Bollinger Bands command failed: {str(e)}")
            
    async def _handle_volume(self, symbol: str, message: Message) -> None:
        """Handle volume analysis command"""
        try:
            data = await self.technical_analyzer.data_manager.get_price_data(
                symbol, TimeFrame.HOUR_1
            )
            volume_analysis = self.technical_analyzer._analyze_volume_trend(data)
            
            response = (
                f"**{symbol} Volume Analysis**\n"
                f"```\n"
                f"Trend: {volume_analysis['description']}\n"
                f"Strength: {volume_analysis['strength']}\n"
                f"Score: {volume_analysis['score']:.2f}\n"
                f"Trading Conditions: {'Favorable' if volume_analysis['is_favorable'] else 'Unfavorable'}\n"
                f"```"
            )
            await message.reply(response)
            
        except Exception as e:
            raise CommandError(f"Volume analysis command failed: {str(e)}")
            
    def _format_signal_strength(self, value: float) -> str:
        """Format signal strength as arrows"""
        if abs(value) < 0.2:
            return "➡️ Neutral"
        elif value > 0:
            return "⬆️ " + ("Strong" if value > 0.5 else "Weak") + " Bullish"
        else:
            return "⬇️ " + ("Strong" if value < -0.5 else "Weak") + " Bearish"