from typing import Optional, List, Union, Literal, Dict, Any, Callable
from datetime import datetime, timedelta
import asyncio
import pandas as pd
import logging
from discord import Message
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
        self.message_formatter = trading_bot.message_formatter
        self.commands = self._initialize_commands()
    
    def _initialize_commands(self) -> Dict[str, callable]:
        """
        Initialize the command mapping dictionary.
        
        Returns:
            Dict mapping command names to their handler methods
        """
        return {
            'add': self._handle_add_coin,
            'addcoin': self._handle_add_coin,
            'remove': self._handle_remove_coin,
            'removecoin': self._handle_remove_coin,
            'list': self._handle_list_coins,
            'position': self._handle_position_details,
            'metrics': self._handle_position_metrics,
            'testapi': self._handle_test_api,
            'price': self._handle_price,
            'rsi': self._handle_rsi,
            'ma': self._handle_ma_analysis,
            'volume': self._handle_volume_analysis,
            'sentiment': self._handle_sentiment_analysis,
            'paper': self._handle_paper_commands,
            'start': self._handle_start_trading,
            'stop': self._handle_stop_trading,
            'status': self._handle_status,
            'balance': self._handle_balance,
            'help': self._handle_help,
            'ping': self._handle_ping,
            'version': self._handle_version,
            'stats': self._handle_stats,
            'bb': self._handle_bb_analysis,
            'conditions': self._handle_market_conditions,
            'commands': self._handle_help,
            'risk': self._handle_risk_analysis,
            'portfolio': self._handle_portfolio_analysis,
            'alerts': self._handle_risk_alerts,
            'limits': self._handle_risk_limits,
            'performance': self._handle_performance,
            'set_risk': self._handle_set_stop_loss,
            'set_max_positions': self._handle_set_max_positions,
            'set_risk_per_trade': self._handle_set_risk_per_trade,
            'set_max_drawdown': self._handle_set_max_drawdown,
            'set_daily_var': self._handle_set_daily_var,
            'set_trailing_stop': self._handle_set_trailing_stop,
            'set_take_profit': self._handle_set_take_profit,
            'signals': self._handle_signals
        }
    
    async def handle_command(self, command: str, *args: str) -> str:
        """
        Handle incoming bot commands.
        
        Args:
            command: Command name without prefix
            *args: Command arguments
            
        Returns:
            str: Response message
            
        Raises:
            CommandError: If command handling fails
        """
        try:
            # Get the command handler
            handler = self.commands.get(command.lower())
            if not handler:
                return self.message_formatter.format_error(f"Unknown command: {command}")

            # Execute the command
            return await handler(*args)
            
        except Exception as e:
            error_msg = f"Command failed: {str(e)}"
            await self.trading_bot.log(error_msg, level="error")
            return self.message_formatter.format_error(error_msg)
    
    async def _handle_help(self, *args) -> str:
        """Get help message with available commands"""
        help_text = (
            "Available Commands:\n"
            "```\n"
            "Basic Commands:\n"
            "!help - Show this help message\n"
            "!status - Get bot status\n"
            "!ping - Check bot response\n"
            "!version - Show bot version\n"
            "!testapi - Test API connection\n\n"
            
            "Trading Commands:\n"
            "!add <symbol> - Add coin to watchlist\n"
            "!remove <symbol> - Remove coin from watchlist\n"
            "!list - List watched coins\n"
            "!paper <on|off> - Control paper trading\n"
            "!start - Start trading\n"
            "!stop - Stop trading\n\n"
            
            "Market Analysis:\n"
            "!price <symbol> - Get current price\n"
            "!signals <symbol> - Get trading signals\n"
            "!conditions <symbol> - Get market conditions\n"
            "!rsi <symbol> - Get RSI indicator\n"
            "!bb <symbol> - Get Bollinger Bands\n"
            "!volume <symbol> - Get volume analysis\n"
            "!sentiment <symbol> - Get market sentiment\n\n"
            
            "Position & Portfolio:\n"
            "!position [symbol] - Get position details\n"
            "!metrics - Get trading metrics\n"
            "!portfolio - Get portfolio analysis\n"
            "!balance - Get account balance\n"
            "!performance - Get performance metrics\n\n"
            
            "Risk Management:\n"
            "!risk [symbol] - Get risk analysis\n"
            "!alerts - Get risk alerts\n"
            "!limits - Show risk limits\n"
            "!set_risk <percentage> - Set stop loss\n"
            "!set_max_positions <count> - Set max positions\n"
            "!set_risk_per_trade <percentage> - Set risk per trade\n"
            "!set_max_drawdown <percentage> - Set max drawdown\n"
            "!set_daily_var <percentage> - Set daily VaR\n"
            "!set_trailing_stop <percentage> - Set trailing stop\n"
            "!set_take_profit <percentage> - Set take profit\n"
            "```\n"
            "Note: <required> [optional] parameters. Default symbol: BTC-USD"
        )
        return help_text
        
    async def _handle_price(self, symbol: str = "BTC-USD") -> str:
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
            return (
                f"**{symbol}**\n"
                f"Price: ${current_price:,.2f}\n"
                f"24h Change: {price_change:+.2f}%"
            )
            
        except Exception as e:
            raise CommandError(f"Price command failed: {str(e)}")

    async def _handle_add_coin(self, symbol: str) -> str:
        """Add a coin to the watchlist"""
        try:
            if await self.trading_bot.add_coin(symbol):
                return self.message_formatter.format_notification(
                    f"Added {symbol} to watchlist", 
                    "success"
                )
            return self.message_formatter.format_error(f"Failed to add {symbol}")
        except Exception as e:
            return self.message_formatter.format_error(str(e))

    async def _handle_remove_coin(self, symbol: str) -> str:
        """Remove a coin from the watchlist"""
        try:
            if await self.trading_bot.remove_coin(symbol):
                return self.message_formatter.format_notification(
                    f"Removed {symbol} from watchlist",
                    "success"
                )
            return self.message_formatter.format_error(f"Failed to remove {symbol}")
        except Exception as e:
            return self.message_formatter.format_error(str(e))

    async def _handle_list_coins(self) -> str:
        """List all watched coins"""
        try:
            coins = self.trading_bot.watched_symbols
            if not coins:
                return "No coins in watchlist"
            return "Watched Coins:\n```\n" + "\n".join(sorted(coins)) + "\n```"
        except Exception as e:
            return self.message_formatter.format_error(str(e))

    async def _handle_position_details(self, symbol: Optional[str] = None) -> str:
        """Get position details for a symbol or all positions"""
        try:
            if symbol:
                position = self.trading_bot.positions.get(symbol.upper())
                if not position:
                    return f"No position found for {symbol}"
                return self.message_formatter.format_position_update(position)
            else:
                positions = self.trading_bot.positions
                if not positions:
                    return "No open positions"
                return "\n".join(
                    self.message_formatter.format_position_update(pos)
                    for pos in positions.values()
                )
        except Exception as e:
            return self.message_formatter.format_error(str(e))

    async def _handle_position_metrics(self) -> str:
        """Get trading metrics"""
        try:
            metrics = await self.trading_bot.get_trading_stats()
            return (
                "Trading Metrics:\n```\n"
                f"Total Trades: {metrics['total_trades']}\n"
                f"Win Rate: {metrics['win_rate']*100:.1f}%\n"
                f"Average Profit: ${metrics['avg_profit']:.2f}\n"
                f"Max Drawdown: {metrics['max_drawdown']*100:.1f}%\n"
                f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
                "```"
            )
        except Exception as e:
            return self.message_formatter.format_error(str(e))

    async def _handle_test_api(self) -> str:
        """Test API connection"""
        try:
            price = await self.trading_bot.test_api_connection()
            return self.message_formatter.format_notification(
                f"API connection successful! BTC Price: ${price:,.2f}",
                "success"
            )
        except Exception as e:
            return self.message_formatter.format_error(f"API test failed: {str(e)}")

    async def _handle_ping(self) -> str:
        """Simple ping command"""
        return "Pong! 🏓"

    async def _handle_version(self) -> str:
        """Get bot version"""
        return "Crypto Trading Bot v1.0.0"

    async def _handle_status(self) -> str:
        """Get bot status"""
        try:
            return await self.trading_bot.get_status()
        except Exception as e:
            return self.message_formatter.format_error(str(e))

    async def _handle_balance(self) -> str:
        """Get account balance"""
        try:
            balance = await self.trading_bot.get_account_balance()
            return f"Account Balance: ${balance:,.2f}"
        except Exception as e:
            return self.message_formatter.format_error(str(e))

    async def _handle_start_trading(self) -> str:
        """Start the trading bot"""
        try:
            if self.trading_bot.trading_active:
                return "Trading already active"
            self.trading_bot.trading_active = True
            return self.message_formatter.format_notification("Trading started", "success")
        except Exception as e:
            return self.message_formatter.format_error(str(e))

    async def _handle_stop_trading(self) -> str:
        """Stop the trading bot"""
        try:
            if not self.trading_bot.trading_active:
                return "Trading already stopped"
            self.trading_bot.trading_active = False
            return self.message_formatter.format_notification("Trading stopped", "success")
        except Exception as e:
            return self.message_formatter.format_error(str(e))

    def _format_signal_strength(self, value: float) -> str:
        """Format signal strength as arrows"""
        if abs(value) < 0.2:
            return "➡️ Neutral"
        elif value > 0:
            return "⬆️ " + ("Strong" if value > 0.5 else "Weak") + " Bullish"
        else:
            return "⬇️ " + ("Strong" if value < -0.5 else "Weak") + " Bearish"

    # Core Command Methods
    async def _handle_signals(self, symbol: str = "BTC-USD") -> str:
        """Get trading signals"""
        try:
            signals = await self.trading_bot.technical_analyzer.get_signals(symbol)
            return (
                f"**{symbol} Trading Signals**\n```\n"
                f"Daily Trend: {self._format_signal_strength(signals['trend']['daily'])}\n"
                f"Hourly Trend: {self._format_signal_strength(signals['trend']['1h'])}\n"
                f"Trend Aligned: {'Yes' if signals['trend']['aligned'] else 'No'}\n"
                f"Strength: {signals['trend']['strength']:.2f}\n"
                "```"
            )
        except Exception as e:
            return self.message_formatter.format_error(str(e))

    async def _handle_market_conditions(self, symbol: str = "BTC-USD") -> str:
        """Get market conditions"""
        try:
            conditions = await self.trading_bot.technical_analyzer.check_market_conditions(symbol)
            return (
                f"**{symbol} Market Conditions**\n```\n"
                f"Volatility: {conditions['volatility']['value']:.2%}\n"
                f"Price Range (7d): {conditions['price_action']['range_7d']:.2f}%\n"
                f"Volume: {conditions['volume']['trend']}\n"
                f"Market Score: {conditions['market_alignment']['score']:.2f}\n\n"
                f"Trading Summary:\n"
                f"  Suitable: {'Yes' if conditions['trading_summary']['suitable'] else 'No'}\n"
                f"  Confidence: {conditions['trading_summary']['confidence']:.2f}\n"
                f"  {conditions['trading_summary']['recommendation']}\n"
                "```"
            )
        except Exception as e:
            return self.message_formatter.format_error(str(e))

    async def _handle_rsi(self, symbol: str = "BTC-USD") -> str:
        """Get RSI analysis"""
        try:
            data = await self.trading_bot.data_manager.get_price_data(symbol, TimeFrame.HOUR_1)
            rsi = await self.trading_bot.technical_analyzer.calculate_rsi(data['close'])
            
            current_rsi = float(rsi.iloc[-1])
            prev_rsi = float(rsi.iloc[-2])
            rsi_change = current_rsi - prev_rsi
            
            # Get RSI zones
            zone = (
                "Overbought" if current_rsi > 70
                else "Oversold" if current_rsi < 30
                else "Neutral"
            )
            
            return (
                f"**{symbol} RSI Analysis**\n```\n"
                f"Current RSI: {current_rsi:.1f}\n"
                f"Change: {rsi_change:+.1f}\n"
                f"Zone: {zone}\n"
                "```"
            )
        except Exception as e:
            return self.message_formatter.format_error(str(e))

    async def _handle_bb_analysis(self, symbol: str = "BTC-USD") -> str:
        """Get Bollinger Bands analysis"""
        try:
            bb = await self.trading_bot.technical_analyzer.calculate_bollinger_bands(symbol)
            return (
                f"**{symbol} Bollinger Bands**\n```\n"
                f"Upper: ${bb['upper']:,.2f}\n"
                f"Middle: ${bb['middle']:,.2f}\n"
                f"Lower: ${bb['lower']:,.2f}\n"
                f"Bandwidth: {bb['bandwidth']:.2f}%\n"
                f"Signal: {bb['signal']}\n"
                "```"
            )
        except Exception as e:
            return self.message_formatter.format_error(str(e))

    async def _handle_volume_analysis(self, symbol: str = "BTC-USD") -> str:
        """Get volume analysis"""
        try:
            analysis = await self.trading_bot.technical_analyzer.analyze_volume_profile(symbol)
            return (
                f"**{symbol} Volume Analysis**\n```\n"
                f"24h Volume: {analysis['volume']:.2f}\n"
                f"Trend: {analysis['volume_trend']['description']}\n"
                f"Strength: {analysis['volume_trend']['strength']}\n"
                "```"
            )
        except Exception as e:
            return self.message_formatter.format_error(str(e))

    async def _handle_sentiment_analysis(self, symbol: str = "BTC-USD") -> str:
        """Get market sentiment analysis"""
        try:
            sentiment = await self.trading_bot.analyze_market_sentiment(symbol)
            return (
                f"**{symbol} Sentiment Analysis**\n```\n"
                f"Overall: {sentiment['overall_sentiment']}\n"
                f"Score: {sentiment['sentiment_score']:.2f}\n"
                f"Strength: {sentiment['strength']:.2f}\n"
                f"Timeframes Aligned: {'Yes' if sentiment['timeframes_aligned'] else 'No'}\n"
                "```"
            )
        except Exception as e:
            return self.message_formatter.format_error(str(e))

    async def _handle_paper_commands(self, action: str = "status") -> str:
        """Handle paper trading commands"""
        try:
            if action == "on":
                self.trading_bot.paper_trading = True
                return self.message_formatter.format_notification("Paper trading enabled", "success")
            elif action == "off":
                self.trading_bot.paper_trading = False
                return self.message_formatter.format_notification("Paper trading disabled", "success")
            else:
                return f"Paper Trading: {'Enabled' if self.trading_bot.paper_trading else 'Disabled'}"
        except Exception as e:
            return self.message_formatter.format_error(str(e))

    async def _handle_risk_analysis(self, symbol: Optional[str] = None) -> str:
        """Get risk analysis"""
        try:
            if symbol:
                position = self.trading_bot.positions.get(symbol.upper())
                if not position:
                    return f"No position found for {symbol}"
                risk = await self.trading_bot.risk_manager.check_position_risk(position)
                return (
                    f"Risk Analysis for {symbol}:\n```\n"
                    f"Risk Level: {risk['level']}\n"
                    f"Acceptable: {'Yes' if risk['acceptable'] else 'No'}\n"
                    f"Reason: {risk['reason']}\n"
                    "```"
                )
            else:
                return (
                    "Portfolio Risk:\n```\n"
                    f"Max Positions: {self.trading_bot.risk_manager.max_positions}\n"
                    f"Current Positions: {len(self.trading_bot.positions)}\n"
                    f"Risk Per Trade: {self.trading_bot.config.RISK_PER_TRADE*100:.1f}%\n"
                    "```"
                )
        except Exception as e:
            return self.message_formatter.format_error(str(e))

    async def _handle_portfolio_analysis(self) -> str:
        """Get portfolio analysis"""
        try:
            exposure = await self.trading_bot.get_total_exposure()
            daily_pnl = await self.trading_bot.get_daily_pnl()
            return (
                "Portfolio Analysis:\n```\n"
                f"Total Exposure: {exposure*100:.1f}%\n"
                f"Daily P/L: ${daily_pnl:,.2f}\n"
                f"Position Count: {len(self.trading_bot.positions)}\n"
                "```"
            )
        except Exception as e:
            return self.message_formatter.format_error(str(e))

    async def _handle_risk_alerts(self) -> str:
        """Get risk alerts"""
        try:
            alerts = []
            for symbol, position in self.trading_bot.positions.items():
                risk = await self.trading_bot.risk_manager.check_position_risk(position)
                if not risk['acceptable']:
                    alerts.append(f"{symbol}: {risk['reason']}")
            
            if not alerts:
                return "No risk alerts"
            return "Risk Alerts:\n```\n" + "\n".join(alerts) + "\n```"
        except Exception as e:
            return self.message_formatter.format_error(str(e))

    async def _handle_risk_limits(self) -> str:
        """Show risk limits"""
        return (
            "Risk Limits:\n```\n"
            f"Max Positions: {self.trading_bot.risk_manager.max_positions}\n"
            f"Risk Per Trade: {self.trading_bot.config.RISK_PER_TRADE*100:.1f}%\n"
            f"Max Drawdown: {self.trading_bot.config.RISK_MAX_DRAWDOWN*100:.1f}%\n"
            f"Daily VaR: {self.trading_bot.config.RISK_DAILY_VAR*100:.1f}%\n"
            "```"
        )

    async def _handle_performance(self) -> str:
        """Get performance metrics"""
        try:
            metrics = await self.trading_bot.calculate_performance_metrics()
            return (
                "Performance Metrics:\n```\n"
                f"Win Rate: {metrics['win_rate']*100:.1f}%\n"
                f"Average Profit: ${metrics['avg_profit']:.2f}\n"
                f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
                f"Max Drawdown: {metrics['max_drawdown']*100:.1f}%\n"
                f"Total Trades: {metrics['total_trades']}\n"
                "```"
            )
        except Exception as e:
            return self.message_formatter.format_error(str(e))

    async def _handle_set_stop_loss(self, percentage: str) -> str:
        """Set stop loss percentage"""
        try:
            value = float(percentage)
            if value <= 0 or value > 20:
                return self.message_formatter.format_error("Invalid stop loss percentage. Must be between 0 and 20")
            self.trading_bot.config.STOP_LOSS_PERCENTAGE = value
            return self.message_formatter.format_notification(f"Stop loss set to {value}%", "success")
        except ValueError:
            return self.message_formatter.format_error("Invalid percentage value")
        except Exception as e:
            return self.message_formatter.format_error(str(e))

    async def _handle_set_max_positions(self, count: str) -> str:
        """Set maximum positions"""
        try:
            value = int(count)
            if value <= 0 or value > 20:
                return self.message_formatter.format_error("Invalid position count. Must be between 1 and 20")
            self.trading_bot.risk_manager.max_positions = value
            return self.message_formatter.format_notification(f"Maximum positions set to {value}", "success")
        except ValueError:
            return self.message_formatter.format_error("Invalid count value")
        except Exception as e:
            return self.message_formatter.format_error(str(e))

    async def _handle_set_risk_per_trade(self, percentage: str) -> str:
        """Set risk per trade percentage"""
        try:
            value = float(percentage)
            if value <= 0 or value > 5:
                return self.message_formatter.format_error("Invalid risk percentage. Must be between 0 and 5")
            self.trading_bot.config.RISK_PER_TRADE = value / 100
            return self.message_formatter.format_notification(f"Risk per trade set to {value}%", "success")
        except ValueError:
            return self.message_formatter.format_error("Invalid percentage value")
        except Exception as e:
            return self.message_formatter.format_error(str(e))

    async def _handle_set_max_drawdown(self, percentage: str) -> str:
        """Set maximum drawdown percentage"""
        try:
            value = float(percentage)
            if value <= 0 or value > 30:
                return self.message_formatter.format_error("Invalid drawdown percentage. Must be between 0 and 30")
            self.trading_bot.config.RISK_MAX_DRAWDOWN = value / 100
            return self.message_formatter.format_notification(f"Maximum drawdown set to {value}%", "success")
        except ValueError:
            return self.message_formatter.format_error("Invalid percentage value")
        except Exception as e:
            return self.message_formatter.format_error(str(e))

    async def _handle_set_daily_var(self, percentage: str) -> str:
        """Set daily Value at Risk percentage"""
        try:
            value = float(percentage)
            if value <= 0 or value > 10:
                return self.message_formatter.format_error("Invalid VaR percentage. Must be between 0 and 10")
            self.trading_bot.config.RISK_DAILY_VAR = value / 100
            return self.message_formatter.format_notification(f"Daily VaR set to {value}%", "success")
        except ValueError:
            return self.message_formatter.format_error("Invalid percentage value")
        except Exception as e:
            return self.message_formatter.format_error(str(e))

    async def _handle_set_trailing_stop(self, percentage: str) -> str:
        """Set trailing stop percentage"""
        try:
            value = float(percentage)
            if value <= 0 or value > 10:
                return self.message_formatter.format_error("Invalid trailing stop percentage. Must be between 0 and 10")
            self.trading_bot.config.TRAILING_STOP_PERCENTAGE = value
            return self.message_formatter.format_notification(f"Trailing stop set to {value}%", "success")
        except ValueError:
            return self.message_formatter.format_error("Invalid percentage value")
        except Exception as e:
            return self.message_formatter.format_error(str(e))

    async def _handle_set_take_profit(self, percentage: str) -> str:
        """Set take profit percentage"""
        try:
            value = float(percentage)
            if value <= 0 or value > 50:
                return self.message_formatter.format_error("Invalid take profit percentage. Must be between 0 and 50")
            self.trading_bot.config.TAKE_PROFIT_PERCENTAGE = value
            return self.message_formatter.format_notification(f"Take profit set to {value}%", "success")
        except ValueError:
            return self.message_formatter.format_error("Invalid percentage value")
        except Exception as e:
            return self.message_formatter.format_error(str(e))

    async def _handle_ma_analysis(self, symbol: str = "BTC-USD") -> str:
        """Get moving average analysis"""
        try:
            analysis = await self.trading_bot.technical_analyzer.get_ma_analysis(symbol)
            return analysis
        except Exception as e:
            return self.message_formatter.format_error(str(e))

    async def _handle_stats(self) -> str:
        """Get trading statistics"""
        try:
            stats = await self.trading_bot.get_trading_stats()
            return (
                "Trading Statistics:\n```\n"
                f"Total Trades: {stats['total_trades']}\n"
                f"Win Rate: {stats['win_rate']*100:.1f}%\n"
                f"Average Profit: ${stats['avg_profit']:.2f}\n"
                f"Max Drawdown: {stats['max_drawdown']*100:.1f}%\n"
                f"Active Positions: {stats['active_positions']}\n"
                f"Closed Positions: {stats['closed_positions']}\n"
                "```"
            )
        except Exception as e:
            return self.message_formatter.format_error(str(e))