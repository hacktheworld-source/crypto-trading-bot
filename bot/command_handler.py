from typing import Optional, List, Union, Literal, Dict, Any
from datetime import datetime, timedelta
import asyncio
import pandas as pd
import logging

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
            'commands': self.get_help
        }
    
    async def handle_command(self, command: str, *args) -> str:
        """Handle a command with arguments."""
        try:
            if command in self.commands:
                return await self.commands[command](*args)
            return self._format_error(f"Unknown command: {command}")
        except Exception as e:
            return self._format_error(str(e))
        
    async def list_coins(self):
        coins = list(self.trading_bot.watched_coins)
        if coins:
            return f"Watched coins: {', '.join(coins)}"
        return "No coins in watchlist"
        
    async def add_coin(self, *symbols: str) -> str:
        """
        Add one or more coins to the watchlist.
        
        Args:
            *symbols: Variable number of cryptocurrency symbols to add
            
        Returns:
            str: Multi-line string containing results for each symbol
            
        Note:
            Symbols are automatically converted to uppercase
            Invalid symbols will be rejected with error messages
        """
        results = []
        for symbol in symbols:
            if await self.trading_bot.add_coin(symbol.upper()):
                results.append(self._format_success(f"Added {symbol.upper()}"))
            else:
                results.append(self._format_error(f"Failed to add {symbol.upper()}"))
        return "\n".join(results)
        
    async def remove_coin(self, *symbols):
        """Remove multiple coins from watchlist"""
        results = []
        for symbol in symbols:
            if await self.trading_bot.remove_coin(symbol):
                results.append(self._format_success(f"Removed {symbol}"))
            else:
                results.append(self._format_error(f"{symbol} not in watchlist"))
        return "\n".join(results)
        
    async def get_rsi(self, symbol: str):
        """Get RSI value for a symbol with interpretation"""
        try:
            symbol = symbol.upper()
            product = self.trading_bot.client.get_product(f"{symbol}-USD")
            current_price = float(product.price)
            rsi = await self.trading_bot.calculate_rsi(symbol)
            
            # Add RSI interpretation
            rsi_status = "Overbought" if rsi > 70 else \
                        "Oversold" if rsi < 30 else \
                        "Neutral"
            
            return f"Analysis for {symbol}:\n```" \
                   f"📊 Technical Analysis:\n" \
                   f"  • Price: ${current_price:,.2f}\n" \
                   f"  • RSI: {rsi:.2f} ({rsi_status})\n" \
                   f"  • Threshold: {self.trading_bot.rsi_oversold} / {self.trading_bot.rsi_overbought}```"
        except Exception as e:
            return self._format_error(str(e))
            
    def set_trade_amount(self, amount):
        if self.trading_bot.set_trade_amount(amount):
            return f"Successfully set trade amount to ${amount}"
        return "Failed to set trade amount. Please provide a valid positive number."
        
    def set_rsi_thresholds(self, oversold, overbought):
        if self.trading_bot.set_rsi_thresholds(oversold, overbought):
            return f"Successfully set RSI thresholds: oversold={oversold}, overbought={overbought}"
        return "Failed to set RSI thresholds. Please provide valid values (0-100, oversold < overbought)"
        
    def set_trading_interval(self, minutes):
        if self.trading_bot.set_trading_interval(minutes):
            return f"Successfully set trading interval to {minutes} minutes"
        return "Failed to set trading interval. Please provide a valid positive number."
        
    def get_trade_history(self):
        history = self.trading_bot.get_trade_history()
        if not history:
            return "No trade history available"
            
        response = "Trade History:\n```"
        for trade in history[-10:]:  # Show last 10 trades
            response += f"{trade['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} - {trade['action']} {trade['symbol']}: ${trade['amount_usd']}\n"
        response += "```"
        return response
        
    async def start_trading(self, *args) -> str:
        """Start trading with specified mode"""
        mode = args[0] if args else None
        return await self.trading_bot.start_trading(mode)
        
    async def stop_trading(self) -> str:
        """Stop any active trading"""
        return await self.trading_bot.stop_trading()
        
    async def get_status(self):
        """
        Get comprehensive bot status with proper error handling.
        
        Returns:
            str: Formatted status message for Discord
        """
        try:
            bot = self.trading_bot
            
            # Get balances with error handling
            try:
                real_balance = await bot.get_account_balance()
            except Exception as e:
                logging.warning(f"Failed to get real balance: {str(e)}")
                real_balance = {'balances': {}, 'total_usd_value': 0.0}
            
            try:
                paper_balance = bot.get_paper_balance() if hasattr(bot, 'get_paper_balance') else {'cash_balance': 0, 'total_value': 0}
            except Exception as e:
                logging.warning(f"Failed to get paper balance: {str(e)}")
                paper_balance = {'cash_balance': 0, 'total_value': 0}
            
            status = "Bot Status:\n```"
            
            # Trading Status
            status += "\n📊 Trading Status:"
            status += f"\n  Trading Active: {'✅' if bot.trading_active else '❌'}"
            status += f"\n  Paper Trading: {'✅' if bot.paper_trading else '❌'}"
            status += f"\n  Check Interval: {bot.trading_interval//60} minutes"
            
            # Paper Trading Status
            if hasattr(bot, 'paper_positions'):
                status += f"\n  Paper Positions: {len(bot.paper_positions)}"
            if hasattr(bot, 'paper_trade_history'):
                status += f"\n  Paper Trades: {len(bot.paper_trade_history)}"
            
            # Trading Configuration
            status += "\n\n⚙️ Configuration:"
            status += f"\n  Trade Amount: ${bot.trade_amount:.2f}"
            status += f"\n  Stop Loss: {bot.stop_loss_percentage}%"
            status += f"\n  Take Profit: {bot.take_profit_percentage}%"
            status += f"\n  Max Position Size: ${bot.max_position_size:.2f}"
            
            # Technical Indicators
            status += "\n\n📈 Technical Indicators:"
            status += f"\n  RSI Period: {bot.rsi_period} days"
            status += f"\n  RSI Thresholds: {bot.rsi_oversold} (oversold) / {bot.rsi_overbought} (overbought)"
            
            # Watched Coins
            status += "\n\n🔍 Watched Coins:"
            if bot.watched_coins:
                for coin in sorted(bot.watched_coins):
                    try:
                        # Use synchronous price fetch
                        product = bot.client.get_product(f"{coin}-USD")
                        current_price = float(product.price)
                        
                        # Calculate RSI using existing method
                        rsi = await bot.calculate_rsi(coin)
                        
                        status += f"\n  {coin}: ${current_price:,.2f} (RSI: {rsi:.1f})"
                    except Exception as e:
                        status += f"\n  {coin}: Error fetching data ({str(e)})"
            else:
                status += "\n  None"
            
            # Real Account Balances (only show if there are balances)
            if real_balance['balances']:
                status += "\n\n💰 Real Account Balances:"
                for symbol, data in real_balance['balances'].items():
                    status += f"\n  {symbol}: {data['balance']:.8f} (${data['usd_value']:.2f})"
                status += f"\n  Total Real Portfolio Value: ${real_balance['total_usd_value']:.2f}"
            
            # Paper Trading Account
            status += "\n\n📝 Paper Trading Account:"
            status += f"\n  Paper Cash: ${paper_balance['cash_balance']:.2f}"
            paper_profit = paper_balance['total_value'] - 1000.0  # Assuming $1000 starting balance
            paper_profit_pct = (paper_profit / 1000.0) * 100
            status += f"\n  Paper Portfolio Value: ${paper_balance['total_value']:.2f}"
            status += f"\n  Paper P/L: ${paper_profit:+.2f} ({paper_profit_pct:+.2f}%)"
            
            # Trading History Summary
            status += "\n\n📜 Trading History:"
            status += f"\n  Total Real Trades: {len(bot.trade_history)}"
            status += f"\n  Total Paper Trades: {len(getattr(bot, 'paper_trade_history', []))}"
            status += f"\n  Active Real Positions: {len(getattr(bot, 'positions', []))}"
            status += f"\n  Active Paper Positions: {len(getattr(bot, 'paper_positions', []))}"
            
            status += "```"
            return status
            
        except Exception as e:
            await self.trading_bot.log(f"Error getting status: {str(e)}", level="error")
            return self._format_error(f"Error getting status: {str(e)}")
        
    async def test_api(self, *args):
        try:
            btc_price = await self.trading_bot.test_api_connection()
            return (
                "✅ Coinbase API Test Results:\n"
                "- Authentication: Success\n"
                "- Price Fetch: Success\n"
                f"- Current BTC price: ${btc_price:,.2f}"
            )
        except Exception as e:
            return self._format_error(str(e))
        
    def get_help(self):
        help_text = "Trading Bot Commands:\n```"
        
        help_text += "\nReal Trading Commands:"
        help_text += "\n!start         - Start the trading bot"
        help_text += "\n!stop          - Stop the trading bot"
        help_text += "\n!status        - Show bot status and portfolio value"
        help_text += "\n!positions     - View all positions (real & paper)"
        help_text += "\n!poshistory    - View position history"
        help_text += "\n!performance   - View trading performance stats"
        
        help_text += "\n\nPaper Trading Commands:"
        help_text += "\n!paper start [balance] - Start paper trading with optional balance"
        help_text += "\n!paper balance         - Show paper trading balance"
        help_text += "\n!paper reset           - Reset paper trading"
        help_text += "\n!paper stats           - Show paper trading statistics"
        help_text += "\n!paper trades          - Show paper trade history"
        help_text += "\n!paper positions       - Show paper positions only"
        help_text += "\n!paper settings        - Show/modify paper settings"
        
        help_text += "\n\nAnalysis Commands:"
        help_text += "\n!price BTC     - Get current price"
        help_text += "\n!rsi BTC       - Get RSI with interpretation"
        help_text += "\n!ma BTC        - Get Moving Average analysis"
        help_text += "\n!bb BTC        - Get Bollinger Bands analysis"
        help_text += "\n!volume BTC    - Get volume analysis"
        help_text += "\n!sentiment BTC - Get market sentiment analysis"
        help_text += "\n!conditions BTC - Get market conditions analysis"
        
        help_text += "\n\nCoin Management:"
        help_text += "\n!addcoin BTC   - Add a coin to watchlist"
        help_text += "\n!removecoin BTC - Remove a coin from watchlist"
        help_text += "\n!listcoins     - Show all watched coins"
        
        help_text += "\n\nRisk Management:"
        help_text += "\n!setrisk <stop_loss> <take_profit> <max_position>"
        help_text += "\n  Example: !setrisk 5 10 1000"
        help_text += "\n  Sets: 5% stop loss, 10% take profit, $1000 max position"
        
        help_text += "\n\nConfiguration:"
        help_text += "\n!setamount 100 - Set trade amount in USD"
        help_text += "\n!setrsi 30 70  - Set RSI thresholds (oversold overbought)"
        help_text += "\n!setinterval 5 - Set check interval in minutes"
        
        help_text += "\n\nSystem Commands:"
        help_text += "\n!testapi       - Test Coinbase API connection"
        help_text += "\n!ping          - Test if bot is responsive"
        help_text += "\n!commands      - Show this help message"
        help_text += "```"
        return help_text
        
    async def get_price(self, symbol: str):
        try:
            # Convert symbol to uppercase
            symbol = symbol.upper()
            price = await self.trading_bot.get_current_price(symbol)
            return f"{symbol} Price: ${price:,.2f}"
        except Exception as e:
            return self._format_error(str(e))
        
    async def get_volume_analysis(self, symbol: str):
        """Get volume analysis with trend interpretation"""
        try:
            symbol = symbol.upper()
            end = datetime.now()
            start = end - timedelta(days=90)
            
            response = self.trading_bot.client.get_candles(
                product_id=f"{symbol}-USD",
                start=int(start.timestamp()),
                end=int(end.timestamp()),
                granularity="ONE_DAY"
            )
            
            volumes = [float(candle.volume) for candle in response.candles]
            current_volume = volumes[0]
            avg_volume = sum(volumes) / len(volumes)
            volume_ratio = current_volume / avg_volume
            
            # Add volume trend interpretation
            trend = "High" if volume_ratio > 1.5 else \
                   "Above Average" if volume_ratio > 1.0 else \
                   "Below Average" if volume_ratio > 0.5 else \
                   "Low"
            
            return f"Volume Analysis for {symbol}:\n" \
                   f"```📈 24h Volume Metrics:\n" \
                   f"  • Current: {current_volume:,.2f} {symbol}\n" \
                   f"  • Average: {avg_volume:,.2f} {symbol}\n" \
                   f"  • Ratio: {volume_ratio:.2f}x average\n" \
                   f"  • Volume Trend: {trend}```"
        except Exception as e:
            return self._format_error(str(e))
        
    def get_positions(self):
        """Show both real and paper positions"""
        positions = self.trading_bot.get_position_info()
        paper_positions = self.trading_bot.paper_positions
        
        if not positions and not paper_positions:
            return "No active positions or holdings"
        
        response = "Current Positions and Holdings:\n```"
        
        # Real positions
        if positions:
            response += "\nReal Positions:"
            for symbol, pos in positions.items():
                response += f"\n{symbol}:"
                response += f"\n  Current Price: ${pos['current_price']:.2f}"
                response += f"\n  Quantity: {pos['quantity']:.8f}"
                response += f"\n  Total Value: ${pos['current_price'] * pos['quantity']:.2f}"
                
                if pos['is_bot_position']:
                    response += f"\n  Entry Price: ${pos['entry_price']:.2f}"
                    response += f"\n  Profit: ${pos['profit_usd']:.2f} ({pos['profit_percentage']:+.2f}%)"
                    response += f"\n  Max Profit: {pos['highest_profit_percentage']:+.2f}%"
                    response += f"\n  Max Drawdown: {pos['drawdown_percentage']:+.2f}%"
                else:
                    response += "\n  (External holding)"
                response += "\n"
        
        # Paper positions
        if paper_positions:
            response += "\nPaper Positions:"
            for symbol, pos in paper_positions.items():
                current_price = float(self.trading_bot.client.get_product(f"{symbol}-USD").price)
                profit_info = pos.calculate_profit(current_price)
                
                response += f"\n{symbol} (Paper):"
                response += f"\n  Current Price: ${current_price:.2f}"
                response += f"\n  Quantity: {pos.quantity:.8f}"
                response += f"\n  Total Value: ${current_price * pos.quantity:.2f}"
                response += f"\n  Entry Price: ${pos.entry_price:.2f}"
                response += f"\n  Profit: ${profit_info['profit_usd']:.2f} ({profit_info['profit_percentage']:+.2f}%)"
                response += f"\n  Fees Paid: ${profit_info['fees_paid']:.2f}"
                response += "\n"
        
        response += "```"
        return response
        
    def get_position_history(self):
        history = self.trading_bot.position_history
        if not history:
            return "No position history available"
            
        response = "Position History:\n```"
        for pos in history[-5:]:  # Show last 5 closed positions
            response += f"\n{pos['symbol']}:"
            response += f"\n  Entry: ${pos['entry_price']:.2f}"
            response += f"\n  Exit: ${pos['exit_price']:.2f}"
            response += f"\n  Profit: ${pos['profit_usd']:.2f} ({pos['profit_percentage']:+.2f}%)"
            response += f"\n  Duration: {pos['exit_time'] - pos['entry_time']}"
            response += "\n"
        response += "```"
        return response
        
    async def get_ma_analysis(self, symbol: str):
        """Get moving average analysis"""
        try:
            symbol = symbol.upper()
            ma_data = await self.trading_bot.calculate_moving_averages(symbol)
            
            # Determine trend emoji
            trend_emoji = "🟢" if ma_data['trend'] == 'bullish' else \
                         "🔴" if ma_data['trend'] == 'bearish' else "⚪"
            
            return f"Moving Average Analysis for {symbol}:\n```" \
                   f"📊 Price Levels:\n" \
                   f"  • Current: ${ma_data['current_price']:,.2f}\n" \
                   f"  • SMA 20: ${ma_data['sma_20']:,.2f}\n" \
                   f"  • SMA 50: ${ma_data['sma_50']:,.2f}\n" \
                   f"  • SMA 200: ${ma_data['sma_200']:,.2f}\n\n" \
                   f"📈 Trend Analysis:\n" \
                   f"  • Status: {trend_emoji} {ma_data['trend'].title()}\n" \
                   f"  • Above 20 MA: {'Yes ✅' if ma_data['current_price'] > ma_data['sma_20'] else 'No ❌'}\n" \
                   f"  • Above 50 MA: {'Yes ✅' if ma_data['current_price'] > ma_data['sma_50'] else 'No ❌'}\n" \
                   f"  • Above 200 MA: {'Yes ✅' if ma_data['current_price'] > ma_data['sma_200'] else 'No ❌'}```"
        except Exception as e:
            return self._format_error(str(e))
        
    def get_performance(self):
        """Show both real and paper trading performance"""
        real_stats = self.trading_bot.get_performance_stats()
        paper_balance = self.trading_bot.get_paper_balance()
        
        response = "Trading Performance:\n```"
        
        # Real trading performance
        response += "\nReal Trading:"
        response += f"\nTotal Trades: {real_stats['total_trades']}"
        response += f"\nActive Positions: {real_stats['active_positions']}"
        response += f"\nClosed Positions: {real_stats['closed_positions']}"
        response += f"\nTotal Profit: ${real_stats['total_profit_usd']:.2f}"
        if real_stats['total_trades'] > 0:
            response += f"\nWin Rate: {real_stats['win_rate']:.1f}%"
            response += f"\nAverage Profit: ${real_stats['average_profit']:.2f}"
        
        # Paper trading performance
        response += "\n\nPaper Trading:"
        paper_profit = paper_balance['total_value'] - 1000.0
        paper_profit_pct = (paper_profit / 1000.0) * 100
        response += f"\nTotal Value: ${paper_balance['total_value']:.2f}"
        response += f"\nTotal P/L: ${paper_profit:+.2f} ({paper_profit_pct:+.2f}%)"
        response += f"\nCash Balance: ${paper_balance['cash_balance']:.2f}"
        
        response += "```"
        return response
        
    async def set_risk_params(self, stop_loss: float, take_profit: float, max_position: float):
        try:
            success = await self.trading_bot.set_risk_parameters(stop_loss, take_profit, max_position)
            if success:
                return self._format_success(f"Risk parameters updated: SL={stop_loss}%, TP={take_profit}%, Max=${max_position}")
            return self._format_error("Invalid risk parameters")
        except Exception as e:
            return self._format_error(str(e))
        
    async def get_sentiment_analysis(self, symbol: str):
        """Get sentiment analysis with detailed interpretation"""
        try:
            symbol = symbol.upper()
            analysis = await self.trading_bot.analyze_market_sentiment(symbol)
            
            # Add sentiment strength interpretation
            strength = "Strong" if abs(analysis['sentiment_score']) > 7 else \
                      "Moderate" if abs(analysis['sentiment_score']) > 3 else \
                      "Weak"
            
            # Add directional emojis
            up_arrow, down_arrow = "↗️", "↘️"
            short_arrow = up_arrow if analysis['momentum']['short_term'] == 'bullish' else down_arrow
            mid_arrow = up_arrow if analysis['momentum']['medium_term'] == 'bullish' else down_arrow
            long_arrow = up_arrow if analysis['momentum']['long_term'] == 'bullish' else down_arrow
            
            return f"Sentiment Analysis for {symbol}:\n```" \
                   f"📊 Market Sentiment:\n" \
                   f"  • Overall: {analysis['overall_sentiment']}\n" \
                   f"  • Score: {analysis['sentiment_score']:+.2f} ({strength} {analysis['overall_sentiment']})\n\n" \
                   f"📈 Momentum Analysis:\n" \
                   f"  • Short-term:  {analysis['momentum']['short_term'].title()} {short_arrow}\n" \
                   f"  • Medium-term: {analysis['momentum']['medium_term'].title()} {mid_arrow}\n" \
                   f"  • Long-term:   {analysis['momentum']['long_term'].title()} {long_arrow}```"
        except Exception as e:
            return self._format_error(str(e))
        
    def get_paper_balance(self):
        """Get paper trading account status"""
        balance = self.trading_bot.get_paper_balance()
        
        response = "Paper Trading Account:\n```"
        response += f"Cash Balance: ${balance['cash_balance']:.2f}\n"
        response += f"Total Value: ${balance['total_value']:.2f}\n"
        
        # Calculate profit/loss
        profit = balance['total_value'] - 1000.0  # Assuming $1000 starting balance
        profit_percentage = (profit / 1000.0) * 100
        
        response += f"Total P/L: ${profit:+.2f} ({profit_percentage:+.2f}%)"
        response += "```"
        return response
        
    def reset_paper_trading(self, initial_balance: float = 1000.0):
        """Reset paper trading with new balance"""
        self.trading_bot.reset_paper_trading(initial_balance)
        return f"Paper trading reset with ${initial_balance:.2f} balance"
        
    def get_paper_stats(self):
        """Get detailed paper trading statistics"""
        paper_balance = self.trading_bot.get_paper_balance()
        paper_trades = self.trading_bot.paper_trade_history
        
        if not paper_trades:
            return "No paper trading history available"
        
        # Calculate statistics
        total_trades = len(paper_trades)
        winning_trades = len([t for t in paper_trades if t.get('profit', 0) > 0])
        total_profit = sum(t.get('profit', 0) for t in paper_trades)
        total_fees = sum(t.get('fees', 0) for t in paper_trades)
        
        response = "Paper Trading Statistics:\n```"
        response += f"Total Trades: {total_trades}\n"
        response += f"Winning Trades: {winning_trades}\n"
        response += f"Win Rate: {(winning_trades/total_trades*100):.1f}%\n"
        response += f"Total Profit: ${total_profit:,.2f}\n"
        response += f"Total Fees Paid: ${total_fees:.2f}\n"
        response += f"Current Balance: ${paper_balance['cash_balance']:,.2f}\n"
        response += f"Portfolio Value: ${paper_balance['total_value']:,.2f}\n"
        
        # Calculate return on initial investment
        roi = ((paper_balance['total_value'] - 1000.0) / 1000.0) * 100
        response += f"Total Return: {roi:+.2f}%"
        response += "```"
        return response
        
    def get_paper_trades(self):
        """Show paper trading history"""
        trades = self.trading_bot.paper_trade_history
        if not trades:
            return "No paper trades yet"
        
        response = "Paper Trading History:\n```"
        # Show last 10 trades
        for trade in trades[-10:]:
            response += f"\n{trade['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"
            response += f"\n{trade['action']} {trade['symbol']}"
            response += f"\nPrice: ${trade['price']:,.2f}"
            response += f"\nQuantity: {trade['quantity']:.8f}"
            response += f"\nTotal: ${trade['amount_usd']:,.2f}"
            response += f"\nFees: ${trade['fees']:.2f}"
            if 'profit' in trade:
                response += f"\nProfit: ${trade['profit']:+,.2f} ({trade['profit_percentage']:+.2f}%)"
            response += "\n"
        response += "```"
        return response
        
    def get_paper_positions(self):
        """Show paper trading positions"""
        if not self.trading_bot.paper_positions:
            return "No paper trading positions"
            
        response = "Paper Trading Positions:\n```"
        
        for symbol, pos in self.trading_bot.paper_positions.items():
            current_price = float(self.trading_bot.client.get_product(f"{symbol}-USD").price)
            profit_info = pos.calculate_profit(current_price)
            
            response += f"\n{symbol}:"
            response += f"\n  Entry Price: ${pos.entry_price:,.2f}"
            response += f"\n  Current Price: ${current_price:,.2f}"
            response += f"\n  Quantity: {pos.quantity:.8f}"
            response += f"\n  Value: ${current_price * pos.quantity:,.2f}"
            response += f"\n  P/L: ${profit_info['profit_usd']:+,.2f} ({profit_info['profit_percentage']:+.2f}%)"
            response += f"\n  Fees Paid: ${profit_info['fees_paid']:.2f}"
            response += f"\n  Holding Time: {datetime.now() - pos.entry_time}"
            response += "\n"
            
        response += f"\nPaper Balance: ${self.trading_bot.paper_balance:,.2f}"
        response += "```"
        return response
        
    def get_balance(self):
        """Get real trading account status"""
        balance = self.trading_bot.get_account_balance()
        
        response = "Real Trading Account:\n```"
        
        # Show individual balances
        for symbol, data in balance['balances'].items():
            response += f"\n{symbol}:"
            response += f"\n  Balance: {data['balance']:.8f}"
            response += f"\n  Value: ${data['usd_value']:.2f}"
        
        response += f"\n\nTotal Portfolio Value: ${balance['total_usd_value']:.2f}"
        response += "```"
        return response
        
    def get_trades(self):
        """Show real trading history"""
        trades = self.trading_bot.trade_history
        if not trades:
            return "No real trades yet"
        
        response = "Real Trading History:\n```"
        # Show last 10 trades
        for trade in trades[-10:]:
            response += f"\n{trade['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"
            response += f"\n{trade['action']} {trade['symbol']}"
            response += f"\nAmount: ${trade['amount_usd']:,.2f}"
            response += "\n"
        response += "```"
        return response
        
    def analyze_coin(self, symbol: str) -> str:
        try:
            signal = self.trading_bot._calculate_trade_signal(symbol)
            
            response = f"Analysis for {symbol}:\n```"
            response += f"\nPrice: ${signal['price']:,.2f}"
            response += f"\nSignal: {signal['action']}"
            response += f"\nScore: {signal['score']:.2f}"
            response += "\n\nSignal Components:"
            response += f"\n• Trend (0.4x):    {signal['signals']['trend']:>6.1f}"
            response += f"\n• Momentum (0.3x): {signal['signals']['momentum']:>6.1f}"
            response += f"\n• Volume (0.2x):   {signal['signals']['volume']:>6.1f}"
            response += f"\n• Risk (0.1x):     {signal['signals']['risk']:>6.1f}"
            response += "```"
            return response
            
        except Exception as e:
            return f"Error analyzing {symbol}: {str(e)}"
        
    def set_trailing_stop(self, percentage: float = None, enabled: bool = None, activation: float = None) -> str:
        """Configure trailing stop settings"""
        try:
            changes = []
            
            if percentage is not None:
                if 0.1 <= percentage <= 20.0:
                    self.trading_bot.trailing_stop_percentage = percentage
                    changes.append(f"percentage: {percentage}%")
                else:
                    return "❌ Percentage must be between 0.1 and 20.0"
                    
            if enabled is not None:
                self.trading_bot.trailing_stop_enabled = enabled
                changes.append(f"enabled: {enabled}")
                
            if activation is not None:
                if 0.0 <= activation <= 10.0:
                    self.trading_bot.trailing_stop_activation = activation
                    changes.append(f"activation: {activation}%")
                else:
                    return "❌ Activation must be between 0.0 and 10.0"
                    
            if changes:
                self.trading_bot.save_config()
                return f"✅ Updated trailing stop settings: {', '.join(changes)}"
            else:
                return (f"Current trailing stop settings:\n"
                       f"• Enabled: {self.trading_bot.trailing_stop_enabled}\n"
                       f"• Percentage: {self.trading_bot.trailing_stop_percentage}%\n"
                       f"• Activation: {self.trading_bot.trailing_stop_activation}%")
                       
        except Exception as e:
            return f"❌ Error updating trailing stop settings: {str(e)}"
        
    def set_take_profit(self, full_tp: float = None, partial_tp: float = None, partial_size: float = None) -> str:
        """Configure take profit settings"""
        try:
            changes = []
            
            if full_tp is not None:
                if 0.1 <= full_tp <= 100.0:
                    self.trading_bot.take_profit_percentage = full_tp
                    changes.append(f"full take profit: {full_tp}%")
                else:
                    return "❌ Full take profit must be between 0.1 and 100.0"
                    
            if partial_tp is not None:
                if 0.1 <= partial_tp <= full_tp:
                    self.trading_bot.partial_tp_percentage = partial_tp
                    changes.append(f"partial take profit: {partial_tp}%")
                else:
                    return "❌ Partial take profit must be between 0.1 and full take profit"
                    
            if partial_size is not None:
                if 0.1 <= partial_size <= 0.9:
                    self.trading_bot.partial_tp_size = partial_size
                    changes.append(f"partial size: {partial_size*100}%")
                else:
                    return "❌ Partial size must be between 0.1 and 0.9"
                    
            if changes:
                self.trading_bot.save_config()
                return f"✅ Updated take profit settings: {', '.join(changes)}"
            else:
                return (f"Current take profit settings:\n"
                       f"• Full TP: {self.trading_bot.take_profit_percentage}%\n"
                       f"• Partial TP: {self.trading_bot.partial_tp_percentage}%\n"
                       f"• Partial Size: {self.trading_bot.partial_tp_size*100}%")
                   
        except Exception as e:
            return f"❌ Error updating take profit settings: {str(e)}"
        
    def set_stop_loss(self, percentage: float = None) -> str:
        """Configure stop loss settings"""
        try:
            if percentage is None:
                return f"Current stop loss: {self.trading_bot.stop_loss_percentage}%"
            
            if 0.1 <= percentage <= 20.0:
                self.trading_bot.stop_loss_percentage = percentage
                self.trading_bot.save_config()
                return f"✅ Updated stop loss to {percentage}%"
            else:
                return "❌ Stop loss must be between 0.1% and 20.0%"
            
        except Exception as e:
            return f"❌ Error updating stop loss: {str(e)}"
        
    async def get_position_details(self, symbol: str = None):
        try:
            positions = self.trading_bot.get_position_info(symbol)
            return self._format_position_details(positions)
        except Exception as e:
            return self._format_error(str(e))
        
    async def get_position_metrics(self, symbol: str = None):
        try:
            metrics = await self.trading_bot.get_position_metrics(symbol)
            return self._format_metrics_response(metrics)
        except Exception as e:
            return self._format_error(str(e))
        
    async def handle_paper_commands(self, subcommand: str, *args):
        """Handle paper trading subcommands"""
        try:
            if subcommand == 'start':
                balance = float(args[0]) if args else 1000.0
                # Reset paper trading first
                self.trading_bot.reset_paper_trading(balance)
                # Then start paper trading
                return await self.trading_bot.start_trading('paper')
            elif subcommand == 'balance':
                return await self.get_paper_balance()
            elif subcommand == 'reset':
                if not self.trading_bot.paper_trading_active:
                    return self._format_error("Paper trading is not active")
                self.trading_bot.reset_paper_trading()
                return self._format_success("Paper trading reset")
            elif subcommand == 'stats':
                return await self.get_paper_stats()
            elif subcommand == 'trades':
                return await self.get_paper_trades()
            elif subcommand == 'positions':
                return await self.get_paper_positions()
            else:
                return self._format_error(f"Unknown paper trading command: {subcommand}")
        except Exception as e:
            return self._format_error(str(e))
        
    async def ping(self):
        return "Pong! 🏓"
        
    async def version(self):
        return f"Trading Bot v1.0.0"
        
    async def get_stats(self):
        try:
            stats = await self.trading_bot.get_trading_stats()
            return self._format_stats_response(stats)
        except Exception as e:
            return self._format_error(str(e))
        
    def _format_error(self, message: str) -> str:
        """
        Format an error message with the error prefix.
        
        Args:
            message (str): The error message to format
        
        Returns:
            str: Formatted error message with prefix
        """
        return f"{self.ERROR_PREFIX}{message}"
        
    def _format_success(self, message: str) -> str:
        """
        Format a success message with the success prefix.
        
        Args:
            message (str): The success message to format
        
        Returns:
            str: Formatted success message with prefix
        """
        return f"{self.SUCCESS_PREFIX}{message}"
        
    async def get_bb_analysis(self, symbol: str):
        """Get Bollinger Bands analysis"""
        try:
            symbol = symbol.upper()
            bb_data = await self.trading_bot.calculate_bollinger_bands(symbol)
            current_price = await self.trading_bot.price_manager.get_current_price(symbol)
            
            # Calculate price position
            price_position = (current_price - bb_data['lower']) / (bb_data['upper'] - bb_data['lower']) * 100
            position_status = "Overbought ⚠️" if price_position > 80 else \
                             "Oversold 🔥" if price_position < 20 else \
                             "Neutral ⚖️"
            
            return f"Bollinger Bands Analysis for {symbol}:\n```" \
                   f"📊 Band Levels:\n" \
                   f"  • Upper Band: ${bb_data['upper']:,.2f}\n" \
                   f"  • Middle Band: ${bb_data['middle']:,.2f}\n" \
                   f"  • Lower Band: ${bb_data['lower']:,.2f}\n\n" \
                   f"📈 Position Analysis:\n" \
                   f"  • Current Price: ${current_price:,.2f}\n" \
                   f"  • Position: {position_status} ({price_position:.1f}%)\n" \
                   f"  • Bandwidth: {bb_data['bandwidth']:.1f}%"
        except Exception as e:
            return self._format_error(str(e))
        
    async def get_market_conditions(self, *args) -> str:
        """Get market conditions analysis for a symbol."""
        try:
            if not args:
                return self._format_error("Please specify a symbol: !conditions <symbol>")
            symbol = args[0].upper()
            conditions = await self.trading_bot.check_market_conditions(symbol)
            
            # Get BTC correlation if not BTC
            correlation = await self.trading_bot._calculate_btc_correlation(symbol) if symbol != 'BTC' else 1.0
            
            return f"Market Conditions Analysis for {symbol}:\n```" \
                   f"📊 Trading Conditions:\n" \
                   f"  • Volatility: {'High ⚠️' if conditions['is_volatile'] else 'Normal ✅'}\n" \
                   f"  • Price Range (7d): {conditions['price_range_7d']:.1f}%\n" \
                   f"  • Market Hours: {'Active 🟢' if conditions['is_high_activity'] else 'Quiet 🔴'}\n\n" \
                   f"📈 Market Alignment:\n" \
                   f"  • BTC Correlation: {correlation:.2f}\n" \
                   f"  • Market Aligned: {'Yes ✅' if conditions['market_aligned'] else 'No ⚠️'}\n" \
                   f"  • Suitable for Trading: {'Yes ✅' if conditions['suitable_for_trading'] else 'No ❌'}```"
        except Exception as e:
            return self._format_error(str(e))