import logging
from datetime import datetime
from typing import Dict, List, Any

class CommandHandler:
    def __init__(self, trading_bot):
        self.trading_bot = trading_bot
        
    def add_coin(self, *symbols):
        """Add multiple coins to watchlist"""
        results = []
        for symbol in symbols:
            if self.trading_bot.add_coin(symbol):
                results.append(f"✅ Added {symbol}")
            else:
                results.append(f"❌ Failed to add {symbol}")
        
        return "\n".join(results)
        
    def remove_coin(self, *symbols):
        """Remove multiple coins from watchlist"""
        results = []
        for symbol in symbols:
            if self.trading_bot.remove_coin(symbol):
                results.append(f"✅ Removed {symbol}")
            else:
                results.append(f"❌ {symbol} not in watchlist")
        
        return "\n".join(results)
        
    def list_coins(self):
        coins = list(self.trading_bot.watched_coins)
        if coins:
            return f"Watched coins: {', '.join(coins)}"
        return "No coins in watchlist"
        
    def get_rsi(self, symbol):
        try:
            rsi = self.trading_bot.calculate_rsi(symbol)
            return f"Current RSI for {symbol}: {rsi:.2f}"
        except Exception as e:
            return f"Error calculating RSI for {symbol}: {str(e)}"
            
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
        
    def start_real_trading(self):
        try:
            # Verify API connection
            account = self.trading_bot.client.get_accounts()
            if not account:
                return "❌ Failed to connect to Coinbase API"
            
            # Check USD balance
            usd_account = next((acc for acc in account.data if acc.currency == 'USD'), None)
            if not usd_account or float(usd_account.available_balance.value) < 10:
                return "❌ Insufficient USD balance for trading"
            
            # Check trading permissions
            if not self.trading_bot.check_trading_permissions():
                return "❌ Account does not have required trading permissions"
            
            return self.trading_bot.start_trading_loop(paper=False)
        
        except Exception as e:
            logging.error(f"Error starting real trading: {str(e)}")
            return f"❌ Error starting real trading: {str(e)}"
        
    def start_paper_trading(self, initial_balance: float = 1000.0):
        """Start paper trading"""
        if self.trading_bot.trading_active:
            return "❌ Cannot start paper trading while real trading is active. Stop real trading first with '!stop real'"
        
        # Reset paper trading with initial balance
        self.trading_bot.reset_paper_trading(initial_balance)
        
        # Start the trading loop
        return self.trading_bot.start_trading_loop(paper=True)
        
    def stop_trading(self, mode: str = 'all'):
        """Stop trading (paper, real, or all)"""
        if mode == 'paper':
            self.trading_bot.paper_trading = False
            return "Paper trading stopped"
        elif mode == 'real':
            self.trading_bot.trading_active = False
            return "Real trading stopped"
        else:  # all
            self.trading_bot.paper_trading = False
            self.trading_bot.trading_active = False
            return "All trading stopped"
        
    def get_status(self):
        bot = self.trading_bot
        real_balance = bot.get_account_balance()
        paper_balance = bot.get_paper_balance()
        
        status = "Bot Status:\n```"
        
        # Trading Status - More detailed now
        status += "\n📊 Trading Status:"
        status += "\n  Trading Loop: " + ("✅ Running" if (bot.trading_active or bot.paper_trading) else "❌ Stopped")
        status += "\n  Real Trading: " + ("✅ Active" if bot.trading_active else "❌ Inactive")
        status += "\n  Paper Trading: " + ("✅ Active" if bot.paper_trading else "❌ Inactive")
        status += f"\n  Check Interval: {bot.trading_interval//60} minutes"
        
        # Add warning if both modes are inactive
        if not bot.trading_active and not bot.paper_trading:
            status += "\n  ⚠️ WARNING: No trading mode is active. Use !start paper or !start real"
        
        # Trading Configuration
        status += "\n\n⚙️ Configuration:"
        status += f"\n  Stop Loss: {bot.stop_loss_percentage}%"
        status += f"\n  Take Profit: {bot.take_profit_percentage}%"
        status += f"\n  Max Position Size: ${bot.max_position_size:.2f}"
        
        # Technical Indicators
        status += "\n\n📈 Technical Indicators:"
        status += f"\n  RSI Period: {bot.rsi_period} days"
        status += f"\n  RSI Thresholds: {bot.rsi_oversold} (oversold) / {bot.rsi_overbought} (overbought)"
        
        # Compact Watched Coins section
        status += "\n\n🔍 Watched Coins:"
        if bot.watched_coins:
            for coin in sorted(bot.watched_coins):
                try:
                    current_price = bot.get_current_price(coin)
                    rsi = bot.calculate_rsi(coin)
                    volume_data = bot.analyze_volume(coin)
                    ma_data = bot.calculate_moving_averages(coin)
                    
                    # Compact single-line format
                    status += f"\n  {coin}: ${current_price:,.2f} | RSI: {rsi:.1f} | Vol: {volume_data['volume_ratio']:.1f}x"
                    if coin in bot.paper_positions or coin in bot.positions:
                        status += " 📍"  # Position indicator
                
                except Exception as e:
                    status += f"\n  {coin}: Error fetching data"
        else:
            status += "\n  None"
        
        # Real Account Balances
        status += "\n\n💰 Real Account Balances:"
        for symbol, data in real_balance['balances'].items():
            status += f"\n  {symbol}: {data['balance']:.8f} (${data['usd_value']:.2f})"
        status += f"\n  Total Real Portfolio Value: ${real_balance['total_usd_value']:.2f}"
        
        # Paper Trading Account
        status += "\n\n📝 Paper Trading Account:"
        status += f"\n  Cash Balance: ${paper_balance['cash_balance']:.2f}"
        status += f"\n  Position Value: ${paper_balance['positions_value']:.2f}"
        status += f"\n  Total Value: ${paper_balance['total_value']:.2f}"
        status += f"\n  Realized P/L: ${paper_balance['realized_pl']:+.2f}"
        status += f"\n  Unrealized P/L: ${paper_balance['unrealized_pl']:+.2f}"
        status += f"\n  Total P/L: ${paper_balance['total_pl']:+.2f} ({paper_balance['pl_percentage']:+.2f}%)"
        status += f"\n  Total Fees: ${paper_balance['total_fees']:.2f}"
        
        # Trading History Summary
        status += "\n\n📜 Trading History:"
        status += f"\n  Total Real Trades: {len(bot.trade_history)}"
        status += f"\n  Total Paper Trades: {len(bot.paper_trade_history)}"
        status += f"\n  Active Real Positions: {len(bot.positions)}"
        status += f"\n  Active Paper Positions: {len(bot.paper_positions)}"
        
        status += "```"
        return status
        
    def test_api(self):
        try:
            btc_price = self.trading_bot.test_api_connection()
            return (
                "✅ Coinbase API Test Results:\n"
                "- Authentication: Success\n"
                "- Price Fetch: Success\n"
                f"- Current BTC price: ${btc_price:,.2f}"
            )
        except Exception as e:
            return (
                "❌ Coinbase API Test Failed\n\n"
                f"Error Details: {str(e)}\n\n"
                "Please verify:\n"
                "1. API Key is correct\n"
                "2. API Secret is correct\n"
                "3. API Keys have correct permissions"
            )
        
    def get_help(self):
        help_text = "🤖 Trading Bot Help Guide\n```"
        
        help_text += "\n📈 Trading Controls:"
        help_text += "\n!start paper [amount]  - Start paper trading with optional initial balance"
        help_text += "\n!start real           - Start real trading (requires confirmation)"
        help_text += "\n!stop [paper/real/all] - Stop trading in specified mode"
        
        help_text += "\n\n📊 Monitoring Commands:"
        help_text += "\n!status              - Full bot status, portfolio, and watched coins"
        help_text += "\n!positions           - View all current positions (real & paper)"
        help_text += "\n!balance             - Show real trading balance and P/L"
        help_text += "\n!trades              - Show real trading history"
        help_text += "\n!paper balance       - Show paper trading balance and P/L"
        help_text += "\n!paper positions     - Show paper trading positions only"
        help_text += "\n!paper trades        - Show paper trading history"
        
        help_text += "\n\n🔍 Analysis Tools:"
        help_text += "\n!price <coin>        - Get current price and 24h change"
        help_text += "\n!rsi <coin>          - Get RSI with overbought/oversold indicators"
        help_text += "\n!volume <coin>       - Get volume analysis and trend confirmation"
        
        help_text += "\n\n⚙️ Configuration:"
        help_text += "\n!addcoin <coin>      - Add coin to watchlist"
        help_text += "\n!removecoin <coin>   - Remove coin from watchlist"
        help_text += "\n!listcoins           - Show all watched coins"
        help_text += "\n!setrsi <low> <high> - Set RSI thresholds (e.g., !setrsi 30 70)"
        help_text += "\n!setinterval <mins>  - Set trading check interval in minutes"
        
        help_text += "\n\n🔧 System Commands:"
        help_text += "\n!testapi             - Test Coinbase API connection"
        help_text += "\n!ping                - Check if bot is responsive"
        help_text += "\n!help or !commands   - Show this help guide"
        
        help_text += "\n\n📝 Notes:"
        help_text += "\n• Paper trading simulates real trading with virtual money"
        help_text += "\n• Real trading requires API keys and confirmation"
        help_text += "\n• Stop losses are set at -5%, trailing stops at +3%"
        help_text += "\n• Position sizes are calculated based on risk management"
        help_text += "```"
        return help_text
        
    def get_price(self, symbol):
        try:
            price = self.trading_bot.get_current_price(symbol)
            return f"Current {symbol} price: ${price:,.2f}"
        except Exception as e:
            return f"Error getting price for {symbol}: {str(e)}"
        
    def get_volume_analysis(self, symbol):
        try:
            analysis = self.trading_bot.analyze_volume(symbol)
            
            response = f"Volume Analysis for {symbol}:\n```"
            response += f"Current Volume: {analysis['current_volume']:,.2f}\n"
            response += f"Average Volume: {analysis['average_volume']:,.2f}\n"
            response += f"Volume Ratio: {analysis['volume_ratio']:.2f}x average\n"
            response += f"Price Change: {analysis['price_change']:+.2f}%\n"
            response += f"Trend Strength: {analysis['trend_strength'].title()}\n"
            response += f"Confirms Trend: {'Yes' if analysis['confirms_trend'] else 'No'}"
            response += "```"
            return response
            
        except Exception as e:
            return f"Error analyzing volume for {symbol}: {str(e)}"
        
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
        
    def get_ma_analysis(self, symbol):
        try:
            analysis = self.trading_bot.calculate_moving_averages(symbol)
            
            response = f"Moving Average Analysis for {symbol}:\n```"
            response += f"Current Price: ${analysis['current_price']:.2f}\n"
            response += f"20-day SMA: ${analysis['sma_20']:.2f}\n"
            response += f"50-day SMA: ${analysis['sma_50']:.2f}\n"
            response += f"12-day EMA: ${analysis['ema_12']:.2f}\n"
            response += f"26-day EMA: ${analysis['ema_26']:.2f}\n\n"
            response += f"Current Trend: {analysis['trend']}\n"
            response += f"SMA Golden Cross: {'Yes' if analysis['sma_cross_bullish'] else 'No'}\n"
            response += f"SMA Death Cross: {'Yes' if analysis['sma_cross_bearish'] else 'No'}\n"
            response += f"EMA Bullish Cross: {'Yes' if analysis['ema_cross_bullish'] else 'No'}\n"
            response += f"EMA Bearish Cross: {'Yes' if analysis['ema_cross_bearish'] else 'No'}"
            response += "```"
            return response
            
        except Exception as e:
            return f"Error analyzing MAs for {symbol}: {str(e)}"
        
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
        
    def set_risk_params(self, stop_loss: float, take_profit: float, max_position: float):
        if self.trading_bot.set_risk_parameters(stop_loss, take_profit, max_position):
            return (f"Risk parameters updated:\n"
                    f"Stop Loss: {stop_loss}%\n"
                    f"Take Profit: {take_profit}%\n"
                    f"Max Position: ${max_position}")
        return "Failed to update risk parameters. Please check values."
        
    def get_sentiment_analysis(self, symbol):
        try:
            analysis = self.trading_bot.analyze_market_sentiment(symbol)
            
            response = f"Market Sentiment Analysis for {symbol}:\n```"
            response += f"Overall Sentiment: {analysis['overall_sentiment']}\n"
            response += f"Sentiment Score: {analysis['sentiment_score']:.1f}\n\n"
            
            response += "Price Changes:\n"
            response += f"  7-Day:  {analysis['price_changes']['short_term']:+.2f}%\n"
            response += f"  30-Day: {analysis['price_changes']['medium_term']:+.2f}%\n"
            response += f"  90-Day: {analysis['price_changes']['long_term']:+.2f}%\n\n"
            
            response += "Momentum:\n"
            response += f"  Short-term:  {analysis['momentum']['short_term'].title()}\n"
            response += f"  Medium-term: {analysis['momentum']['medium_term'].title()}\n"
            response += f"  Long-term:   {analysis['momentum']['long_term'].title()}\n\n"
            
            response += f"Trend Strength: {analysis['trend_strength']:+d}\n"
            response += f"Volume Trend: {analysis['volume_trend'].title()}\n\n"
            
            response += "Technical Indicators:\n"
            response += f"  MA Trend: {analysis['technical_indicators']['ma_trend']}\n"
            response += f"  RSI: {analysis['technical_indicators']['rsi']:.2f}\n"
            response += f"  Volume Ratio: {analysis['technical_indicators']['volume_ratio']:.2f}x average"
            response += "```"
            return response
            
        except Exception as e:
            return f"Error analyzing market sentiment for {symbol}: {str(e)}"
        
    def get_paper_balance(self):
        """Get paper trading account status"""
        balance = self.trading_bot.get_paper_balance()
        
        response = "Paper Trading Account:\n```"
        response += f"Cash Balance: ${balance['cash_balance']:.2f}\n"
        response += f"Position Value: ${balance['positions_value']:.2f}\n"
        response += f"Total Value: ${balance['total_value']:.2f}\n"
        response += f"Realized P/L: ${balance['realized_pl']:+.2f}\n"
        response += f"Unrealized P/L: ${balance['unrealized_pl']:+.2f}\n"
        response += f"Total P/L: ${balance['total_pl']:+.2f} ({balance['pl_percentage']:+.2f}%)\n"
        response += f"Total Fees: ${balance['total_fees']:.2f}"
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
        if total_trades > 0:
            response += f"Winning Trades: {winning_trades}\n"
            response += f"Win Rate: {(winning_trades/total_trades*100):.1f}%\n"
        response += f"Total Profit: ${total_profit:,.2f}\n"
        response += f"Total Fees Paid: ${total_fees:.2f}\n"
        response += f"Current Cash Balance: ${paper_balance['cash_balance']:,.2f}\n"
        response += f"Portfolio Value: ${paper_balance['total_value']:,.2f}\n"
        
        # Calculate ROI using actual initial balance
        roi = ((paper_balance['total_value'] - paper_balance['initial_balance']) / paper_balance['initial_balance']) * 100
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
        """Show current paper positions only"""
        positions = self.trading_bot.paper_positions
        if not positions:
            return "No active paper positions"
        
        response = "Paper Trading Positions:\n```"
        for symbol, pos in positions.items():
            current_price = float(self.trading_bot.client.get_product(f"{symbol}-USD").price)
            profit_info = pos.calculate_profit(current_price)
            
            response += f"\n{symbol}:"
            response += f"\n  Current Price: ${current_price:,.2f}"
            response += f"\n  Entry Price: ${pos.entry_price:,.2f}"
            response += f"\n  Quantity: {pos.quantity:.8f}"
            response += f"\n  Position Value: ${(current_price * pos.quantity):,.2f}"
            response += f"\n  Unrealized P/L: ${profit_info['profit_usd']:+,.2f} ({profit_info['profit_percentage']:+.2f}%)"
            response += f"\n  Fees Paid: ${profit_info['fees_paid']:.2f}"
            response += "\n"
        response += "```"
        return response
        
    def show_commands(self):
        return self.get_help()
        
    def get_real_balance(self):
        """Get real trading account status"""
        try:
            balance = self.trading_bot.get_account_balance()
            positions = self.trading_bot.get_position_info()
            
            response = "Real Trading Account:\n```"
            response += f"Cash Balance (USD): ${balance['balances'].get('USD', {}).get('balance', 0):.2f}\n"
            
            # Calculate total positions value
            positions_value = sum(pos['current_price'] * pos['quantity'] for pos in positions.values())
            response += f"Position Value: ${positions_value:.2f}\n"
            
            # Calculate total value
            total_value = balance['total_usd_value']
            response += f"Total Value: ${total_value:.2f}\n"
            
            # Calculate P/L if we have position history
            if self.trading_bot.position_history:
                realized_pl = sum(pos['profit_usd'] for pos in self.trading_bot.position_history)
                unrealized_pl = sum(pos['profit_usd'] for pos in positions.values())
                total_pl = realized_pl + unrealized_pl
                
                response += f"Realized P/L: ${realized_pl:+.2f}\n"
                response += f"Unrealized P/L: ${unrealized_pl:+.2f}\n"
                response += f"Total P/L: ${total_pl:+.2f}\n"
            
            # Add total fees
            total_fees = sum(pos.get('fees_paid', 0) for pos in self.trading_bot.position_history)
            total_fees += sum(pos.get('fees_paid', 0) for pos in positions.values())
            response += f"Total Fees: ${total_fees:.2f}"
            
            response += "```"
            return response
            
        except Exception as e:
            return f"Error getting real balance: {str(e)}"
        
    def get_real_trades(self):
        """Show real trading history"""
        trades = self.trading_bot.trade_history
        if not trades:
            return "No real trades yet"
        
        response = "Real Trading History:\n```"
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