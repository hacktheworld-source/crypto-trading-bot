class CommandHandler:
    def __init__(self, trading_bot):
        self.trading_bot = trading_bot
        
    def add_coin(self, symbol):
        if self.trading_bot.add_coin(symbol):
            return f"Successfully added {symbol} to watchlist"
        return f"Failed to add {symbol} to watchlist"
        
    def remove_coin(self, symbol):
        if self.trading_bot.remove_coin(symbol):
            return f"Successfully removed {symbol} from watchlist"
        return f"Failed to remove {symbol} from watchlist"
        
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
        
    def start_bot(self):
        return self.trading_bot.start_trading_loop()
        
    def stop_bot(self):
        return self.trading_bot.stop_trading_loop()
        
    def get_status(self):
        bot = self.trading_bot
        status = "Bot Status:\n```"
        status += f"Trading Active: {bot.trading_active}\n"
        status += f"Watched Coins: {', '.join(bot.watched_coins) if bot.watched_coins else 'None'}\n"
        status += f"Trade Amount: ${bot.trade_amount}\n"
        status += f"RSI Settings: Oversold={bot.rsi_oversold}, Overbought={bot.rsi_overbought}\n"
        status += f"Check Interval: {bot.trading_interval//60} minutes\n"
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