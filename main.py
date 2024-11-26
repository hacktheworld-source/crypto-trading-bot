import os
import time
from bot.trading_bot import TradingBot
from bot.command_handler import CommandHandler
from discord.ext import commands
import discord
import asyncio
from keep_alive import keep_alive

def main():
    # Initialize Discord bot
    intents = discord.Intents.all()
    bot = commands.Bot(command_prefix='!', intents=intents)
    trading_bot = TradingBot()
    command_handler = CommandHandler(trading_bot)

    @bot.event
    async def on_ready():
        print(f'Bot is ready! Logged in as {bot.user}')
        
        # Look for notifications channel in all guilds (servers) the bot is in
        notification_channel = None
        for guild in bot.guilds:
            channel = discord.utils.get(guild.text_channels, name='notifications')
            if channel:
                notification_channel = channel
                break
        
        if notification_channel:
            trading_bot.set_discord_channel(notification_channel)
            await trading_bot.send_notification("Trading bot initialized and ready!")
            print(f"Found notifications channel: #{notification_channel.name} in {notification_channel.guild.name}")
        else:
            print("No 'notifications' channel found. Bot will run without sending notifications.")
            trading_bot.set_discord_channel(None)

    # Command to add a coin to watchlist
    @bot.command(name='addcoin')
    async def add_coin(ctx, *symbols: str):
        """Add one or more coins to watchlist"""
        response = command_handler.add_coin(*[s.upper() for s in symbols])
        await ctx.send(response)

    # Command to remove a coin from watchlist
    @bot.command(name='removecoin')
    async def remove_coin(ctx, *symbols: str):
        """Remove one or more coins from watchlist"""
        response = command_handler.remove_coin(*[s.upper() for s in symbols])
        await ctx.send(response)

    # Command to list all watched coins
    @bot.command(name='listcoins')
    async def list_coins(ctx):
        response = command_handler.list_coins()
        await ctx.send(response)

    # Command to get current RSI for a coin
    @bot.command(name='rsi')
    async def get_rsi(ctx, symbol: str):
        response = command_handler.get_rsi(symbol.upper())
        await ctx.send(response)

    # Command to set RSI thresholds
    @bot.command(name='setrsi')
    async def set_rsi(ctx, oversold: float, overbought: float):
        response = command_handler.set_rsi_thresholds(oversold, overbought)
        await ctx.send(response)

    # Command to set trading interval
    @bot.command(name='setinterval')
    async def set_interval(ctx, minutes: int):
        response = command_handler.set_trading_interval(minutes)
        await ctx.send(response)

    # Command to get trade history
    @bot.command(name='history')
    async def trade_history(ctx):
        response = command_handler.get_trade_history()
        await ctx.send(response)

    # Command to start the trading bot
    @bot.command(name='start')
    async def start_bot(ctx, mode: str = 'paper'):
        """Start the trading bot in either paper or real mode"""
        mode = mode.lower()
        if mode not in ['paper', 'real']:
            await ctx.send("Invalid mode. Use '!start paper' or '!start real'")
            return
        
        if mode == 'real':
            # Add confirmation for real trading
            msg = await ctx.send("⚠️ WARNING: You are about to start REAL trading with actual money. Type '!confirm' within 30 seconds to proceed.")
            
            def check(m):
                return m.author == ctx.author and m.content.lower() == '!confirm' and m.channel == ctx.channel
            
            try:
                await bot.wait_for('message', check=check, timeout=30.0)
                response = command_handler.start_real_trading()
            except asyncio.TimeoutError:
                response = "Real trading start cancelled due to timeout."
                await msg.edit(content=f"{msg.content}\n\n❌ {response}")
                return
            
        else:
            response = command_handler.start_paper_trading()
        
        await ctx.send(response)

    # Command to stop the trading bot
    @bot.command(name='stop')
    async def stop_bot(ctx, mode: str = 'all'):
        """Stop the trading bot (paper, real, or all)"""
        mode = mode.lower()
        if mode not in ['paper', 'real', 'all']:
            await ctx.send("Invalid mode. Use '!stop paper', '!stop real', or '!stop all'")
            return
        
        response = command_handler.stop_trading(mode)
        await ctx.send(response)

    # Command to get the bot's current status
    @bot.command(name='status')
    async def get_status(ctx):
        response = command_handler.get_status()
        await ctx.send(response)

    # Command to test the bot's response
    @bot.command(name='ping')
    async def ping(ctx):
        await ctx.send('Pong! Bot is working!')

    # Command to test the Coinbase API connection
    @bot.command(name='testapi')
    async def test_api(ctx):
        response = command_handler.test_api()
        await ctx.send(response)

    # Command to get help
    @bot.command(name='commands')
    async def show_commands(ctx):
        response = command_handler.get_help()
        await ctx.send(response)

    # Command to get the current price of a coin
    @bot.command(name='price')
    async def get_price(ctx, symbol: str):
        response = command_handler.get_price(symbol.upper())
        await ctx.send(response)

    # Command to get volume analysis for a coin
    @bot.command(name='volume')
    async def get_volume(ctx, symbol: str):
        response = command_handler.get_volume_analysis(symbol.upper())
        await ctx.send(response)

    # Command to get positions
    @bot.command(name='positions')
    async def get_positions(ctx):
        response = command_handler.get_positions()
        await ctx.send(response)

    # Command to get position history
    @bot.command(name='poshistory')
    async def get_position_history(ctx):
        response = command_handler.get_position_history()
        await ctx.send(response)

    # Command to get Moving Average analysis for a coin
    @bot.command(name='ma')
    async def get_ma(ctx, symbol: str):
        response = command_handler.get_ma_analysis(symbol.upper())
        await ctx.send(response)

    # Command to get performance stats
    @bot.command(name='performance')
    async def get_performance(ctx):
        response = command_handler.get_performance()
        await ctx.send(response)

    # Command to set risk management parameters
    @bot.command(name='setrisk')
    async def set_risk(ctx, stop_loss: float, take_profit: float, max_position: float):
        """Set risk management parameters"""
        response = command_handler.set_risk_params(stop_loss, take_profit, max_position)
        await ctx.send(response)

    # Command to get market sentiment analysis for a coin
    @bot.command(name='sentiment')
    async def get_sentiment(ctx, symbol: str):
        """Get market sentiment analysis for a coin"""
        response = command_handler.get_sentiment_analysis(symbol.upper())
        await ctx.send(response)

    # Add these new commands

    @bot.command(name='paper')
    async def paper_trading(ctx, action: str, *args):
        """Paper trading commands"""
        action = action.lower()
        
        if action == 'start':
            initial_balance = float(args[0]) if args else 1000.0
            response = command_handler.start_paper_trading(initial_balance)
        elif action == 'balance':
            response = command_handler.get_paper_balance()
        elif action == 'reset':
            initial_balance = float(args[0]) if args else 1000.0
            response = command_handler.reset_paper_trading(initial_balance)
        elif action == 'stats':
            response = command_handler.get_paper_stats()
        elif action == 'trades':
            response = command_handler.get_paper_trades()
        elif action == 'positions':
            response = command_handler.get_paper_positions()
        else:
            response = (
                "Invalid paper trading command. Available commands:\n"
                "!paper start [amount] - Start paper trading with optional initial balance\n"
                "!paper balance        - Show current paper trading balance\n"
                "!paper reset [amount] - Reset paper trading with optional new balance\n"
                "!paper stats          - Show detailed paper trading statistics\n"
                "!paper trades         - Show paper trading history\n"
                "!paper positions      - Show current paper positions"
            )
        
        await ctx.send(response)

    @bot.command(name='balance')
    async def get_balance(ctx):
        """Show real trading balance"""
        response = command_handler.get_real_balance()
        await ctx.send(response)

    @bot.command(name='trades')
    async def get_trades(ctx):
        """Show real trading history"""
        response = command_handler.get_real_trades()
        await ctx.send(response)

    @bot.command(name='confirm')
    async def confirm(ctx):
        """Confirmation command - only used internally by other commands"""
        pass  # Just needs to exist to be recognized

    # Add this before bot.run
    keep_alive()  # Start the Flask server
    
    # Start the bot
    try:
        bot.run(os.getenv('DISCORD_TOKEN'))
    except Exception as e:
        print(f"Error starting bot: {e}")
    
    # Keep the script running
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main() 