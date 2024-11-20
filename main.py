import os
import time
from bot.trading_bot import TradingBot
from bot.command_handler import CommandHandler
from discord.ext import commands
import discord
import asyncio
from keep_alive import keep_alive
import logging
import signal
import sys
import psutil

async def heartbeat(trading_bot):
    """Heartbeat coroutine to monitor bot health"""
    while True:
        try:
            await trading_bot.send_notification("🫀 Bot heartbeat check", is_update=True)
            await asyncio.sleep(3600)  # Check every hour
        except Exception as e:
            logging.error(f"Heartbeat error: {e}")
            await asyncio.sleep(60)

def cleanup_old_processes():
    """Kill any existing bot processes and Flask servers"""
    current_pid = os.getpid()
    killed_count = 0
    
    print(f"Current process ID: {current_pid}")
    print("Searching for old bot processes...")
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'connections']):
        try:
            # Skip our own process
            if proc.pid == current_pid:
                continue
                
            # Check for Python processes
            if proc.info['name'] == 'python' or proc.info['name'] == 'python3':
                kill_process = False
                
                # Check command line for our files
                if proc.info['cmdline']:
                    if any(x in str(proc.info['cmdline']) for x in ['main.py', 'trading_bot.py', 'keep_alive.py']):
                        kill_process = True
                
                # Check if process is using our port
                if proc.info['connections']:
                    if any(conn.laddr.port == 8080 for conn in proc.info['connections']):
                        kill_process = True
                
                if kill_process:
                    print(f"Killing process {proc.pid} ({' '.join(proc.info['cmdline'] or [])})")
                    proc.kill()
                    killed_count += 1
                    
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
            print(f"Error checking process: {e}")
            continue
    
    if killed_count > 0:
        print(f"Killed {killed_count} old processes")
        # Give processes time to fully terminate
        time.sleep(2)
    else:
        print("No old processes found")

def main():
    # Add this at the start of main()
    cleanup_old_processes()
    
    # Initialize Discord bot
    intents = discord.Intents.all()
    bot = commands.Bot(command_prefix='!', intents=intents)
    trading_bot = TradingBot()
    command_handler = CommandHandler(trading_bot)

    def signal_handler(sig, frame):
        print("\nShutting down gracefully...")
        if trading_bot:
            trading_bot.stop_trading_loop()
        if bot:
            asyncio.create_task(bot.close())
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    @bot.event
    async def setup_hook():
        """This is called when the bot starts"""
        bot.loop.create_task(heartbeat(trading_bot))

    @bot.event
    async def on_ready():
        print(f'Bot is ready! Logged in as {bot.user}')
        
        # Look for notifications channel
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

    # Command to set trade amount
    @bot.command(name='setamount')
    async def set_amount(ctx, amount: float):
        response = command_handler.set_trade_amount(amount)
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
            await ctx.send("⚠️ WARNING: You are about to start REAL trading with actual money. Type '!confirm' within 30 seconds to proceed.")
            
            def check(m):
                return m.author == ctx.author and m.content.lower() == '!confirm'
            
            try:
                await bot.wait_for('message', check=check, timeout=30.0)
                response = command_handler.start_real_trading()
            except TimeoutError:
                response = "Real trading start cancelled due to timeout."
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

    # Start the bot
    try:
        bot.run(os.getenv('DISCORD_TOKEN'))
    except Exception as e:
        print(f"Error starting bot: {e}")

if __name__ == "__main__":
    main() 