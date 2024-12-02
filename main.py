import os
from bot.trading_bot import TradingBot
from bot.command_handler import CommandHandler
from discord.ext import commands
import discord
from keep_alive import keep_alive

# Initialize Discord bot
bot = commands.Bot(command_prefix='!', intents=discord.Intents.all())
trading_bot = TradingBot()
command_handler = CommandHandler(trading_bot)

@bot.event
async def on_ready():
    print(f'Bot is ready! Logged in as {bot.user}')
    
    # Initialize trading bot
    await trading_bot.post_init()  # Call the async post-init method
    
    # Look for notifications and logs channels in all guilds (servers) the bot is in
    notification_channel = None
    logs_channel = None
    for guild in bot.guilds:
        if not notification_channel:
            notification_channel = discord.utils.get(guild.text_channels, name='notifications')
        if not logs_channel:
            logs_channel = discord.utils.get(guild.text_channels, name='logs')
    
    if notification_channel:
        trading_bot.set_discord_channel(notification_channel)
        await trading_bot.send_notification("Trading bot initialized and ready!")
        print(f"Found notifications channel: #{notification_channel.name} in {notification_channel.guild.name}")
    else:
        print("No 'notifications' channel found. Bot will run without sending notifications.")
        trading_bot.set_discord_channel(None)
    
    if logs_channel:
        trading_bot.set_logs_channel(logs_channel)
        print(f"Found logs channel: #{logs_channel.name} in {logs_channel.guild.name}")
    else:
        print("No 'logs' channel found. Bot will run without sending logs to Discord.")
        trading_bot.set_logs_channel(None)

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

@bot.command(name='balance')
async def get_balance(ctx):
    """Show real trading account balance"""
    response = command_handler.get_balance()
    await ctx.send(response)

@bot.command(name='trades')
async def get_trades(ctx):
    """Show real trading history"""
    response = command_handler.get_trades()
    await ctx.send(response)

@bot.command(name='coin')
async def analyze_coin(ctx, symbol: str):
    """Get detailed analysis for a specific coin"""
    response = command_handler.analyze_coin(symbol.upper())
    await ctx.send(response)

@bot.command(name='trailing')
async def set_trailing_stop(ctx, setting: str = None, value: float = None):
    """Configure trailing stop settings"""
    if not setting:
        # Show current settings
        response = command_handler.set_trailing_stop()
    elif setting.lower() == 'enable':
        response = command_handler.set_trailing_stop(enabled=True)
    elif setting.lower() == 'disable':
        response = command_handler.set_trailing_stop(enabled=False)
    elif setting.lower() == 'percent' and value is not None:
        response = command_handler.set_trailing_stop(percentage=value)
    elif setting.lower() == 'activation' and value is not None:
        response = command_handler.set_trailing_stop(activation=value)
    else:
        response = (
            "Usage:\n"
            "!trailing              - Show current settings\n"
            "!trailing enable       - Enable trailing stop\n"
            "!trailing disable      - Disable trailing stop\n"
            "!trailing percent 5.0  - Set trailing stop percentage\n"
            "!trailing activation 3 - Set profit % before activation"
        )
    
    await ctx.send(response)

@bot.command(name='takeprofit', aliases=['tp'])
async def set_take_profit(ctx, setting: str = None, value: float = None):
    """Configure take profit settings"""
    if not setting:
        # Show current settings
        response = command_handler.set_take_profit()
    elif setting.lower() == 'full' and value is not None:
        response = command_handler.set_take_profit(full_tp=value)
    elif setting.lower() == 'partial' and value is not None:
        response = command_handler.set_take_profit(partial_tp=value)
    elif setting.lower() == 'size' and value is not None:
        response = command_handler.set_take_profit(partial_size=value)
    else:
        response = (
            "Usage:\n"
            "!tp                - Show current settings\n"
            "!tp full 10       - Set full take profit to 10%\n"
            "!tp partial 7     - Set partial take profit to 7%\n"
            "!tp size 0.5      - Set partial size to 50%"
        )
    
    await ctx.send(response)

# Run the bot
keep_alive()
bot.run(os.getenv('DISCORD_TOKEN')) 