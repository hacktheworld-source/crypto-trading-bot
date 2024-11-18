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
    # Start the trading loop
    trading_bot.start_trading_loop()

# Command to add a coin to watchlist
@bot.command(name='addcoin')
async def add_coin(ctx, symbol: str):
    response = command_handler.add_coin(symbol.upper())
    await ctx.send(response)

# Command to remove a coin from watchlist
@bot.command(name='removecoin')
async def remove_coin(ctx, symbol: str):
    response = command_handler.remove_coin(symbol.upper())
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
async def start_bot(ctx):
    response = command_handler.start_bot()
    await ctx.send(response)

# Command to stop the trading bot
@bot.command(name='stop')
async def stop_bot(ctx):
    response = command_handler.stop_bot()
    await ctx.send(response)

# Command to get the bot's current status
@bot.command(name='status')
async def get_status(ctx):
    response = command_handler.get_status()
    await ctx.send(response)

# Run the bot
keep_alive()
bot.run(os.getenv('DISCORD_TOKEN')) 