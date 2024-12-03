import os
from bot.trading_bot import TradingBot
from bot.command_handler import CommandHandler
from discord.ext import commands
import discord
from keep_alive import keep_alive
import logging

# Add at top of file
logging.basicConfig(
    filename='trading_bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize Discord bot
bot = commands.Bot(command_prefix='!', intents=discord.Intents.all())
trading_bot = TradingBot()
command_handler = CommandHandler(trading_bot)

@bot.event
async def on_ready():
    logging.info(f'Bot is ready! Logged in as {bot.user}')
    
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

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if message.content.startswith('!'):
        command = message.content[1:].split()[0]
        args = message.content.split()[1:]
        response = await command_handler.handle_command(command, *args)
        await message.channel.send(response)

# Run the bot
keep_alive()
bot.run(os.getenv('DISCORD_TOKEN')) 