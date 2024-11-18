import os
import time
import threading
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import requests
from coinbase.wallet.client import Client

# Set up logging
logging.basicConfig(
    filename='trading_bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class TradingBot:
    def __init__(self):
        try:
            # Debug logging
            logging.info("Starting bot initialization...")
            
            # Get API credentials directly
            self.api_key = os.environ['COINBASE_API_KEY']  # Just use the key ID directly
            self.api_secret = os.environ['COINBASE_API_SECRET']
            logging.info("API credentials loaded")
            
            # Initialize client
            logging.info("Initializing Coinbase client...")
            self.client = Client(self.api_key, self.api_secret)
            logging.info("Coinbase client initialized")
            
            # Rest of initialization stays the same
            self.watched_coins = set()
            self.trading_interval = 300  # 5 minutes
            self.rsi_period = 14
            self.rsi_overbought = 70
            self.rsi_oversold = 30
            self.trading_active = False
            self.trade_amount = 100  # Default trade amount in USD
            self.trade_history = []
            self.load_config()
            
        except Exception as e:
            logging.error(f"Failed to initialize trading bot: {str(e)}")
            raise Exception(f"Bot initialization failed: {str(e)}")
        
    def start_trading_loop(self):
        if not self.trading_active:
            self.trading_active = True
            thread = threading.Thread(target=self._trading_loop)
            thread.daemon = True
            thread.start()
            logging.info("Trading loop started")
            return "Trading bot started successfully"
        return "Trading bot is already running"
        
    def stop_trading_loop(self):
        if self.trading_active:
            self.trading_active = False
            logging.info("Trading loop stopped")
            return "Trading bot stopped successfully"
        return "Trading bot is already stopped"
        
    def _trading_loop(self):
        consecutive_errors = 0
        while self.trading_active:
            try:
                for coin in self.watched_coins:
                    try:
                        self._check_and_trade(coin)
                        consecutive_errors = 0
                    except Exception as e:
                        consecutive_errors += 1
                        logging.error(f"Error trading {coin}: {str(e)}")
                        if consecutive_errors >= 3:
                            self.trading_active = False
                            logging.critical("Too many consecutive errors. Stopping trading bot.")
                            return
                time.sleep(self.trading_interval)
            except Exception as e:
                logging.critical(f"Critical error in trading loop: {str(e)}")
                self.trading_active = False
                return
            
    def _check_and_trade(self, symbol):
        rsi = self.calculate_rsi(symbol)
        
        if rsi <= self.rsi_oversold:
            self._place_buy_order(symbol)
        elif rsi >= self.rsi_overbought:
            self._place_sell_order(symbol)
            
    def calculate_rsi(self, symbol):
        # Get historical data
        end = datetime.now()
        start = end - timedelta(days=1)
        prices = self._get_historical_prices(symbol, start, end)
        
        # Calculate RSI
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1]
    
    def _get_historical_prices(self, symbol, start, end):
        try:
            # Get historical prices from Coinbase
            currency_pair = f"{symbol}-USD"
            
            # Get current price data
            price_data = self.client.get_buy_price(currency_pair=currency_pair)
            
            # For demonstration, create a simple price series
            # In production, you'd want to implement proper historical data fetching
            dates = pd.date_range(start=start, end=end, freq='5min')
            prices = pd.Series(float(price_data['amount']), index=dates)
            
            return prices
            
        except Exception as e:
            logging.error(f"Error fetching historical prices for {symbol}: {str(e)}")
            raise
            
    def _place_buy_order(self, symbol):
        try:
            # Get payment methods
            payment_methods = self.client.get_payment_methods()
            if not payment_methods:
                raise Exception("No payment methods available")
                
            # Place buy order
            self.client.buy(self.trade_amount,
                          currency=symbol,
                          payment_method=payment_methods[0]['id'])
            
            trade_info = {
                'timestamp': datetime.now(),
                'action': 'BUY',
                'symbol': symbol,
                'amount_usd': self.trade_amount
            }
            self.trade_history.append(trade_info)
            logging.info(f"Buy order placed for {symbol}: ${self.trade_amount}")
            
        except Exception as e:
            logging.error(f"Error placing buy order for {symbol}: {str(e)}")
            raise
            
    def _place_sell_order(self, symbol):
        try:
            # Get accounts
            accounts = self.client.get_accounts()
            account = next((acc for acc in accounts.data if acc['currency'] == symbol), None)
            
            if not account:
                raise Exception(f"No account found for {symbol}")
                
            # Place sell order
            self.client.sell(self.trade_amount,
                           currency=symbol,
                           payment_method=account['id'])
            
            trade_info = {
                'timestamp': datetime.now(),
                'action': 'SELL',
                'symbol': symbol,
                'amount_usd': self.trade_amount
            }
            self.trade_history.append(trade_info)
            logging.info(f"Sell order placed for {symbol}: ${self.trade_amount}")
            
        except Exception as e:
            logging.error(f"Error placing sell order for {symbol}: {str(e)}")
            raise
            
    def set_trade_amount(self, amount):
        try:
            amount = float(amount)
            if amount <= 0:
                return False
            self.trade_amount = amount
            logging.info(f"Trade amount set to ${amount}")
            return True
        except ValueError:
            return False
            
    def set_rsi_thresholds(self, oversold, overbought):
        try:
            oversold = float(oversold)
            overbought = float(overbought)
            if 0 <= oversold <= overbought <= 100:
                self.rsi_oversold = oversold
                self.rsi_overbought = overbought
                logging.info(f"RSI thresholds updated: oversold={oversold}, overbought={overbought}")
                return True
            return False
        except ValueError:
            return False
            
    def set_trading_interval(self, minutes):
        try:
            minutes = int(minutes)
            if minutes < 1:
                return False
            self.trading_interval = minutes * 60
            logging.info(f"Trading interval set to {minutes} minutes")
            return True
        except ValueError:
            return False
            
    def get_trade_history(self):
        return self.trade_history
        
    def add_coin(self, symbol):
        try:
            # Verify the coin exists by attempting to get its price
            product_id = f"{symbol}-USD"
            self.client.get_product_historic_rates(product_id, start=datetime.now().isoformat(), end=datetime.now().isoformat())
            self.watched_coins.add(symbol)
            logging.info(f"Added {symbol} to watchlist")
            return True
        except Exception as e:
            logging.error(f"Failed to add {symbol}: {str(e)}")
            return False
        
    def remove_coin(self, symbol):
        if symbol in self.watched_coins:
            self.watched_coins.remove(symbol)
            return True
        return False 

    def _check_balance(self, symbol, action='BUY'):
        try:
            accounts = self.client.get_accounts()
            
            if action == 'BUY':
                # Check USD balance
                usd_account = next((acc for acc in accounts.data if acc['currency'] == 'USD'), None)
                if not usd_account:
                    return False
                return float(usd_account['balance']['amount']) >= self.trade_amount
            else:
                # Check crypto balance
                crypto_account = next((acc for acc in accounts.data if acc['currency'] == symbol), None)
                if not crypto_account:
                    return False
                    
                price_data = self.client.get_spot_price(currency_pair=f'{symbol}-USD')
                current_price = float(price_data['amount'])
                return float(crypto_account['balance']['amount']) * current_price >= self.trade_amount
                
        except Exception as e:
            logging.error(f"Error checking balance: {str(e)}")
            return False 

    def save_config(self):
        config = {
            'watched_coins': list(self.watched_coins),
            'trading_interval': self.trading_interval,
            'rsi_period': self.rsi_period,
            'rsi_overbought': self.rsi_overbought,
            'rsi_oversold': self.rsi_oversold,
            'trade_amount': self.trade_amount
        }
        try:
            with open('bot_config.json', 'w') as f:
                json.dump(config, f)
            logging.info("Configuration saved successfully")
        except Exception as e:
            logging.error(f"Error saving configuration: {str(e)}")
            
    def load_config(self):
        try:
            with open('bot_config.json', 'r') as f:
                config = json.load(f)
                self.watched_coins = set(config['watched_coins'])
                self.trading_interval = config['trading_interval']
                self.rsi_period = config['rsi_period']
                self.rsi_overbought = config['rsi_overbought']
                self.rsi_oversold = config['rsi_oversold']
                self.trade_amount = config['trade_amount']
            logging.info("Configuration loaded successfully")
        except FileNotFoundError:
            logging.info("No configuration file found, using defaults")
        except Exception as e:
            logging.error(f"Error loading configuration: {str(e)}")

    def test_api_connection(self):
        try:
            # First, test authentication
            auth_test = self.client.get_accounts()
            logging.info("Authentication successful")
            
            # Then test price fetching
            try:
                btc_price = self.client.get_spot_price(currency_pair='BTC-USD')
                price = float(btc_price.amount)
                logging.info(f"Successfully fetched BTC price: ${price}")
                return price
            except Exception as e:
                logging.error(f"Price fetch failed: {str(e)}")
                raise Exception(f"Could not fetch price: {str(e)}")
            
        except Exception as e:
            logging.error(f"Authentication failed: {str(e)}")
            raise Exception(f"Authentication failed - Please check your API keys: {str(e)}")