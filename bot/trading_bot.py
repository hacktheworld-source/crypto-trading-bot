import os
import time
import threading
import logging
from coinbase.wallet.client import Client
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Set up logging
logging.basicConfig(
    filename='trading_bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class TradingBot:
    def __init__(self):
        self.api_key = os.environ['COINBASE_API_KEY']
        self.api_secret = os.environ['COINBASE_API_SECRET']
        self.client = Client(self.api_key, self.api_secret)
        self.watched_coins = set()
        self.trading_interval = 300  # 5 minutes
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.trading_active = False
        self.trade_amount = 100  # Default trade amount in USD
        self.trade_history = []
        self.load_config()
        
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
            product_id = f"{symbol}-USD"
            granularity = 300  # 5-minute intervals
            
            # Calculate time intervals (Coinbase has a limit of 300 data points)
            time_diff = end - start
            interval_seconds = granularity * 300
            
            prices = []
            current_start = start
            
            while current_start < end:
                current_end = min(current_start + timedelta(seconds=interval_seconds), end)
                
                historical_data = self.client.get_product_historic_rates(
                    product_id,
                    start=current_start.isoformat(),
                    end=current_end.isoformat(),
                    granularity=granularity
                )
                
                prices.extend([[data[0], data[4]] for data in historical_data])  # timestamp and close price
                current_start = current_end
            
            df = pd.DataFrame(prices, columns=['timestamp', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            return df['close']
            
        except Exception as e:
            logging.error(f"Error fetching historical prices for {symbol}: {str(e)}")
            raise
            
    def _place_buy_order(self, symbol):
        if not self._check_balance(symbol, 'BUY'):
            logging.error(f"Insufficient USD balance for buying {symbol}")
            return
        try:
            # Place market buy order
            payment_method = self.client.get_payment_methods()[0]  # Get default payment method
            
            self.client.buy(
                amount=str(self.trade_amount),
                currency="USD",
                payment_method=payment_method.id
            )
            
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
        if not self._check_balance(symbol, 'SELL'):
            logging.error(f"Insufficient {symbol} balance for selling")
            return
        try:
            # Place market sell order
            account = self.client.get_account(symbol)
            
            self.client.sell(
                amount=str(self.trade_amount),
                currency="USD",
                payment_method=account.id
            )
            
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
            if action == 'BUY':
                # Check USD balance
                usd_account = self.client.get_account('USD')
                return float(usd_account.balance) >= self.trade_amount
            else:
                # Check crypto balance
                crypto_account = self.client.get_account(symbol)
                current_price = float(self.client.get_spot_price(currency_pair=f'{symbol}-USD').amount)
                return float(crypto_account.balance) * current_price >= self.trade_amount
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