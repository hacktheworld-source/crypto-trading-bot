import os
import time
import threading
import logging
import pandas as pd
from datetime import datetime, timedelta
import json
from coinbase.rest import RESTClient
from typing import Dict, Any, List, Union, Optional
from decimal import Decimal
from .position import Position
import asyncio
import discord

# Set up logging
logging.basicConfig(
    filename='trading_bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class TradingBot:
    def __init__(self):
        try:
            logging.info("Starting bot initialization...")
            
            if 'COINBASE_API_KEY' not in os.environ:
                raise Exception("COINBASE_API_KEY not found in environment variables")
            if 'COINBASE_API_SECRET' not in os.environ:
                raise Exception("COINBASE_API_SECRET not found in environment variables")
            
            self.client = RESTClient(
                api_key=os.environ['COINBASE_API_KEY'].strip(),
                api_secret=os.environ['COINBASE_API_SECRET'].strip()
            )
            logging.info("Coinbase client initialized successfully")
            
            self.watched_coins: set = set()
            self.trading_interval: int = 300
            self.rsi_period: int = 14
            self.rsi_overbought: float = 70.0
            self.rsi_oversold: float = 30.0
            self.trading_active: bool = False
            self.trade_history: List[Dict[str, Any]] = []
            self.positions: Dict[str, Position] = {}  # Track active positions
            self.position_history: List[Dict[str, Any]] = []  # Track closed positions
            self.stop_loss_percentage = 5.0  # Default 5% stop loss
            self.take_profit_percentage = 10.0  # Default 10% take profit
            self.max_position_size = 1000.0  # Maximum USD in any single position
            self.load_config()
            
            # Paper trading attributes
            self.paper_trading = False  # Don't auto-start paper trading
            self.paper_balance = 1000.0
            self.paper_positions: Dict[str, Position] = {}
            self.paper_trade_history: List[Dict[str, Any]] = []
            self.paper_portfolio_value = self.paper_balance
            
            # Add these new attributes
            self.paper_initial_balance = 0.0  # Will be set when paper trading starts
            self.paper_realized_pl = 0.0      # Track realized P/L
            self.paper_total_fees = 0.0       # Track total fees paid
            
            self.discord_channel = None  # Will be set when bot starts
            
        except Exception as e:
            logging.error(f"Failed to initialize trading bot: {str(e)}")
            raise Exception(f"Bot initialization failed: {str(e)}")
        
    def start_trading_loop(self, paper: bool = True):
        """Start the trading loop in either paper or real mode"""
        try:
            if paper:
                if self.trading_active:
                    return "Real trading is already active"
                self.paper_trading = True
                mode = "Paper"
            else:
                if self.paper_trading:
                    return "Paper trading is already active"
                self.trading_active = True
                mode = "Real"
            
            # Create and start the trading loop task
            loop = asyncio.get_event_loop()
            loop.create_task(self._trading_loop())
            
            logging.info(f"{mode} trading loop started")
            return f"{mode} trading bot started successfully"
            
        except Exception as e:
            logging.error(f"Failed to start trading loop: {str(e)}")
            self.trading_active = False
            self.paper_trading = False
            return f"Error starting trading bot: {str(e)}"
        
    def stop_trading_loop(self):
        """Stop the trading loop"""
        was_active = self.trading_active
        self.trading_active = False
        self.paper_trading = False  # Stop paper trading too
        logging.info("Trading loop stopped")
        return "Trading bot stopped successfully" if was_active else "Trading bot is already stopped"
        
    async def _trading_loop(self):
        """Enhanced main trading loop with better analysis and risk management"""
        while self.trading_active or self.paper_trading:
            try:
                # Add reconnection logic
                if not self.client:
                    logging.info("Attempting to reconnect to Coinbase...")
                    self.client = RESTClient(
                        api_key=os.environ['COINBASE_API_KEY'].strip(),
                        api_secret=os.environ['COINBASE_API_SECRET'].strip()
                    )
                
                mode = "Paper" if self.paper_trading else "Real"
                logging.info(f"{mode} trading loop iteration starting")
                
                # Process trades for each coin
                analysis_results = {}  # Store analysis results to avoid recalculation
                
                for symbol in self.watched_coins:
                    try:
                        # Get comprehensive analysis
                        prediction = self._analyze_price_prediction(symbol)
                        current_price = prediction['current_price']
                        rsi = prediction['rsi']
                        volume_data = self.analyze_volume(symbol)
                        ma_data = self.calculate_moving_averages(symbol)
                        
                        # Calculate signals using centralized method
                        signals = self._calculate_signals(
                            symbol, prediction, rsi, volume_data, ma_data
                        )
                        
                        # Store analysis for interval update
                        analysis_results[symbol] = {
                            'prediction': prediction,
                            'current_price': current_price,
                            'rsi': rsi,
                            'volume_data': volume_data,
                            'ma_data': ma_data,
                            'signals': signals
                        }
                        
                        # Check existing positions first
                        position = self.paper_positions.get(symbol) if self.paper_trading else self.positions.get(symbol)
                        if position:
                            # FIXED: Actually call position management
                            await self._manage_position(symbol, position, prediction, current_price)
                            continue  # Skip entry analysis if we have a position
                        
                        # Entry Analysis (only if no position exists)
                        if signals['buy_signals'] >= signals['required_signals'] and self._should_trade(symbol, 'BUY'):
                            await self._analyze_entry(symbol, prediction, current_price, signals)
                    
                    except Exception as e:
                        logging.error(f"Error analyzing {symbol}: {str(e)}")
                        await self.send_notification(f"❌ Error analyzing {symbol}: {str(e)}")
                        continue
                    
                # Send interval update using stored analysis results
                await self.send_interval_update(analysis_results)
                
                # Wait for next interval
                await asyncio.sleep(self.trading_interval)
                
            except Exception as e:
                logging.error(f"Error in trading loop: {str(e)}")
                await self.send_notification(f"❌ Error in trading loop: {str(e)}")
                await asyncio.sleep(300)  # 5 minutes
                
        logging.info("Trading loop stopped")

    async def _manage_position(self, symbol: str, position: Position, prediction: Dict[str, Any], current_price: float):
        """Enhanced position management with weighted exit signals"""
        try:
            profit_info = position.calculate_profit(current_price)
            position.update_price(current_price)  # Make sure we track highest price
            
            # Check percentage-based stop loss first
            if profit_info['profit_percentage'] <= -self.stop_loss_percentage:
                if self._should_trade(symbol, 'SELL'):
                    decision_factors = [
                        "🛑 Stop Loss Triggered",
                        f"Entry Price: ${position.entry_price:.2f}",
                        f"Current Price: ${current_price:.2f}",
                        f"Loss: {profit_info['profit_percentage']:.2f}%",
                        f"Stop Loss: {self.stop_loss_percentage}%"
                    ]
                    await self._place_sell_order(symbol, position.quantity, decision_factors)
                    return
                
            # Add trailing stop check for profitable positions
            if profit_info['profit_percentage'] > 3:  # Once we're up 3%
                trailing_stop = profit_info['profit_percentage'] * 0.5  # Lock in 50% of gains
                max_price = position.highest_price
                stop_price = max_price * (1 - trailing_stop/100)
                
                if current_price <= stop_price:
                    decision_factors = [
                        "🎯 Trailing Stop Hit",
                        f"Max Price: ${max_price:.2f}",
                        f"Stop Price: ${stop_price:.2f}",
                        f"Current Price: ${current_price:.2f}",
                        f"Profit: {profit_info['profit_percentage']:.2f}%"
                    ]
                    await self._place_sell_order(symbol, position.quantity, decision_factors)  # Fixed: Actually place the sell order
                    return
            
            # Calculate exit signals
            exit_signals = 0
            required_exit_signals = 3
            
            # Get analysis data
            volume_data = self.analyze_volume(symbol)
            ma_data = self.calculate_moving_averages(symbol)
            rsi = prediction['rsi']
            
            # RSI signals
            if rsi >= self.rsi_overbought:
                exit_signals += 2
            elif rsi > 60:
                exit_signals += 1
                
            # Trend reversal signals
            if ma_data['trend'] == 'Strong Downtrend':
                exit_signals += 2
            elif ma_data['trend'] == 'Weak Downtrend':
                exit_signals += 1
                
            # Volume signals
            if volume_data['volume_ratio'] > 1.5 and volume_data['price_change'] < 0:
                exit_signals += 2
            elif volume_data['volume_ratio'] > 1.2 and volume_data['price_change'] < 0:
                exit_signals += 1
                
            # Prediction score signals
            if prediction['prediction_score'] < -50:
                exit_signals += 2
            elif prediction['prediction_score'] < -30:
                exit_signals += 1
                
            # Check stop loss first
            stop_loss = self._calculate_stop_loss(symbol, position.entry_price)
            if current_price <= stop_loss:
                decision_factors = [
                    "🛑 Stop Loss Triggered",
                    f"Entry Price: ${position.entry_price:.2f}",
                    f"Stop Loss Level: ${stop_loss:.2f}",
                    f"Current Price: ${current_price:.2f}",
                    f"Loss: {profit_info['profit_percentage']:.2f}%"
                ]
                await self._place_sell_order(symbol, position.quantity, decision_factors)
                return
                
            # Check take profit with dynamic threshold
            take_profit_threshold = (
                self.take_profit_percentage * 0.7  # Lower threshold if bearish
                if prediction['prediction_score'] < -30
                else self.take_profit_percentage
            )
            
            # Handle exits based on signals and profit
            if exit_signals >= required_exit_signals or profit_info['profit_percentage'] >= take_profit_threshold:
                # Determine if we should take partial or full profits
                should_partial_exit = (
                    prediction['prediction_score'] > 0 and  # Still somewhat bullish
                    profit_info['profit_percentage'] >= take_profit_threshold and
                    not position.partial_exit_taken  # New flag to track partial exits
                )
                
                quantity = position.quantity * (0.5 if should_partial_exit else 1.0)
                decision_factors = [
                    "💰 Taking Profits",
                    f"Exit Signals: {exit_signals}/{required_exit_signals}",
                    f"Entry Price: ${position.entry_price:.2f}",
                    f"Current Price: ${current_price:.2f}",
                    f"Profit: {profit_info['profit_percentage']:.2f}%",
                    f"RSI: {rsi:.2f}",
                    f"Trend: {ma_data['trend']}",
                    f"Volume: {volume_data['volume_ratio']:.1f}x average",
                    f"Prediction Score: {prediction['prediction_score']:.1f}",
                    f"{'Partial' if should_partial_exit else 'Full'} Exit"
                ]
                
                # If doing partial exit, update the position's entry price and mark partial exit taken
                if should_partial_exit:
                    position.partial_exit_taken = True
                    # Update entry price to current price for remaining position
                    position.entry_price = current_price
                    
                await self._place_sell_order(symbol, quantity, decision_factors)
                
        except Exception as e:
            logging.error(f"Error managing position for {symbol}: {str(e)}")
            raise

    async def _analyze_entry(self, symbol: str, prediction: Dict[str, Any], current_price: float, signals: Dict[str, Any]):
        """Enhanced entry analysis using pre-calculated signals"""
        try:
            # Check if we can open new positions
            if not self._can_open_new_position():
                logging.info(f"Cannot open new position for {symbol} - position limit reached")
                return
            
            # Use the signals we already calculated
            if signals['buy_signals'] >= signals['required_signals'] and self._should_trade(symbol, 'BUY'):
                # Calculate position size using standardized method
                quantity = self._calculate_position_size(symbol, self.paper_trading)
                if quantity == 0:
                    logging.warning(f"Position size too small for {symbol}")
                    return
                
                # Double check price hasn't moved significantly
                new_price = float(self.client.get_product(f"{symbol}-USD").price)
                if abs(new_price - current_price) / current_price <= 0.01:  # 1% price movement tolerance
                    # Prepare detailed decision factors
                    decision_factors = [
                        f"Buy Strength: {signals['buy_strength']:.0f}%",
                        f"RSI: {prediction['rsi']:.2f}",
                        f"Trend: {prediction['trend']}",
                        f"Volume: {prediction['volume_ratio']:.1f}x average",
                        f"Prediction Score: {prediction['prediction_score']:.1f}"
                    ]
                    decision_factors.extend(prediction['bullish_signals'])
                    
                    await self._place_buy_order(symbol, quantity, decision_factors)
                else:
                    logging.info(f"Price moved too much for {symbol}, aborting buy")
                
        except Exception as e:
            logging.error(f"Error analyzing entry for {symbol}: {str(e)}")
            raise

    def _calculate_stop_loss(self, symbol: str, entry_price: float) -> float:
        """Calculate dynamic stop loss based on ATR and volatility"""
        try:
            atr = self._calculate_atr(symbol)
            volatility = self._calculate_volatility(symbol)
            
            # Base stop loss (ATR-based)
            atr_stop = entry_price - (atr * 2)
            
            # Percentage-based stop loss
            pct_stop = entry_price * (1 - self.stop_loss_percentage/100)
            
            # Use the more conservative stop loss
            stop_loss = max(atr_stop, pct_stop)
            
            # Adjust for volatility
            if volatility > 0.05:  # High volatility
                stop_loss = entry_price - (atr * 2.5)  # Wider stop for volatile markets
                
            return stop_loss
            
        except Exception as e:
            logging.error(f"Error calculating stop loss: {str(e)}")
            return entry_price * (1 - self.stop_loss_percentage/100)  # Fallback to simple percentage
    
    def _validate_position(self, position, current_price):
        """Validate position data before making trading decisions"""
        try:
            if position is None:
                return False
            if not hasattr(position, 'entry_price') or position.entry_price is None:
                return False
            if not hasattr(position, 'quantity') or position.quantity <= 0:
                return False
            if current_price <= 0:
                return False
            return True
        except Exception as e:
            logging.error(f"Error validating position: {str(e)}")
            return False
    
    async def _check_and_trade(self, symbol):
        """Check trading conditions using weighted signal system"""
        try:
            # Get comprehensive analysis
            prediction = self._analyze_price_prediction(symbol)
            current_price = prediction['current_price']
            rsi = prediction['rsi']
            volume_data = self.analyze_volume(symbol)
            ma_data = self.calculate_moving_averages(symbol)
            
            # Calculate entry/exit signals
            entry_signals = 0
            exit_signals = 0
            required_signals = 3
            
            # Entry signals
            if rsi <= self.rsi_oversold:
                entry_signals += 2
            elif rsi < 40:
                entry_signals += 1
                
            if prediction['prediction_score'] > 30:
                entry_signals += 1
            if volume_data['volume_ratio'] > 1.2:
                entry_signals += 1
            if ma_data['trend'] in ['Strong Uptrend', 'Weak Uptrend']:
                entry_signals += 1
                
            # Exit signals
            if rsi >= self.rsi_overbought:
                exit_signals += 2
            elif rsi > 60:
                exit_signals += 1
                
            if prediction['prediction_score'] < -30:
                exit_signals += 1
            if volume_data['volume_ratio'] > 1.5 and volume_data['price_change'] < 0:
                exit_signals += 1
            if ma_data['trend'] == 'Downtrend':
                exit_signals += 1
            
            # Check if we should trade
            if entry_signals >= required_signals and self._should_trade(symbol, 'BUY'):
                quantity = self._calculate_position_size(symbol, self.paper_trading) / current_price
                decision_factors = [
                    f"Entry Signals: {entry_signals}/{required_signals}",
                    f"RSI: {rsi:.2f}",
                    f"Trend: {ma_data['trend']}",
                    f"Volume: {volume_data['volume_ratio']:.1f}x average",
                    f"Prediction Score: {prediction['prediction_score']:.1f}"
                ]
                await self._place_buy_order(symbol, quantity, decision_factors)
                
            elif exit_signals >= required_signals and self._should_trade(symbol, 'SELL'):
                position = self.paper_positions.get(symbol) if self.paper_trading else self.positions.get(symbol)
                if position:
                    decision_factors = [
                        f"Exit Signals: {exit_signals}/{required_signals}",
                        f"RSI: {rsi:.2f}",
                        f"Trend: {ma_data['trend']}",
                        f"Volume: {volume_data['volume_ratio']:.1f}x average",
                        f"Prediction Score: {prediction['prediction_score']:.1f}"
                    ]
                    await self._place_sell_order(symbol, position.quantity, decision_factors)
        
        except Exception as e:
            logging.error(f"Error checking and trading {symbol}: {str(e)}")
            raise
    
    def calculate_rsi(self, symbol: str) -> float:
        try:
            end = datetime.now()
            start = end - timedelta(days=30)
            prices = self._get_historical_prices(symbol, start, end)
            
            logging.info(f"Calculating RSI for {symbol} with {len(prices)} data points")
            
            if len(prices) < self.rsi_period * 2:  # Need at least 2x RSI period for accuracy
                raise Exception(f"Not enough data points for accurate RSI calculation")
            
            # Calculate price changes
            delta = prices.diff()
            
            # Log some key statistics for verification
            logging.info(f"Price range: ${prices.min():.2f} - ${prices.max():.2f}")
            logging.info(f"Average daily change: ${delta.abs().mean():.2f}")
            
            # Separate gains and losses
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            
            # Calculate exponential moving averages
            avg_gain = gains.ewm(span=self.rsi_period, adjust=False).mean()
            avg_loss = losses.ewm(span=self.rsi_period, adjust=False).mean()
            
            # Calculate RS and RSI
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            # Get the latest RSI value
            current_rsi = float(rsi.iloc[-1])
            
            # Log intermediate values for debugging
            logging.info(f"Latest price: {prices.iloc[-1]:.2f}")
            logging.info(f"Latest gain EMA: {avg_gain.iloc[-1]:.4f}")
            logging.info(f"Latest loss EMA: {avg_loss.iloc[-1]:.4f}")
            logging.info(f"Calculated RSI: {current_rsi:.2f}")
            
            return current_rsi
            
        except Exception as e:
            logging.error(f"Error calculating RSI for {symbol}: {str(e)}")
            raise
    
    def _get_historical_prices(self, symbol: str, start: datetime, end: datetime) -> pd.Series:
        try:
            product_id = f"{symbol}-USD"
            
            # Get current time and calculate start time
            end_time = datetime.now()
            start_time = end_time - timedelta(days=90)  # Get 90 days for better trend analysis
            
            logging.info(f"Fetching candles for {symbol} from {start_time} to {end_time}")
            
            try:
                # Convert to Unix timestamps
                start_unix = int(start_time.timestamp())
                end_unix = int(end_time.timestamp())
                
                # Get daily candles
                response = self.client.get_candles(
                    product_id=product_id,
                    start=start_unix,
                    end=end_unix,
                    granularity="ONE_DAY"  # Daily candles
                )
                
                # Convert response to list and check if we have data
                candles = response.candles if hasattr(response, 'candles') else []
                if not candles:
                    raise Exception(f"No candle data received for {symbol}")
                
                # Convert candles to pandas Series
                prices = pd.Series(
                    [float(candle.close) for candle in reversed(candles)],
                    index=[datetime.fromtimestamp(float(candle.start)) for candle in reversed(candles)]
                )
                
                logging.info(f"Fetched {len(candles)} candles for {symbol}")
                return prices
                    
            except Exception as e:
                logging.error(f"Error fetching candle batch: {str(e)}")
                raise
            
        except Exception as e:
            logging.error(f"Error fetching historical prices for {symbol}: {str(e)}")
            raise
            
    async def _place_buy_order(self, symbol: str, quantity: float, decision_factors: List[str]):
        """Place a buy order with proper error handling and logging"""
        try:
            mode = "Paper" if self.paper_trading else "Real"
            current_price = float(self.client.get_product(f"{symbol}-USD").price)
            
            # Add at start of method
            if self.paper_trading:
                total_cost = quantity * current_price * (1 + 0.006)  # Include fees
                if total_cost > self.max_position_size:
                    logging.warning(f"Order exceeds maximum position size (${self.max_position_size})")
                    return False
                if total_cost > self.paper_balance:
                    logging.warning(f"Insufficient paper balance for {symbol} buy order")
                    return False
            
            # Calculate order details
            amount_usd = quantity * current_price
            
            if self.paper_trading:
                # Paper trading logic
                fees = amount_usd * 0.006  # Simulate 0.6% fees
                self.paper_total_fees += fees  # Add this line to track fees
                
                if amount_usd + fees > self.paper_balance:
                    logging.warning(f"Insufficient paper balance for {symbol} buy order")
                    return False
                    
                # Create paper position
                self.paper_positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=current_price,
                    entry_time=datetime.now(),
                    is_paper=True
                )
                
                # Update paper balance
                self.paper_balance -= (amount_usd + fees)
                
                # Record paper trade
                trade = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'action': 'BUY',
                    'price': current_price,
                    'quantity': quantity,
                    'amount_usd': amount_usd,
                    'fees': fees,
                    'balance_after': self.paper_balance,
                    'is_paper': True
                }
                self.paper_trade_history.append(trade)
                
            else:
                # Real trading logic
                try:
                    # Place market buy order
                    order = self.client.create_order(
                        product_id=f"{symbol}-USD",
                        side='BUY',
                        order_configuration={
                            'market_market_ioc': {
                                'base_size': str(quantity)  # Use base_size for exact quantity
                            }
                        }
                    )
                    
                    if not order:
                        raise Exception("Order creation failed")
                    
                    # Get fill details
                    filled_order = self.client.get_order(order.order_id)
                    
                    # Get actual execution details
                    actual_quantity = float(filled_order.filled_size)
                    actual_price = float(filled_order.filled_value) / actual_quantity
                    actual_amount = float(filled_order.filled_value)
                    actual_fees = float(filled_order.fee)
                    
                    # Create real position with actual fill data
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        quantity=actual_quantity,
                        entry_price=actual_price,
                        entry_time=datetime.now(),
                        is_paper=False
                    )
                    
                    # Record real trade with actual data
                    trade = {
                        'timestamp': datetime.now(),
                        'symbol': symbol,
                        'action': 'BUY',
                        'price': actual_price,
                        'quantity': actual_quantity,
                        'amount_usd': actual_amount,
                        'fees': actual_fees,
                        'order_id': order.order_id,
                        'is_paper': False
                    }
                    self.trade_history.append(trade)
                    
                    # Update local variables for notification
                    current_price = actual_price
                    quantity = actual_quantity
                    amount_usd = actual_amount
                    fees = actual_fees
                    
                except Exception as e:
                    logging.error(f"Error executing real trade: {str(e)}")
                    return False
            
            # Log and notify
            decision_text = "\n".join(decision_factors)
            message = (
                f"💰 {mode} BUY Order {'Simulated' if self.paper_trading else 'Executed'}\n"
                f"Symbol: {symbol}\n"
                f"Price: ${current_price:.2f}\n"
                f"Quantity: {quantity:.8f}\n"
                f"Total: ${amount_usd:.2f}\n"
                f"Fees: ${fees:.2f}\n\n"
                f"Decision Factors:\n{decision_text}"
            )
            
            logging.info(f"{mode} buy order placed for {symbol}")
            await self.send_notification(message)
            return True
            
        except Exception as e:
            error_msg = f"Error placing {mode} buy order for {symbol}: {str(e)}"
            logging.error(error_msg)
            await self.send_notification(f"❌ {error_msg}")
            return False

    async def _place_sell_order(self, symbol: str, quantity: float, decision_factors: List[str]):
        """Place a sell order with proper error handling and logging"""
        try:
            mode = "Paper" if self.paper_trading else "Real"
            current_price = float(self.client.get_product(f"{symbol}-USD").price)
            
            # Get position for profit calculation
            position = self.paper_positions.get(symbol) if self.paper_trading else self.positions.get(symbol)
            if not position:
                logging.warning(f"No {mode} position found for {symbol}, cannot sell")
                return False
            
            # Validate quantity
            if quantity > position.quantity:
                logging.warning(f"Trying to sell more than available: {quantity} > {position.quantity}")
                quantity = position.quantity
            
            if self.paper_trading:
                # Paper trading logic - calculate our own fees and profits
                total_value = quantity * current_price
                profit_info = position.calculate_profit(current_price)
                fees = profit_info['fees_paid']  # Use Position's fee calculation
                actual_value = total_value - fees
                
                # Update realized P/L and fees (only once!)
                self.paper_realized_pl += profit_info['profit_usd']
                self.paper_total_fees += fees
                
                # Update paper balance (only once!)
                self.paper_balance += actual_value
                
                # Record trade (only once!)
                trade_record = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'action': 'SELL',
                    'price': current_price,
                    'quantity': quantity,
                    'amount_usd': total_value,
                    'fees': fees,
                    'profit': profit_info['profit_usd'],
                    'profit_percentage': profit_info['profit_percentage'],
                    'is_paper': True
                }
                self.paper_trade_history.append(trade_record)
                
                # Update or remove position
                if quantity >= position.quantity:
                    del self.paper_positions[symbol]
                else:
                    position.quantity -= quantity
                    
            else:
                # Real trading - get actual order details from Coinbase
                try:
                    order = self.client.create_order(
                        product_id=f"{symbol}-USD",
                        side='SELL',
                        order_configuration={
                            'market_market_ioc': {
                                'base_size': str(quantity)  # Use base_size for exact quantity
                            }
                        }
                    )
                    
                    if not order:
                        raise Exception("Order creation failed")
                    
                    # Wait for order to be filled and get fill details
                    filled_order = self.client.get_order(order.order_id)
                    
                    # Calculate actual values from the fill
                    total_value = float(filled_order.filled_value)
                    fees = float(filled_order.fee)
                    actual_value = total_value - fees
                    
                    # Calculate profit using actual fill price
                    entry_value = position.quantity * position.entry_price
                    entry_fees = entry_value * 0.006  # Entry fees
                    
                    profit_info = {
                        'profit_usd': actual_value - entry_value - entry_fees - fees,
                        'profit_percentage': ((actual_value - entry_value - entry_fees - fees) / entry_value) * 100,
                        'fees_paid': entry_fees + fees
                    }
                    
                except Exception as e:
                    logging.error(f"Error executing real trade: {str(e)}")
                    return False
            
            # Update or remove position
            if quantity >= position.quantity:
                if self.paper_trading:
                    del self.paper_positions[symbol]
                else:
                    # Add to position history with actual trade data
                    self.position_history.append({
                        'symbol': symbol,
                        'entry_price': position.entry_price,
                        'exit_price': current_price,
                        'quantity': position.quantity,
                        'entry_time': position.entry_time,
                        'exit_time': datetime.now(),
                        'profit_usd': profit_info['profit_usd'],
                        'profit_percentage': profit_info['profit_percentage'],
                        'fees_paid': profit_info['fees_paid'],
                        'max_profit_percentage': position.highest_profit_percentage,
                        'max_drawdown': position.drawdown_percentage
                    })
                    del self.positions[symbol]
            else:
                position.quantity -= quantity
            
            # Record trade
            trade_record = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'action': 'SELL',
                'price': current_price,
                'quantity': quantity,
                'amount_usd': total_value,
                'fees': fees,
                'profit': profit_info['profit_usd'],
                'profit_percentage': profit_info['profit_percentage'],
                'is_paper': self.paper_trading
            }
            
            if self.paper_trading:
                self.paper_trade_history.append(trade_record)
                self.paper_balance += actual_value
            else:
                self.trade_history.append(trade_record)
            
            # Log and notify
            decision_text = "\n".join(decision_factors)
            message = (
                f"💰 {mode} SELL Order {'Simulated' if self.paper_trading else 'Executed'}\n"
                f"Symbol: {symbol}\n"
                f"Price: ${current_price:.2f}\n"
                f"Quantity: {quantity:.8f}\n"
                f"Total: ${total_value:.2f}\n"
                f"Fees: ${fees:.2f}\n"
                f"Profit: ${profit_info['profit_usd']:+.2f} ({profit_info['profit_percentage']:+.2f}%)\n\n"
                f"Decision Factors:\n{decision_text}"
            )
            
            logging.info(f"{mode} sell order placed for {symbol}")
            await self.send_notification(message)
            return True
            
        except Exception as e:
            error_msg = f"Error placing {mode} sell order for {symbol}: {str(e)}"
            logging.error(error_msg)
            await self.send_notification(f"❌ {error_msg}")
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
        
    def add_coin(self, symbol: str) -> bool:
        """Add a coin to watchlist"""
        try:
            # Validate the symbol
            self.client.get_product(f"{symbol}-USD")
            self.watched_coins.add(symbol)
            logging.info(f"Added {symbol} to watchlist")
            return True
        except Exception as e:
            logging.error(f"Failed to add {symbol}: {str(e)}")
            return False

    def remove_coin(self, symbol: str) -> bool:
        """Remove a coin from watchlist if not in any positions"""
        if symbol in self.positions or symbol in self.paper_positions:
            logging.warning(f"Cannot remove {symbol} - active position exists")
            return False
        
        if symbol in self.watched_coins:
            self.watched_coins.remove(symbol)
            logging.info(f"Removed {symbol} from watchlist")
            return True
        return False

    def _ensure_positions_watched(self):
        """Ensure all positions are in watchlist"""
        for symbol in self.positions.keys():
            self.watched_coins.add(symbol)
        for symbol in self.paper_positions.keys():
            self.watched_coins.add(symbol)

    def _check_balance(self, symbol: str, action: str) -> bool:
        """Check if we have sufficient balance for the trade"""
        try:
            accounts = self.client.get_accounts()
            
            if action == 'BUY':
                # Check USD balance against calculated position size
                usd_account = next((acc for acc in accounts.data if acc.currency == 'USD'), None)
                if not usd_account:
                    return False
                    
                # Calculate required amount using position sizing
                quantity = self._calculate_position_size(symbol, is_paper=False)
                current_price = float(self.client.get_product(f"{symbol}-USD").price)
                required_amount = quantity * current_price * 1.01  # Add 1% buffer for fees
                
                return float(usd_account.available_balance.value) >= required_amount
                
            else:  # SELL
                crypto_account = next((acc for acc in accounts.data if acc.currency == symbol), None)
                if not crypto_account:
                    return False
                return float(crypto_account.available_balance.value) > 0
                
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
            logging.info("Configuration loaded successfully")
        except FileNotFoundError:
            logging.info("No configuration file found, using defaults")
        except Exception as e:
            logging.error(f"Error loading configuration: {str(e)}")

    def test_api_connection(self):
        try:
            # Test authentication by getting BTC price
            btc_product = self.client.get_product('BTC-USD')
            price = float(btc_product.price)
            logging.info(f"Successfully fetched BTC price: ${price}")
            return price
            
        except Exception as e:
            logging.error(f"API test failed: {str(e)}")
            raise Exception(f"API test failed: {str(e)}")

    def get_account_balance(self) -> Dict[str, Union[Dict[str, float], float]]:
        try:
            # Get accounts from response
            accounts_response = self.client.get_accounts()
            balances = {}
            total_usd_value = 0.0

            # Debug logging
            logging.info("Raw accounts response received")
            
            # Access accounts through the accounts attribute
            if hasattr(accounts_response, 'accounts'):
                for account in accounts_response.accounts:
                    # Debug log each account
                    logging.info(f"Processing account: {account.__dict__}")
                    
                    # Check for balance in both available_balance and hold
                    balance_value = 0.0
                    
                    # Handle available balance
                    if hasattr(account, 'available_balance') and isinstance(account.available_balance, dict):
                        balance_value = float(account.available_balance.get('value', 0))
                    
                    # Add hold balance if it exists
                    if hasattr(account, 'hold') and isinstance(account.hold, dict):
                        balance_value += float(account.hold.get('value', 0))
                    
                    if balance_value > 0:
                        symbol = account.currency
                        balance = balance_value
                        
                        if symbol == 'USD':
                            usd_value = balance
                            logging.info(f"Found USD balance: ${usd_value}")
                        else:
                            try:
                                product = self.client.get_product(f"{symbol}-USD")
                                price = float(product.price)
                                usd_value = balance * price
                                logging.info(f"Calculated {symbol} value: ${usd_value}")
                            except Exception as e:
                                logging.warning(f"Could not get price for {symbol}: {str(e)}")
                                continue
                        
                        if usd_value > 0:
                            balances[symbol] = {
                                'balance': balance,
                                'usd_value': usd_value
                            }
                            total_usd_value += usd_value
                            logging.info(f"Added {symbol} balance: {balance} (${usd_value:.2f})")

            logging.info(f"Total portfolio value: ${total_usd_value:.2f}")
            return {
                'balances': balances,
                'total_usd_value': total_usd_value
            }
        except Exception as e:
            logging.error(f"Error getting account balance: {str(e)}")
            return {'balances': {}, 'total_usd_value': 0.0}

    def get_current_price(self, symbol: str) -> float:
        try:
            product_id = f"{symbol}-USD"
            product = self.client.get_product(product_id)
            price = float(product.price)
            logging.info(f"Current price for {symbol}: ${price}")
            return price
        except Exception as e:
            logging.error(f"Error getting price for {symbol}: {str(e)}")
            raise

    def analyze_volume(self, symbol: str) -> Dict[str, Any]:
        """Analyzes trading volume to confirm price trends"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=90)  # 90 days for good volume baseline
            
            # Get candles with volume data
            response = self.client.get_candles(
                product_id=f"{symbol}-USD",
                start=int(start_time.timestamp()),
                end=int(end_time.timestamp()),
                granularity="ONE_DAY"
            )
            
            candles = response.candles if hasattr(response, 'candles') else []
            if not candles:
                raise Exception("No candle data received")
            
            # Extract volume and price data
            volumes = [float(candle.volume) for candle in candles]
            prices = [float(candle.close) for candle in candles]
            
            # Calculate average volume
            avg_volume = sum(volumes) / len(volumes)
            current_volume = volumes[0]  # Most recent volume
            
            # Calculate price change
            price_change = ((prices[0] - prices[1]) / prices[1]) * 100
            
            # Determine if volume confirms trend
            volume_ratio = current_volume / avg_volume
            
            analysis = {
                'current_volume': current_volume,
                'average_volume': avg_volume,
                'volume_ratio': volume_ratio,
                'price_change': price_change,
                'trend_strength': 'strong' if volume_ratio > 1.5 else 'moderate' if volume_ratio > 1.0 else 'weak',
                'confirms_trend': volume_ratio > 1.0 and abs(price_change) > 1.0
            }
            
            logging.info(f"Volume analysis for {symbol}: {analysis}")
            return analysis
            
        except Exception as e:
            logging.error(f"Error analyzing volume for {symbol}: {str(e)}")
            raise

    def _calculate_position_size(self, symbol: str, is_paper: bool = False) -> float:
        """Calculate appropriate position size based on risk and portfolio value"""
        try:
            # Get available funds and portfolio value
            if is_paper:
                available_funds = self.paper_balance
                portfolio_value = self.get_paper_balance()['total_value']
            else:
                balance = self.get_account_balance()
                available_funds = float(balance['balances'].get('USD', {}).get('balance', 0))
                portfolio_value = balance['total_usd_value']
            
            # Risk per trade (2% of portfolio)
            risk_amount = portfolio_value * 0.02
            
            # Get current price and ATR
            current_price = float(self.client.get_product(f"{symbol}-USD").price)
            atr = self._calculate_atr(symbol)
            volatility = self._calculate_volatility(symbol)
            
            # Base position size on ATR for stop loss
            if atr > 0:
                # Use 2 ATR units for stop loss
                stop_distance = atr * 2
                max_quantity = risk_amount / stop_distance
            else:
                # Fallback to percentage-based
                stop_distance = current_price * (self.stop_loss_percentage / 100)
                max_quantity = risk_amount / stop_distance
            
            # Adjust for volatility
            if volatility > 0.05:  # High volatility
                max_quantity *= 0.5
            elif volatility > 0.03:  # Medium volatility
                max_quantity *= 0.75
            
            # Calculate position value
            position_value = max_quantity * current_price
            
            # Apply limits
            position_value = min(
                position_value,
                available_funds * 0.5,  # Max 50% of available funds
                self.max_position_size,  # Max position size limit
                portfolio_value * 0.1    # Max 10% of portfolio in single position
            )
            
            # Ensure minimum position size
            min_position = 10.0 if not is_paper else 1.0
            if position_value < min_position:
                return 0
            
            # Convert value to quantity
            quantity = position_value / current_price
            
            # Log position sizing details
            logging.info(
                f"Position size calculation for {symbol}:\n"
                f"Portfolio Value: ${portfolio_value:.2f}\n"
                f"Risk Amount: ${risk_amount:.2f}\n"
                f"ATR: {atr:.4f}\n"
                f"Volatility: {volatility:.4f}\n"
                f"Final Quantity: {quantity:.8f}\n"
                f"Position Value: ${position_value:.2f}"
            )
            
            return quantity
            
        except Exception as e:
            logging.error(f"Error calculating position size: {str(e)}")
            return 0

    def _should_trade(self, symbol: str, action: str) -> bool:
        """Enhanced trade validation"""
        try:
            # Add cooldown period for recently traded symbols
            if symbol in self.paper_trade_history[-10:]:  # Check last 10 trades
                last_trade = next(t for t in reversed(self.paper_trade_history) if t['symbol'] == symbol)
                time_since_trade = datetime.now() - last_trade['timestamp']
                if time_since_trade.total_seconds() < 3600:  # 1 hour cooldown
                    logging.info(f"Skipping {symbol} - cooldown period active")
                    return False

            # Add minimum expected profit check
            current_price = float(self.client.get_product(f"{symbol}-USD").price)
            fees = current_price * 0.012  # Round trip fees (1.2%)
            min_price_move = fees * 2  # Need at least 2x fees in price movement
            
            if action == 'BUY':
                # Don't rebuy a coin we recently sold at a loss
                recent_sells = [t for t in self.paper_trade_history[-20:] 
                              if t['symbol'] == symbol and t['action'] == 'SELL']
                if recent_sells and recent_sells[-1].get('profit', 0) < 0:
                    logging.info(f"Skipping {symbol} - recent loss")
                    return False

            return True
            
        except Exception as e:
            logging.error(f"Error in trade validation: {str(e)}")
            return False

    def _is_good_trading_hour(self) -> bool:
        """Check if current hour is good for trading"""
        hour = datetime.utcnow().hour
        return 13 <= hour <= 21  # 9 AM - 5 PM EST

    def _can_open_new_position(self) -> bool:
        """Check if we can open a new position based on portfolio management rules"""
        try:
            # Get current positions and portfolio value
            positions = self.paper_positions if self.paper_trading else self.positions
            
            # Maximum number of concurrent positions
            max_positions = 10
            current_positions = len(positions)
            
            if current_positions >= max_positions:
                logging.info(f"Maximum positions ({max_positions}) reached")
                return False
                
            # Calculate total exposure
            total_exposure = 0
            for symbol, pos in positions.items():
                try:
                    current_price = float(self.client.get_product(f"{symbol}-USD").price)
                    position_value = pos.quantity * current_price
                    total_exposure += position_value
                except Exception as e:
                    logging.error(f"Error calculating position value for {symbol}: {str(e)}")
                    continue
            
            # Get portfolio value
            if self.paper_trading:
                portfolio_value = self.get_paper_balance()['total_value']
            else:
                portfolio_value = self.get_account_balance()['total_usd_value']
            
            # Maximum total exposure (80% of portfolio)
            max_exposure = portfolio_value * 0.8
            
            # Dynamic position sizing - larger positions allowed when fewer positions
            # but still capped for safety
            max_position_pct = min(0.15, 0.8 / (current_positions + 1))
            self.max_position_size = portfolio_value * max_position_pct
            
            if total_exposure >= max_exposure:
                logging.info(f"Maximum exposure reached (${total_exposure:.2f} / ${max_exposure:.2f})")
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error checking position limits: {str(e)}")
            return False

    def _calculate_min_profit_target(self, entry_price: float) -> float:
        """Calculate minimum profit needed to cover fees"""
        total_fee_percentage = 0.012  # 0.6% entry + 0.6% exit
        min_profit = total_fee_percentage * 1.5  # 50% buffer over fees
        return min_profit

    def get_position_info(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get information about current positions and holdings"""
        try:
            positions_info = {}
            
            # Get account balances
            accounts_response = self.client.get_accounts()
            if hasattr(accounts_response, 'accounts'):
                for account in accounts_response.accounts:
                    # Skip USD and empty balances
                    if (account.currency == 'USD' or 
                        not hasattr(account, 'available_balance') or 
                        float(account.available_balance.get('value', 0)) <= 0):
                        continue
                    
                    symbol = account.currency
                    balance = float(account.available_balance.get('value', 0))
                    
                    try:
                        # Get current price
                        current_price = float(self.client.get_product(f"{symbol}-USD").price)
                        
                        # If we have a tracked position, use its data
                        if symbol in self.positions:
                            position = self.positions[symbol]
                            position.update_price(current_price)
                            profit_info = position.calculate_profit(current_price)
                            
                            positions_info[symbol] = {
                                'symbol': symbol,
                                'entry_price': position.entry_price,
                                'current_price': current_price,
                                'quantity': position.quantity,
                                'entry_time': position.entry_time,
                                'is_bot_position': True,
                                **profit_info
                            }
                        else:
                            # For holdings not tracked by bot, use current price as entry price
                            positions_info[symbol] = {
                                'symbol': symbol,
                                'entry_price': current_price,  # We don't know the actual entry price
                                'current_price': current_price,
                                'quantity': balance,
                                'entry_time': None,  # We don't know when it was acquired
                                'is_bot_position': False,
                                'profit_usd': 0,  # Can't calculate profit without entry price
                                'profit_percentage': 0,
                                'highest_profit_percentage': 0,
                                'drawdown_percentage': 0
                            }
                    
                    except Exception as e:
                        logging.warning(f"Could not process position for {symbol}: {str(e)}")
                        continue
                    
            return positions_info
                
        except Exception as e:
            logging.error(f"Error getting position info: {str(e)}")
            return {}

    def calculate_moving_averages(self, symbol: str) -> Dict[str, Any]:
        """Calculate various moving averages and their signals"""
        try:
            # Get 90 days of price data for reliable MA calculation
            end = datetime.now()
            start = end - timedelta(days=90)
            prices = self._get_historical_prices(symbol, start, end)
            
            # Calculate different MAs
            sma_20 = prices.rolling(window=20).mean()  # 20-day SMA
            sma_50 = prices.rolling(window=50).mean()  # 50-day SMA
            ema_12 = prices.ewm(span=12, adjust=False).mean()  # 12-day EMA
            ema_26 = prices.ewm(span=26, adjust=False).mean()  # 26-day EMA
            
            # Get latest values
            current_price = prices.iloc[-1]
            sma_20_current = sma_20.iloc[-1]
            sma_50_current = sma_50.iloc[-1]
            ema_12_current = ema_12.iloc[-1]
            ema_26_current = ema_26.iloc[-1]
            
            # Check for golden/death crosses (SMA)
            sma_cross_bullish = (sma_20.iloc[-2] <= sma_50.iloc[-2] and 
                               sma_20_current > sma_50_current)
            sma_cross_bearish = (sma_20.iloc[-2] >= sma_50.iloc[-2] and 
                               sma_20_current < sma_50_current)
            
            # Check for EMA crosses
            ema_cross_bullish = (ema_12.iloc[-2] <= ema_26.iloc[-2] and 
                               ema_12_current > ema_26_current)
            ema_cross_bearish = (ema_12.iloc[-2] >= ema_26.iloc[-2] and 
                               ema_12_current < ema_26_current)
            
            # Determine trend based on price position relative to MAs
            above_sma_20 = current_price > sma_20_current
            above_sma_50 = current_price > sma_50_current
            
            if above_sma_20 and above_sma_50:
                trend = "Strong Uptrend"
            elif above_sma_20:
                trend = "Weak Uptrend"
            elif above_sma_50:
                trend = "Mixed Trend"
            else:
                trend = "Downtrend"
                
            analysis = {
                'current_price': current_price,
                'sma_20': sma_20_current,
                'sma_50': sma_50_current,
                'ema_12': ema_12_current,
                'ema_26': ema_26_current,
                'sma_cross_bullish': sma_cross_bullish,
                'sma_cross_bearish': sma_cross_bearish,
                'ema_cross_bullish': ema_cross_bullish,
                'ema_cross_bearish': ema_cross_bearish,
                'trend': trend
            }
            
            logging.info(f"MA analysis for {symbol}: {analysis}")
            return analysis
            
        except Exception as e:
            logging.error(f"Error calculating MAs for {symbol}: {str(e)}")
            raise

    def get_performance_stats(self) -> Dict[str, Any]:
        """Calculate overall trading performance statistics"""
        try:
            stats = {
                'total_trades': len(self.trade_history),
                'active_positions': len(self.positions),
                'closed_positions': len(self.position_history),
                'total_profit_usd': 0.0,
                'win_rate': 0.0,
                'average_profit': 0.0,
                'best_trade': None,
                'worst_trade': None,
                'average_hold_time': timedelta(0)
            }
            
            if self.position_history:
                # Calculate profits
                profits = [pos['profit_usd'] for pos in self.position_history]
                winning_trades = len([p for p in profits if p > 0])
                
                # Calculate average hold time
                hold_times = [(pos['exit_time'] - pos['entry_time']) for pos in self.position_history]
                avg_hold_time = sum(hold_times, timedelta(0)) / len(hold_times)
                
                stats.update({
                    'total_profit_usd': sum(profits),
                    'win_rate': (winning_trades / len(profits)) * 100,
                    'average_profit': sum(profits) / len(profits),
                    'best_trade': max(self.position_history, key=lambda x: x['profit_percentage']),
                    'worst_trade': min(self.position_history, key=lambda x: x['profit_percentage']),
                    'average_hold_time': avg_hold_time
                })
                
            return stats
            
        except Exception as e:
            logging.error(f"Error calculating performance stats: {str(e)}")
            return {}

    def set_risk_parameters(self, stop_loss: float, take_profit: float, max_position: float) -> bool:
        """Set risk management parameters"""
        try:
            if 0 < stop_loss < take_profit and max_position > 0:
                self.stop_loss_percentage = stop_loss
                self.take_profit_percentage = take_profit
                self.max_position_size = max_position
                logging.info(f"Risk parameters updated: SL={stop_loss}%, TP={take_profit}%, Max=${max_position}")
                return True
            return False
        except ValueError:
            return False

    def _check_risk_management(self, symbol: str) -> None:
        """Enhanced risk management with trailing stops"""
        try:
            position = self.positions.get(symbol)
            if not position:
                return
            
            current_price = float(self.client.get_product(f"{symbol}-USD").price)
            profit_info = position.calculate_profit(current_price)
            
            # Update trailing stop as profit increases
            if profit_info['profit_percentage'] > 5:  # Above 5% profit
                # Move stop loss to break even
                adjusted_stop = max(
                    position.entry_price,  # Don't go below entry price
                    current_price * (1 - self.stop_loss_percentage/200)  # Tighter stop
                )
            else:
                adjusted_stop = position.entry_price * (1 - self.stop_loss_percentage/100)
            
            # Check stops
            if current_price <= adjusted_stop:
                logging.info(f"Stop loss triggered for {symbol} at {profit_info['profit_percentage']}%")
                self._place_sell_order(symbol)
            
            # Take partial profits at targets
            elif profit_info['profit_percentage'] >= self.take_profit_percentage:
                # Sell half the position
                original_quantity = position.quantity
                self.trade_amount = (original_quantity * current_price) / 2
                logging.info(f"Taking partial profits for {symbol} at {profit_info['profit_percentage']}%")
                self._place_sell_order(symbol)
            
        except Exception as e:
            logging.error(f"Error in risk management: {str(e)}")

    def analyze_market_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Analyze overall market sentiment using multiple indicators"""
        try:
            # Get data for different timeframes
            end = datetime.now()
            start_long = end - timedelta(days=90)   # 90 days for long-term
            start_medium = end - timedelta(days=30)  # 30 days for medium-term
            start_short = end - timedelta(days=7)    # 7 days for short-term
            
            # Get price data for different timeframes
            prices_long = self._get_historical_prices(symbol, start_long, end)
            prices_medium = prices_long[prices_long.index >= start_medium]
            prices_short = prices_long[prices_long.index >= start_short]
            
            # Calculate price changes
            price_change_long = ((prices_long.iloc[-1] - prices_long.iloc[0]) / prices_long.iloc[0]) * 100
            price_change_medium = ((prices_medium.iloc[-1] - prices_medium.iloc[0]) / prices_medium.iloc[0]) * 100
            price_change_short = ((prices_short.iloc[-1] - prices_short.iloc[0]) / prices_short.iloc[0]) * 100
            
            # Get technical indicators
            ma_data = self.calculate_moving_averages(symbol)
            volume_data = self.analyze_volume(symbol)
            rsi = self.calculate_rsi(symbol)
            
            # Determine momentum
            momentum_signals = {
                'short_term': 'bullish' if price_change_short > 0 else 'bearish',
                'medium_term': 'bullish' if price_change_medium > 0 else 'bearish',
                'long_term': 'bullish' if price_change_long > 0 else 'bearish'
            }
            
            # Count bullish signals
            bullish_signals = 0
            total_signals = 4  # Total possible signals
            
            # MA trend signal
            if ma_data['trend'] in ['Strong Uptrend', 'Weak Uptrend']:
                bullish_signals += 1
            elif ma_data['trend'] == 'Downtrend':
                bullish_signals -= 1
                
            # Volume trend signal
            if volume_data['volume_ratio'] > 1.0 and volume_data['price_change'] > 0:
                bullish_signals += 1
            elif volume_data['volume_ratio'] > 1.0 and volume_data['price_change'] < 0:
                bullish_signals -= 1
                
            # RSI signal
            if rsi < 30:  # Oversold
                bullish_signals += 1
            elif rsi > 70:  # Overbought
                bullish_signals -= 1
                
            # Momentum signal (weighted more heavily)
            momentum_score = sum(
                1 if momentum_signals[timeframe] == 'bullish' else -1
                for timeframe in ['short_term', 'medium_term', 'long_term']
            )
            
            # Calculate final sentiment score (-100 to 100)
            base_score = (bullish_signals / total_signals) * 100
            momentum_influence = (momentum_score / 3) * 50  # Momentum can shift score by up to ±50
            sentiment_score = base_score + momentum_influence
            
            # Determine sentiment category
            overall_sentiment = (
                'Strong Buy' if sentiment_score > 75 else
                'Buy' if sentiment_score > 25 else
                'Neutral (Bullish)' if sentiment_score > 0 else
                'Neutral (Bearish)' if sentiment_score > -25 else
                'Sell' if sentiment_score > -75 else
                'Strong Sell'
            )
            
            return {
                'sentiment_score': sentiment_score,
                'overall_sentiment': overall_sentiment,
                'momentum': momentum_signals,
                'price_changes': {
                    'short_term': price_change_short,
                    'medium_term': price_change_medium,
                    'long_term': price_change_long
                }
            }
            
        except Exception as e:
            logging.error(f"Error analyzing market sentiment for {symbol}: {str(e)}")
            raise

    async def _simulate_buy_order(self, symbol: str) -> None:
        """Simulate a buy order with paper trading"""
        try:
            product_id = f"{symbol}-USD"
            current_price = float(self.client.get_product(product_id).price)
            
            # Calculate position size
            trade_amount = self._calculate_position_size(symbol, is_paper=True)
            if trade_amount == 0:
                logging.warning(f"Position size too small for {symbol}")
                return
            
            # Calculate fees (0.6% Coinbase fee)
            fee = trade_amount * 0.006
            actual_trade_amount = trade_amount - fee
            
            # Check if we have enough paper balance
            if self.paper_balance < trade_amount:
                logging.warning(f"Insufficient paper balance for {symbol} buy")
                return
            
            # Calculate quantity after fees
            quantity = actual_trade_amount / current_price
            
            # Create new paper position
            self.paper_positions[symbol] = Position(
                symbol=symbol,
                entry_price=current_price,
                quantity=quantity,
                entry_time=datetime.now(),
                is_paper=True
            )
            
            # Update paper balance
            self.paper_balance -= trade_amount
            
            # Record trade
            trade_record = {
                'timestamp': datetime.now(),
                'action': 'BUY',
                'symbol': symbol,
                'amount_usd': trade_amount,
                'price': current_price,
                'quantity': quantity,
                'fees': fee,
                'is_paper': True,
                'balance_after': self.paper_balance
            }
            self.paper_trade_history.append(trade_record)
            
            # Send notification
            await self.send_trade_notification(
                action='BUY',
                symbol=symbol,
                price=current_price,
                quantity=quantity,
                is_paper=True
            )
            
            logging.info(f"Paper buy order placed for {symbol}: ${trade_amount:.2f} (Fee: ${fee:.2f})")
            
        except Exception as e:
            logging.error(f"Error simulating buy order for {symbol}: {str(e)}")
            raise

    def _simulate_sell_order(self, symbol: str, quantity: float) -> None:
        """Simulate a sell order with paper trading"""
        try:
            position = self.paper_positions.get(symbol)
            if not position:
                logging.warning(f"No paper position found for {symbol}, cannot sell")
                return
            
            # Validate quantity
            if quantity > position.quantity:
                logging.warning(f"Trying to sell more than available: {quantity} > {position.quantity}")
                quantity = position.quantity
            
            product_id = f"{symbol}-USD"
            current_price = float(self.client.get_product(product_id).price)
            
            # Calculate total value and fees
            total_value = quantity * current_price
            fee = total_value * 0.006
            actual_value = total_value - fee
            
            # Calculate profit/loss
            profit_info = position.calculate_profit(current_price)
            
            # Update paper balance
            self.paper_balance += actual_value
            
            # Record trade
            self.paper_trade_history.append({
                'timestamp': datetime.now(),
                'action': 'SELL',
                'symbol': symbol,
                'amount_usd': total_value,
                'price': current_price,
                'quantity': quantity,
                'fees': fee,
                'profit': profit_info['profit_usd'],
                'profit_percentage': profit_info['profit_percentage'],
                'is_paper': True,
                'balance_after': self.paper_balance
            })
            
            # Update or remove position based on quantity sold
            if quantity >= position.quantity:
                del self.paper_positions[symbol]
            else:
                position.quantity -= quantity
            
            logging.info(f"Paper sell order placed for {symbol}: ${total_value:.2f} (Profit: ${profit_info['profit_usd']:.2f})")
            
        except Exception as e:
            logging.error(f"Error simulating sell order for {symbol}: {str(e)}")
            raise

    def get_paper_balance(self) -> Dict[str, float]:
        """Get paper trading account balance with accurate P/L calculation"""
        try:
            # Get current cash
            cash_balance = self.paper_balance
            positions_value = 0.0
            unrealized_pl = 0.0
            
            # Calculate positions value and unrealized P/L
            for symbol, position in self.paper_positions.items():
                current_price = float(self.client.get_product(f"{symbol}-USD").price)
                
                # Calculate current position value
                pos_value = position.quantity * current_price
                positions_value += pos_value
                
                # Calculate unrealized P/L including fees
                profit_info = position.calculate_profit(current_price)
                unrealized_pl += profit_info['profit_usd']
            
            # Calculate total value (should equal initial balance + total P/L)
            total_value = cash_balance + positions_value
            total_pl = self.paper_realized_pl + unrealized_pl
            
            # Verify total value matches P/L calculation
            expected_total = self.paper_initial_balance + total_pl
            if abs(total_value - expected_total) > 0.01:  # Allow for small rounding differences
                logging.warning(f"Total value discrepancy detected: {total_value} vs {expected_total}")
                total_value = expected_total  # Use P/L-based calculation as it's more accurate
            
            # Calculate P/L percentage based on initial balance
            pl_percentage = (total_pl / self.paper_initial_balance) * 100 if self.paper_initial_balance > 0 else 0
            
            return {
                'cash_balance': round(cash_balance, 2),
                'positions_value': round(positions_value, 2),
                'total_value': round(total_value, 2),
                'realized_pl': round(self.paper_realized_pl, 2),
                'unrealized_pl': round(unrealized_pl, 2),
                'total_pl': round(total_pl, 2),
                'pl_percentage': round(pl_percentage, 2),
                'total_fees': round(self.paper_total_fees, 2),
                'initial_balance': self.paper_initial_balance
            }
            
        except Exception as e:
            logging.error(f"Error calculating paper balance: {str(e)}")
            return {
                'cash_balance': self.paper_balance,
                'positions_value': 0.0,
                'total_value': self.paper_balance,
                'realized_pl': 0.0,
                'unrealized_pl': 0.0,
                'total_pl': 0.0,
                'pl_percentage': 0.0,
                'total_fees': self.paper_total_fees,
                'initial_balance': self.paper_initial_balance
            }

    def reset_paper_trading(self, initial_balance: float = 1000.0) -> None:
        """Reset paper trading with new balance"""
        self.paper_initial_balance = initial_balance  # Store initial balance
        self.paper_balance = initial_balance
        self.paper_positions.clear()
        self.paper_trade_history.clear()
        self.paper_realized_pl = 0.0
        self.paper_total_fees = 0.0
        logging.info(f"Paper trading reset with ${initial_balance} balance")

    def set_discord_channel(self, channel):
        """Set the Discord channel for notifications"""
        self.discord_channel = channel
        logging.info(f"Discord notifications channel set")

    async def send_notification(self, message: str, is_update: bool = False, part: tuple = None):
        """Send notification to Discord channel with embeds"""
        try:
            if not self.discord_channel:
                return
                
            # Create embed
            embed = discord.Embed(
                description=message,
                color=discord.Color.blue() if is_update else discord.Color.green()
            )
            
            # Add header based on type
            if is_update:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                embed.title = f"Trading Update ({timestamp})"
                if part:
                    current, total = part
                    embed.title += f" - Part {current}/{total}"
            
            # Split message if too long (Discord embed limit is 4096 characters)
            if len(message) > 4000:
                chunks = [message[i:i+3900] for i in range(0, len(message), 3900)]
                for i, chunk in enumerate(chunks, 1):
                    embed = discord.Embed(
                        title=f"Trading Update - Part {i}/{len(chunks)}",
                        description=chunk,
                        color=discord.Color.blue() if is_update else discord.Color.green()
                    )
                    await self.discord_channel.send(embed=embed)
            else:
                await self.discord_channel.send(embed=embed)
                
        except Exception as e:
            logging.error(f"Error sending notification: {str(e)}")

    async def send_trade_notification(self, action: str, symbol: str, price: float, quantity: float, 
                                    is_paper: bool = False, profit_info: Dict[str, float] = None,
                                    decision_factors: List[str] = None):
        """Send a detailed trade notification"""
        try:
            trade_type = "Paper" if is_paper else "Real"
            
            # Use different emojis for buy/sell
            emoji = "🔔" if action == "BUY" else "💰"
            
            message = f"{emoji} {trade_type} Trade: {action} {symbol}\n"
            message += f"Price: ${price:,.2f}\n"
            message += f"Quantity: {quantity:.8f}\n"
            message += f"Total: ${(price * quantity):,.2f}"
            
            if profit_info and action == 'SELL':
                message += f"\nProfit: ${profit_info['profit_usd']:+,.2f} ({profit_info['profit_percentage']:+.2f}%)"
                message += f"\nFees Paid: ${profit_info['fees_paid']:.2f}"
            
            # Add decision factors if provided
            if decision_factors:
                message += "\n\nDecision Factors:"
                for factor in decision_factors:
                    message += f"\n• {factor}"
            
            await self.send_notification(message)
        except Exception as e:
            logging.error(f"Error sending trade notification: {str(e)}")

    async def send_interval_update(self, analysis_results: Dict[str, Dict[str, Any]]):
        """Send periodic update of analysis results"""
        try:
            if not self.watched_coins:
                return
            
            message = "Periodic Trading Update\n\n"
            
            for symbol, analysis in analysis_results.items():
                try:
                    signals = analysis['signals']
                    prediction = analysis['prediction']
                    current_price = analysis['current_price']
                    rsi = analysis['rsi']
                    volume_data = analysis['volume_data']
                    ma_data = analysis['ma_data']
                    
                    # Determine action based on signals
                    action = "HOLD"
                    if signals['buy_signals'] >= signals['required_signals']:
                        action = "BUY SIGNAL"
                    elif signals['sell_signals'] >= signals['required_signals']:
                        action = "SELL SIGNAL"
                    
                    # Format update message
                    entry = f"{symbol}: ${current_price:.2f}\n"
                    entry += f"Action: {action}\n"
                    entry += f"Buy Strength: {signals['buy_strength']:.0f}%\n"
                    entry += f"Sell Strength: {signals['sell_strength']:.0f}%\n"
                    entry += f"RSI: {rsi:.1f}\n"
                    entry += f"Trend: {ma_data['trend']}\n"
                    entry += f"Volume: {volume_data['volume_ratio']:.1f}x\n"
                    entry += f"Score: {prediction['prediction_score']:.1f}\n\n"
                    
                    message += entry
                    
                except Exception as e:
                    logging.error(f"Error formatting update for {symbol}: {str(e)}")
                    message += f"{symbol}: Error analyzing - {str(e)}\n\n"
                    continue
            
            # Send the formatted message
            await self.send_notification(message, is_update=True)
                
        except Exception as e:
            logging.error(f"Error in interval update: {str(e)}")
            await self.send_notification(f"❌ Error in interval update: {str(e)}")

    async def send_alert(self, symbol: str, alert_type: str, details: str):
        """Send an alert notification"""
        try:
            message = f"Alert for {symbol}\n"
            message += f"Type: {alert_type}\n"
            message += f"Details: {details}"
            await self.send_notification(message)
        except Exception as e:
            logging.error(f"Error sending alert: {str(e)}")

    def _analyze_price_prediction(self, symbol: str) -> Dict[str, Any]:
        """Make a unified price movement prediction"""
        try:
            # Get all indicators
            rsi = self.calculate_rsi(symbol)
            volume_data = self.analyze_volume(symbol)
            ma_data = self.calculate_moving_averages(symbol)
            sentiment = self.analyze_market_sentiment(symbol)
            
            # Determine overall market prediction
            bullish_signals = []
            bearish_signals = []
            
            # Calculate prediction score components
            technical_score = 0
            total_technicals = 3  # RSI, Volume, Trend
            
            # RSI component
            if rsi <= self.rsi_oversold:
                technical_score += 1
                bullish_signals.append(f"RSI oversold ({rsi:.2f})")
            elif rsi >= self.rsi_overbought:
                technical_score -= 1
                bearish_signals.append(f"RSI overbought ({rsi:.2f})")
                
            # Volume component
            if volume_data['volume_ratio'] > 1.5:
                if volume_data['price_change'] > 0:
                    technical_score += 1
                    bullish_signals.append(f"High volume upward movement ({volume_data['volume_ratio']:.1f}x)")
                else:
                    technical_score -= 1
                    bearish_signals.append(f"High volume downward movement ({volume_data['volume_ratio']:.1f}x)")
                    
            # Trend component
            if ma_data['trend'] in ['Strong Uptrend', 'Weak Uptrend']:
                technical_score += 1
                bullish_signals.append(f"Upward trend: {ma_data['trend']}")
            elif ma_data['trend'] == 'Downtrend':
                technical_score -= 1
                bearish_signals.append("Downward trend")
                
            # Calculate final prediction score (-100 to 100)
            technical_influence = (technical_score / total_technicals) * 100
            sentiment_influence = sentiment['sentiment_score']
            
            # Combine technical and sentiment scores (60/40 weight)
            prediction_score = (technical_influence * 0.6) + (sentiment_influence * 0.4)
            
            return {
                'prediction_score': prediction_score,
                'bullish_signals': bullish_signals,
                'bearish_signals': bearish_signals,
                'current_price': float(self.client.get_product(f"{symbol}-USD").price),
                'rsi': rsi,
                'volume_ratio': volume_data['volume_ratio'],
                'trend': ma_data['trend'],
                'sentiment': sentiment['overall_sentiment']
            }
            
        except Exception as e:
            logging.error(f"Error making price prediction for {symbol}: {str(e)}")
            raise

    def _get_recent_highs_lows(self, symbol: str, days: int = 30) -> Dict[str, List[float]]:
        """Get recent significant highs and lows"""
        try:
            end = datetime.now()
            start = end - timedelta(days=days)
            prices = self._get_historical_prices(symbol, start, end)
            
            # Calculate rolling max/min with a 5-day window
            highs = []
            lows = []
            window = 5
            
            for i in range(window, len(prices) - window):
                price_window = prices.iloc[i-window:i+window+1]
                current_price = prices.iloc[i]
                
                if current_price == price_window.max():
                    highs.append(float(current_price))
                if current_price == price_window.min():
                    lows.append(float(current_price))
            
            return {
                'highs': sorted(set(highs))[-5:],  # Last 5 unique highs
                'lows': sorted(set(lows))[:5]      # Last 5 unique lows
            }
        except Exception as e:
            logging.error(f"Error getting highs/lows for {symbol}: {str(e)}")
            return {'highs': [], 'lows': []}

    def _calculate_atr(self, symbol: str, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            end = datetime.now()
            start = end - timedelta(days=period * 2)  # Get extra data for calculation
            
            # Get OHLC data
            response = self.client.get_candles(
                product_id=f"{symbol}-USD",
                start=int(start.timestamp()),
                end=int(end.timestamp()),
                granularity="ONE_DAY"
            )
            
            if not response.candles:
                raise Exception("No candle data received")
            
            # Calculate True Range
            tr_values = []
            prev_close = None
            
            for candle in response.candles:
                high = float(candle.high)
                low = float(candle.low)
                close = float(candle.close)
                
                if prev_close is not None:
                    tr = max(
                        high - low,  # Current high - low
                        abs(high - prev_close),  # Current high - prev close
                        abs(low - prev_close)    # Current low - prev close
                    ) # stop forgetting to add this parenthesis!
                    tr_values.append(tr)
                    
                prev_close = close
            
            # Calculate ATR
            if tr_values:
                atr = sum(tr_values[-period:]) / min(len(tr_values), period)
                return atr
            
            return 0
            
        except Exception as e:
            logging.error(f"Error calculating ATR for {symbol}: {str(e)}")
            return 0

    def _calculate_volatility(self, symbol: str, days: int = 14) -> float:
        """Calculate price volatility"""
        try:
            end = datetime.now()
            start = end - timedelta(days=days)
            prices = self._get_historical_prices(symbol, start, end)
            
            # Calculate daily returns
            returns = prices.pct_change().dropna()
            
            # Calculate volatility (standard deviation of returns)
            volatility = float(returns.std())
            
            return volatility
            
        except Exception as e:
            logging.error(f"Error calculating volatility for {symbol}: {str(e)}")
            return 0

    def _validate_signals(self, buy_signals: int, sell_signals: int, required_signals: int) -> Dict[str, bool]:
        """Validate trading signals"""
        try:
            return {
                'can_buy': buy_signals >= required_signals,
                'can_sell': sell_signals >= required_signals,
                'signal_strength': max(buy_signals, sell_signals) / required_signals
            }
        except Exception as e:
            logging.error(f"Error validating signals: {str(e)}")
            return {'can_buy': False, 'can_sell': False, 'signal_strength': 0.0}

    def _calculate_signals(self, symbol: str, prediction: dict, rsi: float, volume_data: dict, ma_data: dict) -> dict:
        try:
            buy_signals = 0
            sell_signals = 0
            required_signals = 3
            
            # RSI signals - More balanced
            if rsi >= 80:  # Extremely overbought
                sell_signals += 2
                buy_signals = 0  # Still prevent buying at extreme levels
            elif rsi >= 70:  # Overbought
                sell_signals += 1
                buy_signals = max(0, buy_signals - 1)  # Reduce but don't eliminate buy signals
            elif rsi <= 30:  # Oversold
                buy_signals += 2
            elif rsi <= 40:  # Approaching oversold
                buy_signals += 1
            elif rsi >= 65:  # Approaching overbought
                sell_signals += 1
            
            # Volume signals
            if volume_data['volume_ratio'] > 2.0:
                if volume_data['price_change'] > 0:
                    buy_signals += 2
                else:
                    sell_signals += 2
            elif volume_data['volume_ratio'] > 1.2:
                if volume_data['price_change'] > 0:
                    buy_signals += 1
                else:
                    sell_signals += 1
                
            # Trend signals
            if ma_data['trend'] == 'Strong Uptrend':
                buy_signals += 2
            elif ma_data['trend'] == 'Weak Uptrend':
                buy_signals += 1
            elif ma_data['trend'] == 'Strong Downtrend':
                sell_signals += 2
            elif ma_data['trend'] == 'Weak Downtrend':
                sell_signals += 1
            
            # Prediction score signals
            if prediction['prediction_score'] > 50:
                buy_signals += 2
            elif prediction['prediction_score'] > 30:
                buy_signals += 1
            elif prediction['prediction_score'] < -50:
                sell_signals += 2
            elif prediction['prediction_score'] < -30:
                sell_signals += 1
            
            # Only add extra requirements in extreme conditions
            if rsi > 75 or volume_data['volume_ratio'] > 5:
                required_signals += 1
            
            return {
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'required_signals': required_signals,
                'buy_strength': min((buy_signals / required_signals) * 100, 100),
                'sell_strength': min((sell_signals / required_signals) * 100, 100)
            } # stop forgetting this curly brace!
            
        except Exception as e:
            logging.error(f"Error calculating signals for {symbol}: {str(e)}")
            return {
                'buy_signals': 0,
                'sell_signals': 0,
                'required_signals': 3,
                'buy_strength': 0,
                'sell_strength': 0
            }

    def _simulate_slippage(self, price: float, quantity: float, is_buy: bool) -> float:
        """Simulate realistic slippage based on order size"""
        try:
            order_size = price * quantity
            base_slippage = 0.001  # 0.1% base slippage
            
            # More slippage for larger orders
            if order_size > 1000:
                base_slippage *= (order_size / 1000) ** 0.5
            
            # Cap at 1%
            slippage = min(base_slippage, 0.01)
            
            # Buys get worse prices, sells get worse prices
            return price * (1 + slippage) if is_buy else price * (1 - slippage)
        except Exception as e:
            logging.error(f"Error simulating slippage: {str(e)}")
            return price