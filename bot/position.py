from datetime import datetime
from typing import Dict, Any

class Position:
    # Position states
    STATE_OPEN = 'OPEN'
    STATE_PARTIAL_EXIT = 'PARTIAL_EXIT'
    STATE_CLOSED = 'CLOSED'
    
    def __init__(self, trading_bot, symbol: str, entry_price: float, quantity: float, 
                 entry_time: datetime = None, is_paper: bool = False):
        self.trading_bot = trading_bot
        self.symbol = symbol
        self.entry_price = entry_price
        self.quantity = quantity
        self.entry_time = entry_time or datetime.now()
        self.highest_price = entry_price
        self.lowest_price = entry_price
        self.is_paper = is_paper
        self.partial_exit_taken = False
        self.original_quantity = quantity
        
        # New state tracking
        self.state = self.STATE_OPEN
        self.exit_history = []
        self.metrics = PositionMetrics(self)
        self.stops = PositionStops(self)

    def update_metrics(self, current_price: float) -> None:
        """Update position metrics"""
        try:
            # Update high/low prices
            self.highest_price = max(self.highest_price, current_price)
            self.lowest_price = min(self.lowest_price, current_price)
            
            # Update metrics
            self.metrics.update({
                'max_profit_pct': ((self.highest_price - self.entry_price) / self.entry_price) * 100,
                'max_drawdown_pct': ((self.lowest_price - self.entry_price) / self.entry_price) * 100,
                'days_held': (datetime.now() - self.entry_time).days,
                'current_profit_pct': ((current_price - self.entry_price) / self.entry_price) * 100
            })
            
        except Exception as e:
            self.trading_bot.log(f"Error updating position metrics: {str(e)}", level="error")

    def record_exit(self, exit_price: float, exit_quantity: float, reason: str) -> None:
        """Record a full or partial exit"""
        try:
            exit_data = {
                'timestamp': datetime.now(),
                'price': exit_price,
                'quantity': exit_quantity,
                'reason': reason,
                'profit_pct': ((exit_price - self.entry_price) / self.entry_price) * 100,
                'profit_usd': (exit_price - self.entry_price) * exit_quantity - (exit_price * exit_quantity * self.trading_bot.FEE_RATE)
            }
            
            self.exit_history.append(exit_data)
            self.metrics['realized_profit'] += exit_data['profit_usd']
            
            # Update state
            if exit_quantity == self.quantity:
                self.state = self.STATE_CLOSED
            elif not self.partial_exit_taken:
                self.state = self.STATE_PARTIAL_EXIT
                self.partial_exit_taken = True
                
        except Exception as e:
            self.trading_bot.log(f"Error recording exit: {str(e)}", level="error")

    def get_position_info(self) -> Dict[str, Any]:
        """Get comprehensive position information"""
        try:
            current_price = float(self.trading_bot.client.get_product(f"{symbol}-USD").price)
            profit_info = self.calculate_profit(current_price)
            
            return {
                'symbol': self.symbol,
                'state': self.state,
                'entry_price': self.entry_price,
                'current_price': current_price,
                'quantity': self.quantity,
                'original_quantity': self.original_quantity,
                'entry_time': self.entry_time,
                'days_held': self.metrics['days_held'],
                'profit_info': profit_info,
                'metrics': self.metrics,
                'exit_history': self.exit_history,
                'is_paper': self.is_paper
            }
            
        except Exception as e:
            self.trading_bot.log(f"Error getting position info: {str(e)}", level="error")
            return {}

    def update_price(self, current_price: float) -> None:
        """Update position price and metrics"""
        try:
            # Update price tracking
            self.highest_price = max(self.highest_price, current_price)
            self.lowest_price = min(self.lowest_price, current_price)
            
            # Update metrics
            self.metrics.update({
                'max_profit_pct': ((self.highest_price - self.entry_price) / self.entry_price) * 100,
                'max_drawdown_pct': ((self.lowest_price - self.entry_price) / self.entry_price) * 100,
                'days_held': (datetime.now() - self.entry_time).days,
                'current_profit_pct': ((current_price - self.entry_price) / self.entry_price) * 100,
                'total_fees': (self.original_quantity * self.entry_price * self.trading_bot.fee_rate) + 
                             (self.quantity * current_price * self.trading_bot.fee_rate)
            })
            
            # Update trailing stop if needed
            if self.stops.trailing_enabled and current_price > self.highest_price:
                profit_pct = ((current_price - self.entry_price) / self.entry_price) * 100
                if profit_pct >= self.stops.trailing_activation:
                    self.stops.trailing_price = max(
                        self.stops.trailing_price,
                        current_price * (1 - self.stops.trailing_pct/100)
                    )
                    
        except Exception as e:
            self.trading_bot.log(f"Error updating position for {self.symbol}: {str(e)}", level="error")

    def calculate_profit(self, current_price: float) -> Dict[str, float]:
        try:
            current_value = self.quantity * current_price
            initial_value = self.original_quantity * self.entry_price
            
            if initial_value <= 0:
                return self._get_zero_profit_info()
            
            total_fee = (initial_value * self.trading_bot.fee_rate) + (current_value * self.trading_bot.fee_rate)
            position_ratio = self.quantity / self.original_quantity
            profit = current_value - (position_ratio * initial_value) - total_fee
            profit_percentage = (profit / (position_ratio * initial_value)) * 100
            
            return {
                'profit_usd': profit,
                'profit_percentage': profit_percentage,
                'highest_profit_percentage': ((self.highest_price - self.entry_price) / self.entry_price) * 100,
                'drawdown_percentage': ((self.lowest_price - self.entry_price) / self.entry_price) * 100,
                'fees_paid': total_fee,
                'partial_exit': self.partial_exit_taken
            }
        except Exception as e:
            self.trading_bot.log(f"Error calculating profit for {self.symbol}: {str(e)}", level="error")
            return {
                'profit_usd': 0,
                'profit_percentage': 0,
                'highest_profit_percentage': 0,
                'drawdown_percentage': 0,
                'fees_paid': 0,
                'partial_exit': self.partial_exit_taken
            }

    def should_trigger_trailing_stop(self, current_price: float) -> bool:
        if not self.stops.trailing_enabled:
            return False
        
        profit_pct = ((current_price - self.entry_price) / self.entry_price) * 100
        
        # Enhanced trailing stop logic
        if profit_pct >= self.stops.trailing_activation:
            # Dynamic trailing stop based on volatility
            volatility_adjustment = 1.0
            if hasattr(self.trading_bot, '_calculate_bollinger_bands'):
                bb_data = self.trading_bot._calculate_bollinger_bands(self.symbol)
                if bb_data['bandwidth'] > 5.0:  # High volatility
                    volatility_adjustment = 1.2  # Wider trailing stop
            
            adjusted_stop_percentage = self.stops.trailing_pct * volatility_adjustment
            
            if current_price > self.highest_price:
                self.stops.trailing_price = max(
                    self.stops.trailing_price,
                    current_price * (1 - adjusted_stop_percentage/100)
                )
                self.trading_bot.log(f"Updated trailing stop for {self.symbol} to ${self.stops.trailing_price:.2f}")
            
            if current_price < self.stops.trailing_price:
                return True
        
        return False

    def __str__(self) -> str:
        return f"{self.symbol} {'(Paper)' if self.is_paper else ''} Position"

class PositionMetrics:
    """Separate class for position metrics calculations"""
    def __init__(self, position):
        self.position = position
        self.max_profit_pct = 0.0
        self.max_drawdown_pct = 0.0
        self.realized_profit = 0.0
        self.total_fees = 0.0

class PositionStops:
    """Separate class for stop management"""
    def __init__(self, position):
        self.position = position
        self.trailing_enabled = position.trading_bot.config.trailing_stop['enabled']
        self.trailing_pct = position.trading_bot.config.trailing_stop['percentage']
        # ... rest of stops logic ...