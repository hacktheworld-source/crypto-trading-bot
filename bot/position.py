from datetime import datetime
from typing import Dict, Any

class Position:
    def __init__(self, trading_bot, symbol: str, entry_price: float, quantity: float, entry_time: datetime, is_paper: bool = False):
        self.trading_bot = trading_bot
        self.symbol = symbol
        self.entry_price = entry_price
        self.quantity = quantity
        self.entry_time = entry_time
        self.highest_price = entry_price
        self.lowest_price = entry_price
        self.is_paper = is_paper
        self.partial_exit_taken = False
        self.original_quantity = quantity
        
        # Initialize trailing stop based on bot settings
        self.trailing_stop_enabled = trading_bot.trailing_stop_enabled
        self.trailing_stop_percentage = trading_bot.trailing_stop_percentage
        self.trailing_stop_activation = trading_bot.trailing_stop_activation
        self.trailing_stop_price = entry_price * (1 - self.trailing_stop_percentage/100)
        
        # Add stop loss tracking
        self.stop_loss_price = entry_price * (1 - trading_bot.stop_loss_percentage/100)
        
    def update_price(self, current_price: float) -> None:
        self.highest_price = max(self.highest_price, current_price)
        self.lowest_price = min(self.lowest_price, current_price)
        
    def calculate_profit(self, current_price: float) -> Dict[str, float]:
        current_value = self.quantity * current_price
        initial_value = self.original_quantity * self.entry_price
        
        if initial_value <= 0:
            return {
                'profit_usd': 0,
                'profit_percentage': 0,
                'highest_profit_percentage': 0,
                'drawdown_percentage': 0,
                'fees_paid': 0,
                'partial_exit': self.partial_exit_taken
            }
        
        total_fee = (initial_value * 0.006) + (current_value * 0.006)
        profit = current_value - (self.quantity/self.original_quantity * initial_value) - total_fee
        profit_percentage = (profit / (self.quantity/self.original_quantity * initial_value)) * 100
        
        return {
            'profit_usd': profit,
            'profit_percentage': profit_percentage,
            'highest_profit_percentage': ((self.highest_price - self.entry_price) / self.entry_price) * 100,
            'drawdown_percentage': ((self.lowest_price - self.entry_price) / self.entry_price) * 100,
            'fees_paid': total_fee,
            'partial_exit': self.partial_exit_taken
        }

    def should_trigger_trailing_stop(self, current_price: float) -> bool:
        """Enhanced trailing stop with activation threshold"""
        if not self.trailing_stop_enabled:
            return False
        
        # Calculate current profit percentage
        profit_pct = ((current_price - self.entry_price) / self.entry_price) * 100
        
        # Only activate trailing stop after reaching activation threshold
        if profit_pct >= self.trailing_stop_activation:
            # Update trailing stop if we have a new high
            if current_price > self.highest_price:
                self.trailing_stop_price = max(
                    self.trailing_stop_price,
                    current_price * (1 - self.trailing_stop_percentage/100)
                )
                self.trading_bot.log(f"Updated trailing stop for {self.symbol} to ${self.trailing_stop_price:.2f}")
            
            # Check if price fell below trailing stop
            if current_price < self.trailing_stop_price:
                self.trading_bot.log(
                    f"Trailing stop triggered for {self.symbol}",
                    context={
                        'current_price': current_price,
                        'stop_price': self.trailing_stop_price,
                        'profit_pct': profit_pct
                    }
                )
                return True
                
        return False

    def __str__(self) -> str:
        return f"{self.symbol} {'(Paper)' if self.is_paper else ''} Position"