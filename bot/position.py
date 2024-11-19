from datetime import datetime
from typing import Dict, Any

class Position:
    def __init__(self, symbol: str, entry_price: float, quantity: float, entry_time: datetime):
        self.symbol = symbol
        self.entry_price = entry_price
        self.quantity = quantity
        self.entry_time = entry_time
        self.highest_price = entry_price
        self.lowest_price = entry_price
        
    def update_price(self, current_price: float) -> None:
        self.highest_price = max(self.highest_price, current_price)
        self.lowest_price = min(self.lowest_price, current_price)
        
    def calculate_profit(self, current_price: float) -> Dict[str, float]:
        current_value = self.quantity * current_price
        initial_value = self.quantity * self.entry_price
        profit = current_value - initial_value
        profit_percentage = (profit / initial_value) * 100 if initial_value > 0 else 0
        
        return {
            'profit_usd': profit,
            'profit_percentage': profit_percentage,
            'highest_profit_percentage': ((self.highest_price - self.entry_price) / self.entry_price) * 100,
            'drawdown_percentage': ((self.lowest_price - self.entry_price) / self.entry_price) * 100
        } 