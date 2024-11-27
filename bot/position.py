from datetime import datetime
from typing import Dict, Any
import logging

class Position:
    def __init__(self, symbol: str, entry_price: float, quantity: float, entry_time: datetime, is_paper: bool = False):
        self.symbol = symbol
        self.entry_price = entry_price
        self.quantity = quantity
        self.entry_time = entry_time
        self.highest_price = entry_price
        self.lowest_price = entry_price
        self.is_paper = is_paper
        self.partial_exit_taken = False
        
    def update_price(self, current_price: float) -> None:
        """Track highest and lowest prices"""
        self.highest_price = max(self.highest_price, current_price)
        self.lowest_price = min(self.lowest_price, current_price)
        
    def calculate_profit(self, current_price: float) -> Dict[str, float]:
        try:
            # Base calculations (same for both modes)
            current_value = self.quantity * current_price
            cost_basis = self.quantity * self.entry_price
            base_fee = cost_basis * 0.006  # 0.6% Coinbase fee
            
            if not self.is_paper:
                try:
                    # For real trades, get actual data from Coinbase
                    product_id = f"{self.symbol}-USD"
                    filled_orders = self.client.get_fills(product_id=product_id)
                    
                    # Calculate actual cost and fees from filled orders
                    actual_cost = sum(float(o.price) * float(o.size) for o in filled_orders if o.side == 'BUY')
                    actual_fees = sum(float(o.fee) for o in filled_orders)
                    
                    profit = current_value - actual_cost - actual_fees
                    profit_percentage = (profit / actual_cost) * 100 if actual_cost > 0 else 0
                    
                    return {
                        'profit_usd': profit,
                        'profit_percentage': profit_percentage,
                        'fees_paid': actual_fees
                    }
                except Exception as e:
                    logging.error(f"Error getting real trade data: {str(e)}")
                    # Fallback to tracked values if API fails
            
            # Paper trading or fallback calculation
            profit = current_value - cost_basis - base_fee
            profit_percentage = (profit / cost_basis) * 100 if cost_basis > 0 else 0
            
            return {
                'profit_usd': profit,
                'profit_percentage': profit_percentage,
                'fees_paid': base_fee
            }
            
        except Exception as e:
            logging.error(f"Error calculating profit: {str(e)}")
            return {
                'profit_usd': 0,
                'profit_percentage': 0,
                'fees_paid': 0
            }

    def __str__(self) -> str:
        return f"{self.symbol} {'(Paper)' if self.is_paper else ''} Position"