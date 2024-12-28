from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass
from decimal import Decimal

@dataclass
class ScaleLevel:
    """Represents a position scaling level"""
    price: float
    quantity: float
    triggered: bool = False
    timestamp: Optional[datetime] = None

@dataclass
class PartialExit:
    """Represents a partial position exit"""
    price: float
    quantity: float
    profit: float
    timestamp: datetime

class Position:
    """Trading position with advanced management features"""
    
    def __init__(
        self,
        trading_bot,
        symbol: str,
        entry_price: float,
        initial_quantity: float,
        remaining_quantity: float,
        side: str,
        entry_time: datetime,
        fees_paid: float = 0.0  # Add fees tracking
    ):
        self.trading_bot = trading_bot
        self.symbol = symbol
        self.entry_price = entry_price
        self.initial_quantity = initial_quantity
        self.remaining_quantity = remaining_quantity
        self.side = side
        self.entry_time = entry_time
        self.fees_paid = fees_paid  # Track total fees paid
        
        # Initialize metrics
        self.current_price = entry_price
        self.unrealized_pnl = 0.0
        self.unrealized_pnl_pct = 0.0
        self.unrealized_pnl_change = 0.0
        
        # Risk management
        self.stop_loss = entry_price * (1 - self.trading_bot.config.STOP_LOSS_PERCENTAGE/100)
        self.take_profit = entry_price * (1 + self.trading_bot.config.TAKE_PROFIT_PERCENTAGE/100)
        self.trailing_stop = None
        
        # Position management
        self.partial_exits = []
        self.scale_levels = []
        
        # Trailing stop configuration
        self.trailing_stop_enabled = self.trading_bot.config.TRAILING_STOP_ENABLED
        self.trailing_stop_distance = self.trading_bot.config.TRAILING_STOP_PERCENTAGE / 100
        self.trailing_stop_activation = self.trading_bot.config.TAKE_PROFIT_PERCENTAGE / 100 * 0.5  # Activate at 50% of take profit
        self.highest_price = entry_price
        self.trailing_stop_price = None

    async def update_metrics(self, current_price: float) -> None:
        """Update position metrics with current price."""
        try:
            prev_pnl = self.unrealized_pnl
            self.current_price = current_price
            
            # Calculate unrealized P/L (excluding fees until position is closed)
            self.unrealized_pnl = (current_price - self.entry_price) * self.remaining_quantity
            self.unrealized_pnl_pct = ((current_price - self.entry_price) / self.entry_price) * 100
            self.unrealized_pnl_change = self.unrealized_pnl - prev_pnl
            
            # Update trailing stop if needed
            await self._update_trailing_stop(current_price)
            
            # Check and execute scale levels
            await self._check_scale_levels(current_price)
            
            # Check for partial profit taking
            if await self._should_take_partial_profit(current_price):
                await self.execute_partial_exit(current_price)
                
        except Exception as e:
            await self.trading_bot.log(f"Position metrics update error: {str(e)}", level="error")
            
    async def _update_trailing_stop(self, current_price: float) -> None:
        """Update trailing stop based on profit levels"""
        try:
            if not self.trailing_stop_enabled:
                return
                
            profit_percentage = (current_price - self.entry_price) / self.entry_price
            
            # Update trailing distances based on profit level
            if profit_percentage >= 0.07:  # 7%+ profit
                self.trailing_stop_distance = 0.04  # 4% trail
            elif profit_percentage >= 0.05:  # 5%+ profit
                self.trailing_stop_distance = 0.03  # 3% trail
            elif profit_percentage >= 0.03:  # 3%+ profit
                self.trailing_stop_distance = 0.02  # 2% trail
                
            # Update highest price and stop level
            if current_price > self.highest_price:
                self.highest_price = current_price
                self.trailing_stop_price = current_price * (1 - self.trailing_stop_distance)
                
        except Exception as e:
            await self.trading_bot.log(f"Trailing stop update error: {str(e)}", level="error")

    async def should_exit(self, current_price: float) -> bool:
        """Enhanced exit condition checking"""
        try:
            # Regular stop loss check
            if current_price <= self.stop_loss:
                await self.trading_bot.log(f"Stop loss triggered for {self.symbol}", level="warning")
                return True
            
            # Trailing stop check
            if self.trailing_stop_price and current_price <= self.trailing_stop_price:
                await self.trading_bot.log(f"Trailing stop triggered for {self.symbol}", level="info")
                return True
            
            # Take profit check
            if current_price >= self.take_profit:
                await self.trading_bot.log(f"Take profit triggered for {self.symbol}")
                return True
            
            return False
            
        except Exception as e:
            await self.trading_bot.log(f"Exit check error: {str(e)}", level="error")
            return False

    async def add_scale_level(self, price: float, quantity: float) -> bool:
        """Add a new scale level"""
        try:
            scale_level = ScaleLevel(price=price, quantity=quantity)
            self.scale_levels.append(scale_level)
            await self.trading_bot.log(f"Added scale level for {self.symbol} at {price}")
            return True
        except Exception as e:
            await self.trading_bot.log(f"Error adding scale level: {str(e)}", level="error")
            return False

    async def _check_scale_levels(self, current_price: float) -> None:
        """Check and execute scale levels"""
        for level in self.scale_levels:
            if not level.triggered and current_price <= level.price:
                try:
                    # Execute scale in
                    success = await self.trading_bot.execute_scale_in(
                        self.symbol, 
                        level.quantity, 
                        current_price
                    )
                    if success:
                        level.triggered = True
                        level.timestamp = datetime.now()
                        self.remaining_quantity += level.quantity
                except Exception as e:
                    await self.trading_bot.log(f"Scale in execution error: {str(e)}", level="error")

    async def _should_take_partial_profit(self, current_price: float) -> bool:
        """Check if position should take partial profit"""
        try:
            # Calculate profit percentage
            profit_pct = (current_price - self.entry_price) / self.entry_price
            
            # Check if we haven't taken partial profit yet
            no_partial_exits = len(self.partial_exits) == 0
            
            # Take 50% profit at 5% gain
            if profit_pct >= 0.05 and no_partial_exits:
                return True
                
            return False
            
        except Exception as e:
            await self.trading_bot.log(f"Partial profit check error: {str(e)}", level="error")
            return False

    async def execute_partial_exit(self, current_price: float) -> bool:
        """Execute partial position exit"""
        try:
            # Calculate exit quantity (50% of remaining)
            exit_quantity = self.remaining_quantity * 0.5
            
            # Execute the exit
            success = await self.trading_bot.execute_partial_exit(
                self.symbol,
                exit_quantity,
                current_price
            )
            
            if success:
                # Move stop loss to break-even
                self.stop_loss = self.entry_price
                
                await self.trading_bot.log(
                    f"Partial exit executed for {self.symbol} - "
                    f"Quantity: {exit_quantity:.8f}, "
                    f"Price: ${current_price:.2f}",
                    level="info"
                )
                
            return success
            
        except Exception as e:
            await self.trading_bot.log(f"Partial exit execution error: {str(e)}", level="error")
            return False

    def get_info(self) -> Dict[str, Any]:
        """Enhanced position information"""
        return {
            'symbol': self.symbol,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'initial_quantity': self.initial_quantity,
            'remaining_quantity': self.remaining_quantity,
            'unrealized_pnl': self.unrealized_pnl,
            'pnl_pct': (self.current_price - self.entry_price) / self.entry_price * 100,
            'risk_levels': {
                'stop_loss': self.stop_loss,
                'take_profit': self.take_profit,
                'trailing_stop': self.trailing_stop_price
            },
            'scale_levels': [
                {
                    'price': level.price,
                    'quantity': level.quantity,
                    'triggered': level.triggered,
                    'timestamp': level.timestamp
                }
                for level in self.scale_levels
            ],
            'partial_exits': [
                {
                    'price': exit.price,
                    'quantity': exit.quantity,
                    'profit': exit.profit,
                    'timestamp': exit.timestamp
                }
                for exit in self.partial_exits
            ],
            'is_paper': self.is_paper
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get current position metrics."""
        return {
            'symbol': self.symbol,
            'side': self.side,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'initial_quantity': self.initial_quantity,
            'remaining_quantity': self.remaining_quantity,
            'unrealized_pnl': self.unrealized_pnl,
            'unrealized_pnl_pct': self.unrealized_pnl_pct,
            'fees_paid': self.fees_paid,  # Include fees in metrics
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'trailing_stop': self.trailing_stop,
            'entry_time': self.entry_time
        }