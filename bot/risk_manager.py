from typing import Dict, Any
from datetime import datetime
from bot.exceptions import TradingError, RiskError

class RiskManager:
    """Core risk management system"""
    
    def __init__(self, trading_bot):
        self.trading_bot = trading_bot
        self.config = trading_bot.config
        
        # Use new config structure
        self.max_drawdown = self.config.RISK_MAX_DRAWDOWN
        self.max_positions = self.config.RISK_MAX_POSITIONS
        self.daily_var = self.config.RISK_DAILY_VAR
        self.max_exposure = self.config.MAX_PORTFOLIO_EXPOSURE

    async def can_open_position(self, symbol: str) -> bool:
        """Check if new position meets risk criteria"""
        try:
            # Check position count
            if len(self.trading_bot.positions) >= self.max_positions:
                return False
            
            # Check daily drawdown
            if await self._check_daily_drawdown():
                return False
            
            # Check portfolio exposure
            current_exposure = await self.trading_bot.get_total_exposure()
            if current_exposure > 0.8:  # 80% max portfolio usage
                return False
            
            return True
            
        except Exception as e:
            await self.trading_bot.log(f"Risk check error: {str(e)}", level="error")
            return False

    async def calculate_position_size(self, symbol: str, confidence: float) -> float:
        """Enhanced position sizing with confidence adjustment"""
        try:
            account_balance = await self.trading_bot.get_account_balance()
            current_price = await self.trading_bot.data_manager.get_current_price(symbol)
            
            # Risk-adjusted position size
            risk_amount = account_balance * self.config.RISK_PER_TRADE
            adjusted_risk = risk_amount * confidence  # Scale by signal confidence
            
            # Calculate stop distance
            stop_distance = self.config.STOP_LOSS_PERCENTAGE / 100
            
            # Position size calculation
            position_size = adjusted_risk / (current_price * stop_distance)
            
            return position_size
            
        except Exception as e:
            await self.trading_bot.log(f"Position size calculation error: {str(e)}", level="error")
            raise RiskError("Failed to calculate position size", {"error": str(e)})

    async def _check_daily_drawdown(self) -> bool:
        """Check if daily drawdown limit exceeded"""
        try:
            daily_pnl = await self.trading_bot.get_daily_pnl()
            account_value = await self.trading_bot.get_account_balance()
            
            return (daily_pnl / account_value) < -self.daily_var
            
        except Exception as e:
            await self.trading_bot.log(f"Drawdown check error: {str(e)}", level="error")
            return True  # Fail safe
