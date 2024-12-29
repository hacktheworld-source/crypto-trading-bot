from typing import Dict, Any
from datetime import datetime
from bot.exceptions import TradingError, RiskError
from bot.config import TradingConfig
from bot.constants import TradingConstants

class RiskManager:
    """Core risk management system"""
    
    def __init__(self, trading_bot):
        self.trading_bot = trading_bot
        self.config = trading_bot.config
        
        # Use new config structure with position limits
        self.max_positions = self.config.RISK_MAX_POSITIONS
        self.target_positions = TradingConstants.TARGET_POSITIONS
        self.min_positions = TradingConstants.MIN_POSITIONS
        
        # Other risk parameters
        self.max_drawdown = self.config.RISK_MAX_DRAWDOWN
        self.daily_var = self.config.RISK_DAILY_VAR
        self.max_exposure = self.config.MAX_PORTFOLIO_EXPOSURE

    async def can_open_position(self, symbol: str) -> bool:
        """Check if new position meets risk criteria"""
        try:
            current_positions = len(self.trading_bot.positions)
            
            # Check position count against absolute maximum
            if current_positions >= TradingConstants.MAX_POSITIONS:
                await self.trading_bot.log(
                    f"Maximum position limit ({TradingConstants.MAX_POSITIONS}) reached", 
                    level="warning"
                )
                return False
            
            # If above target, apply stricter criteria
            if current_positions >= self.target_positions:
                # Require stronger signals for additional positions
                signal_strength = await self.trading_bot.technical_analyzer.get_signal_strength(symbol)
                if signal_strength < 0.8:  # Require 80% confidence above target position count
                    return False
            
            # Check other risk criteria
            if await self._check_daily_drawdown():
                return False
            
            if await self.trading_bot.get_total_exposure() > self.max_exposure:
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
            daily_pnl = float(await self.trading_bot.get_daily_pnl())
            account_value = float(await self.trading_bot.get_account_balance())
            
            if account_value == 0:
                return True  # Fail safe if account value is zero
                
            drawdown = daily_pnl / account_value
            return drawdown < -self.daily_var
            
        except Exception as e:
            await self.trading_bot.log(f"Drawdown check error: {str(e)}", level="error")
            return True  # Fail safe

    async def validate_position_limits(self) -> bool:
        """Validate current position count against limits"""
        try:
            current_positions = len(self.trading_bot.positions)
            if current_positions > TradingConstants.MAX_POSITIONS:
                raise RiskError(
                    f"Position limit exceeded: {current_positions}/{TradingConstants.MAX_POSITIONS}",
                    {"current": current_positions, "limit": TradingConstants.MAX_POSITIONS}
                )
            return True
        except Exception as e:
            await self.trading_bot.log(f"Position validation error: {str(e)}", level="error")
            return False
