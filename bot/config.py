from typing import Dict, Any
from dataclasses import dataclass, field
import os
from decimal import Decimal

@dataclass
class TradingConfig:
    """Trading configuration with environment overrides"""
    
    # Trading Intervals
    TRADING_INTERVAL: int = int(os.getenv('TRADING_INTERVAL', '300'))
    
    # Position Management
    STOP_LOSS_PERCENTAGE: float = float(os.getenv('STOP_LOSS_PERCENTAGE', '5.0'))
    TAKE_PROFIT_PERCENTAGE: float = float(os.getenv('TAKE_PROFIT_PERCENTAGE', '10.0'))
    TRAILING_STOP_ENABLED: bool = bool(os.getenv('TRAILING_STOP_ENABLED', 'True'))
    TRAILING_STOP_PERCENTAGE: float = float(os.getenv('TRAILING_STOP_PERCENTAGE', '2.0'))
    
    # Risk Management
    RISK_MAX_DRAWDOWN: float = float(os.getenv('RISK_MAX_DRAWDOWN', '0.15'))
    RISK_DAILY_VAR: float = float(os.getenv('RISK_DAILY_VAR', '0.02'))
    RISK_MAX_POSITIONS: int = int(os.getenv('RISK_MAX_POSITIONS', '3'))
    RISK_PER_TRADE: float = float(os.getenv('RISK_PER_TRADE', '0.02'))
    MAX_PORTFOLIO_EXPOSURE: float = float(os.getenv('MAX_PORTFOLIO_EXPOSURE', '0.8'))
    
    # API Keys (from Replit Secrets)
    COINBASE_API_KEY: str = os.environ['COINBASE_API_KEY']
    COINBASE_API_SECRET: str = os.environ['COINBASE_API_SECRET']
    DISCORD_TOKEN: str = os.environ['DISCORD_TOKEN']
    
    # Technical Analysis
    RSI_PERIOD: int = int(os.getenv('RSI_PERIOD', '14'))
    RSI_OVERBOUGHT: float = float(os.getenv('RSI_OVERBOUGHT', '70.0'))
    RSI_OVERSOLD: float = float(os.getenv('RSI_OVERSOLD', '30.0'))
    
    # Timeframes Configuration
    TIMEFRAMES: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        '1h': {'weight': 0.4, 'periods': 24},  # Hourly for entry timing
        '1d': {'weight': 0.6, 'periods': 90}   # Daily for trend direction
    })
    
    # Paper Trading
    PAPER_BALANCE: float = float(os.getenv('PAPER_BALANCE', '10000.0'))
    
    # Exchange Settings
    EXCHANGE_FEE: float = float(os.getenv('EXCHANGE_FEE', '0.004'))
    
    # Cache Settings
    CACHE_TTL: Dict[str, int] = field(default_factory=lambda: {
        '1h': 1800,   # 30 minutes
        '1d': 21600   # 6 hours
    })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            key: getattr(self, key) 
            for key in self.__annotations__
        } 