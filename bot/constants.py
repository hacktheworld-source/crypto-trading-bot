from typing import Dict, Any
from enum import Enum

class TimeFrame(Enum):
    """Trading timeframes supported by Coinbase"""
    HOUR_1 = "ONE_HOUR"    # Short-term analysis
    DAY_1 = "ONE_DAY"      # Long-term trend

class TradingConstants:
    """Trading-related constants"""
    
    # Minimum amounts
    MIN_TRADE_USD = 10.0
    MIN_POSITION_SIZE = 0.001
    
    # Default timeouts
    API_TIMEOUT = 30  # seconds
    WEBSOCKET_TIMEOUT = 60  # seconds
    
    # Rate limiting
    MAX_REQUESTS_PER_SECOND = 5
    RATE_LIMIT_SLEEP = 0.2  # seconds
    
    # Cache settings
    CACHE_SIZE = 1000
    CACHE_TTL = {
        TimeFrame.HOUR_1: 3600,     # 1 hour
        TimeFrame.DAY_1: 86400      # 24 hours
    }
    
    # Technical Analysis
    RSI_PERIOD = 14
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    MA_FAST = 20
    MA_SLOW = 50
    VOLUME_MA = 20
    
    # Timeframe weights for analysis
    TIMEFRAMES = {
        TimeFrame.DAY_1: {'weight': 0.6, 'periods': 90},  # Daily for trend direction
        TimeFrame.HOUR_1: {'weight': 0.4, 'periods': 24}  # Hourly for entry timing
    }
    
    # Risk Management
    MAX_POSITION_SIZE = 0.1  # 10% of portfolio
    MAX_LEVERAGE = 3.0
    STOP_LOSS_DEFAULT = 0.05  # 5%
    TAKE_PROFIT_DEFAULT = 0.1  # 10%
    MAX_PORTFOLIO_RISK = 0.02  # 2% max risk per trade
    MAX_DRAWDOWN = 0.15  # 15% maximum drawdown
    MIN_POSITIONS = 1
    MAX_POSITIONS = 10  # Upper limit of 10 positions
    TARGET_POSITIONS = 7  # Preferred number of positions
    POSITION_SIZE_LIMIT = 0.20  # 20% max position size
    
    # Portfolio Management
    MAX_PORTFOLIO_USAGE = 0.8  # 80% max usage
    POSITION_SIZE_INCREMENT = 0.25  # 25% position size steps
    
    # Fees
    MAKER_FEE = 0.004  # 0.4%
    TAKER_FEE = 0.006  # 0.6%
    
    # Logging
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    LOG_LEVEL = "INFO"
    LOG_FILE = "trading_bot.log"

class ErrorMessages:
    """Centralized error messages"""
    
    API_ERROR = "API request failed: {}"
    DATA_ERROR = "Data retrieval failed: {}"
    VALIDATION_ERROR = "Validation failed: {}"
    RISK_ERROR = "Risk check failed: {}" 