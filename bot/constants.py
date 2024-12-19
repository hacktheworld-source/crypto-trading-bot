from typing import Dict, Any
from enum import Enum

class TimeFrame(Enum):
    """Trading timeframes"""
    MINUTE_1 = "ONE_MINUTE"
    MINUTE_5 = "FIVE_MINUTE"
    MINUTE_15 = "FIFTEEN_MINUTE"
    HOUR_1 = "ONE_HOUR"
    HOUR_6 = "SIX_HOUR"
    DAY_1 = "ONE_DAY"

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
        TimeFrame.MINUTE_1: 300,    # 5 minutes
        TimeFrame.MINUTE_5: 900,    # 15 minutes
        TimeFrame.MINUTE_15: 1800,  # 30 minutes
        TimeFrame.HOUR_1: 3600,     # 1 hour
        TimeFrame.HOUR_6: 21600,    # 6 hours
        TimeFrame.DAY_1: 86400      # 24 hours
    }
    
    # Technical Analysis
    RSI_PERIOD = 14
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    MA_FAST = 20
    MA_SLOW = 50
    VOLUME_MA = 20
    
    # Risk Management
    MAX_POSITION_SIZE = 0.1  # 10% of portfolio
    MAX_LEVERAGE = 3.0
    STOP_LOSS_DEFAULT = 0.05  # 5%
    TAKE_PROFIT_DEFAULT = 0.1  # 10%
    
    # Portfolio Management
    MAX_POSITIONS = 10
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