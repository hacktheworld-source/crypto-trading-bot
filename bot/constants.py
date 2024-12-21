from typing import Dict, Any
from enum import Enum

class TimeFrame(Enum):
    """Trading timeframes supported by Coinbase"""
    HOUR_1 = "ONE_HOUR"    # Short-term momentum
    HOUR_4 = "FOUR_HOURS"  # Medium-term trend
    DAY_1 = "ONE_DAY"      # Long-term structure

class TradingConstants:
    """Trading-related constants"""
    
    # Data Requirements
    MIN_DATA_POINTS = 30  # Minimum data points needed for analysis
    MAX_CANDLES = 300    # Maximum candles to request (Coinbase limit is 350)
    
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
        TimeFrame.HOUR_1: 300,      # 5 minutes
        TimeFrame.HOUR_4: 900,      # 15 minutes
        TimeFrame.DAY_1: 3600       # 1 hour
    }
    
    # Technical Analysis
    RSI_PERIOD = 14
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    EMA_SHORT = 9
    EMA_LONG = 21
    VOLUME_MA = 20
    
    # Timeframe weights for analysis
    TIMEFRAMES = {
        TimeFrame.DAY_1: {'weight': 0.3, 'periods': 90},   # Market structure
        TimeFrame.HOUR_4: {'weight': 0.3, 'periods': 60},  # Trend confirmation
        TimeFrame.HOUR_1: {'weight': 0.4, 'periods': 48}   # Entry timing
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
    
    # Fees (Coinbase Advanced Trade)
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