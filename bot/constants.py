from typing import Dict, Any
from enum import Enum

class TimeFrame(str, Enum):
    MINUTE_1 = "ONE_MINUTE"
    MINUTE_5 = "FIVE_MINUTE"
    MINUTE_15 = "FIFTEEN_MINUTE"
    HOUR_1 = "ONE_HOUR"
    HOUR_4 = "FOUR_HOURS"
    DAY_1 = "ONE_DAY"

class TradingConfig:
    """Trading configuration constants"""
    
    # Risk Management
    MAX_PORTFOLIO_RISK = 0.02  # 2% max risk per trade
    MAX_DRAWDOWN = 0.15  # 15% maximum drawdown
    MIN_POSITIONS = 1
    MAX_POSITIONS = 10  # Upper limit of 10 positions
    TARGET_POSITIONS = 7  # Preferred number of positions
    POSITION_SIZE_LIMIT = 0.20  # 20% max position size
    
    # Technical Analysis
    TIMEFRAMES = {
        "1h": {"weight": 0.4, "periods": 24},
        "4h": {"weight": 0.3, "periods": 30},
        "1d": {"weight": 0.3, "periods": 90}
    }
    
    # Indicators
    INDICATORS = {
        "ma_fast": 20,
        "ma_slow": 50,
        "rsi_period": 14,
        "volume_ma": 20,
        "bb_period": 20,
        "bb_std": 2
    }
    
    # API Settings
    RATE_LIMIT = 0.1  # seconds between requests
    CACHE_TTL = {
        "1h": 1800,   # 30 minutes
        "4h": 7200,   # 2 hours
        "1d": 21600   # 6 hours
    }

class ErrorMessages:
    """Centralized error messages"""
    
    API_ERROR = "API request failed: {}"
    DATA_ERROR = "Data retrieval failed: {}"
    VALIDATION_ERROR = "Validation failed: {}"
    RISK_ERROR = "Risk check failed: {}" 