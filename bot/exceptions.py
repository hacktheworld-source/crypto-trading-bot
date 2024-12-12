from typing import Optional

class TradingError(Exception):
    """Base exception class for trading bot errors"""
    
    def __init__(self, message: str, error_type: str, details: Optional[dict] = None):
        """
        Initialize trading error.
        
        Args:
            message: Error description
            error_type: Category of error (e.g., 'API', 'DATA', 'RISK')
            details: Optional dictionary with additional error context
        """
        self.message = message
        self.error_type = error_type
        self.details = details or {}
        super().__init__(self.message)

class DataError(TradingError):
    """Raised when there are issues with data fetching or processing"""
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, "DATA", details)

class RiskError(TradingError):
    """Raised when risk limits are exceeded or risk checks fail"""
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, "RISK", details)

class APIError(TradingError):
    """Raised for API-related issues"""
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, "API", details)

class ValidationError(TradingError):
    """Raised for input validation failures"""
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, "VALIDATION", details) 