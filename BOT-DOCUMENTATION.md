# Cryptocurrency Trading Bot Documentation

## Overview
A professional-grade cryptocurrency trading bot that combines technical analysis, risk management, and automated execution. Supports both paper trading and live trading with a focus on capital preservation and consistent returns.

## Core Features
- Risk-adjusted position sizing
- Portfolio management
- Paper trading simulation
- Discord command interface
- Real-time monitoring and alerts
- Dual timeframe analysis (1H/1D)

## Trading Strategy

### Core Principles
1. **Trend Following with Confirmation**
   - Daily trend determines bias
   - Hourly signals for entry timing
   - Wait for volume confirmation
   - Exit when trend weakens

2. **Smart Entry Points**
   - Buy pullbacks in uptrends
   - RSI conditions:
     * Trending: 40-60 range
     * Reversal: < 30 oversold
   - Price near support levels

3. **Risk-Optimized Position Management**
   - Scale in: Start with half position
   - Add on confirmation: Add 50% if trend strengthens
   - Lock in profits: Move stop to break-even at 5%
   - Let winners run: Trail stops at key levels

4. **Clear Exit Rules**
   - Technical: Trend reversal
   - Risk: Stop loss hit
   - Profit: Trailing stop triggered
   - Time: Max holding period exceeded

## System Architecture

### 1. Core Components
- **TradingBot**: Central controller and coordination
- **TechnicalAnalyzer**: Signal generation and market analysis
- **RiskManager**: Position sizing and risk control
- **DataManager**: Price data and market information
- **Position**: Trade management and tracking
- **CommandHandler**: User interface and controls

## Trading Logic

### Timeframe Analysis
```python
timeframes = {
    '1h': {'weight': 0.4},  # Entry timing and momentum
    '1d': {'weight': 0.6}   # Overall trend direction
}
```

#### Daily Analysis (60%)
- Determines overall trend direction
- Sets trading bias (long/short/neutral)
- Identifies key support/resistance

#### Hourly Analysis (40%)
- Entry and exit timing
- Volume confirmation
- Short-term momentum

### Signal Generation and Analysis
Each timeframe analysis produces:
- Trend Direction: Moving Average Crossovers (20/50)
- Momentum: RSI as confirmation only
- Volume Profile: Relative to 20-period average
- Support/Resistance: Key price levels
- Final Confidence Score: Weighted combination

Example Signal:
```python
{
    'symbol': 'BTC',
    'action': 'buy',
    'confidence': 0.75,
    'signals': {
        '1d': {'trend': 1, 'momentum': 0.7, 'volume': 1.1},  # Trend bias
        '1h': {'trend': 1, 'momentum': 0.8, 'volume': 1.2}   # Entry timing
    }
}
```

### Signal Weighting
```python
confidence_score = (
    (daily_trend * 0.4) +      # Primary trend direction
    (hourly_signal * 0.3) +    # Entry timing
    (volume_profile * 0.2) +   # Volume confirmation
    (price_level * 0.1)        # Support/Resistance
)
```

### Entry Rules
A valid entry requires:
1. Daily trend aligned (primary requirement)
2. Hourly confirmation (timing)
2. Risk checks pass:
   - Portfolio exposure < 80%
   - Position count < max_positions
   - Daily drawdown within limits
3. Volume confirmation
4. Price position checks:
   - Not near major resistance (2% buffer)
   - Not more than 70% up from daily low
   - RSI between 40-60 (trending) or < 30 (reversal)

## Risk Management

### Position Sizing
```python
position_size = account_value * risk_per_trade / (entry_price * stop_loss_distance)
```

### Risk Limits
- Maximum 2% risk per trade
- Maximum 80% portfolio exposure
- Maximum 5 concurrent positions
- 15% maximum drawdown limit

### Position Management
1. **Trailing Stops**
   - Initial stop: 5% from entry
   - Tightens as profit increases:
     * At 3% profit: Trail 2%
     * At 5% profit: Trail 3%
     * At 7%+ profit: Trail 4%

2. **Partial Exits**
   - Take 50% profit at 5% gain
   - Move stop to breakeven
   - Let remainder run with trailing stop

## Trading Loop Operation

### Main Trading Loop
```python
async def trading_loop(self):
    """Main trading loop - runs every TRADING_INTERVAL seconds"""
    while self.trading_active:
        try:
            # 1. Update Global State
            await self.update_account_state()
            await self.risk_manager.check_portfolio_health()
            
            # 2. Position Management (Priority)
            for symbol, position in self.positions.items():
                signals = await self.technical_analyzer.get_signals(symbol)
                await self.manage_position(position, signals)
            
            # 3. New Opportunities
            for symbol in self.watched_symbols:
                if symbol not in self.positions:
                    # Get fresh analysis
                    signals = await self.technical_analyzer.get_signals(symbol)
                    
                    # Check entry conditions
                    if await self.should_enter(symbol, signals):
                        # Calculate position size
                        size = await self.risk_manager.calculate_position_size(
                            symbol, signals['confidence']
                        )
                        
                        # Execute entry
                        await self.execute_entry(symbol, size, signals)
            
            # 4. Cleanup & Logging
            await self.cleanup_expired_data()
            await self.log_trading_metrics()
            
        except Exception as e:
            await self.log(f"Trading loop error: {str(e)}", level="error")
            
        finally:
            # Maintain loop interval
            await asyncio.sleep(self.config.trading_interval)
```

### Loop Components

1. **State Updates** (Every Loop)
   - Account balance
   - Position values
   - Portfolio exposure
   - Risk metrics

2. **Position Management** (Priority)
   - Update position metrics
   - Check stop loss/take profit
   - Evaluate exit signals
   - Adjust trailing stops

3. **Entry Analysis** (For Watched Symbols)
   - Technical analysis
   - Risk evaluation
   - Entry condition verification
   - Position sizing calculation

4. **System Maintenance**
   - Clean old data
   - Update metrics
   - Log performance
   - Error recovery

### Error Handling
- Comprehensive try/except blocks
- Automatic recovery attempts
- Error logging and alerts
- Safe state maintenance

## User Commands

### Basic Commands
```
/start - Start the trading bot
/stop - Stop the trading bot
/status - Get current bot status
/watch <symbol> - Add symbol to watchlist
/unwatch <symbol> - Remove symbol from watchlist
```

### Trading Commands
```
/paper - Switch to paper trading
/live - Switch to live trading (requires confirmation)
/buy <symbol> - Manual buy order
/sell <symbol> - Manual sell order
/position <symbol> - Get position details
```

### Configuration Commands
```
/set_risk <percentage> - Set risk per trade
/set_max_positions <number> - Set maximum positions
/set_stop_loss <percentage> - Set default stop loss
/set_take_profit <percentage> - Set default take profit
```

### Analysis Commands
```
/analyze <symbol> - Get detailed analysis
/signals - View current signals
/performance - View trading performance
/portfolio - View portfolio status
```

## Configuration

### Environment Variables
```env
# Trading Configuration
TRADING_INTERVAL=300
STOP_LOSS_PERCENTAGE=5.0
TAKE_PROFIT_PERCENTAGE=10.0

# Risk Management
RISK_MAX_DRAWDOWN=0.15
RISK_DAILY_VAR=0.02
RISK_MAX_POSITIONS=5

# Paper Trading
PAPER_BALANCE=1000.0
```

### Technical Analysis Settings
```python
settings = {
    'ma_fast': 20,
    'ma_slow': 50,
    'rsi_period': 14,
    'volume_ma': 20
}
```

## Safety Features

### Error Handling
- Comprehensive error catching
- Automatic position safety checks
- Network failure recovery
- Data validation

### Risk Controls
- Maximum drawdown protection
- Position size limits
- Exchange rate limiting
- Data quality checks

## Performance Monitoring

### Metrics Tracked
- Win/Loss ratio
- Average profit per trade
- Maximum drawdown
- Sharpe ratio
- Portfolio value over time

### Alerts
- Position updates
- Risk threshold warnings
- Technical signal alerts
- Error notifications

## Best Practices

### Trading
1. Always start with paper trading
2. Monitor initial trades closely
3. Start with conservative risk settings
4. Keep proper trading records

### Risk Management
1. Never override stop losses
2. Maintain position size discipline
3. Monitor overall exposure
4. Regular performance review

## Future Enhancements
1. Machine learning integration
2. Advanced order types
3. Multiple exchange support
4. Enhanced backtesting capabilities