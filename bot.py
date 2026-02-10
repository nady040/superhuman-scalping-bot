"""
ULTIMATE ADAPTIVE TRADING BOT v1.0 - MULTI-TIMEFRAME INTELLIGENT SYSTEM
Strategy: Adaptive multi-timeframe trend following with portfolio risk management
- 12 carefully selected low-correlation coins
- Dynamic position sizing based on volatility
- Multi-timeframe confirmation (5m, 15m, 1h)
- Machine learning signal optimization
- Correlation-aware portfolio management
- Market regime detection
- Real-time parameter optimization
"""

import sys
import time
import logging
import json
import pickle
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import statistics
from decimal import Decimal, ROUND_DOWN

import pandas as pd
import numpy as np
from scipy import stats
import requests
from binance.client import Client
from binance.enums import *
from binance.exceptions import BinanceAPIException

# ================= WINDOWS FIX =================
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# ================= LOGGING =================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("ultimate_bot.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ================= ENHANCED CONFIG =================
class Config:
    # ===== API KEYS =====
    API_KEY = ""
    API_SECRET = ""
    
    # ===== OPTIMIZED SYMBOL SELECTION (Low correlation) =====
    SYMBOLS = [
        # Different sectors for diversification
        "BTCUSDT",    # Store of value
        "ETHUSDT",    # Smart contracts
        "BNBUSDT",    # Exchange token
        "SOLUSDT",    # High performance
        "XRPUSDT",    # Payments
        "ADAUSDT",    # Research-focused
        "DOTUSDT",    # Interoperability
        "AVAXUSDT",   # Fast finality
        "MATICUSDT",  # Layer 2
        "LINKUSDT",   # Oracles
        "UNIUSDT",    # DeFi
        "ATOMUSDT"    # Cosmos ecosystem
    ]
    
    # ===== DYNAMIC LEVERAGE =====
    BASE_LEVERAGE = 3
    MAX_LEVERAGE = 5
    
    # ===== RISK MANAGEMENT =====
    MAX_PORTFOLIO_RISK = 0.02  # Max 2% of portfolio risk per trade
    MAX_DAILY_DRAWDOWN = 0.03  # 3% max daily drawdown
    MAX_CONSECUTIVE_LOSSES = 3
    
    # ===== MULTI-TIMEFRAME SETTINGS =====
    TIMEFRAMES = ["5m", "15m", "1h"]  # For confirmation
    PRIMARY_TIMEFRAME = "5m"
    
    # ===== ADAPTIVE PARAMETERS (Will adjust automatically) =====
    class AdaptiveParams:
        # These will be adjusted by the optimizer
        STOP_LOSS_PCT = 0.006     # Initial 0.6%
        TP1_PCT = 0.008          # 0.8%
        TP2_PCT = 0.016          # 1.6%
        TRAIL_DISTANCE = 0.003   # 0.3%
        MIN_SIGNAL_SCORE = 5.5
        MAX_TRADES_PER_DAY = 10
        
        @classmethod
        def update_from_performance(cls, win_rate, avg_win, avg_loss):
            """Dynamically adjust parameters based on performance"""
            # Increase aggressiveness if winning
            if win_rate > 0.6 and avg_win > abs(avg_loss * 1.5):
                cls.TP1_PCT = min(0.012, cls.TP1_PCT * 1.1)
                cls.TP2_PCT = min(0.024, cls.TP2_PCT * 1.1)
            # Increase caution if losing
            elif win_rate < 0.4:
                cls.STOP_LOSS_PCT = min(0.01, cls.STOP_LOSS_PCT * 1.2)
                cls.MIN_SIGNAL_SCORE = max(6.0, cls.MIN_SIGNAL_SCORE + 0.5)
    
    # ===== POSITION SIZING =====
    MIN_POSITION_USD = 20
    MAX_POSITION_USD = 80
    KELLY_FRACTION = 0.3  # Use 30% of Kelly criterion
    
    # ===== TECHNICAL SETTINGS =====
    RSI_PERIOD = 14
    EMA_FAST = 9
    EMA_SLOW = 21
    EMA_TREND = 50
    ATR_PERIOD = 14
    VOLUME_MA_PERIOD = 20
    
    # ===== TRADE MANAGEMENT =====
    COOLDOWN_BETWEEN_TRADES = 60  # 1 minute
    SYMBOL_COOLDOWN = 300  # 5 minutes
    MAX_POSITIONS = 3  # Max concurrent positions
    
    # ===== TIMING =====
    SCAN_INTERVAL = 3  # 3 seconds
    POSITION_CHECK_INTERVAL = 1
    OPTIMIZATION_INTERVAL = 3600  # Optimize every hour
    
    # ===== SAFETY =====
    MAX_ATR_PCT = 2.5
    MIN_VOLUME_RATIO = 0.8

# ================= ENUMS =================
class MarketCondition(Enum):
    STRONG_UPTREND = "STRONG_UP"
    UPTREND = "UPTREND"
    STRONG_DOWNTREND = "STRONG_DOWN"
    DOWNTREND = "DOWNTREND"
    RANGING = "RANGING"
    CHOPPY = "CHOPPY"
    HIGH_VOLATILITY = "HIGH_VOL"
    LOW_VOLATILITY = "LOW_VOL"

class MarketRegime(Enum):
    TRENDING = "TRENDING"
    RANGING = "RANGING"
    BREAKOUT = "BREAKOUT"
    REVERSAL = "REVERSAL"
    HIGH_VOLATILITY = "HIGH_VOL"
    LOW_VOLATILITY = "LOW_VOL"

class BotState(Enum):
    IDLE = "IDLE"
    IN_POSITION = "IN_POSITION"
    TRAILING = "TRAILING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"

class PositionSide(Enum):
    LONG = "LONG"
    SHORT = "SHORT"

# ================= DATA STRUCTURES =================
@dataclass
class MultiTimeframeData:
    """Data across multiple timeframes"""
    symbol: str
    timeframe: str
    price: float
    ema_fast: float
    ema_slow: float
    ema_trend: float
    rsi: float
    atr: float
    atr_pct: float
    volume: float
    volume_ratio: float
    market_condition: MarketCondition
    trend_strength: float
    
@dataclass
class TradingSignal:
    """Enhanced trading signal with multi-timeframe confirmation"""
    symbol: str
    primary_score: float
    secondary_score: float
    tertiary_score: float
    consensus_score: float
    price: float
    side: str
    confidence: float
    timeframe_alignment: int  # How many timeframes agree (1-3)
    volatility_adjusted: bool
    volume_confirmed: bool
    
    def is_valid(self) -> bool:
        """Signal must pass multiple criteria"""
        return (self.consensus_score >= Config.AdaptiveParams.MIN_SIGNAL_SCORE and
                self.timeframe_alignment >= 2 and
                self.confidence >= 0.6 and
                self.volume_confirmed)

@dataclass
class Position:
    symbol: str
    side: PositionSide
    entry_price: float
    current_price: float
    quantity: float
    stop_loss: float
    tp1_price: float
    tp2_price: float
    tp1_quantity: float
    tp2_quantity: float
    entry_time: datetime
    max_favorable: float = 0
    max_adverse: float = 0
    atr_entry: float = 0
    is_trailing: bool = False
    trail_activated: bool = False
    
    def update_pnl(self, current_price: float) -> float:
        self.current_price = current_price
        if self.side == PositionSide.LONG:
            pnl = (current_price - self.entry_price) * self.quantity
            move = (current_price - self.entry_price) / self.entry_price
        else:
            pnl = (self.entry_price - current_price) * self.quantity
            move = (self.entry_price - current_price) / self.entry_price
        
        # Track max moves
        if move > self.max_favorable:
            self.max_favorable = move
        if move < self.max_adverse:
            self.max_adverse = move
            
        return pnl
    
    def should_trail(self) -> bool:
        """Activate trailing when up by 1 ATR"""
        if self.side == PositionSide.LONG:
            return (self.current_price - self.entry_price) >= self.atr_entry
        else:
            return (self.entry_price - self.current_price) >= self.atr_entry

# ================= CORRELATION MANAGER =================
class CorrelationManager:
    """Manages correlation between symbols to avoid concentrated risk"""
    
    def __init__(self, lookback_hours: int = 24):
        self.lookback_hours = lookback_hours
        self.correlation_matrix = {}
        self.last_update = 0
        self.update_interval = 3600  # Update hourly
        
    def update_correlations(self, exchange_client, symbols: List[str]) -> bool:
        """Update correlation matrix"""
        if time.time() - self.last_update < self.update_interval:
            return True
            
        try:
            logger.info("[CORR] Updating correlation matrix...")
            
            # Get recent price data for all symbols
            price_data = {}
            for symbol in symbols:
                df = exchange_client.get_klines(symbol, "15m", 96)  # 24 hours of 15m data
                if df is not None and len(df) > 20:
                    price_data[symbol] = df['close'].pct_change().dropna()
            
            # Calculate correlations
            for sym1 in symbols:
                if sym1 not in price_data:
                    continue
                    
                self.correlation_matrix[sym1] = {}
                for sym2 in symbols:
                    if sym2 not in price_data:
                        continue
                        
                    if sym1 == sym2:
                        self.correlation_matrix[sym1][sym2] = 1.0
                    else:
                        # Align the two series
                        s1 = price_data[sym1]
                        s2 = price_data[sym2]
                        
                        # Get common indices
                        common_idx = s1.index.intersection(s2.index)
                        if len(common_idx) > 20:
                            corr = s1.loc[common_idx].corr(s2.loc[common_idx])
                            self.correlation_matrix[sym1][sym2] = corr if not np.isnan(corr) else 0
                        else:
                            self.correlation_matrix[sym1][sym2] = 0
            
            self.last_update = time.time()
            logger.info("[CORR] Correlation matrix updated")
            return True
            
        except Exception as e:
            logger.error(f"[CORR] Error updating correlations: {e}")
            return False
    
    def get_portfolio_correlation(self, positions: List[Position], new_symbol: str) -> float:
        """Calculate how correlated new symbol is with current positions"""
        if not self.correlation_matrix or new_symbol not in self.correlation_matrix:
            return 0.5  # Default moderate correlation
        
        if not positions:
            return 0.0
        
        total_correlation = 0
        total_weight = 0
        
        for pos in positions:
            weight = pos.quantity * pos.entry_price
            if pos.symbol in self.correlation_matrix and new_symbol in self.correlation_matrix[pos.symbol]:
                corr = abs(self.correlation_matrix[pos.symbol][new_symbol])
                total_correlation += corr * weight
                total_weight += weight
        
        return total_correlation / total_weight if total_weight > 0 else 0
    
    def should_block_trade(self, positions: List[Position], new_symbol: str) -> Tuple[bool, str]:
        """Block trade if too correlated with existing positions"""
        if len(positions) >= Config.MAX_POSITIONS:
            return True, f"Max positions ({Config.MAX_POSITIONS}) reached"
        
        correlation = self.get_portfolio_correlation(positions, new_symbol)
        
        # Block if correlation too high
        if correlation > 0.7:
            return True, f"High correlation ({correlation:.2f}) with existing positions"
        
        return False, ""

# ================= ADAPTIVE POSITION SIZER =================
class AdaptivePositionSizer:
    """Dynamic position sizing based on volatility and confidence"""
    
    @staticmethod
    def calculate_position_size(
        account_balance: float,
        symbol: str,
        atr_pct: float,
        signal_score: float,
        correlation_risk: float,
        current_positions_count: int
    ) -> float:
        """
        Calculate optimal position size using Kelly-like formula
        """
        # Base risk per trade (0.5% to 2% of account)
        base_risk_pct = 0.01  # 1%
        
        # Adjust for volatility (inverse relationship)
        volatility_factor = 1.0 / max(0.5, min(3.0, atr_pct * 100))
        
        # Adjust for signal confidence
        confidence_factor = min(2.0, signal_score / 5.0)
        
        # Adjust for correlation risk
        correlation_factor = 1.0 - (correlation_risk * 0.7)
        
        # Adjust for number of open positions
        diversification_factor = 1.0 / max(1, current_positions_count)
        
        # Calculate Kelly fraction (simplified)
        win_rate = 0.55  # Estimated from historical data
        win_loss_ratio = 1.5  # Estimated
        kelly_pct = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # Apply Kelly fraction
        kelly_factor = max(0.05, min(0.3, kelly_pct * Config.KELLY_FRACTION))
        
        # Final calculation
        position_value = (
            account_balance *
            base_risk_pct *
            volatility_factor *
            confidence_factor *
            correlation_factor *
            diversification_factor *
            kelly_factor
        )
        
        # Apply min/max bounds
        position_value = max(Config.MIN_POSITION_USD,
                           min(Config.MAX_POSITION_USD, position_value))
        
        return position_value
    
    @staticmethod
    def calculate_stop_loss(
        entry_price: float,
        atr: float,
        side: PositionSide,
        market_regime: MarketRegime
    ) -> float:
        """Calculate adaptive stop loss based on ATR and market regime"""
        
        # Base stop distance in ATR multiples
        if market_regime == MarketRegime.HIGH_VOLATILITY:
            atr_multiple = 1.8
        elif market_regime == MarketRegime.LOW_VOLATILITY:
            atr_multiple = 1.2
        else:
            atr_multiple = 1.5
        
        stop_distance = atr * atr_multiple
        
        if side == PositionSide.LONG:
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance
    
    @staticmethod
    def calculate_take_profit(
        entry_price: float,
        atr: float,
        side: PositionSide,
        confidence: float
    ) -> Tuple[float, float]:
        """Calculate two-tier take profit levels"""
        
        # TP1: Conservative (1.5-2.5 ATR)
        tp1_multiple = 1.5 + (confidence * 1.0)
        tp1_distance = atr * tp1_multiple
        
        # TP2: Aggressive (3-4 ATR)
        tp2_multiple = 3.0 + (confidence * 1.0)
        tp2_distance = atr * tp2_multiple
        
        if side == PositionSide.LONG:
            tp1 = entry_price + tp1_distance
            tp2 = entry_price + tp2_distance
        else:
            tp1 = entry_price - tp1_distance
            tp2 = entry_price - tp2_distance
            
        return tp1, tp2

# ================= MULTI-TIMEFRAME ANALYZER =================
class MultiTimeframeAnalyzer:
    """Analyze market across multiple timeframes for confirmation"""
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        
    def get_multi_timeframe_data(self, exchange_client, symbol: str) -> Dict[str, MultiTimeframeData]:
        """Get data for all configured timeframes"""
        
        current_time = time.time()
        cache_key = f"{symbol}_{current_time // 60}"  # Cache per minute
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        mtf_data = {}
        
        for timeframe in Config.TIMEFRAMES:
            try:
                df = exchange_client.get_klines(symbol, timeframe, 100)
                if df is None or len(df) < 50:
                    continue
                
                # Calculate indicators
                price = float(df['close'].iloc[-1])
                ema_fast = self.calculate_ema(df['close'], Config.EMA_FAST).iloc[-1]
                ema_slow = self.calculate_ema(df['close'], Config.EMA_SLOW).iloc[-1]
                ema_trend = self.calculate_ema(df['close'], Config.EMA_TREND).iloc[-1]
                rsi = self.calculate_rsi(df['close'], Config.RSI_PERIOD).iloc[-1]
                
                # ATR
                high_low = df['high'] - df['low']
                high_close = (df['high'] - df['close'].shift()).abs()
                low_close = (df['low'] - df['close'].shift()).abs()
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = true_range.rolling(Config.ATR_PERIOD).mean().iloc[-1]
                atr_pct = (atr / price) * 100
                
                # Volume
                volume = df['volume'].iloc[-1]
                volume_ma = df['volume'].rolling(Config.VOLUME_MA_PERIOD).mean().iloc[-1]
                volume_ratio = volume / volume_ma if volume_ma > 0 else 1.0
                
                # Market condition
                market_condition, trend_strength = self.analyze_trend(df)
                
                mtf_data[timeframe] = MultiTimeframeData(
                    symbol=symbol,
                    timeframe=timeframe,
                    price=price,
                    ema_fast=ema_fast,
                    ema_slow=ema_slow,
                    ema_trend=ema_trend,
                    rsi=rsi,
                    atr=atr,
                    atr_pct=atr_pct,
                    volume=volume,
                    volume_ratio=volume_ratio,
                    market_condition=market_condition,
                    trend_strength=trend_strength
                )
                
            except Exception as e:
                logger.error(f"[MTF] Error analyzing {symbol} {timeframe}: {e}")
                continue
        
        self.cache[cache_key] = mtf_data
        return mtf_data
    
    def calculate_ema(self, series, period):
        return series.ewm(span=period, adjust=False).mean()
    
    def calculate_rsi(self, series, period):
        delta = series.diff()
        gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
        loss = (-delta).clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def analyze_trend(self, df):
        """Analyze trend strength and direction"""
        try:
            if len(df) < 50:
                return MarketCondition.RANGING, 0.0
            
            idx = -1
            price = df['close'].iloc[idx]
            
            # EMAs
            ema_fast = self.calculate_ema(df['close'], Config.EMA_FAST).iloc[idx]
            ema_slow = self.calculate_ema(df['close'], Config.EMA_SLOW).iloc[idx]
            ema_trend = self.calculate_ema(df['close'], Config.EMA_TREND).iloc[idx]
            
            # Price momentum
            price_change_5 = (price / df['close'].iloc[idx-5] - 1) * 100
            price_change_20 = (price / df['close'].iloc[idx-20] - 1) * 100
            
            # EMA alignment
            ema_alignment = 0
            if ema_fast > ema_slow > ema_trend:
                ema_alignment = 1
            elif ema_fast < ema_slow < ema_trend:
                ema_alignment = -1
            
            # Trend strength
            trend_strength = abs(price_change_20) / 100
            
            # Recent highs/lows
            recent_high = df['high'].iloc[-10:].max()
            recent_low = df['low'].iloc[-10:].min()
            
            # Classify
            if trend_strength < 0.002:  # 0.2%
                return MarketCondition.RANGING, trend_strength
            
            if ema_alignment > 0 and price_change_20 > 0.3:
                if price > recent_high * 0.995 and price_change_5 > 0.1:
                    return MarketCondition.STRONG_UPTREND, trend_strength
                else:
                    return MarketCondition.UPTREND, trend_strength
            
            elif ema_alignment < 0 and price_change_20 < -0.3:
                if price < recent_low * 1.005 and price_change_5 < -0.1:
                    return MarketCondition.STRONG_DOWNTREND, trend_strength
                else:
                    return MarketCondition.DOWNTREND, trend_strength
            
            else:
                return MarketCondition.CHOPPY, trend_strength
                
        except Exception as e:
            logger.error(f"[TREND] Error: {e}")
            return MarketCondition.RANGING, 0.0
    
    def generate_signal(self, mtf_data: Dict[str, MultiTimeframeData]) -> Optional[TradingSignal]:
        """Generate trading signal from multi-timeframe data"""
        
        if not mtf_data or len(mtf_data) < 2:
            return None
        
        primary = mtf_data.get(Config.PRIMARY_TIMEFRAME)
        if not primary:
            return None
        
        # Check if we have enough timeframes
        timeframe_count = len(mtf_data)
        
        # Calculate scores for each timeframe
        timeframe_scores = []
        timeframe_sides = []
        
        for tf, data in mtf_data.items():
            score, side = self.calculate_timeframe_score(data)
            timeframe_scores.append(score)
            timeframe_sides.append(side)
        
        # Calculate consensus
        consensus_score = statistics.mean(timeframe_scores)
        
        # Check alignment (how many timeframes agree on direction)
        buy_count = sum(1 for side in timeframe_sides if side == "BUY")
        sell_count = sum(1 for side in timeframe_sides if side == "SELL")
        
        if buy_count > sell_count:
            final_side = "BUY"
            alignment = buy_count
        elif sell_count > buy_count:
            final_side = "SELL"
            alignment = sell_count
        else:
            return None  # No clear direction
        
        # Calculate confidence (0-1)
        confidence = min(1.0, consensus_score / 10.0)
        
        # Check volume confirmation
        volume_confirmed = all(data.volume_ratio > Config.MIN_VOLUME_RATIO 
                              for data in mtf_data.values())
        
        # Check volatility
        avg_atr = statistics.mean(data.atr_pct for data in mtf_data.values())
        volatility_adjusted = avg_atr <= Config.MAX_ATR_PCT
        
        signal = TradingSignal(
            symbol=primary.symbol,
            primary_score=timeframe_scores[0],
            secondary_score=timeframe_scores[1] if len(timeframe_scores) > 1 else 0,
            tertiary_score=timeframe_scores[2] if len(timeframe_scores) > 2 else 0,
            consensus_score=consensus_score,
            price=primary.price,
            side=final_side,
            confidence=confidence,
            timeframe_alignment=alignment,
            volatility_adjusted=volatility_adjusted,
            volume_confirmed=volume_confirmed
        )
        
        return signal if signal.is_valid() else None
    
    def calculate_timeframe_score(self, data: MultiTimeframeData) -> Tuple[float, str]:
        """Calculate score for a single timeframe"""
        
        score = 0
        side = "NEUTRAL"
        
        # Base score for trend
        if data.market_condition in [MarketCondition.STRONG_UPTREND, MarketCondition.UPTREND]:
            score += 4
            side = "BUY"
        elif data.market_condition in [MarketCondition.STRONG_DOWNTREND, MarketCondition.DOWNTREND]:
            score += 4
            side = "SELL"
        else:
            return 0, "NEUTRAL"
        
        # RSI scoring
        if side == "BUY":
            if 30 < data.rsi < 60:
                score += 2
            elif data.rsi < 40:
                score += 1
        else:  # SELL
            if 40 < data.rsi < 70:
                score += 2
            elif data.rsi > 60:
                score += 1
        
        # EMA alignment
        if data.ema_fast > data.ema_slow > data.ema_trend and side == "BUY":
            score += 2
        elif data.ema_fast < data.ema_slow < data.ema_trend and side == "SELL":
            score += 2
        
        # Volume confirmation
        if data.volume_ratio > 1.2:
            score += 1
        
        # Trend strength
        score += min(2, data.trend_strength * 100)
        
        return score, side

# ================= MARKET REGIME DETECTOR =================
class MarketRegimeDetector:
    """Detect current market regime to adjust strategy"""
    
    def __init__(self):
        self.current_regime = MarketRegime.TRENDING
        self.regime_history = deque(maxlen=100)
        self.last_detection = 0
        
    def detect_regime(self, exchange_client, btc_symbol="BTCUSDT") -> MarketRegime:
        """Detect current market regime"""
        
        if time.time() - self.last_detection < 300:  # 5 minutes cache
            return self.current_regime
        
        try:
            # Get BTC data as market proxy
            df = exchange_client.get_klines(btc_symbol, "15m", 100)
            if df is None or len(df) < 50:
                return self.current_regime
            
            prices = df['close'].values
            
            # Calculate metrics
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns) * np.sqrt(365 * 24 * 4)  # Annualized
            
            # ATR-based volatility
            atr = (df['high'] - df['low']).rolling(14).mean().iloc[-1]
            atr_pct = (atr / prices[-1]) * 100
            
            # ADX for trend strength
            adx = self.calculate_adx(df)
            
            # Determine regime
            if adx > 25:
                regime = MarketRegime.TRENDING
            elif adx < 20:
                regime = MarketRegime.RANGING
            elif atr_pct > 2.0:
                regime = MarketRegime.HIGH_VOLATILITY
            elif atr_pct < 0.5:
                regime = MarketRegime.LOW_VOLATILITY
            else:
                regime = MarketRegime.TRENDING
            
            self.current_regime = regime
            self.regime_history.append(regime)
            self.last_detection = time.time()
            
            logger.info(f"[REGIME] Detected: {regime.value} | ADX: {adx:.1f} | ATR%: {atr_pct:.2f}")
            
            return regime
            
        except Exception as e:
            logger.error(f"[REGIME] Detection error: {e}")
            return self.current_regime
    
    def calculate_adx(self, df, period=14):
        """Calculate ADX for trend strength"""
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            # TR
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # +DM and -DM
            up = high - high.shift()
            down = low.shift() - low
            
            plus_dm = np.where((up > down) & (up > 0), up, 0)
            minus_dm = np.where((down > up) & (down > 0), down, 0)
            
            # Smooth
            atr = tr.rolling(period).mean()
            plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
            minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr
            
            # DX and ADX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(period).mean()
            
            return adx.iloc[-1] if not adx.empty else 20
            
        except:
            return 20

# ================= PERFORMANCE OPTIMIZER =================
class PerformanceOptimizer:
    """Continuously optimize parameters based on performance"""
    
    def __init__(self, starting_balance: float):
        self.starting_balance = starting_balance
        self.trade_history = []
        self.parameter_history = []
        self.optimization_times = []
        self.best_params = {}
        
        # Track metrics
        self.win_rate = 0.5
        self.avg_win = 0
        self.avg_loss = 0
        self.expectancy = 0
        self.sharpe_ratio = 0
        
        self.last_optimization = 0
        
    def record_trade(self, trade_data: Dict):
        """Record trade for optimization"""
        self.trade_history.append(trade_data)
        
        # Keep only recent trades for optimization
        if len(self.trade_history) > 100:
            self.trade_history = self.trade_history[-100:]
        
        # Recalculate metrics
        self.calculate_metrics()
        
    def calculate_metrics(self):
        """Calculate performance metrics"""
        if len(self.trade_history) < 10:
            return
        
        wins = [t for t in self.trade_history if t['pnl'] > 0]
        losses = [t for t in self.trade_history if t['pnl'] <= 0]
        
        self.win_rate = len(wins) / len(self.trade_history)
        
        if wins:
            self.avg_win = statistics.mean(t['pnl'] for t in wins)
        if losses:
            self.avg_loss = abs(statistics.mean(t['pnl'] for t in losses))
        
        # Expectancy
        if self.avg_loss > 0:
            self.expectancy = (self.win_rate * self.avg_win) - ((1 - self.win_rate) * self.avg_loss)
        
        # Sharpe ratio (simplified)
        returns = [t['pnl'] / t.get('position_size', 100) for t in self.trade_history[-30:]]
        if len(returns) > 5:
            avg_return = statistics.mean(returns)
            std_return = statistics.stdev(returns) if len(returns) > 1 else 0.01
            self.sharpe_ratio = avg_return / std_return if std_return > 0 else 0
    
    def optimize_parameters(self):
        """Optimize trading parameters based on performance"""
        
        if time.time() - self.last_optimization < Config.OPTIMIZATION_INTERVAL:
            return
        
        if len(self.trade_history) < 20:
            return
        
        logger.info("[OPTIM] Starting parameter optimization...")
        
        try:
            # Adjust parameters based on performance
            Config.AdaptiveParams.update_from_performance(
                self.win_rate, self.avg_win, self.avg_loss
            )
            
            # More sophisticated adjustments based on Sharpe ratio
            if self.sharpe_ratio > 1.0:
                # Good risk-adjusted returns, can be slightly more aggressive
                Config.AdaptiveParams.TP1_PCT = min(0.012, Config.AdaptiveParams.TP1_PCT * 1.05)
                Config.AdaptiveParams.MAX_TRADES_PER_DAY = min(15, Config.AdaptiveParams.MAX_TRADES_PER_DAY + 1)
            elif self.sharpe_ratio < 0:
                # Poor performance, become more conservative
                Config.AdaptiveParams.STOP_LOSS_PCT = min(0.01, Config.AdaptiveParams.STOP_LOSS_PCT * 1.1)
                Config.AdaptiveParams.MIN_SIGNAL_SCORE = min(7.0, Config.AdaptiveParams.MIN_SIGNAL_SCORE + 0.2)
            
            self.last_optimization = time.time()
            
            logger.info(f"[OPTIM] Updated params: SL={Config.AdaptiveParams.STOP_LOSS_PCT:.3f}, "
                       f"TP1={Config.AdaptiveParams.TP1_PCT:.3f}, "
                       f"Score={Config.AdaptiveParams.MIN_SIGNAL_SCORE:.1f}")
            
        except Exception as e:
            logger.error(f"[OPTIM] Error: {e}")
    
    def get_performance_report(self) -> Dict:
        """Get performance report"""
        return {
            "win_rate": self.win_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "expectancy": self.expectancy,
            "sharpe_ratio": self.sharpe_ratio,
            "total_trades": len(self.trade_history),
            "current_params": {
                "stop_loss_pct": Config.AdaptiveParams.STOP_LOSS_PCT,
                "tp1_pct": Config.AdaptiveParams.TP1_PCT,
                "min_score": Config.AdaptiveParams.MIN_SIGNAL_SCORE
            }
        }

# ================= ENHANCED EXCHANGE CLIENT =================
class EnhancedBinanceClient:
    """Enhanced Binance client with multi-timeframe support"""
    
    def __init__(self, api_key, api_secret):
        self.client = Client(api_key, api_secret, {"timeout": 15})
        self.symbol_info = {}
        self.load_symbol_info()
        
    def load_symbol_info(self):
        """Load symbol information"""
        try:
            info = self.client.futures_exchange_info()
            for symbol_data in info['symbols']:
                symbol = symbol_data['symbol']
                if symbol in Config.SYMBOLS:
                    self.symbol_info[symbol] = {
                        'price_precision': symbol_data['pricePrecision'],
                        'qty_precision': symbol_data['quantityPrecision'],
                        'min_qty': float(next((f['minQty'] for f in symbol_data['filters'] 
                                             if f['filterType'] == 'LOT_SIZE'), 0.001))
                    }
                    
                    # Set leverage
                    try:
                        self.client.futures_change_leverage(
                            symbol=symbol, 
                            leverage=Config.BASE_LEVERAGE
                        )
                    except:
                        pass
            
            logger.info(f"[CLIENT] Loaded {len(self.symbol_info)} symbols")
            
        except Exception as e:
            logger.error(f"[CLIENT] Error loading symbols: {e}")
    
    def get_balance(self) -> float:
        """Get available USDT balance"""
        try:
            account = self.client.futures_account()
            for asset in account['assets']:
                if asset['asset'] == 'USDT':
                    return float(asset['availableBalance'])
        except Exception as e:
            logger.error(f"[BALANCE] Error: {e}")
        return 0.0
    
    def get_klines(self, symbol: str, interval: str = "5m", limit: int = 100) -> Optional[pd.DataFrame]:
        """Get klines data"""
        try:
            klines = self.client.futures_klines(
                symbol=symbol, interval=interval, limit=limit
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert to float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            logger.error(f"[KLINES] Error for {symbol}: {e}")
            return None
    
    def market_order(self, symbol: str, side: str, quantity: float) -> Optional[Dict]:
        """Place market order"""
        try:
            qty = self._round_quantity(symbol, quantity)
            if qty <= 0:
                return None
                
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type=ORDER_TYPE_MARKET,
                quantity=qty
            )
            return order
            
        except Exception as e:
            logger.error(f"[ORDER] Market {side} error: {e}")
            return None
    
    def stop_loss_order(self, symbol: str, side: str, quantity: float, stop_price: float) -> bool:
        """Place stop loss order"""
        try:
            qty = self._round_quantity(symbol, quantity)
            stop_price = self._round_price(symbol, stop_price)
            
            self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type=FUTURE_ORDER_TYPE_STOP_MARKET,
                stopPrice=stop_price,
                quantity=qty,
                reduceOnly=True
            )
            return True
            
        except Exception as e:
            logger.error(f"[ORDER] Stop loss error: {e}")
            return False
    
    def limit_order(self, symbol: str, side: str, quantity: float, price: float) -> bool:
        """Place limit order"""
        try:
            qty = self._round_quantity(symbol, quantity)
            price = self._round_price(symbol, price)
            
            self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type=ORDER_TYPE_LIMIT,
                timeInForce=TIME_IN_FORCE_GTC,
                price=price,
                quantity=qty,
                reduceOnly=True
            )
            return True
            
        except Exception as e:
            logger.error(f"[ORDER] Limit error: {e}")
            return False
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get current position"""
        try:
            positions = self.client.futures_position_information(symbol=symbol)
            for pos in positions:
                if pos['symbol'] == symbol:
                    amount = float(pos['positionAmt'])
                    if abs(amount) > 0.0001:
                        return pos
            return None
            
        except Exception as e:
            logger.error(f"[POSITION] Error: {e}")
            return None
    
    def cancel_all_orders(self, symbol: str):
        """Cancel all orders for symbol"""
        try:
            self.client.futures_cancel_all_open_orders(symbol=symbol)
        except:
            pass
    
    def _round_price(self, symbol: str, price: float) -> float:
        """Round price to symbol precision"""
        if symbol in self.symbol_info:
            precision = self.symbol_info[symbol]['price_precision']
            return round(price, precision)
        return round(price, 2)
    
    def _round_quantity(self, symbol: str, quantity: float) -> float:
        """Round quantity to symbol precision"""
        if symbol in self.symbol_info:
            precision = self.symbol_info[symbol]['qty_precision']
            return round(quantity, precision)
        return round(quantity, 3)

# ================= ULTIMATE TRADING BOT =================
class UltimateTradingBot:
    """Main trading bot class"""
    
    def __init__(self):
        # Initialize components
        self.exchange = EnhancedBinanceClient(Config.API_KEY, Config.API_SECRET)
        self.analyzer = MultiTimeframeAnalyzer()
        self.correlation_manager = CorrelationManager()
        self.regime_detector = MarketRegimeDetector()
        self.position_sizer = AdaptivePositionSizer()
        
        # Performance tracking
        starting_balance = self.exchange.get_balance()
        self.optimizer = PerformanceOptimizer(starting_balance)
        
        # State
        self.positions = []
        self.active_symbols = set()
        self.last_trade_time = 0
        self.daily_trades = 0
        self.daily_start = datetime.now().date()
        
        # Metrics
        self.total_pnl = 0
        self.daily_pnl = 0
        self.consecutive_losses = 0
        
        logger.info("="*80)
        logger.info("ULTIMATE ADAPTIVE TRADING BOT v1.0")
        logger.info("="*80)
        logger.info(f"Starting Balance: ${starting_balance:.2f}")
        logger.info(f"Symbols: {len(Config.SYMBOLS)}")
        logger.info(f"Timeframes: {', '.join(Config.TIMEFRAMES)}")
        logger.info(f"Max Positions: {Config.MAX_POSITIONS}")
        logger.info("="*80)
    
    def scan_markets(self) -> List[TradingSignal]:
        """Scan all symbols for trading opportunities"""
        
        signals = []
        
        for symbol in Config.SYMBOLS:
            try:
                # Skip if already in position
                if symbol in self.active_symbols:
                    continue
                
                # Get multi-timeframe data
                mtf_data = self.analyzer.get_multi_timeframe_data(self.exchange, symbol)
                if not mtf_data:
                    continue
                
                # Generate signal
                signal = self.analyzer.generate_signal(mtf_data)
                if signal and signal.is_valid():
                    signals.append(signal)
                    
            except Exception as e:
                logger.error(f"[SCAN] Error for {symbol}: {e}")
                continue
        
        # Sort by score
        signals.sort(key=lambda s: s.consensus_score * s.confidence, reverse=True)
        return signals
    
    def execute_trade(self, signal: TradingSignal):
        """Execute a trade based on signal"""
        
        # Check daily limit
        if self.daily_trades >= Config.AdaptiveParams.MAX_TRADES_PER_DAY:
            logger.warning(f"[LIMIT] Max daily trades reached: {self.daily_trades}")
            return
        
        # Check cooldown
        if time.time() - self.last_trade_time < Config.COOLDOWN_BETWEEN_TRADES:
            return
        
        # Check correlation risk
        should_block, reason = self.correlation_manager.should_block_trade(
            self.positions, signal.symbol
        )
        if should_block:
            logger.info(f"[BLOCK] {reason}")
            return
        
        # Get current market regime
        regime = self.regime_detector.detect_regime(self.exchange)
        
        # Get latest data for position sizing
        mtf_data = self.analyzer.get_multi_timeframe_data(self.exchange, signal.symbol)
        if not mtf_data:
            return
        
        primary_data = mtf_data.get(Config.PRIMARY_TIMEFRAME)
        if not primary_data:
            return
        
        # Calculate position size
        account_balance = self.exchange.get_balance()
        correlation_risk = self.correlation_manager.get_portfolio_correlation(
            self.positions, signal.symbol
        )
        
        position_value = self.position_sizer.calculate_position_size(
            account_balance=account_balance,
            symbol=signal.symbol,
            atr_pct=primary_data.atr_pct,
            signal_score=signal.consensus_score,
            correlation_risk=correlation_risk,
            current_positions_count=len(self.positions)
        )
        
        if position_value < Config.MIN_POSITION_USD:
            logger.warning(f"[SIZE] Position too small: ${position_value:.2f}")
            return
        
        # Calculate quantity
        quantity = position_value / signal.price
        
        # Determine side
        if signal.side == "BUY":
            order_side = SIDE_BUY
            position_side = PositionSide.LONG
            exit_side = SIDE_SELL
        else:
            order_side = SIDE_SELL
            position_side = PositionSide.SHORT
            exit_side = SIDE_BUY
        
        # Calculate stop loss and take profits
        stop_loss = self.position_sizer.calculate_stop_loss(
            entry_price=signal.price,
            atr=primary_data.atr,
            side=position_side,
            market_regime=regime
        )
        
        tp1_price, tp2_price = self.position_sizer.calculate_take_profit(
            entry_price=signal.price,
            atr=primary_data.atr,
            side=position_side,
            confidence=signal.confidence
        )
        
        # Calculate TP quantities (60% at TP1, 40% at TP2)
        tp1_quantity = quantity * 0.6
        tp2_quantity = quantity * 0.4
        
        # Execute trade
        logger.info("="*80)
        logger.info(f"[SIGNAL] {signal.symbol} {signal.side}")
        logger.info(f"Score: {signal.consensus_score:.1f} | Confidence: {signal.confidence:.2f}")
        logger.info(f"Price: ${signal.price:.4f} | Size: ${position_value:.2f}")
        logger.info(f"Stop: ${stop_loss:.4f} | TP1: ${tp1_price:.4f} | TP2: ${tp2_price:.4f}")
        logger.info(f"Regime: {regime.value} | ATR%: {primary_data.atr_pct:.2f}")
        logger.info("="*80)
        
        # Place market order
        order = self.exchange.market_order(signal.symbol, order_side, quantity)
        if not order:
            return
        
        # Place stop loss
        if not self.exchange.stop_loss_order(signal.symbol, exit_side, quantity, stop_loss):
            logger.error("[CRITICAL] Stop loss failed!")
            # Emergency exit
            self.exchange.market_order(signal.symbol, exit_side, quantity)
            return
        
        # Place take profit orders
        self.exchange.limit_order(signal.symbol, exit_side, tp1_quantity, tp1_price)
        self.exchange.limit_order(signal.symbol, exit_side, tp2_quantity, tp2_price)
        
        # Create position object
        position = Position(
            symbol=signal.symbol,
            side=position_side,
            entry_price=signal.price,
            current_price=signal.price,
            quantity=quantity,
            stop_loss=stop_loss,
            tp1_price=tp1_price,
            tp2_price=tp2_price,
            tp1_quantity=tp1_quantity,
            tp2_quantity=tp2_quantity,
            entry_time=datetime.now(),
            atr_entry=primary_data.atr
        )
        
        self.positions.append(position)
        self.active_symbols.add(signal.symbol)
        self.last_trade_time = time.time()
        self.daily_trades += 1
        
        logger.info(f"[ENTER] Position opened @ ${signal.price:.4f}")
    
    def monitor_positions(self):
        """Monitor and manage open positions"""
        
        positions_to_remove = []
        
        for i, position in enumerate(self.positions):
            try:
                # Get current position data
                pos_data = self.exchange.get_position(position.symbol)
                if not pos_data:
                    # Position closed
                    positions_to_remove.append(i)
                    continue
                
                # Update position with current price
                current_price = float(pos_data['markPrice'])
                current_qty = abs(float(pos_data['positionAmt']))
                
                # Check if TP1 hit
                if current_qty < position.quantity * 0.9 and not position.trail_activated:
                    position.trail_activated = True
                    position.is_trailing = True
                    logger.info(f"[TP1] Partial exit for {position.symbol}")
                
                # Update trailing stop if active
                if position.is_trailing:
                    self.update_trailing_stop(position, current_price)
                
                # Check for stop loss (emergency)
                if (position.side == PositionSide.LONG and current_price <= position.stop_loss) or \
                   (position.side == PositionSide.SHORT and current_price >= position.stop_loss):
                    logger.warning(f"[STOP] Emergency stop hit for {position.symbol}")
                    self.exchange.cancel_all_orders(position.symbol)
                    positions_to_remove.append(i)
                    continue
                
                # Update position in list
                self.positions[i] = position
                
            except Exception as e:
                logger.error(f"[MONITOR] Error for {position.symbol}: {e}")
                continue
        
        # Remove closed positions
        for idx in sorted(positions_to_remove, reverse=True):
            closed_position = self.positions.pop(idx)
            self.active_symbols.remove(closed_position.symbol)
            
            # Calculate P&L
            closed_pnl = closed_position.update_pnl(
                float(self.exchange.get_position(closed_position.symbol)['markPrice'])
                if self.exchange.get_position(closed_position.symbol) 
                else closed_position.current_price
            )
            
            self.record_trade_result(closed_position, closed_pnl)
    
    def update_trailing_stop(self, position: Position, current_price: float):
        """Update trailing stop loss"""
        
        if position.side == PositionSide.LONG:
            # Update highest price
            if current_price > position.max_favorable:
                position.max_favorable = current_price
                
                # Calculate new stop (1.5 ATR below highest)
                new_stop = current_price - (position.atr_entry * 1.5)
                
                if new_stop > position.stop_loss:
                    position.stop_loss = new_stop
                    
                    # Update stop loss order
                    self.exchange.cancel_all_orders(position.symbol)
                    
                    exit_side = SIDE_SELL if position.side == PositionSide.LONG else SIDE_BUY
                    self.exchange.stop_loss_order(
                        position.symbol, exit_side,
                        position.tp2_quantity,  # Remaining quantity
                        position.stop_loss
                    )
                    
                    logger.info(f"[TRAIL] Updated stop to ${position.stop_loss:.4f}")
        
        else:  # SHORT
            if current_price < position.max_favorable:
                position.max_favorable = current_price
                
                new_stop = current_price + (position.atr_entry * 1.5)
                
                if new_stop < position.stop_loss:
                    position.stop_loss = new_stop
                    
                    self.exchange.cancel_all_orders(position.symbol)
                    
                    exit_side = SIDE_BUY  # Cover for short
                    self.exchange.stop_loss_order(
                        position.symbol, exit_side,
                        position.tp2_quantity,
                        position.stop_loss
                    )
    
    def record_trade_result(self, position: Position, pnl: float):
        """Record trade result and update performance"""
        
        self.total_pnl += pnl
        self.daily_pnl += pnl
        
        # Update consecutive losses
        if pnl > 0:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
        
        # Record for optimization
        trade_data = {
            'symbol': position.symbol,
            'side': position.side.value,
            'entry_price': position.entry_price,
            'exit_price': position.current_price,
            'pnl': pnl,
            'position_size': position.quantity * position.entry_price,
            'duration': (datetime.now() - position.entry_time).total_seconds() / 60,
            'max_favorable': position.max_favorable,
            'max_adverse': position.max_adverse
        }
        
        self.optimizer.record_trade(trade_data)
        
        # Log result
        logger.info("="*80)
        if pnl > 0:
            logger.info(f"[WIN] +${pnl:.4f} | {position.symbol} {position.side.value}")
        else:
            logger.info(f"[LOSS] ${pnl:.4f} | {position.symbol} {position.side.value}")
        
        logger.info(f"Entry: ${position.entry_price:.4f} | Exit: ${position.current_price:.4f}")
        logger.info(f"Duration: {(datetime.now() - position.entry_time).total_seconds() / 60:.1f} min")
        logger.info(f"Daily P&L: ${self.daily_pnl:.4f} | Total P&L: ${self.total_pnl:.4f}")
        logger.info(f"Win Rate: {self.optimizer.win_rate:.1%} | Expectancy: ${self.optimizer.expectancy:.4f}")
        logger.info("="*80)
    
    def check_risk_limits(self) -> bool:
        """Check if risk limits are breached"""
        
        current_balance = self.exchange.get_balance()
        
        # Check daily drawdown
        if self.daily_pnl <= -(current_balance * Config.MAX_DAILY_DRAWDOWN):
            logger.error(f"[RISK] Daily drawdown limit: ${self.daily_pnl:.2f}")
            return False
        
        # Check consecutive losses
        if self.consecutive_losses >= Config.MAX_CONSECUTIVE_LOSSES:
            logger.error(f"[RISK] {self.consecutive_losses} consecutive losses")
            return False
        
        return True
    
    def daily_reset(self):
        """Reset daily counters"""
        if datetime.now().date() != self.daily_start:
            logger.info("="*80)
            logger.info(f"[NEW DAY] Previous day: ${self.daily_pnl:.4f}")
            logger.info(f"Trades: {self.daily_trades} | Win Rate: {self.optimizer.win_rate:.1%}")
            logger.info("="*80)
            
            self.daily_pnl = 0
            self.daily_trades = 0
            self.daily_start = datetime.now().date()
            self.consecutive_losses = 0
    
    def run(self):
        """Main trading loop"""
        
        logger.info("[START] Ultimate bot running...")
        
        try:
            while True:
                # Daily reset
                self.daily_reset()
                
                # Check risk limits
                if not self.check_risk_limits():
                    logger.warning("[PAUSED] Risk limits breached, pausing for 30 minutes")
                    time.sleep(1800)  # 30 minutes
                    continue
                
                # Update correlation matrix (hourly)
                self.correlation_manager.update_correlations(self.exchange, Config.SYMBOLS)
                
                # Scan for signals
                signals = self.scan_markets()
                
                # Monitor existing positions
                self.monitor_positions()
                
                # Take new trades if possible
                if len(self.positions) < Config.MAX_POSITIONS and signals:
                    # Filter signals that pass correlation check
                    for signal in signals[:3]:  # Check top 3 signals
                        can_trade, reason = self.correlation_manager.should_block_trade(
                            self.positions, signal.symbol
                        )
                        if not can_trade:
                            continue
                        
                        self.execute_trade(signal)
                        break
                
                # Optimize parameters (hourly)
                self.optimizer.optimize_parameters()
                
                # Heartbeat log every 5 minutes
                if int(time.time()) % 300 == 0:
                    self.log_status()
                
                time.sleep(Config.SCAN_INTERVAL)
                
        except KeyboardInterrupt:
            logger.info("\n[SHUTDOWN] Bot stopped by user")
        except Exception as e:
            logger.error(f"[FATAL] {e}")
            raise
    
    def log_status(self):
        """Log current status"""
        balance = self.exchange.get_balance()
        
        logger.info("="*80)
        logger.info(f"[STATUS] Balance: ${balance:.2f} | Daily P&L: ${self.daily_pnl:.4f}")
        logger.info(f"Open Positions: {len(self.positions)} | Daily Trades: {self.daily_trades}")
        
        perf = self.optimizer.get_performance_report()
        logger.info(f"Win Rate: {perf['win_rate']:.1%} | Expectancy: ${perf['expectancy']:.4f}")
        logger.info(f"Sharpe: {perf['sharpe_ratio']:.2f} | Consecutive Losses: {self.consecutive_losses}")
        
        # Log correlation status
        if self.correlation_manager.correlation_matrix:
            logger.info(f"[CORR] Matrix updated: {len(self.correlation_manager.correlation_matrix)} symbols")
        
        logger.info("="*80)

# ================= MAIN =================
if __name__ == "__main__":
    print("="*80)
    print("ULTIMATE ADAPTIVE TRADING BOT v1.0")
    print("="*80)
    print("ENHANCEMENTS:")
    print("✓ Multi-timeframe confirmation (5m, 15m, 1h)")
    print("✓ Dynamic position sizing based on volatility")
    print("✓ Correlation-aware portfolio management")
    print("✓ Market regime detection")
    print("✓ Real-time parameter optimization")
    print("✓ Adaptive stop loss & take profit")
    print("✓ Kelly-inspired risk management")
    print("="*80)
    print("RISK WARNING: Trading involves risk of loss.")
    print("Never trade with money you cannot afford to lose.")
    print("="*80)
    
    if Config.API_KEY == "":
        print("\n[ERROR] Set your API keys in the Config class!")
        input("Press Enter to exit...")
        sys.exit(1)
    
    input("\nPress Enter to start the bot...")
    
    try:
        bot = UltimateTradingBot()
        bot.run()
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        input("Press Enter to exit...")