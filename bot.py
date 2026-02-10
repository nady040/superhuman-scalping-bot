"""
FINAL WORKING BOT v4.3 - COMPLETE VERSION
Strategy: EMA 9/21 crossover with smart filters
- More trades (8-10 per day target)
- Trailing stops
- Limit order entries
- Dynamic stops
- Simple and effective
"""
import os
from dotenv import load_dotenv
load_dotenv()

import sys
import time
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np
from binance.client import Client
from binance.enums import *

# ================= CONFIG =================
class Config:
    API_KEY = os.getenv("BINANCE_API_KEY")
    API_SECRET = os.getenv("BINANCE_API_SECRET")
    
    # ===== SYMBOLS =====
    SYMBOLS = [
        "XRPUSDT", "ADAUSDT", "DOGEUSDT", "MATICUSDT", 
        "LTCUSDT", "TRXUSDT", "ALGOUSDT", "AVAXUSDT",
    ]
    
    # ===== LEVERAGE =====
    LEVERAGE = 3
    
    # ===== POSITION SIZING =====
    MIN_POSITION_USD = 8
    MAX_POSITION_USD = 25
    
    # ===== RISK MANAGEMENT =====
    BASE_STOP_LOSS_PCT = 0.008
    MIN_STOP_LOSS = 0.006
    MAX_STOP_LOSS = 0.012
    
    TP1_PCT = 0.012
    TP2_PCT = 0.020
    
    # TRAILING STOP
    USE_TRAILING_STOP = True
    TRAILING_STOP_ACTIVATION = 0.005
    TRAILING_STOP_DISTANCE = 0.003
    
    # ===== TRADE FREQUENCY =====
    MAX_TRADES_PER_DAY = 10
    MAX_TRADES_PER_SYMBOL = 3
    COOLDOWN_BETWEEN_TRADES = 30
    MAX_POSITIONS = 3
    
    MAX_DAILY_LOSS_PCT = 0.25
    MAX_CONSECUTIVE_LOSSES = 3
    
    # ===== TECHNICAL SETTINGS =====
    RSI_PERIOD = 14
    EMA_FAST = 9
    EMA_SLOW = 21
    ATR_PERIOD = 14
    
    # ===== SIGNAL SETTINGS =====
    MIN_SCORE = 4.8
    MIN_VOLUME_RATIO = 1.15
    MAX_SPREAD_PCT = 0.0025
    
    # ===== SMART FILTERS =====
    AVOID_HOURS = [2, 3, 4]
    MIN_VOLATILITY_PCT = 0.3
    MIN_TREND_STRENGTH = 0.003
    
    # ===== ORDER SETTINGS =====
    USE_LIMIT_ENTRY = True
    LIMIT_SLIPPAGE = 0.0002
    
    # ===== TIMING =====
    SCAN_INTERVAL = 15
    POSITION_CHECK_INTERVAL = 10
    TRAILING_STOP_UPDATE_INTERVAL = 5

# ================= LOGGING SETUP (MUST BE BEFORE LOGGER USAGE) =================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("active_bot_v4.3.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ================= ENUMS =================
class PositionSide(Enum):
    LONG = "LONG"
    SHORT = "SHORT"

# ================= DATA STRUCTURES =================
@dataclass
class TradingSignal:
    symbol: str
    score: float
    price: float
    side: str
    rsi: float
    volume_ratio: float
    spread_pct: float
    atr_pct: float
    trend_strength: float
    best_bid: float
    best_ask: float
    
    def is_valid(self) -> bool:
        if self.score < Config.MIN_SCORE:
            return False
        if self.volume_ratio < Config.MIN_VOLUME_RATIO:
            return False
        if self.spread_pct > Config.MAX_SPREAD_PCT:
            return False
        if self.atr_pct < Config.MIN_VOLATILITY_PCT:
            return False
        if self.trend_strength < Config.MIN_TREND_STRENGTH:
            return False
        return True

@dataclass
class Position:
    symbol: str
    side: PositionSide
    entry_price: float
    quantity: float
    stop_loss: float
    tp1_price: float
    tp2_price: float
    entry_time: datetime
    tp1_hit: bool = False
    
    trailing_stop_activated: bool = False
    highest_price: float = None
    lowest_price: float = None
    trailing_stop_price: float = None
    original_stop_loss: float = None
    
    def __post_init__(self):
        self.original_stop_loss = self.stop_loss
        if self.side == PositionSide.LONG:
            self.highest_price = self.entry_price
        else:
            self.lowest_price = self.entry_price

# ================= SIMPLE BINANCE CLIENT =================
class SimpleBinanceClient:
    def __init__(self, api_key: str, api_secret: str):
        self.client = Client(api_key, api_secret, {"timeout": 20})
        self.symbol_precision = {}
        self.load_precision()
        self.set_leverage()
    
    def load_precision(self):
        try:
            exchange_info = self.client.futures_exchange_info()
            for symbol_data in exchange_info['symbols']:
                symbol = symbol_data['symbol']
                if symbol in Config.SYMBOLS:
                    self.symbol_precision[symbol] = {
                        'price': symbol_data['pricePrecision'],
                        'quantity': symbol_data['quantityPrecision']
                    }
            logger.info(f"[PRECISION] Loaded {len(self.symbol_precision)} symbols")
        except Exception as e:
            logger.error(f"[PRECISION] Error: {e}")
    
    def set_leverage(self):
        for symbol in Config.SYMBOLS:
            try:
                self.client.futures_change_leverage(
                    symbol=symbol,
                    leverage=Config.LEVERAGE
                )
                logger.info(f"[LEVERAGE] {symbol}: {Config.LEVERAGE}x")
            except:
                pass
    
    def get_balance(self) -> float:
        try:
            account = self.client.futures_account()
            for asset in account['assets']:
                if asset['asset'] == 'USDT':
                    return float(asset['availableBalance'])
        except Exception as e:
            logger.error(f"[BALANCE] Error: {e}")
        return 0.0
    
    def get_klines(self, symbol: str, interval: str = "5m", limit: int = 100) -> Optional[pd.DataFrame]:
        for attempt in range(3):
            try:
                klines = self.client.futures_klines(
                    symbol=symbol, interval=interval, limit=limit
                )
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                return df
            except Exception as e:
                if attempt == 2:
                    logger.error(f"[KLINES] Error for {symbol}: {e}")
                time.sleep(1)
        return None
    
    def get_ticker_price(self, symbol: str) -> float:
        try:
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except:
            return 0.0
    
    def get_order_book(self, symbol: str) -> Optional[Dict]:
        try:
            depth = self.client.futures_order_book(symbol=symbol, limit=5)
            if 'asks' in depth and 'bids' in depth and depth['asks'] and depth['bids']:
                best_ask = float(depth['asks'][0][0])
                best_bid = float(depth['bids'][0][0])
                mid_price = (best_ask + best_bid) / 2
                spread_pct = (best_ask - best_bid) / mid_price
                return {
                    'best_ask': best_ask,
                    'best_bid': best_bid,
                    'mid_price': mid_price,
                    'spread_pct': spread_pct
                }
        except Exception as e:
            logger.error(f"[ORDERBOOK] Error for {symbol}: {e}")
        return None
    
    def place_market_order(self, symbol: str, side: str, quantity: float) -> Optional[Dict]:
        try:
            if symbol not in self.symbol_precision:
                logger.error(f"[ORDER] Unknown symbol: {symbol}")
                return None
            precision = self.symbol_precision[symbol]['quantity']
            quantity = round(quantity, precision)
            if quantity <= 0:
                logger.error(f"[ORDER] Invalid quantity: {quantity}")
                return None
            logger.info(f"[ORDER] Placing {side} market: {quantity} {symbol}")
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type=ORDER_TYPE_MARKET,
                quantity=quantity
            )
            logger.info(f"[ORDER] Success: {order['orderId']}")
            return order
        except Exception as e:
            logger.error(f"[ORDER] Market {side} error: {e}")
            return None
    
    def place_stop_loss_order(self, symbol: str, side: str, quantity: float, stop_price: float) -> bool:
        try:
            if symbol not in self.symbol_precision:
                return False
            q_precision = self.symbol_precision[symbol]['quantity']
            p_precision = self.symbol_precision[symbol]['price']
            quantity = round(quantity, q_precision)
            stop_price = round(stop_price, p_precision)
            self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type=FUTURE_ORDER_TYPE_STOP_MARKET,
                stopPrice=stop_price,
                quantity=quantity,
                reduceOnly=True
            )
            logger.info(f"[STOP] Placed at ${stop_price:.6f}")
            return True
        except Exception as e:
            logger.error(f"[STOP] Error: {e}")
            return False
    
    def update_stop_loss_order(self, symbol: str, quantity: float, stop_price: float) -> bool:
        try:
            self.cancel_all_orders(symbol)
            side = SIDE_SELL
            pos_data = self.get_position(symbol)
            if pos_data and pos_data['amount'] < 0:
                side = SIDE_BUY
            return self.place_stop_loss_order(symbol, side, quantity, stop_price)
        except Exception as e:
            logger.error(f"[UPDATE STOP] Error: {e}")
            return False
    
    def place_limit_order(self, symbol: str, side: str, quantity: float, price: float) -> bool:
        try:
            if symbol not in self.symbol_precision:
                return False
            q_precision = self.symbol_precision[symbol]['quantity']
            p_precision = self.symbol_precision[symbol]['price']
            quantity = round(quantity, q_precision)
            price = round(price, p_precision)
            self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type=ORDER_TYPE_LIMIT,
                timeInForce=TIME_IN_FORCE_GTC,
                price=price,
                quantity=quantity,
                reduceOnly=True
            )
            logger.info(f"[LIMIT] {side} at ${price:.6f}")
            return True
        except Exception as e:
            logger.error(f"[LIMIT] Error: {e}")
            return False
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        try:
            positions = self.client.futures_position_information(symbol=symbol)
            for pos in positions:
                if pos['symbol'] == symbol:
                    amount = float(pos['positionAmt'])
                    if abs(amount) > 0.0001:
                        return {
                            'symbol': symbol,
                            'amount': amount,
                            'entry_price': float(pos['entryPrice']),
                            'mark_price': float(pos['markPrice']),
                            'pnl': float(pos['unRealizedProfit'])
                        }
            return None
        except Exception as e:
            logger.error(f"[POSITION] Error: {e}")
            return None
    
    def cancel_all_orders(self, symbol: str):
        try:
            self.client.futures_cancel_all_open_orders(symbol=symbol)
            logger.info(f"[CANCEL] All orders for {symbol}")
        except:
            pass

# ================= PERFORMANCE TRACKER =================
class PerformanceTracker:
    def __init__(self, starting_balance: float):
        self.starting_balance = starting_balance
        self.current_balance = starting_balance
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.trades_today = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.consecutive_losses = 0
        self.day_start = datetime.now(timezone.utc).date()
        self.symbol_trades_today = {}
    
    def can_trade(self) -> bool:
        return self.trades_today < Config.MAX_TRADES_PER_DAY
    
    def can_trade_symbol(self, symbol: str) -> bool:
        trades = self.symbol_trades_today.get(symbol, 0)
        return trades < Config.MAX_TRADES_PER_SYMBOL
    
    def record_trade(self, symbol: str, pnl: float):
        self.trades_today += 1
        self.total_trades += 1
        self.daily_pnl += pnl
        self.total_pnl += pnl
        self.current_balance += pnl
        self.symbol_trades_today[symbol] = self.symbol_trades_today.get(symbol, 0) + 1
        
        if pnl > 0:
            self.winning_trades += 1
            self.consecutive_losses = 0
            logger.info("="*60)
            logger.info(f"[WIN] +${pnl:.2f} | {symbol}")
        else:
            self.consecutive_losses += 1
            logger.info("="*60)
            logger.info(f"[LOSS] ${pnl:.2f} | {symbol}")
        
        logger.info(f"Daily: ${self.daily_pnl:.2f} | Total: ${self.total_pnl:.2f}")
        logger.info(f"Balance: ${self.current_balance:.2f}")
        logger.info(f"Win Rate: {self.get_win_rate():.1f}%")
        logger.info("="*60)
    
    def get_win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100
    
    def should_stop_trading(self) -> Tuple[bool, str]:
        current_date = datetime.now(timezone.utc).date()
        if current_date != self.day_start:
            self.daily_reset()
            return False, ""
        daily_loss_pct = abs(self.daily_pnl) / self.starting_balance
        if self.daily_pnl < 0 and daily_loss_pct >= Config.MAX_DAILY_LOSS_PCT:
            return True, f"Daily loss: {daily_loss_pct:.1%}"
        if self.consecutive_losses >= Config.MAX_CONSECUTIVE_LOSSES:
            return True, f"{self.consecutive_losses} consecutive losses"
        return False, ""
    
    def daily_reset(self):
        logger.info("="*60)
        logger.info(f"[NEW DAY] Previous: ${self.daily_pnl:.2f}")
        logger.info(f"Trades: {self.trades_today} | Win Rate: {self.get_win_rate():.1f}%")
        logger.info("="*60)
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.symbol_trades_today = {}
        self.consecutive_losses = 0
        self.day_start = datetime.now(timezone.utc).date()
        self.starting_balance = self.current_balance

# ================= OPTIMIZED ANALYZER =================
class OptimizedMarketAnalyzer:
    def __init__(self, exchange_client):
        self.exchange = exchange_client
    
    def calculate_atr(self, df, period=14):
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr
    
    def calculate_ema(self, series, period):
        return series.ewm(span=period, adjust=False).mean()
    
    def calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
        loss = (-delta).clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def get_multi_timeframe_score(self, symbol: str) -> float:
        try:
            df_15m = self.exchange.get_klines(symbol, "15m", 30)
            if df_15m is None or len(df_15m) < 15:
                return 1.0
            current_price = df_15m['close'].iloc[-1]
            price_2_bars_ago = df_15m['close'].iloc[-3] if len(df_15m) >= 3 else current_price
            df_5m = self.exchange.get_klines(symbol, "5m", 10)
            if df_5m is not None and len(df_5m) > 5:
                current_price_5m = df_5m['close'].iloc[-1]
                price_2_bars_ago_5m = df_5m['close'].iloc[-3] if len(df_5m) >= 3 else current_price_5m
                trend_15m = current_price > price_2_bars_ago
                trend_5m = current_price_5m > price_2_bars_ago_5m
                if trend_15m == trend_5m:
                    return 1.1
                else:
                    return 0.9
            return 1.0
        except Exception as e:
            logger.debug(f"[QUICK MTF] {symbol}: {e}")
            return 1.0
    
    def analyze_symbol(self, symbol: str) -> Optional[TradingSignal]:
        current_hour = datetime.now(timezone.utc).hour
        if current_hour in Config.AVOID_HOURS:
            logger.debug(f"[TIME] Avoiding {symbol} hour {current_hour}")
            return None
        
        try:
            df = self.exchange.get_klines(symbol, "5m", 50)
            if df is None or len(df) < 30:
                return None
            order_book = self.exchange.get_order_book(symbol)
            if not order_book:
                return None
            price = order_book['mid_price']
            spread_pct = order_book['spread_pct']
            best_bid = order_book['best_bid']
            best_ask = order_book['best_ask']
            if price <= 0:
                return None
            if spread_pct > Config.MAX_SPREAD_PCT:
                logger.debug(f"[SPREAD] {symbol}: {spread_pct:.4%}")
                return None
            
            idx = -1
            rsi = self.calculate_rsi(df['close'], Config.RSI_PERIOD).iloc[idx]
            ema_fast = self.calculate_ema(df['close'], Config.EMA_FAST).iloc[idx]
            ema_slow = self.calculate_ema(df['close'], Config.EMA_SLOW).iloc[idx]
            high_low_range = (df['high'].iloc[idx] - df['low'].iloc[idx]) / price * 100
            atr_pct = high_low_range
            trend_strength = abs(ema_fast - ema_slow) / ema_slow
            volume = df['volume'].iloc[idx]
            volume_ma = df['volume'].rolling(10).mean().iloc[idx]
            volume_ratio = volume / volume_ma if volume_ma > 0 else 1.0
            price_5_bars_ago = df['close'].iloc[idx-5] if idx >= 5 else price
            price_change_5 = (price / price_5_bars_ago - 1) * 100
            
            score = 0
            side = "NEUTRAL"
            mtf_score = self.get_multi_timeframe_score(symbol)
            
            if ema_fast > ema_slow * 1.001:
                side = "BUY"
                score += 3
                if rsi < 70:
                    score += 2
                if rsi < 65:
                    score += 1
                if price_change_5 > 0.1:
                    score += 1
            elif ema_fast < ema_slow * 0.999:
                side = "SELL"
                score += 3
                if rsi > 30:
                    score += 2
                if rsi > 35:
                    score += 1
                if price_change_5 < -0.1:
                    score += 1
            
            score *= mtf_score
            if volume_ratio > 1.4:
                score += 2
            elif volume_ratio > 1.1:
                score += 1
            if trend_strength > 0.008:
                score += 1
            
            if side == "NEUTRAL" or score < Config.MIN_SCORE:
                return None
            
            signal = TradingSignal(
                symbol=symbol,
                score=score,
                price=price,
                side=side,
                rsi=rsi,
                volume_ratio=volume_ratio,
                spread_pct=spread_pct,
                atr_pct=atr_pct,
                trend_strength=trend_strength,
                best_bid=best_bid,
                best_ask=best_ask
            )
            
            return signal if signal.is_valid() else None
            
        except Exception as e:
            logger.debug(f"[ANALYZER] {symbol}: {e}")
            return None

# ================= ACTIVE TRADING BOT =================
class ActiveTradingBot:
    def __init__(self):
        self.exchange = SimpleBinanceClient(Config.API_KEY, Config.API_SECRET)
        self.analyzer = OptimizedMarketAnalyzer(self.exchange)
        
        starting_balance = self.exchange.get_balance()
        self.performance = PerformanceTracker(starting_balance)
        
        self.positions = []
        self.last_trade_time = 0
        self.last_trailing_check = 0
        
        logger.info("="*60)
        logger.info("ACTIVE BOT v4.3 - COMPLETE VERSION")
        logger.info("="*60)
        logger.info(f"Starting Balance: ${starting_balance:.2f}")
        logger.info(f"Max Trades/Day: {Config.MAX_TRADES_PER_DAY}")
        logger.info(f"Max Positions: {Config.MAX_POSITIONS}")
        logger.info("="*60)
    
    def calculate_position_size(self, signal: TradingSignal) -> float:
        balance = self.exchange.get_balance()
        position_value = min(Config.MAX_POSITION_USD, balance * 0.85)
        position_value = max(Config.MIN_POSITION_USD, position_value)
        quantity = position_value / signal.price
        if signal.symbol in self.exchange.symbol_precision:
            precision = self.exchange.symbol_precision[signal.symbol]['quantity']
            quantity = round(quantity, precision)
        if quantity <= 0:
            return 0
        return quantity
    
    def calculate_dynamic_stop_loss(self, entry_price: float, atr_pct: float) -> float:
        base_stop = Config.BASE_STOP_LOSS_PCT
        if atr_pct < 0.4:
            dynamic_stop = base_stop * 0.8
        elif atr_pct > 1.0:
            dynamic_stop = base_stop * 1.3
        else:
            dynamic_stop = base_stop
        dynamic_stop = max(Config.MIN_STOP_LOSS, min(Config.MAX_STOP_LOSS, dynamic_stop))
        return dynamic_stop
    
    def update_trailing_stops(self):
        current_time = time.time()
        if current_time - self.last_trailing_check < Config.TRAILING_STOP_UPDATE_INTERVAL:
            return
        self.last_trailing_check = current_time
        
        if not Config.USE_TRAILING_STOP:
            return
        
        for i, position in enumerate(self.positions):
            try:
                current_price = self.exchange.get_ticker_price(position.symbol)
                if current_price <= 0:
                    continue
                
                if position.side == PositionSide.LONG:
                    if current_price > position.highest_price:
                        position.highest_price = current_price
                    profit_pct = (current_price - position.entry_price) / position.entry_price
                    
                    if not position.trailing_stop_activated and profit_pct >= Config.TRAILING_STOP_ACTIVATION:
                        position.trailing_stop_activated = True
                        position.trailing_stop_price = current_price * (1 - Config.TRAILING_STOP_DISTANCE)
                        logger.info(f"[TRAILING] Activated for {position.symbol} @ ${current_price:.6f}")
                    
                    if position.trailing_stop_activated:
                        new_trailing_stop = current_price * (1 - Config.TRAILING_STOP_DISTANCE)
                        if new_trailing_stop > position.trailing_stop_price:
                            position.trailing_stop_price = new_trailing_stop
                            if position.trailing_stop_price > position.stop_loss:
                                position.stop_loss = position.trailing_stop_price
                                self.exchange.update_stop_loss_order(
                                    position.symbol,
                                    position.quantity,
                                    position.stop_loss
                                )
                                logger.info(f"[TRAILING] {position.symbol} moved to ${position.stop_loss:.6f}")
                
                else:
                    if current_price < position.lowest_price:
                        position.lowest_price = current_price
                    profit_pct = (position.entry_price - current_price) / position.entry_price
                    
                    if not position.trailing_stop_activated and profit_pct >= Config.TRAILING_STOP_ACTIVATION:
                        position.trailing_stop_activated = True
                        position.trailing_stop_price = current_price * (1 + Config.TRAILING_STOP_DISTANCE)
                        logger.info(f"[TRAILING] Activated for {position.symbol} @ ${current_price:.6f}")
                    
                    if position.trailing_stop_activated:
                        new_trailing_stop = current_price * (1 + Config.TRAILING_STOP_DISTANCE)
                        if new_trailing_stop < position.trailing_stop_price:
                            position.trailing_stop_price = new_trailing_stop
                            if position.trailing_stop_price < position.stop_loss:
                                position.stop_loss = position.trailing_stop_price
                                self.exchange.update_stop_loss_order(
                                    position.symbol,
                                    position.quantity,
                                    position.stop_loss
                                )
                                logger.info(f"[TRAILING] {position.symbol} moved to ${position.stop_loss:.6f}")
                
                self.positions[i] = position
                
            except Exception as e:
                logger.error(f"[TRAILING] Error for {position.symbol}: {e}")
                continue
    
    def scan_markets(self) -> List[TradingSignal]:
        signals = []
        for symbol in Config.SYMBOLS:
            if any(pos.symbol == symbol for pos in self.positions):
                continue
            signal = self.analyzer.analyze_symbol(symbol)
            if signal and signal.is_valid():
                signals.append(signal)
                logger.info(f"[SCAN] {signal.symbol} {signal.side} | Score: {signal.score:.1f} | RSI: {signal.rsi:.1f}")
        signals.sort(key=lambda s: s.score, reverse=True)
        return signals
    
    def execute_trade(self, signal: TradingSignal):
        if not self.performance.can_trade():
            logger.warning(f"[LIMIT] Max daily trades: {self.performance.trades_today}")
            return
        if not self.performance.can_trade_symbol(signal.symbol):
            logger.info(f"[LIMIT] Max trades for {signal.symbol} today")
            return
        if time.time() - self.last_trade_time < Config.COOLDOWN_BETWEEN_TRADES:
            logger.info(f"[COOLDOWN] Waiting {Config.COOLDOWN_BETWEEN_TRADES}s")
            return
        if len(self.positions) >= Config.MAX_POSITIONS:
            logger.info(f"[LIMIT] Max positions: {Config.MAX_POSITIONS}")
            return
        
        balance = self.exchange.get_balance()
        if balance < Config.MIN_POSITION_USD:
            logger.warning(f"[BALANCE] Too low: ${balance:.2f}")
            return
        
        quantity = self.calculate_position_size(signal)
        if quantity <= 0:
            logger.error(f"[SIZE] Invalid quantity")
            return
        
        stop_loss_pct = self.calculate_dynamic_stop_loss(signal.price, signal.atr_pct)
        
        if signal.side == "BUY":
            order_side = SIDE_BUY
            position_side = PositionSide.LONG
            exit_side = SIDE_SELL
            
            if Config.USE_LIMIT_ENTRY:
                entry_price = signal.best_ask * (1 + Config.LIMIT_SLIPPAGE)
                logger.info(f"[LIMIT ENTRY] Buying at ${entry_price:.6f}")
            else:
                entry_price = signal.price
            
            stop_price = entry_price * (1 - stop_loss_pct)
            tp1_price = entry_price * (1 + Config.TP1_PCT)
            tp2_price = entry_price * (1 + Config.TP2_PCT)
        else:
            order_side = SIDE_SELL
            position_side = PositionSide.SHORT
            exit_side = SIDE_BUY
            
            if Config.USE_LIMIT_ENTRY:
                entry_price = signal.best_bid * (1 - Config.LIMIT_SLIPPAGE)
                logger.info(f"[LIMIT ENTRY] Selling at ${entry_price:.6f}")
            else:
                entry_price = signal.price
            
            stop_price = entry_price * (1 + stop_loss_pct)
            tp1_price = entry_price * (1 - Config.TP1_PCT)
            tp2_price = entry_price * (1 - Config.TP2_PCT)
        
        logger.info("="*60)
        logger.info(f"[SIGNAL] {signal.symbol} {signal.side}")
        logger.info(f"Score: {signal.score:.1f} | RSI: {signal.rsi:.1f}")
        logger.info(f"ATR: {signal.atr_pct:.2f}% | Trend: {signal.trend_strength:.3%}")
        logger.info(f"Entry: ${entry_price:.6f}")
        logger.info(f"Stop: ${stop_price:.6f} ({stop_loss_pct:.3%})")
        logger.info(f"TP1: ${tp1_price:.6f} ({Config.TP1_PCT:.1%})")
        logger.info(f"TP2: ${tp2_price:.6f} ({Config.TP2_PCT:.1%})")
        logger.info("="*60)
        
        if Config.USE_LIMIT_ENTRY:
            order_placed = self.exchange.place_limit_order(
                signal.symbol, order_side, quantity, entry_price
            )
            if not order_placed:
                logger.warning("[LIMIT] Failed, falling back to market")
                market_order = self.exchange.place_market_order(
                    signal.symbol, order_side, quantity
                )
                if not market_order:
                    return
                pos_data = self.exchange.get_position(signal.symbol)
                if pos_data:
                    entry_price = pos_data['entry_price']
        else:
            market_order = self.exchange.place_market_order(
                signal.symbol, order_side, quantity
            )
            if not market_order:
                return
        
        if not self.exchange.place_stop_loss_order(
            signal.symbol, exit_side, quantity, stop_price
        ):
            logger.error("[ERROR] Stop loss failed")
            self.exchange.place_market_order(signal.symbol, exit_side, quantity)
            return
        
        self.exchange.place_limit_order(
            signal.symbol, exit_side, quantity * 0.6, tp1_price
        )
        self.exchange.place_limit_order(
            signal.symbol, exit_side, quantity * 0.4, tp2_price
        )
        
        position = Position(
            symbol=signal.symbol,
            side=position_side,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=stop_price,
            tp1_price=tp1_price,
            tp2_price=tp2_price,
            entry_time=datetime.now(timezone.utc)
        )
        
        self.positions.append(position)
        self.last_trade_time = time.time()
        logger.info(f"[SUCCESS] Position opened @ ${entry_price:.6f}")
    
    def monitor_positions(self):
        positions_to_remove = []
        for i, position in enumerate(self.positions):
            try:
                pos_data = self.exchange.get_position(position.symbol)
                if not pos_data:
                    positions_to_remove.append(i)
                    continue
                current_qty = abs(pos_data['amount'])
                if current_qty < position.quantity * 0.9 and not position.tp1_hit:
                    position.tp1_hit = True
                    logger.info(f"[TP1] Hit for {position.symbol}")
                self.positions[i] = position
            except Exception as e:
                logger.error(f"[MONITOR] Error: {e}")
                continue
        
        for idx in sorted(positions_to_remove, reverse=True):
            closed_pos = self.positions.pop(idx)
            pos_data = self.exchange.get_position(closed_pos.symbol)
            if pos_data:
                pnl = pos_data['pnl']
            else:
                current_price = self.exchange.get_ticker_price(closed_pos.symbol)
                if closed_pos.side == PositionSide.LONG:
                    pnl = (current_price - closed_pos.entry_price) * closed_pos.quantity
                else:
                    pnl = (closed_pos.entry_price - current_price) * closed_pos.quantity
            
            self.performance.record_trade(closed_pos.symbol, pnl)
            self.exchange.cancel_all_orders(closed_pos.symbol)
    
    def run(self):
        logger.info("[START] Bot running...")
        last_status = 0
        
        try:
            while True:
                should_stop, reason = self.performance.should_stop_trading()
                if should_stop:
                    logger.warning(f"[PAUSE] {reason} - pausing for 30 min")
                    for position in self.positions:
                        self.exchange.cancel_all_orders(position.symbol)
                    time.sleep(1800)
                    self.performance.daily_reset()
                    continue
                
                self.update_trailing_stops()
                signals = self.scan_markets()
                self.monitor_positions()
                
                if (self.performance.can_trade() and
                    len(self.positions) < Config.MAX_POSITIONS and 
                    signals):
                    taken = 0
                    for signal in signals:
                        if taken >= 2:
                            break
                        self.execute_trade(signal)
                        taken += 1
                
                current_time = time.time()
                if current_time - last_status > 45:
                    self.log_status()
                    last_status = current_time
                
                time.sleep(Config.SCAN_INTERVAL)
                
        except KeyboardInterrupt:
            logger.info("\n[STOP] Bot stopped by user")
            for position in self.positions:
                self.exchange.cancel_all_orders(position.symbol)
        except Exception as e:
            logger.error(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()
    
    def log_status(self):
        balance = self.exchange.get_balance()
        logger.info("="*60)
        logger.info(f"[STATUS] Balance: ${balance:.2f}")
        logger.info(f"Daily P&L: ${self.performance.daily_pnl:.2f}")
        logger.info(f"Trades Today: {self.performance.trades_today}/{Config.MAX_TRADES_PER_DAY}")
        logger.info(f"Open Positions: {len(self.positions)}/{Config.MAX_POSITIONS}")
        logger.info(f"Win Rate: {self.performance.get_win_rate():.1f}%")
        logger.info("="*60)

# ================= MAIN =================
if __name__ == "__main__":
    print("="*60)
    print("ACTIVE BOT v4.3 - COMPLETE VERSION")
    print("="*60)
    print("Strategy: EMA 9/21 Crossover")
    print("Target: 8-10 trades per day")
    print(f"Max Trades/Day: {Config.MAX_TRADES_PER_DAY}")
    print(f"Max Positions: {Config.MAX_POSITIONS}")
    print("="*60)
    
    if Config.API_KEY == "" or Config.API_SECRET == "":
        print("\n[ERROR] Set your API keys!")
        input("Press Enter to exit...")
        sys.exit(1)
    
    input("\nPress Enter to start trading...")
    
    try:
        bot = ActiveTradingBot()
        bot.run()
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        input("Press Enter to exit...")
