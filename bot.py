"""
SUPERHUMAN SCALPING BOT v2.1
Heartbeat + Debug Logs + Network Safe
STRATEGY UNCHANGED
"""

import sys
import time
import logging
from datetime import datetime
from typing import Optional
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import requests
from binance.client import Client
from binance.enums import *

# ================= WINDOWS FIX =================
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# ================= LOGGING =================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("scalping_bot.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ================= CONFIG =================
class Config:
    API_KEY = ""
    API_SECRET = ""

    SYMBOL = "ETHUSDT"
    LEVERAGE = 3

    POSITION_SIZE_PCT = 0.95
    MIN_POSITION_SIZE = 0.01
    MAX_POSITION_SIZE = 0.02

    SCALP_PROFIT_PCT = 0.006
    SCALP_STOP_PCT = 0.004

    RSI_PERIOD = 14
    ATR_PERIOD = 14
    VOLUME_MA_PERIOD = 20

    TREND_THRESHOLD = 0.0015
    SIDEWAYS_THRESHOLD = 0.0005

    MIN_BUY_SCORE = 4
    MIN_SELL_SCORE = 4

    SIGNAL_CHECK_INTERVAL = 5

    # Safety
    MAX_ATR_PCT = 2.0
    EMERGENCY_COOLDOWN = 600


# ================= ENUMS =================
class MarketCondition(Enum):
    SIDEWAYS = "SIDEWAYS"
    TRENDING = "TRENDING"
    VOLATILE = "VOLATILE"


class BotState(Enum):
    IDLE = "IDLE"
    IN_POSITION = "IN_POSITION"


class PositionSide(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


# ================= DATA =================
@dataclass
class TradingSignals:
    buy_score: int
    sell_score: int
    price: float
    market_condition: MarketCondition
    rsi: float
    volume_ratio: float
    atr_pct: float


@dataclass
class Position:
    side: PositionSide
    quantity: float
    entry_price: float
    entry_time: datetime


# ================= EXCHANGE =================
class BinanceFuturesClient:
    def __init__(self, api_key, api_secret):
        self.client = Client(api_key, api_secret)
        self.client.futures_change_leverage(
            symbol=Config.SYMBOL,
            leverage=Config.LEVERAGE
        )

    def get_balance(self):
        try:
            for a in self.client.futures_account()["assets"]:
                if a["asset"] == "USDT":
                    return float(a["availableBalance"])
        except Exception as e:
            logger.error(f"Balance fetch error: {e}")
        return 0.0

    def get_klines(self):
        try:
            kl = self.client.futures_klines(
                symbol=Config.SYMBOL,
                interval="1m",
                limit=100
            )

            df = pd.DataFrame(kl, columns=[
                "time", "open", "high", "low", "close", "volume",
                "ct", "qv", "n", "tb", "tq", "i"
            ])
            df[["open", "high", "low", "close", "volume"]] = df[
                ["open", "high", "low", "close", "volume"]
            ].astype(float)

            return df

        except requests.exceptions.ReadTimeout:
            logger.warning("⏳ Binance timeout (klines)")
            return None

        except Exception as e:
            logger.error(f"❌ Kline error: {e}")
            return None

    def market_order(self, side, qty):
        try:
            return self.client.futures_create_order(
                symbol=Config.SYMBOL,
                side=side,
                type=ORDER_TYPE_MARKET,
                quantity=round(qty, 3)
            )
        except Exception as e:
            logger.error(f"Order error: {e}")
            return None


# ================= INDICATORS =================
def calculate_rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


# ================= BOT =================
class SuperhumanScalpingBot:
    def __init__(self):
        self.ex = BinanceFuturesClient(Config.API_KEY, Config.API_SECRET)
        self.state = BotState.IDLE
        self.position: Optional[Position] = None
        self.last_heartbeat = 0

    def detect_market(self, df):
        vol = df["close"].pct_change().abs().rolling(20).mean().iloc[-2]
        if vol > Config.TREND_THRESHOLD:
            return MarketCondition.TRENDING
        if vol < Config.SIDEWAYS_THRESHOLD:
            return MarketCondition.SIDEWAYS
        return MarketCondition.VOLATILE

    def generate_signals(self, df) -> TradingSignals:
        idx = -2
        price = df["close"].iloc[idx]

        rsi_val = calculate_rsi(df["close"], Config.RSI_PERIOD).iloc[idx]

        atr = (df["high"] - df["low"]).rolling(
            Config.ATR_PERIOD
        ).mean().iloc[idx]
        atr_pct = atr / price * 100

        vol_ratio = (
            df["volume"].iloc[idx] /
            df["volume"].rolling(Config.VOLUME_MA_PERIOD).mean().iloc[idx]
        )

        buy = sell = 0

        if rsi_val < 30:
            buy += 2
        if rsi_val > 70:
            sell += 2

        if vol_ratio > 1.2:
            buy += 1
            sell += 1

        return TradingSignals(
            buy_score=buy,
            sell_score=sell,
            price=price,
            market_condition=self.detect_market(df),
            rsi=rsi_val,
            volume_ratio=vol_ratio,
            atr_pct=atr_pct
        )

    def run(self):
        logger.info("🚀 SUPERHUMAN BOT STARTED")

        while True:
            df = self.ex.get_klines()

            if df is None or len(df) < 50:
                time.sleep(10)
                continue

            sig = self.generate_signals(df)
            now = time.time()

            # ❤️ HEARTBEAT every 15 sec
            if now - self.last_heartbeat >= 15:
                logger.info(
                    f"❤️ Alive | Price: {sig.price:.2f} | "
                    f"RSI: {sig.rsi:.1f} | "
                    f"ATR: {sig.atr_pct:.2f}% | "
                    f"Vol: {sig.volume_ratio:.2f}x | "
                    f"Buy: {sig.buy_score} Sell: {sig.sell_score} | "
                    f"Market: {sig.market_condition.value}"
                )
                self.last_heartbeat = now

            # 🚨 Volatility kill switch
            if sig.atr_pct > Config.MAX_ATR_PCT:
                logger.warning("⚠️ Extreme volatility — paused")
                time.sleep(Config.EMERGENCY_COOLDOWN)
                continue

            if self.state == BotState.IDLE:
                if sig.buy_score >= Config.MIN_BUY_SCORE:
                    logger.info("🟢 BUY conditions met")
                    self.enter(SIDE_BUY, sig)

                elif sig.sell_score >= Config.MIN_SELL_SCORE:
                    logger.info("🔴 SELL conditions met")
                    self.enter(SIDE_SELL, sig)

                else:
                    logger.info(
                        f"🟡 No trade | Buy={sig.buy_score} "
                        f"Sell={sig.sell_score} "
                        f"(min={Config.MIN_BUY_SCORE})"
                    )

            time.sleep(Config.SIGNAL_CHECK_INTERVAL)

    def enter(self, side, sig):
        balance = self.ex.get_balance()
        if balance <= 0:
            logger.error("❌ No balance")
            return

        qty = min(
            Config.MAX_POSITION_SIZE,
            (balance * Config.POSITION_SIZE_PCT) / sig.price
        )

        order = self.ex.market_order(side, qty)
        if not order:
            return

        self.position = Position(
            side=PositionSide.LONG if side == SIDE_BUY else PositionSide.SHORT,
            quantity=qty,
            entry_price=sig.price,
            entry_time=datetime.utcnow()
        )

        self.state = BotState.IN_POSITION
        logger.info(f"✅ ENTERED {side} @ {sig.price:.2f}")


# ================= MAIN =================
if __name__ == "__main__":
    bot = SuperhumanScalpingBot()
    bot.run()
