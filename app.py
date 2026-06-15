from __future__ import annotations

import csv
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


st.set_page_config(page_title="AI进场助手", page_icon="AI", layout="wide")

APP_VERSION = "direction-okx-pnl-1-2026-06-15"
TRADE_RECORDS_FILE = Path(__file__).with_name("trade_records.csv")
if st.session_state.get("app_version") != APP_VERSION:
    for key in [
        "daily_plan",
        "forced_plan",
        "signals",
        "coin",
        "unit",
        "order_helper_result",
        "order_helper_inputs",
        "long_short_scores",
        "risk_inputs",
        "selected_plan",
    ]:
        st.session_state.pop(key, None)
    st.session_state.app_version = APP_VERSION

COIN_OPTIONS = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
    "BNB": "BNB-USD",
    "XRP": "XRP-USD",
    "DOGE": "DOGE-USD",
    "ADA": "ADA-USD",
}

TIMEFRAMES = {
    "15分钟": {"interval": "15m", "period": "60d"},
    "1小时": {"interval": "1h", "period": "730d"},
    "4小时": {"interval": "1h", "period": "730d", "resample": "4h"},
    "日线": {"interval": "1d", "period": "2y"},
}


@dataclass
class TimeframeSignal:
    name: str
    price: float
    ma20: float
    ma60: float
    support: float
    resistance: float
    previous_high: float
    previous_low: float
    rsi: float
    macd_hist: float
    direction: str
    score: int


@dataclass
class TradeSetup:
    title: str
    wait_area: str
    trigger: str
    entry: float
    stop: float
    target1: float
    target2: float
    reward_risk: float
    risk_level: str
    max_position_usdt: float


@dataclass
class DailyPlan:
    symbol: str
    one_liner: str
    market_state: str
    state_color: str
    state_reason: str
    aggressive: TradeSetup
    conservative: TradeSetup
    donts: list[str]
    max_loss: float
    recommended_leverage: int
    max_position_usdt: float
    analysis_reason: str
    current_price: float


@dataclass
class ForcedPlan:
    title: str
    source: str
    side: str
    order_type: str
    entry: float
    stop: float
    target1: float
    target2: float
    leverage: int
    quantity: float
    position_usdt: float
    margin_usdt: float
    max_loss_usdt: float
    loss_rmb: float
    profit1_usdt: float
    profit1_rmb: float
    profit2_usdt: float
    profit2_rmb: float
    reward_risk: float
    warning: str


@dataclass
class OrderHelperResult:
    usdt_capital: float
    max_loss_usdt: float
    position_usdt: float
    quantity: float
    margin_usdt: float
    target1: float
    target2: float
    estimated_loss: float
    profit1_usdt: float
    profit2_usdt: float
    open_fee_usdt: float
    close_fee_stop_usdt: float
    close_fee_target1_usdt: float
    close_fee_target2_usdt: float
    total_fee_stop_usdt: float
    total_fee_target1_usdt: float
    total_fee_target2_usdt: float
    net_loss_stop_usdt: float
    net_profit1_usdt: float
    net_profit2_usdt: float
    risk_reward_1: float
    risk_reward_2: float
    risk_too_high: bool
    warning: str


def symbol_to_yfinance(symbol: str) -> tuple[str, str]:
    cleaned = symbol.strip().upper().replace(" ", "")
    display = cleaned
    cleaned = cleaned.replace("/", "-")
    base = cleaned.split("-")[0].replace("USDT", "").replace("USD", "")
    if base in COIN_OPTIONS:
        return COIN_OPTIONS[base], f"{base}/USDT"
    if cleaned.endswith("-USDT"):
        return cleaned.replace("-USDT", "-USD"), display.replace("-", "/")
    if cleaned.endswith("USDT"):
        return cleaned.replace("USDT", "-USD"), f"{base}/USDT"
    if cleaned.endswith("-USD"):
        return cleaned, display.replace("-USD", "/USDT")
    return f"{cleaned}-USD", f"{cleaned}/USDT"


def base_coin(symbol: str) -> str:
    return symbol_to_yfinance(symbol)[1].split("/")[0]


@st.cache_data(ttl=120, show_spinner=False)
def load_market_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    data = yf.download(tickers=ticker, period=period, interval=interval, auto_adjust=False, progress=False, threads=False)
    if data.empty:
        return data
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.rename(columns=str.title)
    cols = ["Open", "High", "Low", "Close", "Volume"]
    data = data[[col for col in cols if col in data.columns]].dropna()
    data.index = pd.to_datetime(data.index)
    return data


def load_all_timeframes(ticker: str) -> dict[str, pd.DataFrame]:
    result = {}
    for name, cfg in TIMEFRAMES.items():
        raw = load_market_data(ticker, cfg["period"], cfg["interval"])
        if raw.empty:
            result[name] = raw
            continue
        if cfg.get("resample") == "4h":
            raw = raw.resample("4h").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
        result[name] = add_indicators(raw)
    return result


def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()
    df["RSI14"] = calculate_rsi(df["Close"])
    macd, signal, hist = calculate_macd(df["Close"])
    df["MACD"] = macd
    df["MACD_SIGNAL"] = signal
    df["MACD_HIST"] = hist
    return df


def calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def calculate_macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal, macd - signal


def support_resistance(df: pd.DataFrame, window: int = 80) -> tuple[float, float, float, float]:
    recent = df.tail(window)
    current = float(recent["Close"].iloc[-1])
    lows = recent["Low"]
    highs = recent["High"]
    below = lows[lows <= current]
    above = highs[highs >= current]
    support = float(below.quantile(0.25)) if not below.empty else float(lows.min())
    resistance = float(above.quantile(0.75)) if not above.empty else float(highs.max())
    previous_high = float(highs.iloc[:-1].max()) if len(highs) > 1 else float(highs.max())
    previous_low = float(lows.iloc[:-1].min()) if len(lows) > 1 else float(lows.min())
    return support, resistance, previous_high, previous_low


def analyze_timeframe(name: str, df: pd.DataFrame) -> TimeframeSignal | None:
    if df.empty or len(df) < 70:
        return None
    latest = df.iloc[-1]
    price = float(latest["Close"])
    ma20 = float(latest["MA20"])
    ma60 = float(latest["MA60"])
    rsi = float(latest["RSI14"])
    macd_hist = float(latest["MACD_HIST"])
    support, resistance, previous_high, previous_low = support_resistance(df)
    score = 50
    score += 12 if price > ma20 else -12
    score += 12 if price > ma60 else -12
    score += 10 if ma20 > ma60 else -8
    score += 8 if macd_hist > 0 else -8
    score += 6 if 42 <= rsi <= 68 else -8 if rsi < 30 else -12 if rsi > 75 else 0
    if price > ma20 and ma20 > ma60 and macd_hist > 0:
        direction = "多"
    elif price < ma20 and ma20 < ma60 and macd_hist < 0:
        direction = "空"
    else:
        direction = "震荡"
    return TimeframeSignal(name, price, ma20, ma60, support, resistance, previous_high, previous_low, rsi, macd_hist, direction, int(np.clip(score, 0, 100)))


def pick_signal(signals: list[TimeframeSignal], name: str) -> TimeframeSignal:
    return next((signal for signal in signals if signal.name == name), signals[0])


def money_risk_setup(title: str, wait_low: float, wait_high: float, entry: float, stop: float, target1: float, target2: float, risk_level: str, max_loss: float) -> TradeSetup:
    risk_per_coin = abs(entry - stop)
    reward_risk = abs(target1 - entry) / max(risk_per_coin, entry * 0.003)
    max_position = max_loss / max(risk_per_coin / entry, 0.003)
    return TradeSetup(title, f"{wait_low:,.2f} - {wait_high:,.2f}", "", entry, stop, target1, target2, reward_risk, risk_level, max_position)


def build_daily_plan(symbol: str, signals: list[TimeframeSignal], capital: float, risk_pct: float) -> DailyPlan:
    short, one, four, daily = pick_signal(signals, "15分钟"), pick_signal(signals, "1小时"), pick_signal(signals, "4小时"), pick_signal(signals, "日线")
    current = short.price
    max_loss = capital * risk_pct / 100
    directions = [short.direction, one.direction, four.direction]
    avg_score = int(round(np.mean([short.score, one.score, four.score, daily.score])))
    aligned_long = all(item == "多" for item in directions)
    aligned_short = all(item == "空" for item in directions)
    if avg_score >= 68 and aligned_long:
        state, color, reason = "🟢 强势上涨", "green", "短周期方向一致偏多，优先等回踩或突破确认。"
    elif avg_score <= 42 and aligned_short:
        state, color, reason = "🔴 弱势下跌", "red", "短周期方向一致偏弱，抄底只能等支撑企稳。"
    else:
        state, color, reason = "🟡 震荡等待", "orange", "15分钟、1小时、4小时方向不完全一致，重点等关键价位确认。"
    support_low, support_high = min(one.support, four.support) * 0.998, max(one.support, four.support) * 1.004
    entry_a = (support_low + support_high) / 2
    stop_a = min(one.previous_low, support_low * 0.988)
    risk_a = max(entry_a - stop_a, entry_a * 0.004)
    aggressive = money_risk_setup("方案A：激进交易（抄底）", support_low, support_high, entry_a, stop_a, entry_a + risk_a * 2, entry_a + risk_a * 3, "高", max_loss)
    aggressive.trigger = "15分钟出现放量阳线，并重新站上短期均线；只是阴跌到区域，不接。"
    entry_b = max(one.previous_high, one.ma20, four.ma20, current * 1.006)
    stop_b = min(entry_b * 0.976, max(support_high, one.ma20 * 0.99))
    risk_b = max(entry_b - stop_b, entry_b * 0.004)
    conservative = money_risk_setup("方案B：稳健交易（趋势确认）", entry_b * 0.997, entry_b * 1.006, entry_b, stop_b, entry_b + risk_b * 2, entry_b + risk_b * 3, "中", max_loss)
    conservative.trigger = "1小时K线收盘站稳突破价，回踩不破，再考虑轻仓跟进。"
    one_liner = f"{symbol.split('/')[0]} 今天不在中间价追单；等 {support_low:,.0f}-{support_high:,.0f} 企稳，或者突破 {entry_b:,.0f} 后再重新评估。"
    donts = [f"不要在 {support_high:,.0f}-{entry_b:,.0f} 中间位置追涨杀跌。", f"不要跌破 {stop_a:,.0f} 还继续抄底。", "不要因为FOMO临时加仓。"]
    max_position = min(max(aggressive.max_position_usdt, conservative.max_position_usdt), capital * 2)
    analysis_reason = f"日线方向：{daily.direction}，4小时方向：{four.direction}，1小时方向：{one.direction}，15分钟方向：{short.direction}。{reason}"
    return DailyPlan(symbol, one_liner, state, color, reason, aggressive, conservative, donts, max_loss, 1 if avg_score <= 42 else 2, max_position, analysis_reason, current)


def normalize_side(side: str) -> str:
    text = str(side).strip().lower()
    if text in {"short", "sell"} or "空" in text:
        return "short"
    return "long"


def side_label(side: str) -> str:
    return "做空" if normalize_side(side) == "short" else "做多"


def okx_side_label(side: str) -> str:
    return "开空" if normalize_side(side) == "short" else "开多"


def score_long_short_signal(signal: TimeframeSignal) -> tuple[float, float]:
    long_score = 50.0
    short_score = 50.0
    if signal.price > signal.ma20:
        long_score += 12
        short_score -= 8
    else:
        long_score -= 8
        short_score += 12
    if signal.price > signal.ma60:
        long_score += 10
        short_score -= 8
    else:
        long_score -= 8
        short_score += 10
    if signal.ma20 > signal.ma60:
        long_score += 8
        short_score -= 6
    else:
        long_score -= 6
        short_score += 8
    if signal.macd_hist > 0:
        long_score += 8
        short_score -= 6
    else:
        long_score -= 6
        short_score += 8
    if 45 <= signal.rsi <= 68:
        long_score += 5
    elif signal.rsi > 72:
        long_score -= 10
        short_score += 4
    elif signal.rsi < 32:
        short_score -= 8
        long_score += 2
    return float(np.clip(long_score, 0, 100)), float(np.clip(short_score, 0, 100))


def calculate_long_short_scores(signals: list[TimeframeSignal]) -> tuple[float, float]:
    weights = [0.25, 0.35, 0.30, 0.10]
    long_total = 0.0
    short_total = 0.0
    weight_total = 0.0
    for index, signal in enumerate(signals[:4]):
        weight = weights[index] if index < len(weights) else 0.20
        long_score, short_score = score_long_short_signal(signal)
        long_total += long_score * weight
        short_total += short_score * weight
        weight_total += weight
    if weight_total == 0:
        return 50.0, 50.0
    return long_total / weight_total, short_total / weight_total


def choose_direction(direction_mode: str, long_score: float, short_score: float) -> str:
    mode = str(direction_mode).strip().lower()
    if mode == "long" or "多" in mode:
        return "long"
    if mode == "short" or "空" in mode:
        return "short"
    return "long" if long_score >= short_score else "short"


def calc_position(entry: float, stop: float, target1: float, target2: float, rmb_capital: float, risk_pct: float, leverage: int, rate: float, side: str = "long") -> tuple[float, float, float, float, float, float, float]:
    usdt_capital = rmb_capital / rate
    max_loss = usdt_capital * risk_pct / 100
    side_code = normalize_side(side)
    if side_code == "short":
        stop_distance = max(stop - entry, entry * 0.002)
        profit1_distance = max(entry - target1, 0)
        profit2_distance = max(entry - target2, 0)
    else:
        stop_distance = max(entry - stop, entry * 0.002)
        profit1_distance = max(target1 - entry, 0)
        profit2_distance = max(target2 - entry, 0)
    quantity = max_loss / stop_distance
    position = quantity * entry
    margin = position / leverage
    profit1 = profit1_distance * quantity
    profit2 = profit2_distance * quantity
    rr = profit1 / max(max_loss, 0.000001)
    return max_loss, quantity, position, margin, profit1, profit2, rr


def forced_plan(title: str, source: str, side: str, order_type: str, entry: float, stop: float, target1: float, target2: float, leverage: int, rmb_capital: float, risk_pct: float, rate: float, warning: str) -> ForcedPlan:
    side_code = normalize_side(side)
    if side_code == "short":
        stop = max(stop, entry * 1.002)
        target1 = min(target1, entry * 0.998)
        target2 = min(target2, target1 * 0.998)
    else:
        stop = min(stop, entry * 0.998)
        target1 = max(target1, entry * 1.002)
        target2 = max(target2, target1 * 1.002)
    max_loss, quantity, position, margin, profit1, profit2, rr = calc_position(entry, stop, target1, target2, rmb_capital, risk_pct, leverage, rate, side_code)
    return ForcedPlan(title, source, side_code, order_type, entry, stop, target1, target2, leverage, quantity, position, margin, max_loss, max_loss * rate, profit1, profit1 * rate, profit2, profit2 * rate, rr, warning)


def assert_plan_direction(plan: ForcedPlan) -> ForcedPlan:
    side_code = normalize_side(plan.side)
    if side_code == "long" and not (plan.stop < plan.entry and plan.target1 > plan.entry and plan.target2 > plan.entry):
        raise ValueError("做多方案必须止损低于入场，目标高于入场")
    if side_code == "short" and not (plan.stop > plan.entry and plan.target1 < plan.entry and plan.target2 < plan.entry):
        raise ValueError("做空方案必须止损高于入场，目标低于入场")
    return plan


def choose_side(signals: list[TimeframeSignal], choice: str) -> str:
    long_score, short_score = calculate_long_short_scores(signals)
    return choose_direction(choice, long_score, short_score)


def build_long_plan(current_price: float, support: float, resistance: float, risk_params: dict) -> ForcedPlan:
    entry = float(risk_params.get("entry", support if risk_params.get("mode") == "limit" else current_price))
    stop = float(risk_params.get("stop", min(support * 0.99, entry * 0.992)))
    stop = min(stop, entry * 0.998)
    risk = max(entry - stop, entry * 0.004)
    target1 = float(risk_params.get("target1", entry + risk * 2))
    target2 = float(risk_params.get("target2", entry + risk * 3))
    target1 = max(target1, entry + risk * 2)
    target2 = max(target2, target1 + risk)
    return assert_plan_direction(forced_plan(
        risk_params.get("title", "今日强制交易方案"),
        risk_params.get("source", "方案A"),
        "long",
        risk_params.get("order_type", "限价挂单"),
        entry,
        stop,
        target1,
        target2,
        int(risk_params["leverage"]),
        float(risk_params["rmb_capital"]),
        float(risk_params["risk_pct"]),
        float(risk_params["rate"]),
        risk_params.get("warning", f"现在不要市价追，挂 {entry:,.2f} 限价多单，等价格回来再成交。"),
    ))


def build_short_plan(current_price: float, support: float, resistance: float, risk_params: dict) -> ForcedPlan:
    entry = float(risk_params.get("entry", resistance if risk_params.get("mode") == "limit" else current_price))
    stop = float(risk_params.get("stop", max(resistance * 1.01, entry * 1.008)))
    stop = max(stop, entry * 1.002)
    risk = max(stop - entry, entry * 0.004)
    target1 = float(risk_params.get("target1", entry - risk * 2))
    target2 = float(risk_params.get("target2", entry - risk * 3))
    target1 = min(target1, entry - risk * 2)
    target2 = min(target2, target1 - risk)
    return assert_plan_direction(forced_plan(
        risk_params.get("title", "今日强制交易方案"),
        risk_params.get("source", "方案A"),
        "short",
        risk_params.get("order_type", "限价挂单"),
        entry,
        stop,
        max(target1, 0.000001),
        max(target2, 0.000001),
        int(risk_params["leverage"]),
        float(risk_params["rmb_capital"]),
        float(risk_params["risk_pct"]),
        float(risk_params["rate"]),
        risk_params.get("warning", f"现在不要市价追空，挂 {entry:,.2f} 限价空单，等反弹再成交。"),
    ))


def build_candidates(signals: list[TimeframeSignal], side: str, rmb_capital: float, risk_pct: float, leverage: int, rate: float) -> list[ForcedPlan]:
    short, one, four = pick_signal(signals, "15分钟"), pick_signal(signals, "1小时"), pick_signal(signals, "4小时")
    current = short.price
    side_code = normalize_side(side)
    if side_code == "long":
        support_low, support_high = min(one.support, four.support) * 0.998, max(one.support, four.support) * 1.004
        entry_a = (support_low + support_high) / 2
        stop_a = min(one.previous_low, support_low * 0.988)
        risk_a = max(entry_a - stop_a, entry_a * 0.004)
        entry_b = max(one.previous_high, one.ma20, four.ma20, current * 1.006)
        stop_b = min(entry_b * 0.976, max(support_high, one.ma20 * 0.99))
        risk_b = max(entry_b - stop_b, entry_b * 0.004)
        return [
            build_long_plan(current, support_low, one.resistance, {"title": "今日强制交易方案", "source": "方案A", "order_type": "限价挂单", "entry": entry_a, "stop": stop_a, "target1": entry_a + risk_a * 2, "target2": entry_a + risk_a * 3, "leverage": leverage, "rmb_capital": rmb_capital, "risk_pct": risk_pct, "rate": rate, "warning": f"现在不要市价追，挂 {entry_a:,.2f} 限价多单，等价格回来再成交。"}),
            build_long_plan(current, support_low, one.resistance, {"title": "今日强制交易方案", "source": "方案B", "order_type": "条件单", "entry": entry_b, "stop": stop_b, "target1": entry_b + risk_b * 2, "target2": entry_b + risk_b * 3, "leverage": leverage, "rmb_capital": rmb_capital, "risk_pct": risk_pct, "rate": rate, "warning": f"只有突破并站稳 {entry_b:,.2f} 才做多，不提前追。"}),
        ]
    resistance_low, resistance_high = min(one.resistance, four.resistance) * 0.996, max(one.resistance, four.resistance) * 1.002
    entry_a = (resistance_low + resistance_high) / 2
    stop_a = max(one.previous_high, resistance_high * 1.012)
    risk_a = max(stop_a - entry_a, entry_a * 0.004)
    entry_b = min(one.previous_low, one.ma20, four.ma20, current * 0.994)
    stop_b = max(entry_b * 1.024, min(resistance_low, one.ma20 * 1.01))
    risk_b = max(stop_b - entry_b, entry_b * 0.004)
    return [
        build_short_plan(current, one.support, resistance_high, {"title": "今日强制交易方案", "source": "方案A", "order_type": "限价挂单", "entry": entry_a, "stop": stop_a, "target1": entry_a - risk_a * 2, "target2": entry_a - risk_a * 3, "leverage": leverage, "rmb_capital": rmb_capital, "risk_pct": risk_pct, "rate": rate, "warning": f"现在不要市价追空，挂 {entry_a:,.2f} 限价空单，等反弹再成交。"}),
        build_short_plan(current, one.support, resistance_high, {"title": "今日强制交易方案", "source": "方案B", "order_type": "条件单", "entry": entry_b, "stop": stop_b, "target1": entry_b - risk_b * 2, "target2": entry_b - risk_b * 3, "leverage": leverage, "rmb_capital": rmb_capital, "risk_pct": risk_pct, "rate": rate, "warning": f"只有跌破并站稳 {entry_b:,.2f} 下方才做空。"}),
    ]


def choose_candidate(candidates: list[ForcedPlan], current_price: float) -> ForcedPlan:
    pool = [item for item in candidates if item.reward_risk >= 2] or candidates
    if len(pool) == 1:
        return pool[0]
    nearest = min(pool, key=lambda item: abs(item.entry - current_price) / current_price)
    plan_b = next((item for item in pool if item.source == "方案B"), None)
    if plan_b and abs(plan_b.entry - current_price) / current_price <= abs(nearest.entry - current_price) / current_price * 1.5:
        return plan_b
    return nearest


def build_now_plan(signals: list[TimeframeSignal], side: str, rmb_capital: float, risk_pct: float, leverage: int, rate: float) -> ForcedPlan:
    short, one = pick_signal(signals, "15分钟"), pick_signal(signals, "1小时")
    entry = short.price
    leverage = min(leverage, 3)
    risk_pct = min(risk_pct, 0.5)
    side_code = normalize_side(side)
    if side_code == "long":
        stop = min(one.support, short.previous_low, entry * 0.99)
        risk = max(entry - stop, entry * 0.004)
        target1, target2 = entry + risk * 2, entry + risk * 3
        return build_long_plan(entry, one.support, one.resistance, {"title": "现在强制进场方案", "source": "立即轻仓", "order_type": "限价/市价附近成交", "entry": entry, "stop": stop, "target1": target1, "target2": target2, "leverage": leverage, "rmb_capital": rmb_capital, "risk_pct": risk_pct, "rate": rate, "warning": "当前属于强制进场单，不是最佳位置，只能小仓，不能加仓。"})
    else:
        stop = max(one.resistance, short.previous_high, entry * 1.01)
        risk = max(stop - entry, entry * 0.004)
        target1, target2 = max(entry - risk * 2, 0), max(entry - risk * 3, 0)
        return build_short_plan(entry, one.support, one.resistance, {"title": "现在强制进场方案", "source": "立即轻仓", "order_type": "限价/市价附近成交", "entry": entry, "stop": stop, "target1": target1, "target2": target2, "leverage": leverage, "rmb_capital": rmb_capital, "risk_pct": risk_pct, "rate": rate, "warning": "当前属于强制进场单，不是最佳位置，只能小仓，不能加仓。"})


def build_price_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="K线"))
    for name, color in [("MA20", "#16a34a"), ("MA60", "#dc2626")]:
        fig.add_trace(go.Scatter(x=df.index, y=df[name], mode="lines", line=dict(width=1.6, color=color), name=name))
    fig.update_layout(title=f"{ticker} 高级K线", height=520, margin=dict(l=10, r=10, t=50, b=10), xaxis_rangeslider_visible=False, legend=dict(orientation="h"))
    return fig


def build_indicator_chart(df: pd.DataFrame) -> go.Figure:
    colors = np.where(df["MACD_HIST"] >= 0, "#16a34a", "#dc2626")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI14"], name="RSI 14", line=dict(color="#7c3aed")))
    fig.add_hline(y=70, line_dash="dash", line_color="#dc2626", annotation_text="偏热")
    fig.add_hline(y=30, line_dash="dash", line_color="#16a34a", annotation_text="偏冷")
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_HIST"], name="MACD柱", marker_color=colors, yaxis="y2"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="#2563eb"), yaxis="y2"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_SIGNAL"], name="Signal", line=dict(color="#f97316"), yaxis="y2"))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10), yaxis=dict(title="RSI", range=[0, 100]), yaxis2=dict(title="MACD", overlaying="y", side="right", showgrid=False), legend=dict(orientation="h"))
    return fig


def extract_trades(bt: pd.DataFrame) -> list[float]:
    trades, entry = [], None
    for _, row in bt.iterrows():
        if row["trade"] > 0 and row["position"] == 1 and entry is None:
            entry = row["Close"]
        elif row["trade"] > 0 and row["position"] == 0 and entry is not None:
            trades.append(row["Close"] / entry - 1)
            entry = None
    if entry is not None:
        trades.append(bt["Close"].iloc[-1] / entry - 1)
    return trades


def run_backtest(df: pd.DataFrame, initial_cash: float, fee_pct: float) -> tuple[dict, pd.DataFrame]:
    bt = df.dropna(subset=["MA20", "MA60"]).copy()
    if bt.empty:
        return {}, pd.DataFrame()
    bt["signal"] = np.where(bt["MA20"] > bt["MA60"], 1, 0)
    bt["position"] = bt["signal"].shift(1).fillna(0)
    bt["return"] = bt["Close"].pct_change().fillna(0)
    bt["trade"] = bt["position"].diff().abs().fillna(bt["position"])
    bt["strategy_return"] = bt["position"] * bt["return"] - bt["trade"] * fee_pct / 100
    bt["equity"] = initial_cash * (1 + bt["strategy_return"]).cumprod()
    bt["buy_hold_equity"] = initial_cash * (1 + bt["return"]).cumprod()
    bt["drawdown"] = bt["equity"] / bt["equity"].cummax() - 1
    trades = extract_trades(bt)
    wins = [trade for trade in trades if trade > 0]
    return {"策略收益率": bt["equity"].iloc[-1] / initial_cash - 1, "买入持有收益率": bt["buy_hold_equity"].iloc[-1] / initial_cash - 1, "最大回撤": bt["drawdown"].min(), "交易次数": len(trades), "胜率": len(wins) / len(trades) if trades else 0}, bt


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def calculate_order_helper(rmb_capital: float, current_price: float, side: str, entry_price: float, stop_price: float, target1_price: float, target2_price: float, leverage: int, risk_pct: float, fee_pct: float, rmb_per_usdt: float = 7.25) -> OrderHelperResult:
    max_loss, quantity, position, margin, profit1, profit2, rr1 = calc_position(entry_price, stop_price, target1_price, target2_price, rmb_capital, risk_pct, leverage, rmb_per_usdt, side)
    estimated_loss = max_loss
    fee_rate = fee_pct / 100
    open_fee = position * fee_rate
    close_stop = quantity * stop_price * fee_rate
    close_t1 = quantity * target1_price * fee_rate
    close_t2 = quantity * target2_price * fee_rate
    total_stop = open_fee + close_stop
    total_t1 = open_fee + close_t1
    total_t2 = open_fee + close_t2
    warning = "风险在可计算范围内，但仍建议逐仓、小仓、严格止损。"
    risk_too_high = False
    if risk_pct > 2 or leverage > 5 or margin > rmb_capital / rmb_per_usdt:
        risk_too_high = True
        warning = "风险太高，建议不要开。"
    elif rr1 < 2:
        warning = "目标1盈利小于止损亏损的2倍，这单不值得做。"
    elif total_t1 > profit1 * 0.2:
        warning = "手续费占目标1盈利太高，仓位太小或目标太近。"
    return OrderHelperResult(rmb_capital / rmb_per_usdt, max_loss, position, quantity, margin, target1_price, target2_price, estimated_loss, profit1, profit2, open_fee, close_stop, close_t1, close_t2, total_stop, total_t1, total_t2, estimated_loss + total_stop, profit1 - total_t1, profit2 - total_t2, rr1, profit2 / max(max_loss, 0.000001), risk_too_high, warning)


def market_snapshot_from_data(data: dict[str, pd.DataFrame]) -> dict[str, float] | None:
    for name in ["15分钟", "1小时", "4小时", "日线"]:
        df = data.get(name)
        if df is not None and not df.empty and {"Close", "High", "Low"}.issubset(df.columns):
            recent = df.tail(min(len(df), 80))
            current = float(recent["Close"].iloc[-1])
            previous = recent.iloc[:-1] if len(recent) > 1 else recent
            resistance = float(previous["High"].max())
            support = float(previous["Low"].min())
            short_window = recent.tail(min(len(recent), 8))
            start = float(short_window["Close"].iloc[0])
            short_gain = current / start - 1 if start > 0 else 0
            return {
                "current": current,
                "support": min(support, current * 0.98),
                "resistance": max(resistance, current * 0.995),
                "short_gain": short_gain,
            }
    return None


def is_strong_trend(signals: list[TimeframeSignal], daily: DailyPlan | None, candidates: list[ForcedPlan] | None = None) -> bool:
    if not signals:
        return False
    current = signals[0].price
    resistance = max(signal.resistance for signal in signals[:3])
    strong_breakout = current > resistance * 1.012
    fast_gain = False
    if len(signals) >= 1:
        fast_gain = signals[0].price > signals[0].ma20 * 1.018 and signals[0].macd_hist > 0
    far_from_plan = False
    if daily is not None:
        a_low, a_high = setup_range(daily.aggressive)
        b_low, b_high = setup_range(daily.conservative)
        nearest = min(distance_to_range(current, a_low, a_high), distance_to_range(current, b_low, b_high))
        far_from_plan = nearest / max(current, 0.000001) > 0.035
    return strong_breakout or (fast_gain and far_from_plan)


def build_strong_trend_plans(current: float, support: float, resistance: float, rmb_capital: float, risk_pct: float, leverage: int, rate: float, must_mode: str = "normal") -> dict[str, ForcedPlan | str | float]:
    leverage = min(int(leverage), 3)
    base_risk = min(float(risk_pct), 1.0)
    pullback_low = resistance * 0.997
    pullback_high = resistance * 1.006
    pullback_entry = (pullback_low + pullback_high) / 2
    pullback_stop = min(resistance * 0.988, pullback_entry * 0.99)
    pullback_risk = max(pullback_entry - pullback_stop, pullback_entry * 0.006)
    pullback = build_long_plan(
        current,
        support,
        resistance,
        {
            "title": "强趋势观察模式",
            "source": "强趋势回踩",
            "order_type": "限价回踩单",
            "entry": pullback_entry,
            "stop": pullback_stop,
            "target1": pullback_entry + pullback_risk * 2,
            "target2": pullback_entry + pullback_risk * 3,
            "leverage": leverage,
            "rmb_capital": rmb_capital,
            "risk_pct": base_risk,
            "rate": rate,
            "warning": "回踩突破位不破再轻仓进，不回踩不追。",
        },
    )
    trigger = max(current * 1.006, resistance * 1.012)
    pursuit_entry = trigger
    pursuit_stop = max(resistance * 0.995, pursuit_entry * 0.986)
    pursuit_risk = max(pursuit_entry - pursuit_stop, pursuit_entry * 0.008)
    pursuit = build_long_plan(
        current,
        support,
        resistance,
        {
            "title": "强趋势突破追随单",
            "source": "强趋势追随",
            "order_type": "条件单",
            "entry": pursuit_entry,
            "stop": pursuit_stop,
            "target1": pursuit_entry + pursuit_risk * 2,
            "target2": pursuit_entry + pursuit_risk * 3,
            "leverage": leverage,
            "rmb_capital": rmb_capital,
            "risk_pct": min(base_risk, 0.5),
            "rate": rate,
            "warning": "只有价格放量站稳新高，才允许小仓追随；这是高风险追随单，不是最佳位置。",
        },
    )
    if must_mode == "now":
        now_stop = min(max(support, current * 0.986), current * 0.995)
        now_risk = max(current - now_stop, current * 0.006)
        forced_now = build_long_plan(
            current,
            support,
            resistance,
            {
                "title": "强趋势现在强制进场方案",
                "source": "强趋势强制轻仓",
                "order_type": "计划委托",
                "entry": current,
                "stop": now_stop,
                "target1": current + now_risk * 2,
                "target2": current + now_risk * 3,
                "leverage": leverage,
                "rmb_capital": rmb_capital,
                "risk_pct": 0.5,
                "rate": rate,
                "warning": "强制追涨单，风险高。仓位减半，风险比例降到0.5%，不能加仓，不能取消止损。",
            },
        )
        selected = forced_now
    else:
        selected = pullback
    return {
        "current": current,
        "breakout": resistance,
        "pullback_low": pullback_low,
        "pullback_high": pullback_high,
        "invalid": resistance * 0.99,
        "pullback": pullback,
        "pursuit": pursuit,
        "selected": selected,
        "abandon": f"如果价格跌回 {resistance * 0.99:,.2f} 下方，或者放量冲高后快速回落，放弃本次交易，等待重新整理。",
    }


def render_strong_trend_mode(symbol: str, mode_data: dict[str, ForcedPlan | str | float], coin: str, unit: str, margin_mode: str, fee_pct: float, rmb_capital: float, rate: float, must_mode: str) -> None:
    selected = mode_data["selected"]
    pullback = mode_data["pullback"]
    pursuit = mode_data["pursuit"]
    status = "现在强制轻仓" if must_mode == "now" else "强势拉升中，禁止盲目追高"
    st.markdown(
        f"""
        <div class='hero-card'>
            <div class='hero-symbol'>强趋势观察模式 · {symbol}</div>
            <div class='hero-price'>{status}</div>
            <div class='hero-line'>价格已经快速偏离原计划区，直接市价追单风险高。等待回踩突破位不破，或者等待下一根K线确认。</div>
            <div>
                <span class='pill pill-yellow'>当前建议：不市价追，等回踩确认</span>
                <span class='pill pill-green'>最优方案：回踩突破位不破再轻仓做多</span>
                <span class='pill pill-red'>价格已经跑远，不追。</span>
            </div>
            <div class='kv-grid' style='margin-top:18px'>
                <div class='kv'><div class='kv-label'>当前价格</div><div class='kv-value'>{mode_data["current"]:,.2f}</div></div>
                <div class='kv'><div class='kv-label'>突破位</div><div class='kv-value yellow'>{mode_data["breakout"]:,.2f}</div></div>
                <div class='kv'><div class='kv-label'>回踩观察区</div><div class='kv-value yellow'>{mode_data["pullback_low"]:,.2f} - {mode_data["pullback_high"]:,.2f}</div></div>
                <div class='kv'><div class='kv-label'>失效价</div><div class='kv-value red'>{mode_data["invalid"]:,.2f}</div></div>
                <div class='kv'><div class='kv-label'>目标1</div><div class='kv-value green'>{selected.target1:,.2f}</div></div>
                <div class='kv'><div class='kv-label'>目标2</div><div class='kv-value green'>{selected.target2:,.2f}</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div class='section-title'>强趋势三个处理方案</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        html_card(
            "方案1：回踩确认多单（推荐）",
            f"""
            <div class='kv-grid'>
                <div class='kv'><div class='kv-label'>等待回踩</div><div class='kv-value yellow'>{mode_data["pullback_low"]:,.2f} - {mode_data["pullback_high"]:,.2f}</div></div>
                <div class='kv'><div class='kv-label'>理想入场</div><div class='kv-value'>{pullback.entry:,.2f}</div></div>
                <div class='kv'><div class='kv-label'>止损</div><div class='kv-value red'>{pullback.stop:,.2f}</div></div>
                <div class='kv'><div class='kv-label'>目标1</div><div class='kv-value green'>{pullback.target1:,.2f}</div></div>
                <div class='kv'><div class='kv-label'>目标2</div><div class='kv-value green'>{pullback.target2:,.2f}</div></div>
                <div class='kv'><div class='kv-label'>仓位</div><div class='kv-value'>{pullback.position_usdt:,.2f} USDT</div></div>
                <div class='kv'><div class='kv-label'>最多亏损</div><div class='kv-value red'>{pullback.max_loss_usdt:,.2f} USDT / {pullback.loss_rmb:,.2f} RMB</div></div>
                <div class='kv'><div class='kv-label'>预计盈利</div><div class='kv-value green'>{pullback.profit1_usdt:,.2f} / {pullback.profit2_usdt:,.2f} USDT</div></div>
            </div>
            <div class='notice'>回踩不破再轻仓进，不回踩不追。</div>
            """,
            "setup-card risk-low",
        )
    with col2:
        html_card(
            "方案2：突破追随单（高风险）",
            f"""
            <div class='kv-grid'>
                <div class='kv'><div class='kv-label'>触发价</div><div class='kv-value yellow'>{pursuit.entry:,.2f}</div></div>
                <div class='kv'><div class='kv-label'>入场价</div><div class='kv-value'>{pursuit.entry:,.2f}</div></div>
                <div class='kv'><div class='kv-label'>止损</div><div class='kv-value red'>{pursuit.stop:,.2f}</div></div>
                <div class='kv'><div class='kv-label'>目标1</div><div class='kv-value green'>{pursuit.target1:,.2f}</div></div>
                <div class='kv'><div class='kv-label'>目标2</div><div class='kv-value green'>{pursuit.target2:,.2f}</div></div>
                <div class='kv'><div class='kv-label'>仓位减半</div><div class='kv-value'>{pursuit.position_usdt:,.2f} USDT</div></div>
                <div class='kv'><div class='kv-label'>风险等级</div><div class='kv-value red'>高</div></div>
            </div>
            <div class='notice danger-notice'>只有价格放量站稳 {pursuit.entry:,.2f} 上方，才允许小仓追随。这不是最佳位置。</div>
            """,
            "setup-card risk-high",
        )
    html_card("方案3：放弃条件", f"<div class='notice danger-notice'>{mode_data['abandon']}</div>")
    if must_mode == "now":
        st.error("强制追涨单，风险高：仓位减半，风险比例降到0.5%，杠杆不超过3x，不能加仓。")
    render_okx_fill(selected, coin, unit, margin_mode, float(mode_data["current"]))
    render_profit_preview(selected, fee_pct, rmb_capital, rate)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #0b0f19;
            --panel: #111827;
            --panel-2: #0f172a;
            --border: #1f2937;
            --text: #e5e7eb;
            --muted: #94a3b8;
            --green: #22c55e;
            --red: #ef4444;
            --yellow: #f59e0b;
            --blue: #38bdf8;
        }
        html, body, [data-testid="stAppViewContainer"], .stApp {
            background-color: #0b0f19 !important;
            color: #ffffff !important;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(56,189,248,.13), transparent 34%),
                linear-gradient(180deg, #05070b 0%, #0b1120 48%, #0b0f19 100%) !important;
            color: var(--text) !important;
        }
        header[data-testid="stHeader"] {
            background: transparent !important;
            height: 0rem !important;
            min-height: 0rem !important;
            max-height: 0rem !important;
            display: none !important;
            visibility: hidden !important;
        }
        div[data-testid="stToolbar"] {
            visibility: hidden !important;
            height: 0rem !important;
            min-height: 0rem !important;
            max-height: 0rem !important;
            display: none !important;
        }
        div[data-testid="stDecoration"] {
            display: none !important;
        }
        #MainMenu {
            visibility: hidden !important;
        }
        footer {
            visibility: hidden !important;
        }
        div[data-testid="stHorizontalBlock"],
        div[data-testid="stVerticalBlock"] {
            background-color: transparent !important;
        }
        div[data-testid="stMainBlockContainer"],
        div[data-testid="stAppViewBlockContainer"],
        div[data-testid="stMain"],
        main {
            background-color: transparent !important;
        }
        section[data-testid="stSidebar"] {
            background: #070a0f !important;
            border-right: 1px solid var(--border) !important;
        }
        section[data-testid="stSidebar"] * {
            color: var(--text) !important;
        }
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 2rem !important;
            max-width: 1200px !important;
        }
        div[data-testid="stElementContainer"],
        div[data-testid="stVerticalBlockBorderWrapper"],
        div[data-testid="stForm"],
        div[data-testid="stExpander"],
        div[data-testid="stTabs"],
        div[data-testid="stAlert"] {
            background-color: transparent !important;
            color: var(--text) !important;
        }
        div[data-testid="stForm"] {
            border: 1px solid #1f2937 !important;
            border-radius: 16px !important;
            background: #111827 !important;
            padding: 12px !important;
        }
        h1, h2, h3 {
            letter-spacing: 0;
            color: #f8fafc;
        }
        div[data-testid="stMetric"] {
            background: #111827 !important;
            border: 1px solid #1f2937 !important;
            border-radius: 16px !important;
            padding: 14px 16px;
            box-shadow: 0 14px 34px rgba(0,0,0,.20);
        }
        div[data-testid="stMetricLabel"] p {
            color: var(--muted);
            font-size: 14px;
        }
        div[data-testid="stMetricValue"] {
            color: #f8fafc;
            font-size: 28px;
            font-weight: 800;
        }
        .stButton > button, .stFormSubmitButton > button {
            border-radius: 12px;
            border: 1px solid rgba(56,189,248,.55);
            background: linear-gradient(135deg, #06b6d4 0%, #2563eb 100%);
            color: #ffffff !important;
            font-weight: 800;
            min-height: 46px;
            box-shadow: 0 14px 28px rgba(37,99,235,.24);
        }
        .stTextInput input, .stNumberInput input {
            background: #0b1220 !important;
            border: 1px solid #334155 !important;
            border-radius: 10px;
            color: #f8fafc !important;
            min-height: 42px;
        }
        div[role="radiogroup"] label {
            background: #0b1220 !important;
            border: 1px solid #263244 !important;
            border-radius: 10px;
            padding: 6px 10px;
            margin-bottom: 6px;
        }
        .trade-card {
            background: #111827 !important;
            border: 1px solid #1f2937 !important;
            border-radius: 16px !important;
            padding: 20px;
            box-shadow: 0 22px 60px rgba(0,0,0,.28);
            margin-bottom: 18px;
        }
        .hero-card {
            background: linear-gradient(135deg, rgba(14,165,233,.18), rgba(17,24,39,.92) 36%, rgba(2,6,23,.96));
            border: 1px solid rgba(56,189,248,.32);
            border-radius: 22px;
            padding: 26px;
            box-shadow: 0 26px 70px rgba(0,0,0,.35);
            margin-bottom: 18px;
        }
        .hero-symbol {
            font-size: 18px;
            color: var(--muted);
            font-weight: 700;
        }
        .hero-price {
            font-size: 48px;
            line-height: 1.05;
            font-weight: 900;
            color: #f8fafc;
        }
        .hero-line {
            font-size: 26px;
            line-height: 1.35;
            font-weight: 800;
            margin-top: 14px;
            color: #f8fafc;
        }
        .pill {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 8px 13px;
            font-size: 15px;
            font-weight: 800;
            margin-right: 8px;
            margin-top: 10px;
        }
        .pill-green { color: #bbf7d0; background: rgba(34,197,94,.14); border: 1px solid rgba(34,197,94,.42); }
        .pill-red { color: #fecaca; background: rgba(239,68,68,.14); border: 1px solid rgba(239,68,68,.42); }
        .pill-yellow { color: #fde68a; background: rgba(245,158,11,.14); border: 1px solid rgba(245,158,11,.42); }
        .pill-blue { color: #bae6fd; background: rgba(56,189,248,.14); border: 1px solid rgba(56,189,248,.42); }
        .setup-card {
            min-height: 410px;
            border-radius: 16px;
            padding: 20px;
            background: #111827;
            box-shadow: 0 22px 60px rgba(0,0,0,.24);
        }
        .risk-high { border: 1px solid rgba(239,68,68,.70); }
        .risk-mid { border: 1px solid rgba(245,158,11,.75); }
        .risk-low { border: 1px solid rgba(34,197,94,.70); }
        .card-title {
            font-size: 22px;
            font-weight: 900;
            margin-bottom: 14px;
            color: #f8fafc;
        }
        .kv-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 12px;
        }
        .kv {
            background: rgba(2,6,23,.62);
            border: 1px solid rgba(148,163,184,.18);
            border-radius: 14px;
            padding: 12px;
        }
        .kv-label {
            color: var(--muted);
            font-size: 13px;
            margin-bottom: 4px;
        }
        .kv-value {
            color: #f8fafc;
            font-size: 21px;
            font-weight: 850;
        }
        .green { color: var(--green) !important; }
        .red { color: var(--red) !important; }
        .yellow { color: var(--yellow) !important; }
        .muted { color: var(--muted); }
        .big-money {
            font-size: 34px;
            font-weight: 900;
            line-height: 1.1;
        }
        .section-title {
            font-size: 25px;
            font-weight: 900;
            margin: 22px 0 12px;
            color: #f8fafc;
        }
        .notice {
            border-radius: 16px;
            border: 1px solid rgba(245,158,11,.42);
            background: rgba(245,158,11,.11);
            padding: 16px;
            color: #fde68a;
            font-size: 17px;
            font-weight: 700;
            margin: 12px 0;
        }
        .danger-notice {
            border-color: rgba(239,68,68,.42);
            background: rgba(239,68,68,.12);
            color: #fecaca;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def html_card(title: str, body: str, css_class: str = "trade-card") -> None:
    st.markdown(f"<div class='{css_class}'><div class='card-title'>{title}</div>{body}</div>", unsafe_allow_html=True)


def money_html(label: str, usdt: float, rate: float, positive: bool) -> str:
    color = "#22c55e" if positive else "#ef4444"
    sign = "" if positive else "-"
    return f"<div class='trade-card'><div class='muted'>{label}</div><div class='big-money' style='color:{color}'>{sign}{abs(usdt):,.2f} USDT</div><div style='font-size:19px;color:{color};font-weight:800'>约 {sign}{abs(usdt * rate):,.2f} 人民币</div></div>"


def risk_css(risk_level: str) -> str:
    if "高" in risk_level:
        return "risk-high"
    if "低" in risk_level:
        return "risk-low"
    return "risk-mid"


def market_pill_class(market_state: str) -> str:
    if "强" in market_state or "上涨" in market_state:
        return "pill-green"
    if "弱" in market_state or "下跌" in market_state:
        return "pill-red"
    return "pill-yellow"


def current_action(daily: DailyPlan, forced: ForcedPlan | None) -> tuple[str, str]:
    if forced is not None:
        if normalize_side(forced.side) == "short":
            return "可轻仓做空", "pill-red"
        return "可轻仓做多", "pill-green"
    if "强" in daily.market_state:
        return "等待回踩", "pill-yellow"
    if "弱" in daily.market_state:
        return "不碰", "pill-red"
    return "等待", "pill-yellow"


def setup_range(setup: TradeSetup) -> tuple[float, float]:
    parts = [item.strip().replace(",", "") for item in setup.wait_area.split("-")]
    try:
        low, high = float(parts[0]), float(parts[1])
    except (IndexError, ValueError):
        low = high = setup.entry
    return min(low, high), max(low, high)


def wait_area_from_plan(plan: ForcedPlan) -> str:
    side = normalize_side(plan.side)
    if plan.source == "方案A":
        if side == "long":
            low, high = plan.entry * 0.994, plan.entry * 1.004
        else:
            low, high = plan.entry * 0.996, plan.entry * 1.006
    else:
        if side == "long":
            low, high = plan.entry * 0.997, plan.entry * 1.006
        else:
            low, high = plan.entry * 0.994, plan.entry * 1.003
    return f"{min(low, high):,.2f} - {max(low, high):,.2f}"


def setup_from_plan(plan: ForcedPlan) -> TradeSetup:
    side = normalize_side(plan.side)
    if plan.source == "方案A":
        title = "方案A：激进交易（抄底）" if side == "long" else "方案A：激进交易（摸顶）"
        risk_level = "高"
        trigger = "15分钟出现放量阳线，并重新站上短期均线；只是阴跌到区域，不接。" if side == "long" else "15分钟反弹到压力区后放量转弱，跌回短期均线下方才考虑；只是急涨到区域，不追空。"
    else:
        title = "方案B：稳健交易（趋势确认）"
        risk_level = "中"
        trigger = "1小时K线收盘站稳突破价，回踩不破，再考虑轻仓跟进。" if side == "long" else "1小时K线收盘跌破关键位，反抽不回去，再考虑轻仓跟进。"
    return TradeSetup(
        title=title,
        wait_area=wait_area_from_plan(plan),
        trigger=trigger,
        entry=plan.entry,
        stop=plan.stop,
        target1=plan.target1,
        target2=plan.target2,
        reward_risk=plan.reward_risk,
        risk_level=risk_level,
        max_position_usdt=plan.position_usdt,
    )


def daily_with_direction_plans(daily: DailyPlan, candidates: list[ForcedPlan], side: str) -> DailyPlan:
    plan_a = next((item for item in candidates if item.source == "方案A"), candidates[0])
    plan_b = next((item for item in candidates if item.source == "方案B"), candidates[-1])
    aggressive = setup_from_plan(plan_a)
    conservative = setup_from_plan(plan_b)
    side_code = normalize_side(side)
    if side_code == "short":
        one_liner = f"{daily.symbol.split('/')[0]} 今天不在中间价追空；等压力区反弹转弱，或者跌破 {plan_b.entry:,.0f} 后再重新评估。"
        donts = [
            f"不要在没有跌破 {plan_b.entry:,.0f} 前追空。",
            f"不要突破 {plan_a.stop:,.0f} 还继续扛空。",
            "不要因为FOMO临时加仓。",
        ]
    else:
        one_liner = daily.one_liner
        donts = daily.donts
    max_position = min(max(aggressive.max_position_usdt, conservative.max_position_usdt), daily.max_position_usdt)
    return DailyPlan(
        daily.symbol,
        one_liner,
        daily.market_state,
        daily.state_color,
        daily.state_reason,
        aggressive,
        conservative,
        donts,
        daily.max_loss,
        daily.recommended_leverage,
        max_position,
        daily.analysis_reason,
        daily.current_price,
    )


def distance_to_range(current: float, low: float, high: float) -> float:
    if low <= current <= high:
        return 0.0
    if current < low:
        return low - current
    return current - high


def setup_body(setup: TradeSetup) -> str:
    extra = ""
    extra_notice = ""
    if "方案B" in setup.title:
        low, high = setup_range(setup)
        extra = (
            f"<div class='kv'><div class='kv-label'>触发价</div><div class='kv-value yellow'>{high:,.2f}</div></div>"
            f"<div class='kv'><div class='kv-label'>理想回踩入场</div><div class='kv-value yellow'>{setup.entry:,.2f}</div></div>"
            f"<div class='kv'><div class='kv-label'>追突破入场</div><div class='kv-value red'>{high:,.2f}</div></div>"
        )
        extra_notice = "<div class='notice danger-notice'>如果价格没有站上突破区，只显示等待，不允许直接进。</div>"
    return (
        "<div class='kv-grid'>"
        f"<div class='kv'><div class='kv-label'>等待区域</div><div class='kv-value yellow'>{setup.wait_area}</div></div>"
        f"<div class='kv'><div class='kv-label'>风险等级</div><div class='kv-value'>{setup.risk_level}</div></div>"
        f"<div class='kv'><div class='kv-label'>入场价</div><div class='kv-value'>{setup.entry:,.2f}</div></div>"
        f"<div class='kv'><div class='kv-label'>止损价</div><div class='kv-value red'>{setup.stop:,.2f}</div></div>"
        f"<div class='kv'><div class='kv-label'>目标1</div><div class='kv-value green'>{setup.target1:,.2f}</div></div>"
        f"<div class='kv'><div class='kv-label'>目标2</div><div class='kv-value green'>{setup.target2:,.2f}</div></div>"
        f"<div class='kv'><div class='kv-label'>盈亏比</div><div class='kv-value'>1 : {setup.reward_risk:.2f}</div></div>"
        f"<div class='kv'><div class='kv-label'>最大仓位</div><div class='kv-value'>{setup.max_position_usdt:,.2f} USDT</div></div>"
        f"{extra}"
        "</div>"
        f"<div class='notice'>触发条件：{setup.trigger}</div>"
        f"{extra_notice}"
    )


def okx_order_type(plan: ForcedPlan, current_price: float | None = None) -> str:
    side = normalize_side(plan.side)
    if "立即" in plan.source or "市价" in plan.order_type:
        return "计划委托"
    if current_price is None:
        if "条件" in plan.order_type:
            return "条件单"
        return "限价单"
    if side == "long" and plan.entry > current_price:
        return "突破条件单"
    if side == "long" and plan.entry <= current_price:
        return "限价挂单"
    if side == "short" and plan.entry < current_price:
        return "跌破条件单"
    return "限价挂空"


def render_okx_fill(plan: ForcedPlan, coin: str, unit: str, margin_mode: str = "逐仓", current_price: float | None = None) -> None:
    direction = okx_side_label(plan.side)
    qty_text = f"{plan.position_usdt:,.2f} USDT" if unit == "USDT" else f"{plan.quantity:.6f} {coin}"
    order_type = okx_order_type(plan, current_price)
    html_card(
        "欧易怎么填",
        f"""
        <div class='kv-grid'>
            <div class='kv'><div class='kv-label'>模式</div><div class='kv-value'>{margin_mode}</div></div>
            <div class='kv'><div class='kv-label'>杠杆</div><div class='kv-value'>{plan.leverage}x</div></div>
            <div class='kv'><div class='kv-label'>方向</div><div class='kv-value {'green' if direction == '开多' else 'red'}'>{direction}</div></div>
            <div class='kv'><div class='kv-label'>委托类型</div><div class='kv-value'>{order_type}</div></div>
            <div class='kv'><div class='kv-label'>触发价</div><div class='kv-value yellow'>{plan.entry:,.2f}</div></div>
            <div class='kv'><div class='kv-label'>委托价</div><div class='kv-value yellow'>{plan.entry:,.2f}</div></div>
            <div class='kv'><div class='kv-label'>数量</div><div class='kv-value'>{qty_text}</div></div>
            <div class='kv'><div class='kv-label'>数量单位</div><div class='kv-value'>{unit}</div></div>
            <div class='kv'><div class='kv-label'>仓位价值</div><div class='kv-value'>{plan.position_usdt:,.2f} USDT</div></div>
            <div class='kv'><div class='kv-label'>预计保证金</div><div class='kv-value'>{plan.margin_usdt:,.2f} USDT</div></div>
            <div class='kv'><div class='kv-label'>止损</div><div class='kv-value red'>{plan.stop:,.2f}</div></div>
            <div class='kv'><div class='kv-label'>止盈1</div><div class='kv-value green'>{plan.target1:,.2f}</div></div>
            <div class='kv'><div class='kv-label'>止盈2</div><div class='kv-value green'>{plan.target2:,.2f}</div></div>
        </div>
        """,
    )


def render_profit_preview(plan: ForcedPlan, fee_pct: float = 0.05, rmb_capital: float = 1000.0, rate: float = 7.2) -> None:
    usdt_capital = rmb_capital / rate
    fee_rate = fee_pct / 100
    open_fee = plan.position_usdt * fee_rate
    close_stop_fee = plan.quantity * plan.stop * fee_rate
    close_target1_fee = plan.quantity * plan.target1 * fee_rate
    close_target2_fee = plan.quantity * plan.target2 * fee_rate
    total_stop_fee = open_fee + close_stop_fee
    total_target1_fee = open_fee + close_target1_fee
    total_target2_fee = open_fee + close_target2_fee
    net_loss = plan.max_loss_usdt + total_stop_fee
    net_profit1 = plan.profit1_usdt - total_target1_fee
    net_profit2 = plan.profit2_usdt - total_target2_fee
    st.markdown("<div class='section-title'>这单赚亏一眼看懂</div>", unsafe_allow_html=True)
    html_card("本金", f"<div class='big-money'>{rmb_capital:,.2f} RMB ≈ {usdt_capital:,.2f} USDT</div>")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"<div class='trade-card'><div class='muted'>如果止损</div><div class='big-money red'>亏损 {plan.max_loss_usdt:,.2f} USDT</div><div class='red' style='font-size:26px;font-weight:900'>≈ {plan.loss_rmb:,.2f} RMB</div></div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"<div class='trade-card'><div class='muted'>如果到目标1</div><div class='big-money green'>盈利 {plan.profit1_usdt:,.2f} USDT</div><div class='green' style='font-size:26px;font-weight:900'>≈ {plan.profit1_rmb:,.2f} RMB</div></div>",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"<div class='trade-card'><div class='muted'>如果到目标2</div><div class='big-money green'>盈利 {plan.profit2_usdt:,.2f} USDT</div><div class='green' style='font-size:26px;font-weight:900'>≈ {plan.profit2_rmb:,.2f} RMB</div></div>",
            unsafe_allow_html=True,
        )
    html_card(
        "手续费",
        f"""
        <div class='kv-grid'>
            <div class='kv'><div class='kv-label'>费率</div><div class='kv-value'>{fee_pct:.3f}%</div></div>
            <div class='kv'><div class='kv-label'>开仓手续费</div><div class='kv-value'>{open_fee:,.3f} USDT</div></div>
            <div class='kv'><div class='kv-label'>止损平仓手续费</div><div class='kv-value'>{close_stop_fee:,.3f} USDT</div></div>
            <div class='kv'><div class='kv-label'>止损手续费合计</div><div class='kv-value red'>{total_stop_fee:,.3f} USDT</div></div>
            <div class='kv'><div class='kv-label'>目标1平仓手续费</div><div class='kv-value'>{close_target1_fee:,.3f} USDT</div></div>
            <div class='kv'><div class='kv-label'>目标1手续费合计</div><div class='kv-value'>{total_target1_fee:,.3f} USDT</div></div>
        </div>
        """,
    )
    html_card(
        "手续费后",
        f"""
        <div class='kv-grid'>
            <div class='kv'><div class='kv-label'>止损实际亏损</div><div class='kv-value red'>{net_loss:,.2f} USDT ≈ {net_loss * rate:,.2f} RMB</div></div>
            <div class='kv'><div class='kv-label'>目标1实际盈利</div><div class='kv-value green'>{net_profit1:,.2f} USDT ≈ {net_profit1 * rate:,.2f} RMB</div></div>
            <div class='kv'><div class='kv-label'>目标2实际盈利</div><div class='kv-value green'>{net_profit2:,.2f} USDT ≈ {net_profit2 * rate:,.2f} RMB</div></div>
        </div>
        <div class='notice'>先看亏多少钱，再看能赚多少钱。</div>
        """,
    )


def render_forced_plan(plan: ForcedPlan, title: str | None = None, warning: str | None = None) -> None:
    display_title = title or plan.title
    warning_text = warning or plan.warning
    direction_text = side_label(plan.side)
    entry_mode_text = plan.order_type
    if "立即" in plan.source:
        direction_text = "轻仓做空" if normalize_side(plan.side) == "short" else "轻仓做多"
        entry_mode_text = "当前价附近成交"
    st.markdown(f"<div class='section-title'>{display_title}</div>", unsafe_allow_html=True)
    if "强制" in display_title or "立即" in plan.source:
        st.error(warning_text)
    else:
        st.warning(warning_text)
    st.markdown("<div class='section-title'>仓位先看这里</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("仓位价值", f"{plan.position_usdt:,.2f} USDT")
    col2.metric("币数量", f"{plan.quantity:.6f}")
    col3.metric("需要保证金", f"{plan.margin_usdt:,.2f} USDT")
    html_card(
        "执行方案",
        f"""
        <div class='kv-grid'>
            <div class='kv'><div class='kv-label'>最推荐方案</div><div class='kv-value'>{plan.source}</div></div>
            <div class='kv'><div class='kv-label'>方向</div><div class='kv-value'>{direction_text}</div></div>
            <div class='kv'><div class='kv-label'>入场方式</div><div class='kv-value'>{entry_mode_text}</div></div>
            <div class='kv'><div class='kv-label'>模式</div><div class='kv-value'>逐仓</div></div>
            <div class='kv'><div class='kv-label'>具体入场价</div><div class='kv-value yellow'>{plan.entry:,.2f}</div></div>
            <div class='kv'><div class='kv-label'>止损价</div><div class='kv-value red'>{plan.stop:,.2f}</div></div>
            <div class='kv'><div class='kv-label'>目标1</div><div class='kv-value green'>{plan.target1:,.2f}</div></div>
            <div class='kv'><div class='kv-label'>目标2</div><div class='kv-value green'>{plan.target2:,.2f}</div></div>
            <div class='kv'><div class='kv-label'>杠杆</div><div class='kv-value'>{plan.leverage}x</div></div>
            <div class='kv'><div class='kv-label'>数量</div><div class='kv-value'>{plan.quantity:.6f}</div></div>
            <div class='kv'><div class='kv-label'>仓位价值</div><div class='kv-value'>{plan.position_usdt:,.2f} USDT</div></div>
            <div class='kv'><div class='kv-label'>需要保证金</div><div class='kv-value'>{plan.margin_usdt:,.2f} USDT</div></div>
            <div class='kv'><div class='kv-label'>如果止损亏</div><div class='kv-value red'>{plan.max_loss_usdt:,.2f} USDT / {plan.loss_rmb:,.2f} RMB</div></div>
            <div class='kv'><div class='kv-label'>到目标1赚</div><div class='kv-value green'>{plan.profit1_usdt:,.2f} USDT / {plan.profit1_rmb:,.2f} RMB</div></div>
            <div class='kv'><div class='kv-label'>到目标2赚</div><div class='kv-value green'>{plan.profit2_usdt:,.2f} USDT / {plan.profit2_rmb:,.2f} RMB</div></div>
        </div>
        """,
    )


def render_trade_value(plan: ForcedPlan, risk_pct: float, leverage: int) -> None:
    is_worth = plan.reward_risk >= 2 and risk_pct <= 2 and leverage <= 5
    verdict = "值得观察，但只按计划做" if is_worth else "这笔不值得硬做"
    css = "green" if is_worth else "red"
    reasons = []
    if plan.reward_risk < 2:
        reasons.append("盈亏比低于 1:2")
    if risk_pct > 2:
        reasons.append("单笔风险超过 2%")
    if leverage > 5:
        reasons.append("杠杆超过 5x")
    if not reasons:
        reasons.append("盈亏比达到 1:2，风险参数在普通人可控范围内")
    html_card(
        "这笔交易值不值得做",
        f"""
        <div class='big-money {css}'>{verdict}</div>
        <div class='kv-grid' style='margin-top:14px'>
            <div class='kv'><div class='kv-label'>盈亏比</div><div class='kv-value'>1 : {plan.reward_risk:.2f}</div></div>
            <div class='kv'><div class='kv-label'>单笔风险</div><div class='kv-value'>{risk_pct:.2f}%</div></div>
            <div class='kv'><div class='kv-label'>杠杆</div><div class='kv-value'>{leverage}x</div></div>
            <div class='kv'><div class='kv-label'>判断原因</div><div class='kv-value'>{'；'.join(reasons)}</div></div>
        </div>
        """,
    )


def build_today_donts(daily: DailyPlan, plan: ForcedPlan) -> list[str]:
    side = normalize_side(plan.side)
    a_low, a_high = setup_range(daily.aggressive)
    b_low, b_high = setup_range(daily.conservative)
    middle_low, middle_high = sorted([a_high, b_low])
    if side == "short":
        return [
            f"❌ 不要在 {middle_low:,.2f} - {middle_high:,.2f} 中间位置追空或追多。",
            f"❌ 不要突破 {plan.stop:,.2f} 后还继续扛空单。",
            f"❌ 不要在没有跌破 {plan.entry:,.2f} 前提前重仓做空。",
            "❌ 不要取消止损。",
            "❌ 不要一亏损就反手开单。",
        ]
    return [
        f"❌ 不要在 {middle_low:,.2f} - {middle_high:,.2f} 中间位置追涨杀跌。",
        f"❌ 不要跌破 {plan.stop:,.2f} 后继续抄底。",
        f"❌ 不要没站稳 {plan.entry:,.2f} 就提前重仓追多。",
        "❌ 不要取消止损。",
        "❌ 不要一亏损就反手开单。",
    ]


def render_today_donts(daily: DailyPlan, plan: ForcedPlan) -> None:
    st.markdown("<div class='section-title'>今天不要做什么</div>", unsafe_allow_html=True)
    dont_html = "".join(f"<div class='notice danger-notice'>{item}</div>" for item in build_today_donts(daily, plan)[:5])
    st.markdown(dont_html, unsafe_allow_html=True)


def render_discipline(risk_pct: float, leverage: int, reward_risk: float, must_mode: str, now_force_count: int = 0) -> None:
    rows = [
        ("notice", "先看亏多少钱，再决定开多少仓。"),
        ("notice", "先看有没有好点位，再决定要不要进场。"),
        ("notice", "不到计划位置，不开仓。"),
        ("notice danger-notice", "没有止损，不开仓。"),
    ]
    if risk_pct > 5:
        rows.append(("notice danger-notice", "单笔风险超过 5%：风险过高，不建议。"))
    elif risk_pct > 2:
        rows.append(("notice", "单笔风险超过 2%：风险偏高，建议降到 1% 附近。"))
    if leverage > 5:
        rows.append(("notice danger-notice", "杠杆超过 5x：普通人不建议高杠杆。"))
    if reward_risk < 2:
        rows.append(("notice danger-notice", "盈亏比低于 1:2：这单不值得做。"))
    if must_mode == "now":
        rows.append(("notice danger-notice", "这是情绪单，只能小仓，不能加仓。"))
    if now_force_count >= 3:
        rows.append(("notice danger-notice", "连续 3 次使用“现在必须进”：你正在冲动交易，建议暂停。"))
    html_card("交易纪律", "".join(f"<div class='{css}'>{text}</div>" for css, text in rows))


def trade_record_mode_label(must_mode: str) -> str:
    return {"normal": "正常", "today": "今天必须做", "now": "现在必须进"}.get(must_mode, must_mode)


def trade_record_plan_label(plan: ForcedPlan) -> str:
    if "A" in plan.source:
        return "方案A"
    if "B" in plan.source:
        return "方案B"
    return "强制轻仓"


def build_trade_record(daily: DailyPlan, plan: ForcedPlan, mode: str, margin_mode: str, fee_pct: float) -> dict[str, object]:
    return {
        "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
        "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "币种": daily.symbol,
        "当前价格": round(daily.current_price, 6),
        "方向": okx_side_label(plan.side),
        "入场价": round(plan.entry, 6),
        "止损价": round(plan.stop, 6),
        "目标1": round(plan.target1, 6),
        "目标2": round(plan.target2, 6),
        "数量": round(plan.quantity, 8),
        "仓位价值": round(plan.position_usdt, 4),
        "保证金": round(plan.margin_usdt, 4),
        "预计亏损": round(plan.max_loss_usdt, 4),
        "预计盈利": round(plan.profit1_usdt, 4),
        "模式": trade_record_mode_label(mode),
        "方案": trade_record_plan_label(plan),
        "保证金模式": margin_mode,
        "杠杆": plan.leverage,
        "手续费率": fee_pct,
        "结果": "",
        "实际盈亏金额": "",
        "是否按计划执行": "",
    }


def append_trade_record(record: dict[str, object]) -> None:
    fieldnames = list(record.keys())
    exists = TRADE_RECORDS_FILE.exists()
    with TRADE_RECORDS_FILE.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(record)


def read_trade_records() -> pd.DataFrame:
    if not TRADE_RECORDS_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(TRADE_RECORDS_FILE, encoding="utf-8-sig", dtype={"id": str})


def save_trade_records(df: pd.DataFrame) -> None:
    df.to_csv(TRADE_RECORDS_FILE, index=False, encoding="utf-8-sig")


def max_consecutive_losses(df: pd.DataFrame) -> int:
    if df.empty or "实际盈亏金额" not in df.columns:
        return 0
    pnl = pd.to_numeric(df["实际盈亏金额"], errors="coerce").dropna()
    max_streak = streak = 0
    for value in pnl:
        if value < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return int(max_streak)


def win_rate(df: pd.DataFrame) -> float:
    if df.empty or "实际盈亏金额" not in df.columns:
        return 0.0
    pnl = pd.to_numeric(df["实际盈亏金额"], errors="coerce").dropna()
    if pnl.empty:
        return 0.0
    return float((pnl > 0).mean())


def group_win_rate(df: pd.DataFrame, column: str, value: str) -> float:
    if df.empty or column not in df.columns:
        return 0.0
    return win_rate(df[df[column] == value])


def review_suggestions(df: pd.DataFrame) -> list[str]:
    if df.empty or "实际盈亏金额" not in df.columns:
        return ["先保存几笔计划并填写结果，复盘建议会自动生成。"]
    work = df.copy()
    if "id" in work.columns:
        work["id"] = work["id"].astype(str)
    work["实际盈亏金额"] = pd.to_numeric(work["实际盈亏金额"], errors="coerce")
    closed = work.dropna(subset=["实际盈亏金额"])
    if closed.empty:
        return ["保存的计划还没有填写实际盈亏，先补结果再复盘。"]
    suggestions = []
    if "模式" in closed.columns:
        mode_loss = closed.groupby("模式")["实际盈亏金额"].sum().sort_values()
        if not mode_loss.empty and mode_loss.iloc[0] < 0:
            suggestions.append(f"你在“{mode_loss.index[0]}”模式亏损最多，建议减少这种交易。")
    if "方案" in closed.columns:
        scheme_rates = closed.groupby("方案").apply(win_rate).sort_values(ascending=False)
        if not scheme_rates.empty:
            suggestions.append(f"目前“{scheme_rates.index[0]}”胜率更高，可以优先观察这类方案。")
    if "是否按计划执行" in closed.columns:
        off_plan = closed[closed["是否按计划执行"] == "否"]
        if not off_plan.empty and off_plan["实际盈亏金额"].sum() < 0:
            suggestions.append("没有按计划执行的交易整体亏损，尤其要避免取消止损和临时加仓。")
    if max_consecutive_losses(closed) >= 3:
        suggestions.append("已经出现连续亏损 3 次，今天停止交易。")
    return suggestions or ["交易样本还少，先继续记录，重点看是否按计划执行。"]


def render_trade_review() -> None:
    st.title("交易记录与复盘")
    st.caption("保存计划、填写结果，然后看自己到底亏在什么模式上。")
    df = read_trade_records()
    if df.empty:
        html_card("还没有交易记录", f"<div class='notice'>先在 AI进场助手 首页点击“保存本次计划”。CSV 会保存到：{TRADE_RECORDS_FILE}</div>")
        return

    work = df.copy()
    if "实际盈亏金额" in work.columns:
        work["实际盈亏金额"] = pd.to_numeric(work["实际盈亏金额"], errors="coerce")
    closed = work.dropna(subset=["实际盈亏金额"]) if "实际盈亏金额" in work.columns else pd.DataFrame()
    total = len(work)
    total_pnl = float(closed["实际盈亏金额"].sum()) if not closed.empty else 0.0
    avg_profit = float(closed.loc[closed["实际盈亏金额"] > 0, "实际盈亏金额"].mean()) if not closed.empty and (closed["实际盈亏金额"] > 0).any() else 0.0
    avg_loss = float(closed.loc[closed["实际盈亏金额"] < 0, "实际盈亏金额"].mean()) if not closed.empty and (closed["实际盈亏金额"] < 0).any() else 0.0

    cols = st.columns(5)
    cols[0].metric("总交易次数", total)
    cols[1].metric("胜率", f"{win_rate(closed) * 100:.1f}%")
    cols[2].metric("总盈亏", f"{total_pnl:,.2f} USDT")
    cols[3].metric("平均盈利", f"{avg_profit:,.2f} USDT")
    cols[4].metric("平均亏损", f"{avg_loss:,.2f} USDT")

    cols = st.columns(5)
    cols[0].metric("最大连续亏损", max_consecutive_losses(closed))
    cols[1].metric("做多胜率", f"{group_win_rate(closed, '方向', '开多') * 100:.1f}%")
    cols[2].metric("做空胜率", f"{group_win_rate(closed, '方向', '开空') * 100:.1f}%")
    cols[3].metric("方案A胜率", f"{group_win_rate(closed, '方案', '方案A') * 100:.1f}%")
    cols[4].metric("方案B胜率", f"{group_win_rate(closed, '方案', '方案B') * 100:.1f}%")

    cols = st.columns(2)
    cols[0].metric("强制进场胜率", f"{group_win_rate(closed, '方案', '强制轻仓') * 100:.1f}%")
    if not closed.empty and "模式" in closed.columns:
        worst_mode = closed.groupby("模式")["实际盈亏金额"].sum().sort_values().index[0]
    else:
        worst_mode = "暂无"
    cols[1].metric("最容易亏钱的模式", worst_mode)

    html_card("复盘建议", "".join(f"<div class='notice'>{item}</div>" for item in review_suggestions(work)))

    st.markdown("<div class='section-title'>填写交易结果</div>", unsafe_allow_html=True)
    selected_id = st.selectbox("选择要更新的计划", work["id"].astype(str).tolist(), format_func=lambda x: f"{x} · {work.loc[work['id'].astype(str) == x, '币种'].iloc[0]} · {work.loc[work['id'].astype(str) == x, '时间'].iloc[0]}")
    row_mask = work["id"].astype(str) == selected_id
    current = work.loc[row_mask].iloc[0]
    with st.form("review_update_form"):
        col1, col2, col3 = st.columns(3)
        result = col1.selectbox("结果", ["", "止损", "目标1", "目标2", "手动平仓"], index=["", "止损", "目标1", "目标2", "手动平仓"].index(str(current.get("结果", ""))) if str(current.get("结果", "")) in ["", "止损", "目标1", "目标2", "手动平仓"] else 0)
        actual_pnl = col2.number_input("实际盈亏金额 USDT", value=float(pd.to_numeric(current.get("实际盈亏金额", 0), errors="coerce") if pd.notna(pd.to_numeric(current.get("实际盈亏金额", 0), errors="coerce")) else 0.0), step=1.0)
        followed = col3.selectbox("是否按计划执行", ["", "是", "否"], index=["", "是", "否"].index(str(current.get("是否按计划执行", ""))) if str(current.get("是否按计划执行", "")) in ["", "是", "否"] else 0)
        submitted = st.form_submit_button("保存复盘结果", type="primary", use_container_width=True)
    if submitted:
        df.loc[df["id"].astype(str) == selected_id, "结果"] = result
        df.loc[df["id"].astype(str) == selected_id, "实际盈亏金额"] = actual_pnl
        df.loc[df["id"].astype(str) == selected_id, "是否按计划执行"] = followed
        save_trade_records(df)
        st.success("复盘结果已保存。")
        st.rerun()

    st.markdown("<div class='section-title'>交易记录 CSV</div>", unsafe_allow_html=True)
    st.dataframe(work, use_container_width=True, hide_index=True)
    st.download_button("下载交易记录 CSV", data=TRADE_RECORDS_FILE.read_bytes(), file_name="trade_records.csv", mime="text/csv", use_container_width=True)


def build_plan_text(daily: DailyPlan, plan: ForcedPlan, coin: str, unit: str, fee_pct: float, margin_mode: str) -> str:
    qty_text = f"{plan.position_usdt:,.2f} USDT" if unit == "USDT" else f"{plan.quantity:.6f} {coin}"
    return f"""普通人合约交易计划

币种：{daily.symbol}
当前价格：{daily.current_price:,.2f}
今日一句话：{daily.one_liner}

方向：{okx_side_label(plan.side)}
入场方式：{okx_order_type(plan, daily.current_price)}
入场价：{plan.entry:,.2f}
止损价：{plan.stop:,.2f}
目标1：{plan.target1:,.2f}
目标2：{plan.target2:,.2f}

欧易填写：
模式：{margin_mode}
杠杆：{plan.leverage}x
方向：{okx_side_label(plan.side)}
委托类型：{okx_order_type(plan, daily.current_price)}
价格：{plan.entry:,.2f}
数量：{qty_text}
仓位价值：{plan.position_usdt:,.2f} USDT
保证金：{plan.margin_usdt:,.2f} USDT
止损：{plan.stop:,.2f}
止盈1：{plan.target1:,.2f}
止盈2：{plan.target2:,.2f}

如果止损：-{plan.max_loss_usdt:,.2f} USDT / -{plan.loss_rmb:,.2f} RMB
目标1盈利：+{plan.profit1_usdt:,.2f} USDT / +{plan.profit1_rmb:,.2f} RMB
目标2盈利：+{plan.profit2_usdt:,.2f} USDT / +{plan.profit2_rmb:,.2f} RMB
手续费率：{fee_pct:.3f}%

今天不要做：
{chr(10).join('- ' + item for item in daily.donts)}

纪律：
先看亏多少钱，再决定开多少仓。
先看有没有好点位，再决定要不要进场。
不到计划位置，不开仓。
没有止损，不开仓。

仅供参考，不构成投资建议，不自动下单。
"""


def render_setup(setup: TradeSetup) -> None:
    st.markdown(
        f"<div class='setup-card {risk_css(setup.risk_level)}'><div class='card-title'>{setup.title}</div>{setup_body(setup)}</div>",
        unsafe_allow_html=True,
    )


def plan_trigger_area(daily: DailyPlan, plan: ForcedPlan) -> str:
    if plan.source == "方案A":
        return daily.aggressive.wait_area
    if plan.source == "方案B":
        return daily.conservative.wait_area
    return f"{plan.entry:,.2f} 附近"


def optimal_advice(daily: DailyPlan, forced: ForcedPlan | None, selected_plan: ForcedPlan) -> dict[str, str]:
    current = daily.current_price
    side = normalize_side(selected_plan.side)
    a_low, a_high = setup_range(daily.aggressive)
    b_low, b_high = setup_range(daily.conservative)
    dist_a = distance_to_range(current, a_low, a_high)
    dist_b = distance_to_range(current, b_low, b_high)

    if forced is not None:
        if "立即" in forced.source:
            return {
                "suggestion": "现在强制轻仓",
                "explain": "这是强制进场单，不是最佳位置，只能小仓，不能加仓，不能取消止损。",
                "best": "强制轻仓方案",
                "entry_mode": "现在轻仓",
            }
        return {
            "suggestion": "可轻仓做空" if normalize_side(forced.side) == "short" else "可轻仓做多",
            "explain": f"今天必须做时，系统选择 {forced.source}。如果当前价格不在好位置，现在不要市价追，挂单等 {forced.entry:,.2f}。",
            "best": forced.source,
            "entry_mode": forced.order_type,
        }

    if daily.aggressive.reward_risk < 2 and daily.conservative.reward_risk < 2:
        return {
            "suggestion": "不碰",
            "explain": "方案A和方案B盈亏比都不够，不值得硬做，今天没有好位置。",
            "best": "无",
            "entry_mode": "等待",
        }
    if side == "short":
        if a_low <= current <= a_high:
            return {
                "suggestion": "等待方案A",
                "explain": "当前已经到方案A压力观察区，但还需要转弱确认。观察确认，不要急着追空。",
                "best": "方案A",
                "entry_mode": "限价挂空",
            }
        if b_low <= current <= b_high:
            return {
                "suggestion": "等待方案B",
                "explain": "当前已经到方案B跌破区，但还需要1小时K线跌破并站稳下方。观察确认，不要急着追空。",
                "best": "方案B",
                "entry_mode": "跌破条件单",
            }
        if current < b_low:
            return {
                "suggestion": "可轻仓做空",
                "explain": f"当前价格已经跌破方案B触发区下沿 {b_low:,.2f} 附近，若反抽不回去，可以轻仓执行方案B。",
                "best": "方案B",
                "entry_mode": "跌破条件单",
            }
        if dist_b <= dist_a and daily.conservative.risk_level != "高":
            return {
                "suggestion": "等待方案B",
                "explain": f"当前价格距离方案B跌破区只差 {dist_b:,.2f} USDT，方案A距离 {dist_a:,.2f} USDT。现在不要市价追，等跌破 {b_low:,.2f}-{b_high:,.2f} 并确认后再考虑轻仓开空。",
                "best": "方案B",
                "entry_mode": "跌破条件单",
            }
        return {
            "suggestion": "等待方案A",
            "explain": f"当前价格离方案A压力区更近，等待 {a_low:,.2f}-{a_high:,.2f} 出现转弱信号。不到计划位置，不开仓。",
            "best": "方案A",
            "entry_mode": "限价挂空",
        }
    if a_high < current < b_low:
        nearest = "方案B" if dist_b <= dist_a else "方案A"
        return {
            "suggestion": f"等待{nearest}",
            "explain": f"现在在中间位置，不追单。方案A距离 {dist_a:,.2f} USDT，方案B距离 {dist_b:,.2f} USDT，等价格进入计划区域。",
            "best": nearest,
            "entry_mode": "条件单" if nearest == "方案B" else "限价挂单",
        }
    if a_low <= current <= a_high:
        return {
            "suggestion": "等待方案A",
            "explain": "当前已经到方案A观察区，但还需要确认信号。观察确认，不要急着追。",
            "best": "方案A",
            "entry_mode": "限价挂单",
        }
    if b_low <= current <= b_high:
        return {
            "suggestion": "等待方案B",
            "explain": "当前已经到方案B突破区，但还需要1小时K线站稳。观察确认，不要急着追。",
            "best": "方案B",
            "entry_mode": "条件单",
        }
    if current > b_high:
        return {
            "suggestion": "可轻仓做多",
            "explain": f"当前价格已经站上方案B突破区上沿 {b_high:,.2f} 附近，若回踩不破，可以轻仓执行方案B。",
            "best": "方案B",
            "entry_mode": "条件单",
        }
    if dist_b < dist_a and daily.conservative.risk_level != "高":
        return {
            "suggestion": "等待方案B",
            "explain": f"当前价格距离方案B突破区只差 {dist_b:,.2f} USDT，方案A距离 {dist_a:,.2f} USDT。现在不要市价追，等突破 {b_low:,.2f}-{b_high:,.2f} 并站稳后再考虑轻仓开多。",
            "best": "方案B",
            "entry_mode": "条件单",
        }
    return {
        "suggestion": "等待方案A",
        "explain": f"当前价格离方案A更近，等待 {a_low:,.2f}-{a_high:,.2f} 出现确认信号。不到计划位置，不开仓。",
        "best": "方案A",
        "entry_mode": "限价挂单",
    }


def render_top_card(daily: DailyPlan, forced: ForcedPlan | None, selected_plan: ForcedPlan, long_score: float, short_score: float, choppy_wait: bool = False) -> None:
    advice = optimal_advice(daily, forced, selected_plan)
    if choppy_wait and forced is None:
        advice = {
            "suggestion": "震荡等待",
            "explain": f"系统判断多空都不明显，多头评分 {long_score:.0f}，空头评分 {short_score:.0f}。现在不要硬做，等价格进入方案A/方案B计划区后再重新评估。",
            "best": "等待确认",
            "entry_mode": "等待",
        }
    suggestion = advice["suggestion"]
    action_class = "pill-red" if suggestion == "不碰" or "做空" in suggestion else "pill-green" if "可轻仓" in suggestion or "强制" in suggestion else "pill-yellow"
    market_class = market_pill_class(daily.market_state)
    trigger_area = plan_trigger_area(daily, selected_plan)
    st.markdown(
        f"""
        <div class='hero-card'>
            <div class='hero-symbol'>当前最优建议 · {daily.symbol}</div>
            <div class='hero-price'>{suggestion}</div>
            <div class='hero-line'>{advice["explain"]}</div>
            <div>
                <span class='pill {action_class}'>当前建议：{suggestion}</span>
                <span class='pill {market_class}'>市场状态：{daily.market_state}</span>
                <span class='pill pill-blue'>多 {long_score:.0f} / 空 {short_score:.0f}</span>
            </div>
            <div class='kv-grid' style='margin-top:18px'>
                <div class='kv'><div class='kv-label'>当前价格</div><div class='kv-value'>{daily.current_price:,.2f}</div></div>
                <div class='kv'><div class='kv-label'>最优方案</div><div class='kv-value'>{advice["best"]}</div></div>
                <div class='kv'><div class='kv-label'>方向</div><div class='kv-value'>{okx_side_label(selected_plan.side)}</div></div>
                <div class='kv'><div class='kv-label'>入场方式</div><div class='kv-value'>{advice["entry_mode"]}</div></div>
                <div class='kv'><div class='kv-label'>触发区</div><div class='kv-value yellow'>{trigger_area}</div></div>
                <div class='kv'><div class='kv-label'>理想入场</div><div class='kv-value yellow'>{selected_plan.entry:,.2f}</div></div>
                <div class='kv'><div class='kv-label'>止损</div><div class='kv-value red'>{selected_plan.stop:,.2f}</div></div>
                <div class='kv'><div class='kv-label'>目标1</div><div class='kv-value green'>{selected_plan.target1:,.2f}</div></div>
                <div class='kv'><div class='kv-label'>目标2</div><div class='kv-value green'>{selected_plan.target2:,.2f}</div></div>
                <div class='kv'><div class='kv-label'>最大仓位</div><div class='kv-value'>{selected_plan.position_usdt:,.2f} USDT</div></div>
                <div class='kv'><div class='kv-label'>欧易委托类型</div><div class='kv-value'>{okx_order_type(selected_plan, daily.current_price)}</div></div>
                <div class='kv'><div class='kv-label'>一句话</div><div class='kv-value'>{daily.one_liner}</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def distance_to_setup(current: float, setup: TradeSetup) -> tuple[float, str]:
    low, high = setup_range(setup)
    if low <= current <= high:
        return 0.0, "已经在观察区，等触发条件，不盲目点。"
    if current < low:
        return low - current, "还没到突破/观察价，继续等。"
    return current - high, "价格已经离计划区偏远，别追单。"


def render_distance_panel(daily: DailyPlan) -> None:
    dist_a, note_a = distance_to_setup(daily.current_price, daily.aggressive)
    dist_b, note_b = distance_to_setup(daily.current_price, daily.conservative)
    a_low, a_high = setup_range(daily.aggressive)
    b_low, b_high = setup_range(daily.conservative)
    if a_high < daily.current_price < b_low:
        conclusion = "现在不动，等价格进入计划区域。"
    elif dist_a == 0 or dist_b == 0:
        conclusion = "价格已经进入计划区，观察确认，不要急着追。"
    elif daily.current_price > b_high or daily.current_price < a_low:
        conclusion = "价格已经偏离计划区，不追单。"
    elif dist_b < dist_a:
        conclusion = f"方案A太远，暂时不用管。方案B只差 {dist_b:,.2f} USDT，重点盯突破。"
    else:
        conclusion = f"方案A更近，差 {dist_a:,.2f} USDT，等支撑区确认。"
    max_dist = max(dist_a, dist_b, daily.current_price * 0.01, 0.000001)
    progress_a = int(max(0, min(100, 100 - dist_a / max_dist * 100)))
    progress_b = int(max(0, min(100, 100 - dist_b / max_dist * 100)))
    html_card(
        "当前离计划还差多少",
        f"""
        <div class='kv-grid'>
            <div class='kv'><div class='kv-label'>当前价</div><div class='kv-value'>{daily.current_price:,.2f}</div></div>
            <div class='kv'><div class='kv-label'>结论</div><div class='kv-value yellow'>{conclusion}</div></div>
            <div class='kv'><div class='kv-label'>距离方案A入场区</div><div class='kv-value'>{dist_a:,.2f} USDT</div><div class='muted'>{note_a}</div></div>
            <div class='kv'><div class='kv-label'>距离方案B突破区</div><div class='kv-value'>{dist_b:,.2f} USDT</div><div class='muted'>{note_b}</div></div>
        </div>
        <div style='margin-top:14px'>
            <div class='muted'>方案A接近度</div>
            <div style='height:10px;border-radius:999px;background:#1f2937;overflow:hidden'><div style='height:10px;width:{progress_a}%;background:#f59e0b'></div></div>
            <div class='muted' style='margin-top:10px'>方案B接近度</div>
            <div style='height:10px;border-radius:999px;background:#1f2937;overflow:hidden'><div style='height:10px;width:{progress_b}%;background:#38bdf8'></div></div>
        </div>
        """,
    )


def sidebar_inputs() -> tuple[str, str, str, float, float, int, float, str, float, str, bool]:
    st.sidebar.title("交易参数")
    st.sidebar.caption("输入放这里，首页专心看结论。")
    with st.sidebar.form("entry_form"):
        symbol_input = st.text_input("币种", value=st.session_state.get("symbol_input_v5", "ETH/USDT"), key="symbol_input_v5")
        rmb_capital = st.number_input("人民币本金", min_value=10.0, value=1000.0, step=100.0, key="rmb_capital_v5")
        rate = st.number_input("USDT/CNY 汇率", min_value=0.1, value=7.2, step=0.01, key="rate_v5")
        risk_pct = st.number_input("单笔最大风险 %", min_value=0.1, max_value=20.0, value=1.0, step=0.1, key="risk_pct_v5")
        leverage = st.number_input("杠杆", min_value=1, max_value=50, value=3, step=1, key="leverage_v5")
        direction_options = {"auto": "系统判断", "long": "只做多", "short": "只做空"}
        direction_choice = st.radio("做多/做空", list(direction_options.keys()), index=0, format_func=lambda value: direction_options[value], key="direction_choice_v5")
        must_options = {"normal": "否，正常模式", "today": "是，今天必须做", "now": "是，现在必须进"}
        must_mode = st.radio("是否必须做这一单", list(must_options.keys()), index=0, format_func=lambda value: must_options[value], key="must_mode_v5")
        unit = st.radio("欧易数量单位", ["ETH", "USDT"], horizontal=True, index=0, key="unit_v5")
        fee_pct = st.number_input("手续费率 %", min_value=0.0, max_value=1.0, value=0.05, step=0.01, key="fee_pct_v1")
        margin_mode = st.radio("保证金模式", ["逐仓", "全仓"], index=0, horizontal=True, key="margin_mode_v1")
        clicked = st.form_submit_button("生成交易面板", type="primary", use_container_width=True)
    if risk_pct > 5:
        st.sidebar.error("风险过高，不建议。")
    elif risk_pct > 2:
        st.sidebar.warning("单笔风险超过 2%，建议降到 1% 附近。")
    if leverage > 5:
        st.sidebar.error("普通人不建议高杠杆。")
    if margin_mode == "全仓":
        st.sidebar.error("默认建议逐仓，不建议普通人使用全仓。")
    return symbol_input, must_mode, direction_choice, rmb_capital, risk_pct, int(leverage), rate, unit, fee_pct, margin_mode, clicked


def render_home() -> None:
    st.title("合约交易决策面板")
    st.caption("像交易员的每日计划：先看动作，再看点位，最后看最多亏多少。")
    symbol_input, must_mode, direction_choice, rmb_capital, risk_pct, leverage, rate, unit, fee_pct, margin_mode, clicked = sidebar_inputs()
    required_state = ["daily_plan", "forced_plan", "signals", "coin", "unit"]
    if not clicked and any(key not in st.session_state for key in required_state):
        st.markdown("<div class='notice'>在左侧填写参数，然后点击“生成交易面板”。默认逐仓、3x、单笔风险1%。</div>", unsafe_allow_html=True)
        return
    if risk_pct > 5:
        st.error("风险过高，不建议。单笔风险超过5%很容易几次亏损就伤到账户。")
    ticker, symbol = symbol_to_yfinance(symbol_input)
    if clicked:
        for key in ["daily_plan", "forced_plan", "signals", "coin", "unit", "long_short_scores", "selected_plan", "strong_trend_data", "strong_trend_symbol"]:
            st.session_state.pop(key, None)
        with st.spinner("正在分析并生成方案..."):
            data = load_all_timeframes(ticker)
            signals = [analyze_timeframe(name, df) for name, df in data.items()]
            signals = [item for item in signals if item is not None]
        if len(signals) < 3:
            snapshot = market_snapshot_from_data(data)
            if snapshot is None:
                st.warning("行情数据暂时拿不到完整K线。保守处理：不追高，等回踩，仓位减半；刷新后再重新评估。")
                return
            strong_data = build_strong_trend_plans(
                snapshot["current"],
                snapshot["support"],
                snapshot["resistance"],
                rmb_capital,
                risk_pct,
                int(leverage),
                rate,
                must_mode,
            )
            st.session_state.strong_trend_data = strong_data
            st.session_state.strong_trend_symbol = symbol
            st.session_state.coin = base_coin(symbol)
            st.session_state.unit = unit
            st.session_state.risk_inputs = {"rmb_capital": rmb_capital, "risk_pct": risk_pct, "leverage": min(int(leverage), 3), "rate": rate, "direction_choice": "long", "fee_pct": fee_pct, "margin_mode": margin_mode, "must_mode": must_mode}
            st.session_state.forced_plan = strong_data["selected"] if must_mode == "now" else None
            st.session_state.selected_plan = strong_data["selected"]
            st.session_state.signals = signals
            render_strong_trend_mode(symbol, strong_data, base_coin(symbol), unit, margin_mode, fee_pct, rmb_capital, rate, must_mode)
            st.markdown("<div class='notice'>仅供参考，不构成投资建议，不自动下单。先确定最多亏多少钱，再决定开多少仓。</div>", unsafe_allow_html=True)
            st.markdown("<div class='notice danger-notice'>不到点位不开仓，没有止损不开仓，亏损金额不能接受不开仓。</div>", unsafe_allow_html=True)
            return
        base_daily = build_daily_plan(symbol, signals, rmb_capital / rate, min(risk_pct, 1))
        long_score, short_score = calculate_long_short_scores(signals)
        side = choose_direction(direction_choice, long_score, short_score)
        candidate_plans = build_candidates(signals, side, rmb_capital, risk_pct, int(leverage), rate)
        daily = daily_with_direction_plans(base_daily, candidate_plans, side)
        if normalize_side(side) == "long" and is_strong_trend(signals, daily, candidate_plans):
            current = daily.current_price
            resistance = max(signal.resistance for signal in signals[:3])
            support = min(signal.support for signal in signals[:3])
            strong_data = build_strong_trend_plans(current, support, resistance, rmb_capital, risk_pct, int(leverage), rate, must_mode)
            if must_mode == "now":
                st.session_state.now_force_count = st.session_state.get("now_force_count", 0) + 1
            st.session_state.strong_trend_data = strong_data
            st.session_state.strong_trend_symbol = symbol
            st.session_state.coin = base_coin(symbol)
            st.session_state.unit = unit
            st.session_state.risk_inputs = {"rmb_capital": rmb_capital, "risk_pct": min(risk_pct, 0.5) if must_mode == "now" else risk_pct, "leverage": min(int(leverage), 3), "rate": rate, "direction_choice": "long", "fee_pct": fee_pct, "margin_mode": margin_mode, "must_mode": must_mode}
            st.session_state.forced_plan = strong_data["selected"] if must_mode == "now" else None
            st.session_state.selected_plan = strong_data["selected"]
            st.session_state.signals = signals
            return
        is_choppy_choice = direction_choice == "auto" and abs(long_score - short_score) < 5
        if must_mode == "now":
            st.session_state.now_force_count = st.session_state.get("now_force_count", 0) + 1
            forced = build_now_plan(signals, side, rmb_capital, risk_pct, int(leverage), rate)
        elif must_mode == "today":
            forced = choose_candidate(candidate_plans, daily.current_price)
        else:
            st.session_state.now_force_count = 0
            forced = None
        if forced is not None and is_choppy_choice:
            forced.warning = f"震荡，只能轻仓。多头评分 {long_score:.0f}，空头评分 {short_score:.0f}。{forced.warning}"
        selected_plan = forced or choose_candidate(candidate_plans, daily.current_price)
        st.session_state.daily_plan = daily
        st.session_state.forced_plan = forced
        st.session_state.selected_plan = selected_plan
        st.session_state.signals = signals
        st.session_state.coin = base_coin(symbol)
        st.session_state.unit = unit
        st.session_state.long_short_scores = (long_score, short_score)
        st.session_state.risk_inputs = {"rmb_capital": rmb_capital, "risk_pct": risk_pct, "leverage": int(leverage), "rate": rate, "direction_choice": direction_choice, "fee_pct": fee_pct, "margin_mode": margin_mode, "must_mode": must_mode}
    if st.session_state.get("strong_trend_data") is not None:
        risk_inputs = st.session_state.get("risk_inputs", {"rmb_capital": 1000, "risk_pct": 1, "leverage": 3, "rate": 7.2, "fee_pct": 0.05, "margin_mode": "逐仓", "must_mode": "normal"})
        render_strong_trend_mode(
            st.session_state.get("strong_trend_symbol", symbol),
            st.session_state.strong_trend_data,
            st.session_state.get("coin", base_coin(symbol)),
            st.session_state.get("unit", unit),
            risk_inputs["margin_mode"],
            risk_inputs["fee_pct"],
            risk_inputs["rmb_capital"],
            risk_inputs["rate"],
            risk_inputs.get("must_mode", "normal"),
        )
        st.markdown("<div class='notice'>仅供参考，不构成投资建议，不自动下单。先确定最多亏多少钱，再决定开多少仓。</div>", unsafe_allow_html=True)
        st.markdown("<div class='notice danger-notice'>不到点位不开仓，没有止损不开仓，亏损金额不能接受不开仓。</div>", unsafe_allow_html=True)
        return
    if any(key not in st.session_state for key in required_state):
        st.markdown("<div class='notice'>页面状态已更新，请在左侧重新点击“生成交易面板”。</div>", unsafe_allow_html=True)
        return

    daily, forced, signals = st.session_state.daily_plan, st.session_state.forced_plan, st.session_state.signals
    coin, unit = st.session_state.coin, st.session_state.unit
    long_score, short_score = st.session_state.get("long_short_scores", calculate_long_short_scores(signals))
    risk_inputs = st.session_state.get("risk_inputs", {"rmb_capital": 1000, "risk_pct": 1, "leverage": 3, "rate": 7.2, "direction_choice": "auto", "fee_pct": 0.05, "margin_mode": "逐仓", "must_mode": "normal"})
    selected_plan = st.session_state.get("selected_plan")
    if selected_plan is None:
        selected_side = choose_direction(risk_inputs["direction_choice"], long_score, short_score)
        selected_plan = choose_candidate(
            build_candidates(signals, selected_side, risk_inputs["rmb_capital"], risk_inputs["risk_pct"], risk_inputs["leverage"], risk_inputs["rate"]),
            daily.current_price,
        )
    choppy_wait = risk_inputs.get("direction_choice") == "auto" and abs(long_score - short_score) < 5
    render_top_card(daily, forced, selected_plan, long_score, short_score, choppy_wait)
    render_distance_panel(daily)

    st.markdown("<div class='section-title'>今日两个交易方案</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        render_setup(daily.aggressive)
    with col2:
        render_setup(daily.conservative)

    if forced is None:
        render_forced_plan(
            selected_plan,
            "如果必须做这一单，参考这个方案",
            "正常模式下，不到计划价格不做；如果你非要做，只能照这个方案小仓执行。",
        )
    else:
        render_forced_plan(selected_plan)

    render_okx_fill(selected_plan, coin, unit, risk_inputs["margin_mode"], daily.current_price)
    render_profit_preview(selected_plan, risk_inputs["fee_pct"], risk_inputs["rmb_capital"], risk_inputs["rate"])
    render_trade_value(selected_plan, risk_inputs["risk_pct"], risk_inputs["leverage"])

    render_today_donts(daily, selected_plan)
    render_discipline(
        risk_inputs["risk_pct"],
        risk_inputs["leverage"],
        selected_plan.reward_risk,
        risk_inputs.get("must_mode", "normal"),
        st.session_state.get("now_force_count", 0),
    )
    if max_consecutive_losses(read_trade_records()) >= 3:
        st.error("交易记录显示已经连续亏损 3 次：今天停止交易。")
    html_card("AI分析理由", f"<div style='font-size:18px;line-height:1.7'>{daily.analysis_reason}</div>")
    if st.button("打开高级指标", use_container_width=True):
        st.session_state.page = "高级指标"
        st.rerun()
    if st.button("保存本次计划", use_container_width=True):
        append_trade_record(build_trade_record(daily, selected_plan, risk_inputs.get("must_mode", "normal"), risk_inputs["margin_mode"], risk_inputs["fee_pct"]))
        st.success(f"已保存到本地 CSV：{TRADE_RECORDS_FILE}")
    st.download_button(
        "下载交易计划 TXT",
        data=build_plan_text(daily, selected_plan, coin, unit, risk_inputs["fee_pct"], risk_inputs["margin_mode"]),
        file_name=f"{coin}_trade_plan.txt",
        mime="text/plain",
        use_container_width=True,
    )
    st.markdown("<div class='notice'>仅供参考，不构成投资建议，不自动下单。先确定最多亏多少钱，再决定开多少仓。</div>", unsafe_allow_html=True)
    st.markdown("<div class='notice danger-notice'>不到点位不开仓，没有止损不开仓，亏损金额不能接受不开仓。</div>", unsafe_allow_html=True)


def render_order_helper() -> None:
    st.title("合约下单助手")
    st.caption("普通人版：先看亏多少钱，再看能赚多少钱。")
    with st.form("order_helper_form"):
        col1, col2, col3 = st.columns(3)
        symbol = col1.text_input("币种", value=st.session_state.get("symbol_input", "ETH/USDT"))
        rmb_capital = col2.number_input("人民币本金", min_value=10.0, value=1000.0, step=100.0)
        rate = col3.number_input("USDT/CNY 汇率", min_value=0.1, value=7.2, step=0.01)
        col1, col2, col3 = st.columns(3)
        current_price = col1.number_input("当前币种价格", min_value=0.0001, value=1662.0, step=1.0)
        side = col2.radio("做多/做空", ["做多", "做空"], horizontal=True)
        leverage = col3.number_input("杠杆倍数", min_value=1, max_value=50, value=3, step=1)
        col1, col2, col3, col4 = st.columns(4)
        entry = col1.number_input("入场价", min_value=0.0001, value=1662.0, step=1.0)
        stop = col2.number_input("止损价", min_value=0.0001, value=1645.0, step=1.0)
        target1 = col3.number_input("目标价1", min_value=0.0001, value=1696.0, step=1.0)
        target2 = col4.number_input("目标价2", min_value=0.0001, value=1713.0, step=1.0)
        col1, col2 = st.columns(2)
        risk_pct = col1.slider("单笔最大亏损比例", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        fee_pct = col2.number_input("手续费率 %", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
        submitted = st.form_submit_button("计算这单赚亏", type="primary", use_container_width=True)
    if not submitted and "order_helper_result" not in st.session_state:
        st.info("填好入场、止损、目标和手续费后点按钮。")
        return
    if submitted:
        st.session_state.order_helper_result = calculate_order_helper(rmb_capital, current_price, side, entry, stop, target1, target2, int(leverage), risk_pct, fee_pct, rate)
        st.session_state.order_helper_inputs = {"coin": base_coin(symbol), "side": side, "entry": entry, "stop": stop, "leverage": int(leverage), "rmb_capital": rmb_capital, "rate": rate}
    result, inputs = st.session_state.order_helper_result, st.session_state.order_helper_inputs
    if result.risk_too_high:
        st.error(result.warning)
    else:
        st.success(result.warning)
    st.subheader("这单赚亏一眼看懂")
    st.markdown(f"**本金：{inputs['rmb_capital']:,.2f} 人民币 ≈ {result.usdt_capital:,.2f} USDT**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(money_html("如果错了：预计亏损", result.estimated_loss, inputs["rate"], False), unsafe_allow_html=True)
    with col2:
        st.markdown(money_html("如果到目标1：预计盈利", result.profit1_usdt, inputs["rate"], True), unsafe_allow_html=True)
    with col3:
        st.markdown(money_html("如果到目标2：预计盈利", result.profit2_usdt, inputs["rate"], True), unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1.container(border=True):
        st.subheader("手续费预估")
        st.metric("开仓手续费", f"{result.open_fee_usdt:,.2f} USDT")
        st.metric("平仓手续费（止损）", f"{result.close_fee_stop_usdt:,.2f} USDT")
        st.metric("合计（按止损）", f"{result.total_fee_stop_usdt:,.2f} USDT")
    with col2.container(border=True):
        st.subheader("扣除手续费后")
        st.markdown(money_html("止损实际亏损", result.net_loss_stop_usdt, inputs["rate"], False), unsafe_allow_html=True)
        st.markdown(money_html("目标1实际盈利", result.net_profit1_usdt, inputs["rate"], True), unsafe_allow_html=True)
        st.markdown(money_html("目标2实际盈利", result.net_profit2_usdt, inputs["rate"], True), unsafe_allow_html=True)
    plan = forced_plan("欧易填写", "手动计算", inputs["side"], "限价", inputs["entry"], inputs["stop"], result.target1, result.target2, inputs["leverage"], inputs["rmb_capital"], 1, inputs["rate"], "")
    plan.quantity, plan.position_usdt, plan.margin_usdt = result.quantity, result.position_usdt, result.margin_usdt
    render_okx_fill(plan, inputs["coin"], inputs["coin"])
    st.info("先看亏多少钱，再看能赚多少钱。")


def render_advanced(ticker: str) -> None:
    st.header("高级指标")
    period = st.selectbox("历史周期", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
    interval = st.selectbox("K线级别", ["1d", "1h", "30m", "15m"], index=0)
    raw = load_market_data(ticker, period, interval)
    if raw.empty or len(raw) < 70:
        st.error("没有获取到足够的K线数据。请稍后刷新，或换一个主流币种。")
        return
    df = add_indicators(raw)
    latest = df.iloc[-1]
    cols = st.columns(4)
    cols[0].metric("当前价格", f"${latest['Close']:,.2f}")
    cols[1].metric("MA20", f"${latest['MA20']:,.2f}")
    cols[2].metric("MA60", f"${latest['MA60']:,.2f}")
    cols[3].metric("RSI14", f"{latest['RSI14']:.1f}")
    st.plotly_chart(build_price_chart(df, ticker), use_container_width=True)
    st.plotly_chart(build_indicator_chart(df), use_container_width=True)
    st.subheader("简单回测")
    col1, col2 = st.columns(2)
    initial_cash = col1.number_input("初始资金 USDT", min_value=100.0, value=10000.0, step=500.0)
    fee_pct = col2.number_input("单次换仓手续费 %", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
    metrics, bt = run_backtest(df, initial_cash, fee_pct)
    if metrics:
        cols = st.columns(5)
        cols[0].metric("策略收益率", format_pct(metrics["策略收益率"]))
        cols[1].metric("买入持有", format_pct(metrics["买入持有收益率"]))
        cols[2].metric("最大回撤", format_pct(metrics["最大回撤"]))
        cols[3].metric("胜率", format_pct(metrics["胜率"]))
        cols[4].metric("交易次数", f"{metrics['交易次数']}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=bt.index, y=bt["equity"], name="策略资金曲线", line=dict(color="#2563eb")))
        fig.add_trace(go.Scatter(x=bt.index, y=bt["buy_hold_equity"], name="买入持有", line=dict(color="#16a34a")))
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    inject_css()
    if "page" not in st.session_state:
        st.session_state.page = "AI进场助手"
    nav1, nav2, nav3, nav4 = st.columns(4)
    if nav1.button("AI进场助手", use_container_width=True):
        st.session_state.page = "AI进场助手"
    if nav2.button("合约下单助手", use_container_width=True):
        st.session_state.page = "合约下单助手"
    if nav3.button("交易记录与复盘", use_container_width=True):
        st.session_state.page = "交易记录与复盘"
    if nav4.button("高级指标", use_container_width=True):
        st.session_state.page = "高级指标"
    if st.session_state.page == "AI进场助手":
        render_home()
    elif st.session_state.page == "合约下单助手":
        render_order_helper()
    elif st.session_state.page == "交易记录与复盘":
        render_trade_review()
    else:
        advanced_symbol = st.text_input("高级指标币种", value=st.session_state.get("symbol_input", "ETH/USDT"))
        ticker, _ = symbol_to_yfinance(advanced_symbol)
        render_advanced(ticker)


if __name__ == "__main__":
    main()
