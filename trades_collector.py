"""
Trades WebSocket collector.
Connects to MMT trades channel, stores every trade, flags large ones via z-score.
Detects TWAP patterns (regular interval + consistent size).

Run alongside collector.py:
    python3 trades_collector.py
"""
import os
import json
import time
import threading
import numpy as np
from collections import deque
from dotenv import load_dotenv

load_dotenv()

try:
    import websocket
except ImportError:
    print("Installing websocket-client...")
    import subprocess
    subprocess.check_call(["pip", "install", "websocket-client"])
    import websocket

from config import EXCHANGE, SYMBOL
from db import get_conn, insert_trades_batch

API_KEY = os.getenv("MMT_API_KEY")
WS_URL = f"wss://eu-central-1.mmt.gg/api/v1/ws?api_key={API_KEY}&format=json"

# Z-score config
ZSCORE_LOOKBACK = 12 * 3600  # 12 hours of trades for z-score reference
ZSCORE_THRESHOLD = 2.0  # flag trades above this z-score
TWAP_CHECK_INTERVAL = 300  # check for TWAP patterns every 5 min

# Rolling buffer: (timestamp, size_usd) — keeps 12h window
trade_history = []  # list of (ts, size_usd), pruned periodically
# Buffer for batch inserts
trade_buffer = []
BUFFER_FLUSH_INTERVAL = 10  # flush every 10 seconds

# TWAP detection state
recent_buys = deque(maxlen=200)  # (timestamp, size) of recent buys
recent_sells = deque(maxlen=200)


def prune_history():
    """Remove trades older than 12h from the reference window."""
    global trade_history
    cutoff = time.time() - ZSCORE_LOOKBACK
    trade_history = [(ts, s) for ts, s in trade_history if ts >= cutoff]


def compute_z_score(value):
    """Compute z-score of value against last 12h of trades."""
    prune_history()
    if len(trade_history) < 30:
        return 0.0
    sizes = np.array([s for _, s in trade_history])
    mean = sizes.mean()
    std = sizes.std()
    if std == 0:
        return 0.0
    return (value - mean) / std


def detect_twap(trades_deque, min_trades=10, max_cv_size=0.3, max_cv_interval=0.5):
    """
    Detect TWAP pattern: regular intervals + consistent sizes.
    Returns (is_twap, details) tuple.
    CV = coefficient of variation (std/mean). Lower = more consistent.
    """
    if len(trades_deque) < min_trades:
        return False, None

    trades = list(trades_deque)
    # Only look at last 5 minutes
    cutoff = time.time() - 300
    trades = [(t, s) for t, s in trades if t >= cutoff]
    if len(trades) < min_trades:
        return False, None

    sizes = [s for _, s in trades]
    timestamps = [t for t, _ in trades]

    # Check size consistency
    size_mean = np.mean(sizes)
    size_std = np.std(sizes)
    size_cv = size_std / size_mean if size_mean > 0 else 999

    # Check interval consistency
    intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
    if not intervals:
        return False, None
    int_mean = np.mean(intervals)
    int_std = np.std(intervals)
    int_cv = int_std / int_mean if int_mean > 0 else 999

    is_twap = size_cv < max_cv_size and int_cv < max_cv_interval

    details = {
        "trades": len(trades),
        "avg_size": round(size_mean, 2),
        "size_cv": round(size_cv, 3),
        "avg_interval": round(int_mean, 2),
        "interval_cv": round(int_cv, 3),
        "duration_sec": round(timestamps[-1] - timestamps[0], 1),
    }
    return is_twap, details


def on_message(ws, message):
    try:
        msg = json.loads(message)
    except json.JSONDecodeError:
        return

    if msg.get("type") != "data" or msg.get("channel") != "trades":
        return

    data = msg.get("data", {})
    if not data:
        return

    ts = data.get("t")
    price = data.get("p")
    qty = data.get("q")  # quantity in base token
    # b = is buyer maker. False = market buy (taker bought), True = market sell (taker sold)
    side = "sell" if data.get("b") else "buy"

    if not all([ts, price, qty]):
        return

    # Timestamp is in milliseconds
    if ts > 1e12:
        ts = ts / 1000

    size_usd = price * qty
    trade_history.append((ts, size_usd))
    z = compute_z_score(size_usd)
    is_large = 1 if z >= ZSCORE_THRESHOLD else 0

    trade = {
        "ts": ts,
        "price": price,
        "size_usd": size_usd,
        "side": side,
        "is_large": is_large,
        "z_score": round(z, 2),
    }
    trade_buffer.append(trade)

    # Track for TWAP
    if side == "buy":
        recent_buys.append((trade["ts"], size_usd))
    else:
        recent_sells.append((trade["ts"], size_usd))

    if is_large:
        print(f"  🐋 LARGE {side.upper()} ${size_usd:,.0f} @ {price:.4f} (z={z:.1f})")


def flush_buffer():
    """Periodically flush trade buffer to DB."""
    conn = get_conn()
    while True:
        time.sleep(BUFFER_FLUSH_INTERVAL)
        if trade_buffer:
            batch = trade_buffer.copy()
            trade_buffer.clear()
            try:
                insert_trades_batch(conn, EXCHANGE, SYMBOL, batch)
            except Exception as e:
                print(f"DB flush error: {e}")


def twap_monitor():
    """Periodically check for TWAP patterns."""
    while True:
        time.sleep(TWAP_CHECK_INTERVAL)
        for label, dq in [("BUY", recent_buys), ("SELL", recent_sells)]:
            is_twap, details = detect_twap(dq)
            if is_twap:
                print(f"  ⚠️  TWAP {label} detected: {details['trades']} trades, "
                      f"avg ${details['avg_size']:,.0f}, "
                      f"interval {details['avg_interval']:.1f}s, "
                      f"over {details['duration_sec']:.0f}s")


def on_error(ws, error):
    print(f"WS error: {error}")


def on_close(ws, close_status_code, close_msg):
    print(f"WS closed: {close_status_code} {close_msg}")


def on_open(ws):
    sub = {
        "type": "subscribe",
        "channel": "trades",
        "exchange": EXCHANGE,
        "symbol": SYMBOL,
    }
    ws.send(json.dumps(sub))
    print(f"Subscribed to trades: {EXCHANGE} {SYMBOL}")


def main():
    print(f"Trades collector starting: {EXCHANGE} {SYMBOL}")
    print(f"Z-score threshold: {ZSCORE_THRESHOLD}, window: {ZSCORE_WINDOW}")

    # Start background threads
    flush_thread = threading.Thread(target=flush_buffer, daemon=True)
    flush_thread.start()

    twap_thread = threading.Thread(target=twap_monitor, daemon=True)
    twap_thread.start()

    while True:
        try:
            ws = websocket.WebSocketApp(
                WS_URL,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
            )
            ws.run_forever(ping_interval=30, ping_timeout=10)
        except Exception as e:
            print(f"Connection error: {e}")

        print("Reconnecting in 5s...")
        time.sleep(5)


if __name__ == "__main__":
    main()
