"""
WebSocket collector for trades + order book depth.
- Trades: stores every trade, flags large ones (z-score > 2σ vs 12h), detects TWAP
- Depth: maintains local order book, detects significant liquidity changes,
  cross-references with trades to determine fill vs pull

Uses 2 WS connections (trades + depth) out of 5 available.

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
    import subprocess
    subprocess.check_call(["pip", "install", "websocket-client"])
    import websocket

from config import EXCHANGE, SYMBOL
from db import get_conn, insert_trades_batch, insert_depth_events

API_KEY = os.getenv("MMT_API_KEY")
WS_URL = f"wss://eu-central-1.mmt.gg/api/v1/ws?api_key={API_KEY}&format=json"

# ── Trade tracking ──────────────────────────────────────────────────────────

ZSCORE_LOOKBACK = 12 * 3600  # 12h
ZSCORE_THRESHOLD = 2.0
TWAP_CHECK_INTERVAL = 300

trade_history = []
trade_buffer = []
BUFFER_FLUSH_INTERVAL = 10

recent_buys = deque(maxlen=200)
recent_sells = deque(maxlen=200)

# Recent trades for cross-referencing with depth (last 60s)
recent_trade_prices = deque(maxlen=500)  # (ts, price, side, size_usd)


def prune_history():
    global trade_history
    cutoff = time.time() - ZSCORE_LOOKBACK
    trade_history = [(ts, s) for ts, s in trade_history if ts >= cutoff]


def compute_z_score(value):
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
    if len(trades_deque) < min_trades:
        return False, None
    trades = list(trades_deque)
    cutoff = time.time() - 300
    trades = [(t, s) for t, s in trades if t >= cutoff]
    if len(trades) < min_trades:
        return False, None
    sizes = [s for _, s in trades]
    timestamps = [t for t, _ in trades]
    size_cv = np.std(sizes) / np.mean(sizes) if np.mean(sizes) > 0 else 999
    intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
    if not intervals:
        return False, None
    int_cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 999
    is_twap = size_cv < max_cv_size and int_cv < max_cv_interval
    details = {
        "trades": len(trades), "avg_size": round(np.mean(sizes), 2),
        "avg_interval": round(np.mean(intervals), 2),
        "duration_sec": round(timestamps[-1] - timestamps[0], 1),
    }
    return is_twap, details


# ── Order book tracking ─────────────────────────────────────────────────────

# Local order book: {price: qty}
local_bids = {}
local_asks = {}
book_initialized = False
last_price = 0.0

# Minimum USD value change to log as an event
MIN_EVENT_USD = 500  # only track changes > $500

# Buffer for depth events
depth_event_buffer = []


def was_price_traded(price, side, lookback_sec=10):
    """Check if a trade happened at/near this price recently."""
    now = time.time()
    cutoff = now - lookback_sec
    for ts, tp, ts_side, _ in recent_trade_prices:
        if ts < cutoff:
            continue
        # For bids being removed: was there a sell trade at this price?
        # For asks being removed: was there a buy trade at this price?
        if side == "bid" and ts_side == "sell" and abs(tp - price) < 0.0005:
            return True
        if side == "ask" and ts_side == "buy" and abs(tp - price) < 0.0005:
            return True
    return False


def process_depth_update(data):
    """Process incremental order book update from depth channel."""
    global book_initialized, last_price

    # Handle snapshot vs update
    if not book_initialized:
        # First message or snapshot — initialize
        if "b" in data and "a" in data:
            local_bids.clear()
            local_asks.clear()
            for entry in data.get("b", []):
                price, qty = entry[0], entry[1]
                if qty > 0:
                    local_bids[price] = qty
            for entry in data.get("a", []):
                price, qty = entry[0], entry[1]
                if qty > 0:
                    local_asks[price] = qty
            if data.get("lp"):
                last_price = data["lp"]
            book_initialized = True
            print(f"  Book initialized: {len(local_bids)} bids, {len(local_asks)} asks")
            return

    # Incremental update
    if data.get("lp"):
        last_price = data["lp"]

    now = time.time()

    for entry in data.get("b", []):
        price, qty = entry[0], entry[1]
        old_qty = local_bids.get(price, 0)
        size_usd_change = abs(qty - old_qty) * price

        if size_usd_change >= MIN_EVENT_USD:
            if qty == 0 and old_qty > 0:
                # Bid removed
                filled = was_price_traded(price, "bid")
                depth_event_buffer.append({
                    "ts": now, "price": price, "side": "bid",
                    "type": "filled" if filled else "pulled",
                    "size_before": old_qty * price, "size_after": 0,
                    "size_usd": old_qty * price, "filled": 1 if filled else 0,
                })
                if filled:
                    print(f"  FILLED bid ${old_qty * price:,.0f} @ {price:.4f}")
                else:
                    print(f"  PULLED bid ${old_qty * price:,.0f} @ {price:.4f}")
            elif qty > old_qty:
                # Bid added/increased
                added_usd = (qty - old_qty) * price
                if added_usd >= MIN_EVENT_USD:
                    depth_event_buffer.append({
                        "ts": now, "price": price, "side": "bid",
                        "type": "added",
                        "size_before": old_qty * price, "size_after": qty * price,
                        "size_usd": added_usd, "filled": 0,
                    })
                    print(f"  NEW bid +${added_usd:,.0f} @ {price:.4f} (total ${qty * price:,.0f})")
            elif qty < old_qty and qty > 0:
                # Bid reduced
                reduced_usd = (old_qty - qty) * price
                if reduced_usd >= MIN_EVENT_USD:
                    filled = was_price_traded(price, "bid")
                    depth_event_buffer.append({
                        "ts": now, "price": price, "side": "bid",
                        "type": "partially_filled" if filled else "reduced",
                        "size_before": old_qty * price, "size_after": qty * price,
                        "size_usd": reduced_usd, "filled": 1 if filled else 0,
                    })

        if qty == 0:
            local_bids.pop(price, None)
        else:
            local_bids[price] = qty

    for entry in data.get("a", []):
        price, qty = entry[0], entry[1]
        old_qty = local_asks.get(price, 0)
        size_usd_change = abs(qty - old_qty) * price

        if size_usd_change >= MIN_EVENT_USD:
            if qty == 0 and old_qty > 0:
                filled = was_price_traded(price, "ask")
                depth_event_buffer.append({
                    "ts": now, "price": price, "side": "ask",
                    "type": "filled" if filled else "pulled",
                    "size_before": old_qty * price, "size_after": 0,
                    "size_usd": old_qty * price, "filled": 1 if filled else 0,
                })
                if filled:
                    print(f"  FILLED ask ${old_qty * price:,.0f} @ {price:.4f}")
                else:
                    print(f"  PULLED ask ${old_qty * price:,.0f} @ {price:.4f}")
            elif qty > old_qty:
                added_usd = (qty - old_qty) * price
                if added_usd >= MIN_EVENT_USD:
                    depth_event_buffer.append({
                        "ts": now, "price": price, "side": "ask",
                        "type": "added",
                        "size_before": old_qty * price, "size_after": qty * price,
                        "size_usd": added_usd, "filled": 0,
                    })
                    print(f"  NEW ask +${added_usd:,.0f} @ {price:.4f} (total ${qty * price:,.0f})")
            elif qty < old_qty and qty > 0:
                reduced_usd = (old_qty - qty) * price
                if reduced_usd >= MIN_EVENT_USD:
                    filled = was_price_traded(price, "ask")
                    depth_event_buffer.append({
                        "ts": now, "price": price, "side": "ask",
                        "type": "partially_filled" if filled else "reduced",
                        "size_before": old_qty * price, "size_after": qty * price,
                        "size_usd": reduced_usd, "filled": 1 if filled else 0,
                    })

        if qty == 0:
            local_asks.pop(price, None)
        else:
            local_asks[price] = qty


# ── WS message handler ──────────────────────────────────────────────────────

def on_message(ws, message):
    try:
        msg = json.loads(message)
    except json.JSONDecodeError:
        return

    if msg.get("type") != "data":
        return

    channel = msg.get("channel")
    data = msg.get("data", {})
    if not data:
        return

    if channel == "trades":
        ts = data.get("t")
        price = data.get("p")
        qty = data.get("q")
        side = "sell" if data.get("b") else "buy"

        if not all([ts, price, qty]):
            return
        if ts > 1e12:
            ts = ts / 1000

        size_usd = price * qty
        trade_history.append((ts, size_usd))
        recent_trade_prices.append((ts, price, side, size_usd))
        z = compute_z_score(size_usd)
        is_large = 1 if z >= ZSCORE_THRESHOLD else 0

        trade_buffer.append({
            "ts": ts, "price": price, "size_usd": size_usd,
            "side": side, "is_large": is_large, "z_score": round(z, 2),
        })

        if side == "buy":
            recent_buys.append((ts, size_usd))
        else:
            recent_sells.append((ts, size_usd))

        if is_large:
            print(f"  LARGE {side.upper()} ${size_usd:,.0f} @ {price:.4f} (z={z:.1f})")

    elif channel in ("depth", "orderbook"):
        process_depth_update(data)


def flush_buffers():
    """Periodically flush trade and depth event buffers to DB."""
    conn = get_conn()
    while True:
        time.sleep(BUFFER_FLUSH_INTERVAL)
        if trade_buffer:
            batch = trade_buffer.copy()
            trade_buffer.clear()
            try:
                insert_trades_batch(conn, EXCHANGE, SYMBOL, batch)
            except Exception as e:
                print(f"Trade DB error: {e}")
        if depth_event_buffer:
            batch = depth_event_buffer.copy()
            depth_event_buffer.clear()
            try:
                insert_depth_events(conn, EXCHANGE, SYMBOL, batch)
            except Exception as e:
                print(f"Depth event DB error: {e}")


def twap_monitor():
    while True:
        time.sleep(TWAP_CHECK_INTERVAL)
        for label, dq in [("BUY", recent_buys), ("SELL", recent_sells)]:
            is_twap, details = detect_twap(dq)
            if is_twap:
                print(f"  TWAP {label}: {details['trades']} trades, "
                      f"avg ${details['avg_size']:,.0f}, "
                      f"interval {details['avg_interval']:.1f}s, "
                      f"over {details['duration_sec']:.0f}s")


def on_error(ws, error):
    print(f"WS error: {error}")


def on_close(ws, close_status_code, close_msg):
    global book_initialized
    book_initialized = False
    print(f"WS closed: {close_status_code} {close_msg}")


def on_open(ws):
    # Subscribe to trades
    ws.send(json.dumps({
        "type": "subscribe", "channel": "trades",
        "exchange": EXCHANGE, "symbol": SYMBOL,
    }))
    # Subscribe to depth (incremental order book)
    ws.send(json.dumps({
        "type": "subscribe", "channel": "depth",
        "exchange": EXCHANGE, "symbol": SYMBOL,
    }))
    print(f"Subscribed to trades + depth: {EXCHANGE} {SYMBOL}")


def main():
    print(f"WS collector starting: {EXCHANGE} {SYMBOL}")
    print(f"Trades: z-score > {ZSCORE_THRESHOLD}σ vs 12h | Depth: events > ${MIN_EVENT_USD}")

    flush_thread = threading.Thread(target=flush_buffers, daemon=True)
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
