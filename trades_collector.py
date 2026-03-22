"""
WebSocket collector for trades + order book depth across all spot exchanges.
- Trades WS: all 4 exchanges (binance, bybit, coinbase, okx)
- Depth WS: binance only (primary, most liquid)
- Total: 5 WS connections (= max on Basic plan)

Trades: stores every trade, flags large ones (z-score > 2σ vs 12h), detects TWAP
Depth: maintains local order book, detects significant liquidity changes,
  cross-references with trades to determine fill vs pull

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

from config import EXCHANGES, SYMBOL, PRIMARY_EXCHANGE
from db import get_conn, insert_trades_batch, insert_depth_events

API_KEY = os.getenv("MMT_API_KEY")
WS_URL = f"wss://eu-central-1.mmt.gg/api/v1/ws?api_key={API_KEY}&format=json"

# ── Trade tracking (per exchange) ──────────────────────────────────────────

ZSCORE_LOOKBACK = 1 * 3600  # 1h
ZSCORE_THRESHOLD = 2.0
TWAP_CHECK_INTERVAL = 300
BUFFER_FLUSH_INTERVAL = 10

# Per-exchange state
class ExchangeState:
    def __init__(self, exchange):
        self.exchange = exchange
        self.trade_history = []  # (ts, size_usd) for z-score
        self.trade_buffer = []
        self.recent_buys = deque(maxlen=200)
        self.recent_sells = deque(maxlen=200)
        self.recent_trade_prices = deque(maxlen=500)

    def prune_history(self):
        cutoff = time.time() - ZSCORE_LOOKBACK
        self.trade_history = [(ts, s) for ts, s in self.trade_history if ts >= cutoff]

    def compute_z_score(self, value):
        self.prune_history()
        if len(self.trade_history) < 30:
            return 0.0
        sizes = np.array([s for _, s in self.trade_history])
        mean = sizes.mean()
        std = sizes.std()
        if std == 0:
            return 0.0
        return (value - mean) / std


exchange_states = {exch: ExchangeState(exch) for exch in EXCHANGES}


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


# ── Order book tracking (primary exchange only) ───────────────────────────

local_bids = {}
local_asks = {}
book_initialized = False
last_price = 0.0
MIN_EVENT_USD = 500
depth_event_buffer = []


def was_price_traded(price, side, lookback_sec=10):
    """Check if a trade happened at/near this price recently (primary exchange)."""
    state = exchange_states[PRIMARY_EXCHANGE]
    now = time.time()
    cutoff = now - lookback_sec
    for ts, tp, ts_side, _ in state.recent_trade_prices:
        if ts < cutoff:
            continue
        if side == "bid" and ts_side == "sell" and abs(tp - price) < 0.0005:
            return True
        if side == "ask" and ts_side == "buy" and abs(tp - price) < 0.0005:
            return True
    return False


def process_depth_update(data):
    global book_initialized, last_price

    if not book_initialized:
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
            print(f"  [{PRIMARY_EXCHANGE}] Book initialized: {len(local_bids)} bids, {len(local_asks)} asks")
            return

    if data.get("lp"):
        last_price = data["lp"]

    now = time.time()

    for entry in data.get("b", []):
        price, qty = entry[0], entry[1]
        old_qty = local_bids.get(price, 0)
        size_usd_change = abs(qty - old_qty) * price

        if size_usd_change >= MIN_EVENT_USD:
            if qty == 0 and old_qty > 0:
                filled = was_price_traded(price, "bid")
                depth_event_buffer.append({
                    "ts": now, "price": price, "side": "bid",
                    "type": "filled" if filled else "pulled",
                    "size_before": old_qty * price, "size_after": 0,
                    "size_usd": old_qty * price, "filled": 1 if filled else 0,
                })
                tag = "FILLED" if filled else "PULLED"
                print(f"  [{PRIMARY_EXCHANGE}] {tag} bid ${old_qty * price:,.0f} @ {price:.4f}")
            elif qty > old_qty:
                added_usd = (qty - old_qty) * price
                if added_usd >= MIN_EVENT_USD:
                    depth_event_buffer.append({
                        "ts": now, "price": price, "side": "bid",
                        "type": "added",
                        "size_before": old_qty * price, "size_after": qty * price,
                        "size_usd": added_usd, "filled": 0,
                    })
                    print(f"  [{PRIMARY_EXCHANGE}] NEW bid +${added_usd:,.0f} @ {price:.4f}")
            elif qty < old_qty and qty > 0:
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
                tag = "FILLED" if filled else "PULLED"
                print(f"  [{PRIMARY_EXCHANGE}] {tag} ask ${old_qty * price:,.0f} @ {price:.4f}")
            elif qty > old_qty:
                added_usd = (qty - old_qty) * price
                if added_usd >= MIN_EVENT_USD:
                    depth_event_buffer.append({
                        "ts": now, "price": price, "side": "ask",
                        "type": "added",
                        "size_before": old_qty * price, "size_after": qty * price,
                        "size_usd": added_usd, "filled": 0,
                    })
                    print(f"  [{PRIMARY_EXCHANGE}] NEW ask +${added_usd:,.0f} @ {price:.4f}")
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


# ── WS connection per exchange ─────────────────────────────────────────────

def make_ws_handler(exchange, subscribe_depth=False):
    """Create WS handlers for a specific exchange."""

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
            state = exchange_states[exchange]
            ts = data.get("t")
            price = data.get("p")
            qty = data.get("q")
            side = "sell" if data.get("b") else "buy"

            if not all([ts, price, qty]):
                return
            if ts > 1e12:
                ts = ts / 1000

            size_usd = price * qty
            state.trade_history.append((ts, size_usd))
            state.recent_trade_prices.append((ts, price, side, size_usd))
            z = state.compute_z_score(size_usd)
            is_large = 1 if z >= ZSCORE_THRESHOLD else 0

            state.trade_buffer.append({
                "ts": ts, "price": price, "size_usd": size_usd,
                "side": side, "is_large": is_large, "z_score": round(z, 2),
            })

            if side == "buy":
                state.recent_buys.append((ts, size_usd))
            else:
                state.recent_sells.append((ts, size_usd))

            if is_large:
                print(f"  [{exchange}] LARGE {side.upper()} ${size_usd:,.0f} @ {price:.4f} (z={z:.1f})")

        elif channel in ("depth", "orderbook") and subscribe_depth:
            process_depth_update(data)

    def on_error(ws, error):
        print(f"  [{exchange}] WS error: {error}")

    def on_close(ws, code, msg):
        global book_initialized
        if subscribe_depth:
            book_initialized = False
        print(f"  [{exchange}] WS closed: {code} {msg}")

    def on_open(ws):
        ws.send(json.dumps({
            "type": "subscribe", "channel": "trades",
            "exchange": exchange, "symbol": SYMBOL,
        }))
        channels = ["trades"]
        if subscribe_depth:
            ws.send(json.dumps({
                "type": "subscribe", "channel": "depth",
                "exchange": exchange, "symbol": SYMBOL,
            }))
            channels.append("depth")
        print(f"  [{exchange}] Subscribed: {', '.join(channels)}")

    return on_open, on_message, on_error, on_close


def run_ws(exchange, subscribe_depth=False):
    """Run a WS connection for one exchange, auto-reconnects."""
    on_open, on_message, on_error, on_close = make_ws_handler(exchange, subscribe_depth)

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
            print(f"  [{exchange}] Connection error: {e}")
        print(f"  [{exchange}] Reconnecting in 5s...")
        time.sleep(5)


# ── Buffer flushing ────────────────────────────────────────────────────────

def flush_buffers():
    conn = get_conn()
    while True:
        time.sleep(BUFFER_FLUSH_INTERVAL)
        for exch, state in exchange_states.items():
            if state.trade_buffer:
                batch = state.trade_buffer.copy()
                state.trade_buffer.clear()
                try:
                    insert_trades_batch(conn, exch, SYMBOL, batch)
                except Exception as e:
                    print(f"  [{exch}] Trade DB error: {e}")
        if depth_event_buffer:
            batch = depth_event_buffer.copy()
            depth_event_buffer.clear()
            try:
                insert_depth_events(conn, PRIMARY_EXCHANGE, SYMBOL, batch)
            except Exception as e:
                print(f"  [{PRIMARY_EXCHANGE}] Depth event DB error: {e}")


def twap_monitor():
    while True:
        time.sleep(TWAP_CHECK_INTERVAL)
        for exch, state in exchange_states.items():
            for label, dq in [("BUY", state.recent_buys), ("SELL", state.recent_sells)]:
                is_twap, details = detect_twap(dq)
                if is_twap:
                    print(f"  [{exch}] TWAP {label}: {details['trades']} trades, "
                          f"avg ${details['avg_size']:,.0f}, "
                          f"interval {details['avg_interval']:.1f}s, "
                          f"over {details['duration_sec']:.0f}s")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    exchanges = list(EXCHANGES.keys())
    print(f"WS collector: {len(exchanges)} exchanges × {SYMBOL}")
    print(f"Trades: all | Depth: {PRIMARY_EXCHANGE} only")
    print(f"WS connections: {len(exchanges) + 1} (trades×{len(exchanges)} + depth×1)")

    # Start flush + TWAP threads
    threading.Thread(target=flush_buffers, daemon=True).start()
    threading.Thread(target=twap_monitor, daemon=True).start()

    # Start WS threads — one per exchange
    threads = []
    for exch in exchanges:
        depth = (exch == PRIMARY_EXCHANGE)
        t = threading.Thread(target=run_ws, args=(exch, depth), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(1)  # stagger connections

    print(f"\nAll {len(threads)} WS connections started")

    # Keep main thread alive
    try:
        while True:
            time.sleep(60)
            # Print stats
            parts = []
            for exch, state in exchange_states.items():
                n = len(state.trade_history)
                parts.append(f"{exch}:{n}")
            print(f"[{time.strftime('%H:%M:%S')}] trades in memory: {' | '.join(parts)}")
    except KeyboardInterrupt:
        print("\nStopping...")


if __name__ == "__main__":
    main()
