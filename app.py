"""
Flow Monitor Dashboard
Tracks market buying (CVD) and limit order activity (depth) for FLOW/USD on Binance spot.
"""
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import json
import time
from config import EXCHANGE, SYMBOL, TICK_SIZE, DEPTH_LEVELS, HISTORY_DAYS
from db import (
    get_conn, get_vd_history, get_stats_history, get_candle_history,
    get_custom_depth_history, get_trades_history, get_trade_stats,
)

st.set_page_config(page_title="FLOW Monitor", layout="wide")

SMOOTH_WINDOW = 15  # 15-minute rolling average


# ── Helpers ──────────────────────────────────────────────────────────────────

def compute_cvd(vd_rows):
    if not vd_rows:
        return [], []
    timestamps = [r[0] for r in vd_rows]
    deltas = [r[1] for r in vd_rows]
    cvd = np.cumsum(deltas).tolist()
    return timestamps, cvd


def compute_price_series(candle_rows):
    if not candle_rows:
        return [], []
    return [r[0] for r in candle_rows], [r[4] for r in candle_rows]


def valid_level_indices(last_price):
    if not last_price or last_price <= 0:
        return list(range(7))
    return [i for i, pct in enumerate(DEPTH_LEVELS) if last_price * (pct / 100) >= TICK_SIZE]


def ts_to_label(ts):
    return time.strftime("%m/%d %H:%M", time.gmtime(ts))


def downsample(timestamps, values, max_points=500):
    if len(timestamps) <= max_points:
        return timestamps, values
    step = len(timestamps) // max_points
    return timestamps[::step], values[::step]


def smooth(values, window):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='valid').tolist()


# ── Market Activity Section (CVD + Volume + Divergence) ─────────────────────

def render_market_activity(vd_rows, candle_rows, hours=24):
    """Stacked: Price on top, raw CVD below, buy/sell volume bars at bottom."""
    cutoff = int(time.time()) - hours * 3600
    vd_filtered = [(t, v) for t, v in vd_rows if t >= cutoff]
    candle_filtered = [r for r in candle_rows if r[0] >= cutoff]

    if not vd_filtered or not candle_filtered:
        st.warning("Not enough data")
        return

    ts_cvd, cvd_raw = compute_cvd(vd_filtered)
    ts_price, prices = compute_price_series(candle_filtered)

    # Convert CVD from USD to coins using price at each point
    # Align prices to CVD timestamps
    price_map = {r[0]: r[4] for r in candle_filtered}
    cvd_coins = []
    deltas_usd = [vd_filtered[i][1] for i in range(len(vd_filtered))]
    cumulative_coins = 0.0
    for i, (ts, delta_usd) in enumerate(vd_filtered):
        p = price_map.get(ts, prices[-1] if prices else 1)
        if p > 0:
            cumulative_coins += delta_usd / p
        cvd_coins.append(cumulative_coins)

    # Current CVD values
    current_cvd_coins = cvd_coins[-1] if cvd_coins else 0
    current_cvd_usd = cvd_raw[-1] if cvd_raw else 0
    current_price = prices[-1] if prices else 0

    # Show current CVD as metric
    col1, col2 = st.columns(2)
    col1.metric("CVD (coins)", f"{current_cvd_coins:+,.0f} FLOW")
    col2.metric("CVD (USD)", f"${current_cvd_usd:+,.0f}")

    # Buy/sell volume per minute from candles
    buy_vols = [r[5] for r in candle_filtered]
    sell_vols = [-r[6] for r in candle_filtered]  # negative for display
    ts_vol = [r[0] for r in candle_filtered]

    # Downsample all
    ts_price_ds, prices_ds = downsample(ts_price, prices)
    ts_cvd_ds, cvd_coins_ds = downsample(ts_cvd, cvd_coins)
    ts_vol_ds, buy_ds = downsample(ts_vol, buy_vols)
    _, sell_ds = downsample(ts_vol, sell_vols)

    # Price chart
    st.markdown("**Price**")
    df_price = pd.DataFrame({"Price": prices_ds}, index=[ts_to_label(t) for t in ts_price_ds])
    st.line_chart(df_price, height=200, use_container_width=True)

    # Raw CVD in coins
    st.markdown("**CVD (net coins bought)**")
    df_cvd = pd.DataFrame({"CVD (FLOW)": cvd_coins_ds}, index=[ts_to_label(t) for t in ts_cvd_ds])
    st.line_chart(df_cvd, height=200, use_container_width=True)

    # Buy/sell volume bars
    st.markdown("**Buy / Sell Volume ($)**")
    df_vol = pd.DataFrame(
        {"Buy": buy_ds, "Sell": sell_ds},
        index=[ts_to_label(t) for t in ts_vol_ds]
    )
    st.bar_chart(df_vol, height=150, use_container_width=True, color=["#4caf50", "#f44336"])


def render_divergence(vd_rows, candle_rows, window=60):
    """CVD vs Price divergence detector."""
    cutoff = int(time.time()) - 24 * 3600
    vd_filtered = [(t, v) for t, v in vd_rows if t >= cutoff]
    candle_filtered = [r for r in candle_rows if r[0] >= cutoff]

    if len(vd_filtered) < window + 1 or len(candle_filtered) < window + 1:
        st.info("Not enough data for divergence (need 24h+)")
        return

    _, cvd = compute_cvd(vd_filtered)
    ts_price, prices = compute_price_series(candle_filtered)

    n = min(len(cvd), len(prices))
    cvd = cvd[:n]
    prices = prices[:n]

    # Latest 60-min rate of change
    cvd_change = cvd[-1] - cvd[-1 - window]
    price_start = prices[-1 - window]
    price_roc = ((prices[-1] - price_start) / price_start * 100) if price_start > 0 else 0
    cvd_abs_mean = np.mean(np.abs(cvd[-window:]))
    cvd_roc = (cvd_change / cvd_abs_mean * 100) if cvd_abs_mean > 0 else 0

    if cvd_roc > 10 and price_roc < 1:
        state = "CVD rising, price flat → absorption (bids absorbing sells)"
        color = "#4caf50"
    elif cvd_roc < -10 and price_roc > -1:
        state = "CVD falling, price flat → distribution (asks absorbing buys)"
        color = "#f44336"
    elif price_roc > 1 and cvd_roc < -5:
        state = "Price rising, CVD falling → rising on thin air"
        color = "#ff9800"
    elif price_roc < -1 and cvd_roc > 5:
        state = "Price falling, CVD rising → selling on thin air"
        color = "#ff9800"
    else:
        state = "No significant divergence"
        color = "#888"

    st.markdown(f'<span style="color:{color};font-size:14px">**{state}**</span>', unsafe_allow_html=True)
    st.caption(f"60-min CVD RoC: {cvd_roc:+.1f}% | Price RoC: {price_roc:+.2f}%")


# ── Trades Section ──────────────────────────────────────────────────────────

def render_notable_trades(conn, hours=24):
    """Show large trades (z-score > 2) as a feed."""
    cutoff = int(time.time()) - hours * 3600
    large_trades = get_trades_history(conn, EXCHANGE, SYMBOL, from_ts=cutoff, large_only=True)

    if not large_trades:
        st.info("No large trades detected yet. Start trades_collector.py to begin tracking.")
        return

    # Build HTML table of notable trades (most recent first)
    header = "<tr><th>Time</th><th>Side</th><th>Size</th><th>Price</th><th>Z-Score</th></tr>"
    rows_html = ""
    for row in reversed(large_trades[-50:]):  # last 50
        ts, price, size_usd, side, is_large, z_score = row
        side_color = "#4caf50" if side == "buy" else "#f44336"
        rows_html += (
            f'<tr>'
            f'<td>{time.strftime("%m/%d %H:%M:%S", time.gmtime(ts))}</td>'
            f'<td style="color:{side_color};font-weight:bold">{side.upper()}</td>'
            f'<td>${size_usd:,.0f}</td>'
            f'<td>{price:.4f}</td>'
            f'<td>{z_score:.1f}σ</td>'
            f'</tr>'
        )

    html = f"""<html><body style="margin:0;background:#0e1117;color:#fafafa;font-family:monospace;font-size:13px">
    <table style="width:100%;border-collapse:collapse;text-align:right">
    <thead style="border-bottom:1px solid #333">{header}</thead>
    <tbody>{rows_html}</tbody>
    </table></body></html>"""

    n_rows = min(len(large_trades), 50)
    components.html(html, height=40 + 26 * n_rows, scrolling=True)

    # Summary stats
    buy_large = [r for r in large_trades if r[3] == "buy"]
    sell_large = [r for r in large_trades if r[3] == "sell"]
    buy_vol = sum(r[2] for r in buy_large)
    sell_vol = sum(r[2] for r in sell_large)

    col1, col2, col3 = st.columns(3)
    col1.metric("Large Buys", f"{len(buy_large)} (${buy_vol:,.0f})")
    col2.metric("Large Sells", f"{len(sell_large)} (${sell_vol:,.0f})")
    net = buy_vol - sell_vol
    col3.metric("Net Large", f"${net:+,.0f}")


def render_trade_intensity(conn, hours=24):
    """Trades per minute chart."""
    now = int(time.time())
    cutoff = now - hours * 3600
    buckets = get_trade_stats(conn, EXCHANGE, SYMBOL, cutoff, now)

    if not buckets:
        return

    timestamps = [b[0] for b in buckets]
    counts = [b[1] for b in buckets]
    buy_vols = [b[2] for b in buckets]
    sell_vols = [b[3] for b in buckets]

    timestamps, counts = downsample(timestamps, counts, max_points=300)

    st.markdown("**Trade Intensity (trades/min)**")
    df = pd.DataFrame({"Trades/min": counts}, index=[ts_to_label(t) for t in timestamps])
    st.line_chart(df, height=150, use_container_width=True)


# ── Depth Section ───────────────────────────────────────────────────────────

def smooth_depth(stats_rows, window=SMOOTH_WINDOW):
    if len(stats_rows) < window:
        window = len(stats_rows)
    recent = stats_rows[-window:]
    bid_avg = [0.0] * 7
    ask_avg = [0.0] * 7
    for row in recent:
        bids = json.loads(row[2])
        asks = json.loads(row[3])
        for i in range(7):
            bid_avg[i] += bids[i] if i < len(bids) else 0
            ask_avg[i] += asks[i] if i < len(asks) else 0
    return [v / window for v in bid_avg], [v / window for v in ask_avg]


def smooth_depth_at_time(stats_rows, target_ts, window=SMOOTH_WINDOW):
    start_ts = target_ts - window * 60
    relevant = [r for r in stats_rows if start_ts <= r[0] <= target_ts]
    if not relevant:
        return None, None
    n = len(relevant)
    bid_avg = [0.0] * 7
    ask_avg = [0.0] * 7
    for row in relevant:
        bids = json.loads(row[2])
        asks = json.loads(row[3])
        for i in range(7):
            bid_avg[i] += bids[i] if i < len(bids) else 0
            ask_avg[i] += asks[i] if i < len(asks) else 0
    return [v / n for v in bid_avg], [v / n for v in ask_avg]


def smooth_custom_depth(custom_rows, window=SMOOTH_WINDOW):
    if not custom_rows:
        return None, None
    w = min(window, len(custom_rows))
    recent = custom_rows[-w:]
    return sum(r[2] for r in recent) / w, sum(r[3] for r in recent) / w


def smooth_custom_depth_at_time(custom_rows, target_ts, window=SMOOTH_WINDOW):
    start_ts = target_ts - window * 60
    relevant = [r for r in custom_rows if start_ts <= r[0] <= target_ts]
    if not relevant:
        return None, None
    return sum(r[2] for r in relevant) / len(relevant), sum(r[3] for r in relevant) / len(relevant)


def render_depth_change_table(stats_rows, custom_depth_rows=None):
    if len(stats_rows) < 2:
        st.warning("Not enough stats data")
        return

    last_price = stats_rows[-1][1]
    valid = valid_level_indices(last_price)
    now_ts = stats_rows[-1][0]
    now_bids, now_asks = smooth_depth(stats_rows)

    has_custom = custom_depth_rows and len(custom_depth_rows) >= 2
    custom_bid_now, custom_ask_now = (None, None)
    if has_custom:
        custom_bid_now, custom_ask_now = smooth_custom_depth(custom_depth_rows)

    windows = [(60, "1h"), (240, "4h"), (720, "12h"), (1440, "24h")]
    refs = {}
    custom_refs = {}
    for minutes, label in windows:
        target_ts = now_ts - minutes * 60
        bid_ref, ask_ref = smooth_depth_at_time(stats_rows, target_ts)
        if bid_ref is not None:
            refs[label] = (bid_ref, ask_ref)
        if has_custom:
            cb, ca = smooth_custom_depth_at_time(custom_depth_rows, target_ts)
            if cb is not None:
                custom_refs[label] = (cb, ca)

    header = "<tr><th>Level</th><th>Bid $</th><th>Ask $</th>"
    for _, label in windows:
        if label in refs:
            header += f"<th>Bid Δ {label}</th><th>Ask Δ {label}</th>"
    header += "</tr>"

    rows_html = ""
    for i in valid:
        bid_now = now_bids[i]
        ask_now = now_asks[i]
        row = f"<tr><td>{DEPTH_LEVELS[i]}%</td>"
        row += f"<td>${bid_now:,.0f}</td><td>${ask_now:,.0f}</td>"
        for _, label in windows:
            if label not in refs:
                continue
            bid_ref = refs[label][0][i]
            ask_ref = refs[label][1][i]
            bid_pct = ((bid_now - bid_ref) / bid_ref * 100) if bid_ref > 0 else 0
            ask_pct = ((ask_now - ask_ref) / ask_ref * 100) if ask_ref > 0 else 0
            bc = "#4caf50" if bid_pct > 5 else "#f44336" if bid_pct < -5 else "#888"
            ac = "#4caf50" if ask_pct > 5 else "#f44336" if ask_pct < -5 else "#888"
            row += f'<td style="color:{bc}">{bid_pct:+.1f}%</td>'
            row += f'<td style="color:{ac}">{ask_pct:+.1f}%</td>'
        row += "</tr>"
        rows_html += row

    if has_custom and custom_bid_now is not None:
        row = f'<tr style="border-top:1px solid #444"><td>25% ★</td>'
        row += f"<td>${custom_bid_now:,.0f}</td><td>${custom_ask_now:,.0f}</td>"
        for _, label in windows:
            if label not in refs:
                continue
            if label in custom_refs:
                cb, ca = custom_refs[label]
                bp = ((custom_bid_now - cb) / cb * 100) if cb > 0 else 0
                ap = ((custom_ask_now - ca) / ca * 100) if ca > 0 else 0
                bc = "#4caf50" if bp > 5 else "#f44336" if bp < -5 else "#888"
                ac = "#4caf50" if ap > 5 else "#f44336" if ap < -5 else "#888"
                row += f'<td style="color:{bc}">{bp:+.1f}%</td>'
                row += f'<td style="color:{ac}">{ap:+.1f}%</td>'
            else:
                row += "<td>—</td><td>—</td>"
        row += "</tr>"
        rows_html += row

    html = f"""<html><body style="margin:0;background:#0e1117;color:#fafafa;font-family:monospace;font-size:13px">
    <table style="width:100%;border-collapse:collapse;text-align:right">
    <thead style="border-bottom:1px solid #333">{header}</thead>
    <tbody>{rows_html}</tbody>
    </table></body></html>"""
    components.html(html, height=40 + 30 * rows_html.count("<tr>"))


def render_depth_chart(stats_rows, level_idx, side, hours=24):
    cutoff = int(time.time()) - hours * 3600
    filtered = [r for r in stats_rows if r[0] >= cutoff]
    if len(filtered) < SMOOTH_WINDOW:
        st.info("Not enough data for smoothed depth chart")
        return

    field_idx = 2 if side == "bid" else 3
    raw = [json.loads(row[field_idx])[level_idx] for row in filtered]
    smoothed = smooth(raw, SMOOTH_WINDOW)
    timestamps = [r[0] for r in filtered[SMOOTH_WINDOW - 1:]]
    timestamps, smoothed = downsample(timestamps, smoothed)

    label = f"{'Bid' if side == 'bid' else 'Ask'} @ {DEPTH_LEVELS[level_idx]}% (15m avg)"
    df = pd.DataFrame({label: smoothed}, index=[ts_to_label(t) for t in timestamps])
    st.line_chart(df, height=200)


def render_custom_depth_chart(custom_rows, hours=24):
    cutoff = int(time.time()) - hours * 3600
    filtered = [r for r in custom_rows if r[0] >= cutoff]
    if len(filtered) < SMOOTH_WINDOW:
        st.info("Not enough custom depth data yet. Run collector.py to build 25% history.")
        return

    bid_smooth = smooth([r[2] for r in filtered], SMOOTH_WINDOW)
    ask_smooth = smooth([r[3] for r in filtered], SMOOTH_WINDOW)
    ts = [r[0] for r in filtered[SMOOTH_WINDOW - 1:]]

    ts_ds, bid_ds = downsample(ts, bid_smooth)
    _, ask_ds = downsample(ts, ask_smooth)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Bid depth @ 25% (15m avg)**")
        df = pd.DataFrame({"Bid $": bid_ds}, index=[ts_to_label(t) for t in ts_ds])
        st.line_chart(df, height=200)
    with col2:
        st.markdown("**Ask depth @ 25% (15m avg)**")
        df = pd.DataFrame({"Ask $": ask_ds}, index=[ts_to_label(t) for t in ts_ds])
        st.line_chart(df, height=200)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    conn = get_conn()

    vd_rows = get_vd_history(conn, EXCHANGE, SYMBOL)
    stats_rows = get_stats_history(conn, EXCHANGE, SYMBOL)
    candle_rows = get_candle_history(conn, EXCHANGE, SYMBOL)
    custom_depth_rows = get_custom_depth_history(conn, EXCHANGE, SYMBOL)

    st.markdown("### FLOW/USD — Binance Spot")

    if not vd_rows or not stats_rows:
        st.error("No data yet. Run backfill.py first, then start collector.py.")
        return

    # Header
    last_stats = stats_rows[-1]
    last_price = last_stats[1]
    total_records = len(vd_rows)
    days_of_data = (vd_rows[-1][0] - vd_rows[0][0]) / 86400

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Price", f"${last_price:.4f}")
    col2.metric("Records", f"{total_records:,}")
    col3.metric("Days", f"{days_of_data:.1f}")
    col4.metric("Tick", f"{TICK_SIZE}")

    hours = st.selectbox(
        "Time window", [6, 12, 24, 48, 72, 168], index=2,
        format_func=lambda h: f"{h}h" if h < 48 else f"{h//24}d"
    )

    st.divider()

    # ── 1. MARKET ACTIVITY ──
    st.markdown("## 1. Market Activity")
    st.caption("Stacked view: Price → CVD (smoothed) → CVD Slope → Buy/Sell Volume")

    render_market_activity(vd_rows, candle_rows, hours=hours)

    # Divergence
    st.markdown("### CVD vs Price Divergence")
    render_divergence(vd_rows, candle_rows)

    st.divider()

    # ── 2. NOTABLE TRADES ──
    st.markdown("## 2. Notable Trades (z-score > 2σ)")
    st.caption("Large individual fills detected via WebSocket. Whale buys/sells and TWAP patterns.")

    render_notable_trades(conn, hours=hours)
    render_trade_intensity(conn, hours=hours)

    st.divider()

    # ── 3. LIMIT ORDER ACTIVITY ──
    st.markdown("## 3. Limit Order Activity (Depth Changes)")
    st.caption("15-min smoothed averages. Green = depth increased >5%, Red = decreased >5%.")

    render_depth_change_table(stats_rows, custom_depth_rows)

    valid = valid_level_indices(last_price)
    level_options = [(i, f"{DEPTH_LEVELS[i]}%") for i in valid] + [(-1, "25% (custom)")]
    level_choice = st.selectbox(
        "Depth chart level",
        [o[0] for o in level_options],
        format_func=lambda i: dict(level_options)[i]
    )

    if level_choice is not None and level_choice >= 0:
        col1, col2 = st.columns(2)
        with col1:
            render_depth_chart(stats_rows, level_choice, "bid", hours=hours)
        with col2:
            render_depth_chart(stats_rows, level_choice, "ask", hours=hours)
    elif level_choice == -1:
        render_custom_depth_chart(custom_depth_rows, hours=hours)

    # Auto-refresh
    st.divider()
    st.caption(f"Last update: {ts_to_label(stats_rows[-1][0])} UTC | Auto-refresh: 60s")
    time.sleep(60)
    st.rerun()


main()
