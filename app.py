"""
Flow Monitor Dashboard
Tracks market buying (CVD) and limit order activity (depth) for FLOW/USD across all spot exchanges.
"""
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import json
import time
from datetime import datetime, timezone
from config import EXCHANGES, SYMBOL, TICK_SIZE, DEPTH_LEVELS, HISTORY_DAYS, PRIMARY_EXCHANGE, EXCHANGE
from db import (
    get_conn, get_vd_history, get_stats_history, get_candle_history,
    get_trades_history, get_trade_stats, get_depth_events,
    get_vd_history_multi, get_candle_history_multi, get_vd_by_exchange,
)

ALL_EXCHANGES = list(EXCHANGES.keys())

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

def ts_to_datetime(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def make_chart(df, y_col, color="#4fc3f7", height=200, y_zero=False):
    """Create an Altair line chart with proper axis scaling."""
    y_scale = alt.Scale(zero=y_zero)
    chart = (
        alt.Chart(df)
        .mark_line(strokeWidth=1.5, color=color)
        .encode(
            x=alt.X("time:T", axis=alt.Axis(format="%m/%d %H:%M", labelAngle=-45, tickCount=8, title=None)),
            y=alt.Y(f"{y_col}:Q", scale=y_scale, title=y_col),
            tooltip=[alt.Tooltip("time:T", format="%m/%d %H:%M"), alt.Tooltip(f"{y_col}:Q", format=",.2f")],
        )
        .properties(height=height)
    )
    return chart


def make_bar_chart(df, height=150):
    """Buy/sell volume bar chart — layered approach for Altair 6 compatibility."""
    base = alt.Chart(df).encode(
        x=alt.X("time:T", axis=alt.Axis(format="%m/%d %H:%M", labelAngle=-45, tickCount=8, title=None)),
    )
    buy_bars = base.mark_bar(color="#4caf50", opacity=0.7).encode(
        y=alt.Y("Buy:Q", title="Volume ($)"),
        tooltip=[alt.Tooltip("time:T", format="%m/%d %H:%M"), alt.Tooltip("Buy:Q", format=",.0f", title="Buy $")],
    )
    sell_bars = base.mark_bar(color="#f44336", opacity=0.7).encode(
        y=alt.Y("Sell:Q"),
        tooltip=[alt.Tooltip("time:T", format="%m/%d %H:%M"), alt.Tooltip("Sell:Q", format=",.0f", title="Sell $")],
    )
    return alt.layer(buy_bars, sell_bars).properties(height=height)


def render_market_activity(conn, vd_rows, candle_rows, hours=24, sig_events=None):
    """Stacked: Price → CVD → Depth → per-exchange CVD → volume bars."""
    cutoff = int(time.time()) - hours * 3600
    vd_filtered = [(t, v) for t, v in vd_rows if t >= cutoff]
    candle_filtered = [r for r in candle_rows if r[0] >= cutoff]

    if not vd_filtered or not candle_filtered:
        st.warning("Not enough data")
        return

    ts_cvd, cvd_raw = compute_cvd(vd_filtered)
    ts_price, prices = compute_price_series(candle_filtered)

    # Convert CVD from USD to coins
    price_map = {r[0]: r[4] for r in candle_filtered}
    cvd_coins = []
    cumulative_coins = 0.0
    for ts, delta_usd in vd_filtered:
        p = price_map.get(ts, prices[-1] if prices else 1)
        if p > 0:
            cumulative_coins += delta_usd / p
        cvd_coins.append(cumulative_coins)

    current_cvd_coins = cvd_coins[-1] if cvd_coins else 0
    current_cvd_usd = cvd_raw[-1] if cvd_raw else 0

    # Metrics
    col1, col2 = st.columns(2)
    col1.metric("Aggregate CVD (coins)", f"{current_cvd_coins:+,.0f} FLOW")
    col2.metric("Aggregate CVD (USD)", f"${current_cvd_usd:+,.0f}")

    # Downsample
    ts_price_ds, prices_ds = downsample(ts_price, prices)
    ts_cvd_ds, cvd_coins_ds = downsample(ts_cvd, cvd_coins)

    buy_vols = [r[5] for r in candle_filtered]
    sell_vols = [r[6] for r in candle_filtered]
    ts_vol = [r[0] for r in candle_filtered]

    # Price chart with significant event markers
    st.markdown("**Price (avg across exchanges)**")
    df_price = pd.DataFrame({"time": [ts_to_datetime(t) for t in ts_price_ds], "Price": prices_ds})
    price_line = make_chart(df_price, "Price", color="#4fc3f7", height=300)

    if sig_events:
        evt_times, evt_prices, evt_labels = [], [], []
        for e in sig_events:
            # Find closest price to event timestamp
            closest_price = None
            for i, ts in enumerate(ts_price):
                if abs(ts - e["ts"]) < 120:
                    closest_price = prices[i]
                    break
            if closest_price:
                evt_times.append(ts_to_datetime(e["ts"]))
                evt_prices.append(closest_price)
                evt_labels.append(e["triggers"][0] if e["triggers"] else "event")

        if evt_times:
            df_evt = pd.DataFrame({"time": evt_times, "Price": evt_prices, "event": evt_labels})
            dots = (
                alt.Chart(df_evt)
                .mark_point(size=100, filled=True, color="#ff9800")
                .encode(
                    x="time:T",
                    y=alt.Y("Price:Q", scale=alt.Scale(zero=False)),
                    tooltip=[alt.Tooltip("time:T", format="%m/%d %H:%M"),
                             alt.Tooltip("Price:Q", format=".4f"),
                             alt.Tooltip("event:N", title="Trigger")],
                )
            )
            st.altair_chart(alt.layer(price_line, dots).properties(height=300), use_container_width=True)
        else:
            st.altair_chart(price_line, use_container_width=True)
    else:
        st.altair_chart(price_line, use_container_width=True)

    # Aggregate CVD chart
    st.markdown("**Aggregate CVD (net coins bought — all exchanges)**")
    df_cvd = pd.DataFrame({"time": [ts_to_datetime(t) for t in ts_cvd_ds], "CVD": cvd_coins_ds})
    cvd_color = "#4caf50" if current_cvd_coins >= 0 else "#f44336"
    st.altair_chart(make_chart(df_cvd, "CVD", color=cvd_color, height=250), use_container_width=True)

    # Aggregate depth chart (3rd from top)
    render_aggregate_depth_chart(conn, hours=hours)

    # Per-exchange CVD breakdown
    exch_colors = {"binance": "#F0B90B", "bybit": "#f7a600", "coinbase": "#0052FF", "okx": "#00C853"}
    vd_by_exch = get_vd_by_exchange(conn, ALL_EXCHANGES, SYMBOL, from_ts=cutoff)

    if len(vd_by_exch) > 1:
        st.markdown("**CVD per exchange**")
        dfs = []
        for exch, rows in vd_by_exch.items():
            ts_list, cvd_vals = compute_cvd(rows)
            if ts_list:
                ts_ds, cvd_ds = downsample(ts_list, cvd_vals)
                df = pd.DataFrame({
                    "time": [ts_to_datetime(t) for t in ts_ds],
                    "CVD": cvd_ds,
                    "Exchange": exch,
                })
                dfs.append(df)

        if dfs:
            df_all = pd.concat(dfs)
            chart = (
                alt.Chart(df_all)
                .mark_line(strokeWidth=1.5)
                .encode(
                    x=alt.X("time:T", axis=alt.Axis(format="%m/%d %H:%M", labelAngle=-45, tickCount=8, title=None)),
                    y=alt.Y("CVD:Q", title="CVD ($)"),
                    color=alt.Color("Exchange:N",
                        scale=alt.Scale(
                            domain=list(exch_colors.keys()),
                            range=list(exch_colors.values()))),
                    tooltip=[alt.Tooltip("time:T", format="%m/%d %H:%M"),
                             alt.Tooltip("CVD:Q", format=",.0f"), "Exchange:N"],
                )
                .properties(height=250)
            )
            st.altair_chart(chart, use_container_width=True)

        # Per-exchange CVD metrics
        cols = st.columns(len(vd_by_exch))
        for i, (exch, rows) in enumerate(vd_by_exch.items()):
            _, cvd_vals = compute_cvd(rows)
            if cvd_vals:
                cols[i].metric(f"{exch}", f"${cvd_vals[-1]:+,.0f}")

    # Volume bars — aggregate into hourly buckets
    st.markdown("**Buy / Sell Volume ($) — hourly (all exchanges)**")
    vol_df = pd.DataFrame({"time": [ts_to_datetime(t) for t in ts_vol], "Buy": buy_vols, "Sell": sell_vols})
    vol_df["hour"] = vol_df["time"].dt.floor("h")
    vol_hourly = vol_df.groupby("hour").agg({"Buy": "sum", "Sell": "sum"}).reset_index()
    vol_hourly.rename(columns={"hour": "time"}, inplace=True)
    st.altair_chart(make_bar_chart(vol_hourly), use_container_width=True)


def render_divergence(vd_rows, candle_rows, hours=24):
    """CVD vs Price divergence detector over the selected time window."""
    cutoff = int(time.time()) - hours * 3600
    vd_filtered = [(t, v) for t, v in vd_rows if t >= cutoff]
    candle_filtered = [r for r in candle_rows if r[0] >= cutoff]

    if len(vd_filtered) < 60 or len(candle_filtered) < 60:
        st.info("Not enough data for divergence")
        return

    _, cvd = compute_cvd(vd_filtered)
    _, prices = compute_price_series(candle_filtered)

    n = min(len(cvd), len(prices))
    cvd = cvd[:n]
    prices = prices[:n]

    # Rate of change over the full window
    price_roc = ((prices[-1] - prices[0]) / prices[0] * 100) if prices[0] > 0 else 0
    cvd_direction = "rising" if cvd[-1] > cvd[n // 2] else "falling"
    cvd_total = cvd[-1] - cvd[0]

    # Classify based on actual price movement + CVD direction
    price_flat = abs(price_roc) < 0.5
    price_up = price_roc > 0.5
    price_down = price_roc < -0.5
    cvd_up = cvd_total > 0
    cvd_down = cvd_total < 0

    if cvd_up and price_flat:
        state = "CVD rising, price flat → absorption (bids absorbing sells)"
        color = "#4caf50"
    elif cvd_down and price_flat:
        state = "CVD falling, price flat → distribution (asks absorbing buys)"
        color = "#f44336"
    elif price_up and cvd_down:
        state = "Price rising, CVD falling → rising on thin air"
        color = "#ff9800"
    elif price_down and cvd_up:
        state = "Price falling, CVD rising → buying into weakness"
        color = "#ff9800"
    elif price_down and cvd_down:
        state = "Price down, CVD down → selling pressure confirmed"
        color = "#f44336"
    elif price_up and cvd_up:
        state = "Price up, CVD up → buying pressure confirmed"
        color = "#4caf50"
    else:
        state = "No significant divergence"
        color = "#888"

    st.markdown(f'<span style="color:{color};font-size:14px">**{state}**</span>', unsafe_allow_html=True)
    st.caption(f"Window: {hours}h | Price: {price_roc:+.2f}% | Net CVD: {cvd_total:+,.0f} coins")

    return {"price_roc": price_roc, "cvd_total": cvd_total, "state": state}


def get_4h_buckets(candle_rows, bucket_seconds=4*3600):
    """Split candle data into 4h buckets with OHLC and high/low timestamps."""
    if not candle_rows:
        return []

    buckets = []
    first_ts = candle_rows[0][0]
    # Align to 4h boundary
    bucket_start = (first_ts // bucket_seconds) * bucket_seconds

    while bucket_start < candle_rows[-1][0]:
        bucket_end = bucket_start + bucket_seconds
        rows = [r for r in candle_rows if bucket_start <= r[0] < bucket_end]
        if rows:
            prices = [r[4] for r in rows]
            high_val = max(prices)
            low_val = min(prices)
            high_idx = prices.index(high_val)
            low_idx = prices.index(low_val)
            buckets.append({
                "start_ts": bucket_start,
                "end_ts": bucket_end,
                "open": rows[0][4],
                "close": rows[-1][4],
                "high": high_val,
                "low": low_val,
                "high_ts": rows[high_idx][0],
                "low_ts": rows[low_idx][0],
                "candles": rows,
            })
        bucket_start = bucket_end

    return buckets


def render_aggregate_depth_chart(conn, hours=24):
    """Separate bid/ask depth chart (aggregate across exchanges, 15m smoothed) — sits below CVD."""
    cutoff = int(time.time()) - hours * 3600

    all_stats = {}
    for exch in ALL_EXCHANGES:
        rows = get_stats_history(conn, exch, SYMBOL, from_ts=cutoff)
        if rows:
            all_stats[exch] = rows

    if not all_stats:
        return

    level_idx = 5  # 5.0%

    # Collect all unique timestamps
    all_ts = set()
    for rows in all_stats.values():
        for r in rows:
            all_ts.add(r[0])
    all_ts = sorted(all_ts)

    if len(all_ts) < 30:
        return

    # Sum depth across exchanges per timestamp, normalize by exchange count
    # to prevent crash when some exchanges stop having data
    bid_series, ask_series, ts_depth = [], [], []
    # Track how many exchanges have data at the most recent timestamp to set baseline
    max_exchanges = len(all_stats)

    for ts in all_ts:
        total_bid, total_ask = 0.0, 0.0
        exch_count = 0
        for exch, rows in all_stats.items():
            for r in rows:
                if abs(r[0] - ts) <= 90:
                    bids = json.loads(r[2])
                    asks = json.loads(r[3])
                    if level_idx < len(bids):
                        total_bid += bids[level_idx]
                        total_ask += asks[level_idx]
                        exch_count += 1
                    break
        if exch_count > 0:
            # Scale up to full exchange count so chart doesn't drop when some go offline
            scale = max_exchanges / exch_count
            ts_depth.append(ts)
            bid_series.append(total_bid * scale)
            ask_series.append(total_ask * scale)

    # Smooth
    if len(bid_series) > SMOOTH_WINDOW:
        bid_sm = smooth(bid_series, SMOOTH_WINDOW)
        ask_sm = smooth(ask_series, SMOOTH_WINDOW)
        ts_sm = ts_depth[SMOOTH_WINDOW - 1:]
    else:
        bid_sm, ask_sm, ts_sm = bid_series, ask_series, ts_depth

    ts_ds, bid_ds = downsample(ts_sm, bid_sm)
    _, ask_ds = downsample(ts_sm, ask_sm)

    st.markdown(f"**Aggregate Book Depth @ {DEPTH_LEVELS[level_idx]}% (all exchanges, 15m smoothed)**")

    df = pd.DataFrame({
        "time": [ts_to_datetime(t) for t in ts_ds],
        "Bid Depth": bid_ds,
        "Ask Depth": ask_ds,
    })

    bid_line = (
        alt.Chart(df).mark_line(strokeWidth=1.5, color="#4caf50")
        .encode(
            x=alt.X("time:T", axis=alt.Axis(format="%m/%d %H:%M", labelAngle=-45, tickCount=8, title=None)),
            y=alt.Y("Bid Depth:Q", scale=alt.Scale(zero=False), title="Depth ($)"),
            tooltip=[alt.Tooltip("time:T", format="%m/%d %H:%M"), alt.Tooltip("Bid Depth:Q", format=",.0f")],
        )
    )
    ask_line = (
        alt.Chart(df).mark_line(strokeWidth=1.5, color="#f44336")
        .encode(
            x="time:T",
            y=alt.Y("Ask Depth:Q", scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("time:T", format="%m/%d %H:%M"), alt.Tooltip("Ask Depth:Q", format=",.0f")],
        )
    )

    chart = alt.layer(bid_line, ask_line).properties(height=250)
    st.altair_chart(chart, use_container_width=True)
    st.caption("Green = bid depth | Red = ask depth")


def compute_significant_events(conn, candle_rows, hours=24):
    """Compute significant events: bid/ask depth ±50% in 1h, or price ±5% in 1h. Returns list of events."""
    cutoff = int(time.time()) - hours * 3600

    DEPTH_THRESHOLD = 50
    PRICE_THRESHOLD = 5
    WINDOW_SEC = 3600

    all_stats = {}
    for exch in ALL_EXCHANGES:
        rows = get_stats_history(conn, exch, SYMBOL, from_ts=cutoff)
        if rows:
            all_stats[exch] = rows

    candle_filtered = [r for r in candle_rows if r[0] >= cutoff]

    if not all_stats or len(candle_filtered) < 30:
        return

    level_idx = 5  # 5.0%
    events = []

    # Slide 1h window in 15-min steps
    check_ts = cutoff
    now = int(time.time())

    def agg_depth_at(ts):
        total_bid, total_ask = 0.0, 0.0
        for exch, rows in all_stats.items():
            for r in rows:
                if abs(r[0] - ts) <= 120:
                    bids = json.loads(r[2])
                    asks = json.loads(r[3])
                    if level_idx < len(bids):
                        total_bid += bids[level_idx]
                        total_ask += asks[level_idx]
                    break
        return total_bid, total_ask

    while check_ts < now - WINDOW_SEC:
        window_end = check_ts + WINDOW_SEC

        bid_start, ask_start = agg_depth_at(check_ts)
        bid_end, ask_end = agg_depth_at(window_end)

        prices_in_window = [r[4] for r in candle_filtered if check_ts <= r[0] < window_end]

        if bid_start > 0 and ask_start > 0 and prices_in_window:
            bid_chg = (bid_end - bid_start) / bid_start * 100
            ask_chg = (ask_end - ask_start) / ask_start * 100
            price_start = prices_in_window[0]
            price_end = prices_in_window[-1]
            price_chg = (price_end - price_start) / price_start * 100

            # What triggered this event?
            triggers = []
            if abs(bid_chg) >= DEPTH_THRESHOLD:
                triggers.append(f"bid depth {bid_chg:+.0f}% in 1h")
            if abs(ask_chg) >= DEPTH_THRESHOLD:
                triggers.append(f"ask depth {ask_chg:+.0f}% in 1h")
            if abs(price_chg) >= PRICE_THRESHOLD:
                triggers.append(f"price {price_chg:+.1f}% in 1h")

            if triggers:
                # Context interpretation
                context = ""
                if price_chg < -1 and bid_chg < -20:
                    context = "bids eaten on the way down"
                elif price_chg < -1 and bid_chg > 20:
                    context = "new bids placed — buying the dip"
                elif price_chg > 1 and ask_chg < -20:
                    context = "asks lifted — breakout buying"
                elif price_chg > 1 and ask_chg > 20:
                    context = "new asks placed — selling into strength"
                elif abs(price_chg) < 1 and bid_chg < -30:
                    context = "bids pulled without price move — spoofing?"
                elif abs(price_chg) < 1 and ask_chg < -30:
                    context = "asks pulled without price move — spoofing?"
                elif abs(price_chg) < 1 and bid_chg > 30:
                    context = "large bid wall added"
                elif abs(price_chg) < 1 and ask_chg > 30:
                    context = "large ask wall added"

                events.append({
                    "ts": check_ts,
                    "price_chg": price_chg,
                    "bid_chg": bid_chg,
                    "ask_chg": ask_chg,
                    "bid_usd": bid_end,
                    "ask_usd": ask_end,
                    "triggers": triggers,
                    "context": context,
                })

        check_ts += 15 * 60  # step 15 min

    return events


def render_significant_events(events):
    """Render the significant events table."""
    if not events:
        st.caption("No significant events (triggers: depth ±50% or price ±5% in 1h)")
        return

    st.markdown(f"**Significant events ({len(events)} detected):**")
    st.caption("Triggers: bid/ask depth ±50% in 1h OR price ±5% in 1h")

    rows_html = ""
    for e in reversed(events):
        time_str = time.strftime("%m/%d %H:%M", time.gmtime(e["ts"]))
        pc = e["price_chg"]
        pc_color = "#4caf50" if pc > 0 else "#f44336" if pc < 0 else "#888"
        bc = "#4caf50" if e["bid_chg"] > 5 else "#f44336" if e["bid_chg"] < -5 else "#888"
        ac = "#4caf50" if e["ask_chg"] > 5 else "#f44336" if e["ask_chg"] < -5 else "#888"
        trigger_str = " + ".join(e["triggers"])

        row = f'<tr>'
        row += f'<td style="text-align:left">{time_str}</td>'
        row += f'<td style="color:{pc_color};font-weight:bold">{pc:+.1f}%</td>'
        row += f'<td style="color:{bc}">{e["bid_chg"]:+.0f}%</td>'
        row += f'<td>${e["bid_usd"]:,.0f}</td>'
        row += f'<td style="color:{ac}">{e["ask_chg"]:+.0f}%</td>'
        row += f'<td>${e["ask_usd"]:,.0f}</td>'
        row += f'<td style="text-align:left;font-size:11px;color:#aaa">{trigger_str}</td>'
        row += f'<td style="text-align:left;font-size:12px">{e["context"]}</td>'
        row += '</tr>'
        rows_html += row

    header = "<tr><th>Time</th><th>Price</th><th>Bid Δ</th><th>Bid $</th><th>Ask Δ</th><th>Ask $</th><th>Trigger</th><th>What happened</th></tr>"
    html = f"""<table style="width:100%;border-collapse:collapse;text-align:right;font-family:monospace;font-size:13px;line-height:2.2">
    <thead style="border-bottom:1px solid #333">{header}</thead>
    <tbody>{rows_html}</tbody>
    </table>"""
    st.markdown(html, unsafe_allow_html=True)


# ── Trades Section ──────────────────────────────────────────────────────────

def render_notable_trades(conn, hours=24):
    """Show large trades (z-score > 2) from all exchanges."""
    cutoff = int(time.time()) - hours * 3600

    # Gather from all exchanges
    all_large = []
    for exch in ALL_EXCHANGES:
        trades = get_trades_history(conn, exch, SYMBOL, from_ts=cutoff, large_only=True)
        for t in trades:
            all_large.append((exch, *t))

    if not all_large:
        st.info("No large trades detected yet. Start trades_collector.py to begin tracking.")
        return

    # Sort by timestamp, most recent first
    all_large.sort(key=lambda x: x[1], reverse=True)

    header = "<tr><th>Time</th><th>Exchange</th><th>Side</th><th>Size</th><th>Price</th><th>Z-Score</th></tr>"
    rows_html = ""
    for row in all_large[:50]:
        exch, ts, price, size_usd, side, is_large, z_score = row
        side_color = "#4caf50" if side == "buy" else "#f44336"
        rows_html += (
            f'<tr>'
            f'<td>{time.strftime("%m/%d %H:%M:%S", time.gmtime(ts))}</td>'
            f'<td>{exch}</td>'
            f'<td style="color:{side_color};font-weight:bold">{side.upper()}</td>'
            f'<td>${size_usd:,.0f}</td>'
            f'<td>{price:.4f}</td>'
            f'<td>{z_score:.1f}σ</td>'
            f'</tr>'
        )

    html = f"""<table style="width:100%;border-collapse:collapse;text-align:right;font-family:monospace;font-size:13px;line-height:2.0">
    <thead style="border-bottom:1px solid #333">{header}</thead>
    <tbody>{rows_html}</tbody>
    </table>"""
    st.markdown(html, unsafe_allow_html=True)

    # Summary stats
    buy_large = [r for r in all_large if r[4] == "buy"]
    sell_large = [r for r in all_large if r[4] == "sell"]
    buy_vol = sum(r[3] for r in buy_large)
    sell_vol = sum(r[3] for r in sell_large)

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


def render_depth_change_table(stats_rows, label=None):
    if len(stats_rows) < 2:
        return

    last_price = stats_rows[-1][1]
    valid = valid_level_indices(last_price)
    now_ts = stats_rows[-1][0]
    now_bids, now_asks = smooth_depth(stats_rows)

    windows = [(60, "1h"), (240, "4h"), (720, "12h"), (1440, "24h")]
    refs = {}
    for minutes, wlabel in windows:
        target_ts = now_ts - minutes * 60
        bid_ref, ask_ref = smooth_depth_at_time(stats_rows, target_ts)
        if bid_ref is not None:
            refs[wlabel] = (bid_ref, ask_ref)

    header = "<tr><th>Level</th><th>Bid $</th><th>Ask $</th>"
    for _, wlabel in windows:
        if wlabel in refs:
            header += f"<th>Bid Δ {wlabel}</th><th>Ask Δ {wlabel}</th>"
    header += "</tr>"

    rows_html = ""
    for i in valid:
        bid_now = now_bids[i]
        ask_now = now_asks[i]
        row = f"<tr><td>{DEPTH_LEVELS[i]}%</td>"
        row += f"<td>${bid_now:,.0f}</td><td>${ask_now:,.0f}</td>"
        for _, wlabel in windows:
            if wlabel not in refs:
                continue
            bid_ref = refs[wlabel][0][i]
            ask_ref = refs[wlabel][1][i]
            bid_pct = ((bid_now - bid_ref) / bid_ref * 100) if bid_ref > 0 else 0
            ask_pct = ((ask_now - ask_ref) / ask_ref * 100) if ask_ref > 0 else 0
            bc = "#4caf50" if bid_pct > 5 else "#f44336" if bid_pct < -5 else "#888"
            ac = "#4caf50" if ask_pct > 5 else "#f44336" if ask_pct < -5 else "#888"
            row += f'<td style="color:{bc}">{bid_pct:+.1f}%</td>'
            row += f'<td style="color:{ac}">{ask_pct:+.1f}%</td>'
        row += "</tr>"
        rows_html += row

    if label:
        st.markdown(f"**{label}**")
    html = f"""<table style="width:100%;border-collapse:collapse;text-align:right;font-family:monospace;font-size:13px;line-height:2.2">
    <thead style="border-bottom:1px solid #333">{header}</thead>
    <tbody>{rows_html}</tbody>
    </table>"""
    st.markdown(html, unsafe_allow_html=True)


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

    side_label = "Bid" if side == "bid" else "Ask"
    level_label = f"{side_label} depth {DEPTH_LEVELS[level_idx]} pct"
    color = "#4caf50" if side == "bid" else "#f44336"
    st.markdown(f"**{side_label} @ {DEPTH_LEVELS[level_idx]}% (15m avg)**")
    df = pd.DataFrame({"time": [ts_to_datetime(t) for t in timestamps], level_label: smoothed})
    st.altair_chart(make_chart(df, level_label, color=color, height=250), use_container_width=True)


# ── Depth Events (WS fill/pull tracking) ───────────────────────────────────

def render_depth_events(conn, hours=24):
    """Show recent order book events: fills, pulls, new orders from WS depth tracking."""
    cutoff = int(time.time()) - hours * 3600
    events = get_depth_events(conn, EXCHANGE, SYMBOL, from_ts=cutoff, min_usd=500)

    if not events:
        st.caption("No depth events yet. Start trades_collector.py for live tracking. WebSocket only — does not backfill.")
        return

    st.markdown("**Order Book Events (live)**")
    st.caption("Real-time fill/pull detection from WebSocket depth stream")

    # events: (timestamp, price, side, event_type, size_before, size_after, size_usd, filled)
    header = "<tr><th>Time</th><th>Side</th><th>Type</th><th>Price</th><th>Size $</th><th>Before→After</th></tr>"
    rows_html = ""
    for row in events[:50]:
        ts, price, side, event_type, size_before, size_after, size_usd, filled = row
        type_colors = {
            "filled": "#4caf50", "pulled": "#f44336", "added": "#4fc3f7",
            "partially_filled": "#ff9800", "reduced": "#ff9800",
        }
        tc = type_colors.get(event_type, "#888")
        sc = "#4caf50" if side == "bid" else "#f44336"
        rows_html += (
            f'<tr>'
            f'<td>{time.strftime("%m/%d %H:%M:%S", time.gmtime(ts))}</td>'
            f'<td style="color:{sc}">{side.upper()}</td>'
            f'<td style="color:{tc};font-weight:bold">{event_type.upper()}</td>'
            f'<td>{price:.4f}</td>'
            f'<td>${size_usd:,.0f}</td>'
            f'<td>${size_before:,.0f}→${size_after:,.0f}</td>'
            f'</tr>'
        )

    html = f"""<table style="width:100%;border-collapse:collapse;text-align:right;font-family:monospace;font-size:13px;line-height:2.0">
    <thead style="border-bottom:1px solid #333">{header}</thead>
    <tbody>{rows_html}</tbody>
    </table>"""
    st.markdown(html, unsafe_allow_html=True)

    # Summary: fills vs pulls
    fills = [e for e in events if e[3] in ("filled", "partially_filled")]
    pulls = [e for e in events if e[3] == "pulled"]
    added = [e for e in events if e[3] == "added"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Filled", f"{len(fills)} (${sum(e[6] for e in fills):,.0f})")
    col2.metric("Pulled", f"{len(pulls)} (${sum(e[6] for e in pulls):,.0f})")
    col3.metric("Added", f"{len(added)} (${sum(e[6] for e in added):,.0f})")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    conn = get_conn()

    # Aggregate data across all exchanges
    vd_rows = get_vd_history_multi(conn, ALL_EXCHANGES, SYMBOL)
    candle_rows = get_candle_history_multi(conn, ALL_EXCHANGES, SYMBOL)
    # Depth stats from primary exchange only (most liquid)
    stats_rows = get_stats_history(conn, PRIMARY_EXCHANGE, SYMBOL)

    # Count how many exchanges have data
    exch_with_data = []
    for exch in ALL_EXCHANGES:
        count = conn.execute("SELECT COUNT(*) FROM vd WHERE exchange=? AND symbol=?", (exch, SYMBOL)).fetchone()[0]
        if count > 0:
            exch_with_data.append(exch)

    st.markdown(f"### FLOW/USD — Spot ({len(exch_with_data)} exchanges)")

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
    col2.metric("Exchanges", f"{', '.join(exch_with_data)}")
    col3.metric("Records", f"{total_records:,}")
    col4.metric("Days", f"{days_of_data:.1f}")

    hours = st.selectbox(
        "Time window", [6, 12, 24, 48, 72, 168], index=2,
        format_func=lambda h: f"{h}h" if h < 48 else f"{h//24}d"
    )

    st.divider()

    # ── 1. MARKET ACTIVITY ──
    st.markdown("## 1. Market Activity")
    st.caption(f"Aggregate across {len(exch_with_data)} spot exchanges: {', '.join(exch_with_data)}")

    # Compute significant events first so price chart can show markers
    sig_events = compute_significant_events(conn, candle_rows, hours=hours)

    render_market_activity(conn, vd_rows, candle_rows, hours=hours, sig_events=sig_events)

    st.markdown("### CVD vs Price Divergence")
    render_divergence(vd_rows, candle_rows, hours=hours)
    render_significant_events(sig_events)

    # Depth events (fill/pull from WS — primary exchange only)
    render_depth_events(conn, hours=hours)

    st.divider()

    # ── 2. NOTABLE TRADES ──
    st.markdown("## 2. Notable Trades (z-score > 2σ)")
    st.caption("Trades >2σ vs last 1h of trades. WebSocket only — does not backfill when offline.")

    render_notable_trades(conn, hours=hours)
    render_trade_intensity(conn, hours=hours)

    st.divider()

    # ── 3. LIMIT ORDER ACTIVITY ──
    st.markdown("## 3. Limit Order Activity (Depth Changes)")
    st.caption("15-min smoothed averages. Green = depth increased >5%, Red = decreased >5%.")

    # Show last reading per exchange
    now_ts = int(time.time())
    last_parts = []
    for exch in ALL_EXCHANGES:
        row = conn.execute("SELECT MAX(timestamp) FROM stats WHERE exchange=? AND symbol=?", (exch, SYMBOL)).fetchone()
        if row[0]:
            age_min = (now_ts - row[0]) / 60
            color = "#4caf50" if age_min < 5 else "#ff9800" if age_min < 60 else "#f44336"
            last_parts.append(f'<span style="color:{color}">{exch}: {age_min:.0f}m ago</span>')
    if last_parts:
        st.markdown("Last reading: " + " | ".join(last_parts), unsafe_allow_html=True)

    # Aggregate depth across all exchanges
    all_stats = {}
    for exch in ALL_EXCHANGES:
        rows = get_stats_history(conn, exch, SYMBOL)
        if len(rows) > 2:
            all_stats[exch] = rows

    if all_stats:
        # Aggregate: sum depth across exchanges at each level
        st.markdown("### Total (all exchanges)")
        # Use the exchange with most data to build aggregate
        # For aggregate, we sum the current depth from each exchange
        agg_bid = [0.0] * 7
        agg_ask = [0.0] * 7
        for exch, rows in all_stats.items():
            bids, asks = smooth_depth(rows)
            for i in range(7):
                agg_bid[i] += bids[i]
                agg_ask[i] += asks[i]

        valid = valid_level_indices(last_price)
        # Show aggregate totals
        header = "<tr><th>Level</th><th>Total Bid $</th><th>Total Ask $</th>"
        for exch in all_stats:
            header += f"<th>{exch} Bid</th><th>{exch} Ask</th>"
        header += "</tr>"

        rows_html = ""
        for i in valid:
            row = f"<tr><td>{DEPTH_LEVELS[i]}%</td>"
            row += f'<td style="font-weight:bold">${agg_bid[i]:,.0f}</td>'
            row += f'<td style="font-weight:bold">${agg_ask[i]:,.0f}</td>'
            for exch, srows in all_stats.items():
                bids, asks = smooth_depth(srows)
                row += f"<td>${bids[i]:,.0f}</td><td>${asks[i]:,.0f}</td>"
            row += "</tr>"
            rows_html += row

        html = f"""<table style="width:100%;border-collapse:collapse;text-align:right;font-family:monospace;font-size:13px;line-height:2.2">
        <thead style="border-bottom:1px solid #333">{header}</thead>
        <tbody>{rows_html}</tbody>
        </table>"""
        st.markdown(html, unsafe_allow_html=True)

        # Per-exchange depth change tables
        st.markdown("### Per-exchange depth changes")
        for exch, rows in all_stats.items():
            render_depth_change_table(rows, label=exch)
    else:
        render_depth_change_table(stats_rows, label=PRIMARY_EXCHANGE)

    valid = valid_level_indices(last_price)
    level_choice = st.selectbox(
        "Depth chart level",
        valid,
        format_func=lambda i: f"{DEPTH_LEVELS[i]}%"
    )

    if level_choice is not None:
        col1, col2 = st.columns(2)
        with col1:
            render_depth_chart(stats_rows, level_choice, "bid", hours=hours)
        with col2:
            render_depth_chart(stats_rows, level_choice, "ask", hours=hours)

    # Auto-refresh
    st.divider()
    st.caption(f"Last update: {ts_to_label(stats_rows[-1][0])} UTC | Auto-refresh: 60s")
    time.sleep(60)
    st.rerun()


main()
