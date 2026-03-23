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
import subprocess
import signal
import os
from datetime import datetime, timezone
from config import EXCHANGES, SYMBOL, TICK_SIZE, DEPTH_LEVELS, HISTORY_DAYS, PRIMARY_EXCHANGE, EXCHANGE
from db import (
    get_conn, get_vd_history, get_stats_history, get_candle_history,
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


def make_x_scale(x_domain=None):
    """Shared x-axis scale — forces all charts to the same time range."""
    if x_domain:
        return alt.Scale(domain=x_domain)
    return alt.Scale()


def make_chart(df, y_col, color="#4fc3f7", height=200, y_zero=False, x_domain=None):
    """Create an Altair line chart with proper axis scaling."""
    y_scale = alt.Scale(zero=y_zero)
    chart = (
        alt.Chart(df)
        .mark_line(strokeWidth=1.5, color=color)
        .encode(
            x=alt.X("time:T", scale=make_x_scale(x_domain), axis=alt.Axis(format="%m/%d %H:%M", labelAngle=-45, tickCount=8, title=None)),
            y=alt.Y(f"{y_col}:Q", scale=y_scale, title=y_col),
            tooltip=[alt.Tooltip("time:T", format="%m/%d %H:%M"), alt.Tooltip(f"{y_col}:Q", format=",.2f")],
        )
        .properties(height=height)
    )
    return chart


def make_bar_chart(df, height=150, x_domain=None):
    """Buy/sell volume bar chart — layered approach for Altair 6 compatibility."""
    base = alt.Chart(df).encode(
        x=alt.X("time:T", scale=make_x_scale(x_domain), axis=alt.Axis(format="%m/%d %H:%M", labelAngle=-45, tickCount=8, title=None)),
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

    # Shared x-axis domain for all charts — use pandas Timestamps for Altair compatibility
    x_domain = [pd.Timestamp(ts_to_datetime(cutoff)), pd.Timestamp(ts_to_datetime(int(time.time())))]

    # Price chart with significant event markers
    st.markdown("**Price (avg across exchanges)**")
    df_price = pd.DataFrame({"time": [ts_to_datetime(t) for t in ts_price_ds], "Price": prices_ds})
    price_line = make_chart(df_price, "Price", color="#4fc3f7", height=300, x_domain=x_domain)

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
                    x=alt.X("time:T", scale=make_x_scale(x_domain)),
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
    st.altair_chart(make_chart(df_cvd, "CVD", color=cvd_color, height=250, x_domain=x_domain), use_container_width=True)

    # Aggregate depth chart (3rd from top)
    render_aggregate_depth_chart(conn, hours=hours, x_domain=x_domain)

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
                    x=alt.X("time:T", scale=make_x_scale(x_domain), axis=alt.Axis(format="%m/%d %H:%M", labelAngle=-45, tickCount=8, title=None)),
                    y=alt.Y("CVD:Q", title="CVD ($)"),
                    color=alt.Color("Exchange:N",
                        scale=alt.Scale(
                            domain=list(exch_colors.keys()),
                            range=list(exch_colors.values())),
                        legend=alt.Legend(orient="bottom", direction="horizontal")),
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
    st.altair_chart(make_bar_chart(vol_hourly, x_domain=x_domain), use_container_width=True)


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



def _build_depth_index(all_stats, level_idx):
    """Build O(1) lookup: {exchange: {rounded_ts: (bid, ask)}} from stats rows.
    Timestamps are rounded to nearest 60s for fuzzy matching."""
    index = {}
    for exch, rows in all_stats.items():
        exch_idx = {}
        for r in rows:
            rounded = round(r[0] / 60) * 60  # round to nearest minute
            bids = json.loads(r[2])
            asks = json.loads(r[3])
            if level_idx < len(bids):
                exch_idx[rounded] = (bids[level_idx], asks[level_idx])
        index[exch] = exch_idx
    return index


def _agg_depth_at_fast(depth_index, ts):
    """O(1) aggregate depth lookup at a timestamp. Returns (bid, ask) — raw sums, no scaling."""
    rounded = round(ts / 60) * 60
    total_bid, total_ask = 0.0, 0.0
    for exch, exch_idx in depth_index.items():
        # Try exact, then ±1, ±2 min (collector polls every 60s but can drift)
        for offset in [0, 60, -60, 120, -120]:
            entry = exch_idx.get(rounded + offset)
            if entry:
                total_bid += entry[0]
                total_ask += entry[1]
                break
    return total_bid, total_ask


def render_aggregate_depth_chart(conn, hours=24, x_domain=None):
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

    # Build O(1) index
    depth_index = _build_depth_index(all_stats, level_idx)


    # Collect all unique timestamps
    all_ts = sorted({r[0] for rows in all_stats.values() for r in rows})

    if len(all_ts) < 30:
        return

    bid_series, ask_series, ts_depth = [], [], []

    for ts in all_ts:
        bid, ask = _agg_depth_at_fast(depth_index, ts)
        if bid > 0 or ask > 0:
            ts_depth.append(ts)
            bid_series.append(bid)
            ask_series.append(ask)

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
            x=alt.X("time:T", scale=make_x_scale(x_domain), axis=alt.Axis(format="%m/%d %H:%M", labelAngle=-45, tickCount=8, title=None)),
            y=alt.Y("Bid Depth:Q", scale=alt.Scale(zero=False), title="Depth ($)"),
            tooltip=[alt.Tooltip("time:T", format="%m/%d %H:%M"), alt.Tooltip("Bid Depth:Q", format=",.0f")],
        )
    )
    ask_line = (
        alt.Chart(df).mark_line(strokeWidth=1.5, color="#f44336")
        .encode(
            x=alt.X("time:T", scale=make_x_scale(x_domain)),
            y=alt.Y("Ask Depth:Q", scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("time:T", format="%m/%d %H:%M"), alt.Tooltip("Ask Depth:Q", format=",.0f")],
        )
    )

    chart = alt.layer(bid_line, ask_line).properties(height=250)
    st.altair_chart(chart, use_container_width=True)
    st.caption("Green = bid depth | Red = ask depth")


def compute_significant_events(conn, candle_rows, hours=24):
    """Compute significant events: bid/ask depth ±100% in 1h, or price ±5% in 1h.
    Rate of change filter: discard depth events that reverse within 30 min (MM noise)."""
    cutoff = int(time.time()) - hours * 3600

    DEPTH_THRESHOLD = 100
    PRICE_THRESHOLD = 5
    WINDOW_SEC = 3600
    REVERSAL_WINDOW = 30 * 60  # 30 min lookahead for rate-of-change filter
    REVERSAL_TOLERANCE = 0.20   # if depth returns within 20% of original, it's MM noise

    all_stats = {}
    for exch in ALL_EXCHANGES:
        rows = get_stats_history(conn, exch, SYMBOL, from_ts=cutoff)
        if rows:
            all_stats[exch] = rows

    candle_filtered = [r for r in candle_rows if r[0] >= cutoff]

    if not all_stats or len(candle_filtered) < 30:
        return []

    level_idx = 6  # 10.0%
    events = []

    # Build O(1) depth index
    depth_index = _build_depth_index(all_stats, level_idx)


    # Slide 1h window in 15-min steps
    check_ts = cutoff
    now = int(time.time())

    while check_ts < now - WINDOW_SEC:
        window_end = check_ts + WINDOW_SEC

        bid_start, ask_start = _agg_depth_at_fast(depth_index, check_ts)
        bid_end, ask_end = _agg_depth_at_fast(depth_index, window_end)

        prices_in_window = [r[4] for r in candle_filtered if check_ts <= r[0] < window_end]

        if bid_start > 0 and ask_start > 0 and bid_end > 0 and ask_end > 0 and prices_in_window:
            bid_chg = (bid_end - bid_start) / bid_start * 100
            ask_chg = (ask_end - ask_start) / ask_start * 100
            price_start = prices_in_window[0]
            price_end = prices_in_window[-1]
            price_chg = (price_end - price_start) / price_start * 100

            # What triggered this event?
            triggers = []
            is_depth_event = False
            if abs(bid_chg) >= DEPTH_THRESHOLD:
                triggers.append(f"bid depth {bid_chg:+.0f}% in 1h")
                is_depth_event = True
            if abs(ask_chg) >= DEPTH_THRESHOLD:
                triggers.append(f"ask depth {ask_chg:+.0f}% in 1h")
                is_depth_event = True
            if abs(price_chg) >= PRICE_THRESHOLD:
                triggers.append(f"price {price_chg:+.1f}% in 1h")

            if triggers:
                # Rate of change filter: check if depth change reversed within 30 min
                if is_depth_event and abs(price_chg) < PRICE_THRESHOLD:
                    # Only apply reversal filter to pure depth events (no price trigger)
                    bid_30, ask_30 = _agg_depth_at_fast(depth_index, window_end + REVERSAL_WINDOW)
                    reversed_bid = False
                    reversed_ask = False
                    if bid_30 > 0 and bid_start > 0 and abs(bid_chg) >= DEPTH_THRESHOLD:
                        # Did bid return close to where it started?
                        bid_recovery = abs(bid_30 - bid_start) / bid_start
                        reversed_bid = bid_recovery < REVERSAL_TOLERANCE
                    if ask_30 > 0 and ask_start > 0 and abs(ask_chg) >= DEPTH_THRESHOLD:
                        ask_recovery = abs(ask_30 - ask_start) / ask_start
                        reversed_ask = ask_recovery < REVERSAL_TOLERANCE
                    # If all depth triggers reversed, skip this event
                    depth_triggers_count = (1 if abs(bid_chg) >= DEPTH_THRESHOLD else 0) + (1 if abs(ask_chg) >= DEPTH_THRESHOLD else 0)
                    reversed_count = (1 if reversed_bid else 0) + (1 if reversed_ask else 0)
                    if reversed_count >= depth_triggers_count and depth_triggers_count > 0:
                        check_ts += 15 * 60
                        continue  # MM noise — skip

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
                elif abs(price_chg) < 1 and bid_chg < -50:
                    context = "bids pulled without price move — spoofing?"
                elif abs(price_chg) < 1 and ask_chg < -50:
                    context = "asks pulled without price move — spoofing?"
                elif abs(price_chg) < 1 and bid_chg > 50:
                    context = "large bid wall added"
                elif abs(price_chg) < 1 and ask_chg > 50:
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

    # Deduplicate: merge events within 30 min of each other (keep strongest)
    if not events:
        return []
    deduped = [events[0]]
    for e in events[1:]:
        prev = deduped[-1]
        if e["ts"] - prev["ts"] < 30 * 60:
            # Keep the one with larger total change
            prev_score = abs(prev["bid_chg"]) + abs(prev["ask_chg"]) + abs(prev["price_chg"]) * 10
            e_score = abs(e["bid_chg"]) + abs(e["ask_chg"]) + abs(e["price_chg"]) * 10
            if e_score > prev_score:
                deduped[-1] = e
        else:
            deduped.append(e)
    return deduped


def render_significant_events(events):
    """Render the significant events table."""
    if not events:
        st.caption("No significant events (triggers: depth ±100% or price ±5% in 1h, with 30-min reversal filter)")
        return

    st.markdown(f"**Significant events ({len(events)} detected):**")
    st.caption("Triggers: bid/ask depth ±100% at 10% level in 1h OR price ±5% in 1h · Rate of change filter: ignores depth changes that reverse within 30 min")

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


def render_depth_chart(conn, level_idx, side, hours=24):
    """Aggregate depth chart across all exchanges for a given level and side."""
    cutoff = int(time.time()) - hours * 3600

    all_stats = {}
    for exch in ALL_EXCHANGES:
        rows = get_stats_history(conn, exch, SYMBOL, from_ts=cutoff)
        if rows:
            all_stats[exch] = rows

    if not all_stats:
        st.info("Not enough data for depth chart")
        return

    depth_index = _build_depth_index(all_stats, level_idx)

    all_ts = sorted({r[0] for rows in all_stats.values() for r in rows})

    if len(all_ts) < SMOOTH_WINDOW:
        st.info("Not enough data for smoothed depth chart")
        return

    field = 0 if side == "bid" else 1  # index into (bid, ask) tuple
    raw = []
    ts_valid = []
    for ts in all_ts:
        bid, ask = _agg_depth_at_fast(depth_index, ts)
        val = bid if side == "bid" else ask
        if val > 0:
            raw.append(val)
            ts_valid.append(ts)

    if len(raw) < SMOOTH_WINDOW:
        return

    smoothed = smooth(raw, SMOOTH_WINDOW)
    timestamps = ts_valid[SMOOTH_WINDOW - 1:]
    timestamps, smoothed = downsample(timestamps, smoothed)

    side_label = "Bid" if side == "bid" else "Ask"
    level_label = f"{side_label} depth {DEPTH_LEVELS[level_idx]} pct"
    color = "#4caf50" if side == "bid" else "#f44336"
    st.markdown(f"**{side_label} @ {DEPTH_LEVELS[level_idx]}% (15m avg, all exchanges)**")
    df = pd.DataFrame({"time": [ts_to_datetime(t) for t in timestamps], level_label: smoothed})
    st.altair_chart(make_chart(df, level_label, color=color, height=250), use_container_width=True)



# ── Process Control ──────────────────────────────────────────────────────────

def _find_process(script_name):
    """Find a running python process by script name. Returns (pid, uptime) or None."""
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid,etime,command"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.strip().split("\n"):
            if script_name in line and "python" in line.lower() and "grep" not in line and "ps -eo" not in line:
                parts = line.split()
                pid = int(parts[0])
                uptime = parts[1]
                return pid, uptime
    except Exception:
        pass
    return None


def _start_process(script_name):
    """Start a collector script in the background."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(script_dir, f"{script_name.replace('.py', '')}.log")
    with open(log_file, "a") as log:
        proc = subprocess.Popen(
            ["python3", os.path.join(script_dir, script_name)],
            stdout=log, stderr=log,
            cwd=script_dir,
            start_new_session=True  # survives parent exit
        )
    return proc.pid


def _stop_process(script_name):
    """Stop all processes matching script name."""
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid,command"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.strip().split("\n"):
            if script_name in line and "python" in line.lower() and "grep" not in line:
                pid = int(line.split()[0])
                os.kill(pid, signal.SIGTERM)
        return True
    except Exception:
        return False


def _render_control_panel():
    """Status table showing collector health + data freshness."""
    conn = get_conn()
    collector_proc = _find_process("collector.py")
    trades_proc = _find_process("trades_collector.py")
    now = time.time()

    rows = []
    # Collector row
    if collector_proc:
        # Get latest data timestamp
        latest = conn.execute("SELECT MAX(timestamp) FROM candles WHERE symbol=?", (SYMBOL,)).fetchone()[0]
        data_age = f"{int((now - latest) / 60)}m ago" if latest else "no data"
        rows.append({"Process": "collector.py", "Status": "Running", "Uptime": collector_proc[1],
                      "Data": f"REST (VD, stats, candles)", "Last data": data_age, "Backfills": "Yes"})
    else:
        rows.append({"Process": "collector.py", "Status": "Stopped", "Uptime": "—",
                      "Data": "REST (VD, stats, candles)", "Last data": "—", "Backfills": "Yes"})

    # Trades collector row
    if trades_proc:
        latest = conn.execute("SELECT MAX(timestamp) FROM trades WHERE symbol=?", (SYMBOL,)).fetchone()[0]
        data_age = f"{int((now - latest) / 60)}m ago" if latest else "no data"
        rows.append({"Process": "trades_collector.py", "Status": "Running", "Uptime": trades_proc[1],
                      "Data": "WS (trades, depth events)", "Last data": data_age, "Backfills": "No"})
    else:
        rows.append({"Process": "trades_collector.py", "Status": "Stopped", "Uptime": "—",
                      "Data": "WS (trades, depth events)", "Last data": "—", "Backfills": "No"})

    df = pd.DataFrame(rows)

    # Color the status
    def color_status(val):
        if val == "Running":
            return "color: #4caf50"
        return "color: #f44336"

    st.dataframe(
        df.style.map(color_status, subset=["Status"]),
        use_container_width=True, hide_index=True, height=110
    )

    # Start buttons if something is stopped
    if not collector_proc or not trades_proc:
        cols = st.columns(4)
        if not collector_proc:
            if cols[0].button("Start collector"):
                _start_process("collector.py")
                time.sleep(1)
                st.rerun()
        if not trades_proc:
            if cols[1].button("Start trades"):
                _start_process("trades_collector.py")
                time.sleep(1)
                st.rerun()


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

    # ── Control Panel ──
    _render_control_panel()

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

    st.divider()

    # ── 2. LIMIT ORDER ACTIVITY ──
    st.markdown("## 2. Limit Order Activity (Depth Changes)")
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
            render_depth_chart(conn, level_choice, "bid", hours=hours)
        with col2:
            render_depth_chart(conn, level_choice, "ask", hours=hours)

    # Auto-refresh with countdown
    st.divider()
    st.caption(f"Last update: {ts_to_label(stats_rows[-1][0])} UTC")
    countdown = st.empty()
    for remaining in range(60, 0, -1):
        countdown.caption(f"↻ Refreshing in {remaining}s")
        time.sleep(1)
    st.rerun()


main()
