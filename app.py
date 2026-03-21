"""
Flow Monitor Dashboard
Tracks market buying (CVD) and limit order activity (depth) for FLOW/USD on Binance spot.
"""
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import json
import time
from config import EXCHANGE, SYMBOL, TICK_SIZE, DEPTH_LEVELS, HISTORY_DAYS
from db import get_conn, get_vd_history, get_stats_history, get_candle_history

st.set_page_config(page_title="FLOW Monitor", layout="wide")

# --- Helpers ---

def compute_cvd(vd_rows):
    """Compute CVD from VD close values (cumulative sum of delta)."""
    if not vd_rows:
        return [], []
    timestamps = [r[0] for r in vd_rows]
    deltas = [r[1] for r in vd_rows]
    cvd = np.cumsum(deltas).tolist()
    return timestamps, cvd


def compute_price_series(candle_rows):
    """Extract close prices from candle rows."""
    if not candle_rows:
        return [], []
    timestamps = [r[0] for r in candle_rows]
    prices = [r[4] for r in candle_rows]
    return timestamps, prices


def compute_depth_series(stats_rows):
    """Extract bid/ask depth over time at each level."""
    if not stats_rows:
        return [], {}, {}
    timestamps = [r[0] for r in stats_rows]
    bid_series = {i: [] for i in range(7)}
    ask_series = {i: [] for i in range(7)}
    for row in stats_rows:
        bids = json.loads(row[2])
        asks = json.loads(row[3])
        for i in range(7):
            bid_series[i].append(bids[i] if i < len(bids) else 0)
            ask_series[i].append(asks[i] if i < len(asks) else 0)
    return timestamps, bid_series, ask_series


def valid_level_indices(last_price):
    """Filter out sub-tick depth levels."""
    if not last_price or last_price <= 0:
        return list(range(7))
    indices = []
    for i, pct in enumerate(DEPTH_LEVELS):
        spread = last_price * (pct / 100)
        if spread >= TICK_SIZE:
            indices.append(i)
    return indices


def ts_to_label(ts):
    return time.strftime("%m/%d %H:%M", time.gmtime(ts))


def downsample(timestamps, values, max_points=500):
    """Downsample for chart performance."""
    if len(timestamps) <= max_points:
        return timestamps, values
    step = len(timestamps) // max_points
    return timestamps[::step], values[::step]


# --- Chart renderers (using Streamlit native charts via st.line_chart with dataframes) ---

def render_cvd_vs_price(vd_rows, candle_rows, hours=24):
    """CVD vs Price dual-axis chart."""
    cutoff = int(time.time()) - hours * 3600
    vd_filtered = [(t, v) for t, v in vd_rows if t >= cutoff]
    candle_filtered = [(t, o, h, l, c, vb, vs) for t, o, h, l, c, vb, vs in candle_rows if t >= cutoff]

    if not vd_filtered or not candle_filtered:
        st.warning("Not enough data for CVD vs Price chart")
        return

    ts_cvd, cvd = compute_cvd(vd_filtered)
    ts_price, prices = compute_price_series(candle_filtered)

    ts_cvd, cvd = downsample(ts_cvd, cvd)
    ts_price, prices = downsample(ts_price, prices)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**CVD (Cumulative Volume Delta)**")
        import pandas as pd
        df_cvd = pd.DataFrame({"CVD ($)": cvd}, index=[ts_to_label(t) for t in ts_cvd])
        st.line_chart(df_cvd, height=250)
    with col2:
        st.markdown("**Price**")
        df_price = pd.DataFrame({"Price": prices}, index=[ts_to_label(t) for t in ts_price])
        st.line_chart(df_price, height=250)


def render_depth_chart(stats_rows, level_idx, side, hours=24):
    """Depth over time for a specific level."""
    cutoff = int(time.time()) - hours * 3600
    filtered = [r for r in stats_rows if r[0] >= cutoff]
    if not filtered:
        return

    timestamps = [r[0] for r in filtered]
    field_idx = 2 if side == "bid" else 3
    values = []
    for row in filtered:
        arr = json.loads(row[field_idx])
        values.append(arr[level_idx] if level_idx < len(arr) else 0)

    timestamps, values = downsample(timestamps, values)

    import pandas as pd
    label = f"{'Bid' if side == 'bid' else 'Ask'} Depth @ {DEPTH_LEVELS[level_idx]}%"
    df = pd.DataFrame({label: values}, index=[ts_to_label(t) for t in timestamps])
    st.line_chart(df, height=200)


def render_depth_change_table(stats_rows):
    """Show how bid/ask depth has changed over different time windows."""
    if len(stats_rows) < 2:
        st.warning("Not enough stats data")
        return

    now_row = stats_rows[-1]
    now_bids = json.loads(now_row[2])
    now_asks = json.loads(now_row[3])
    last_price = now_row[1]
    valid = valid_level_indices(last_price)

    # Time windows: 1h, 4h, 12h, 24h
    windows = [(60, "1h"), (240, "4h"), (720, "12h"), (1440, "24h")]
    now_ts = now_row[0]

    # Find reference rows for each window
    refs = {}
    for minutes, label in windows:
        target_ts = now_ts - minutes * 60
        # Find closest row
        closest = min(stats_rows, key=lambda r: abs(r[0] - target_ts))
        if abs(closest[0] - target_ts) < 120:  # within 2 min
            refs[label] = closest

    # Build HTML table
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
            ref_bids = json.loads(refs[label][2])
            ref_asks = json.loads(refs[label][3])
            bid_ref = ref_bids[i]
            ask_ref = ref_asks[i]

            bid_pct_change = ((bid_now - bid_ref) / bid_ref * 100) if bid_ref > 0 else 0
            ask_pct_change = ((ask_now - ask_ref) / ask_ref * 100) if ask_ref > 0 else 0

            bid_color = "#4caf50" if bid_pct_change > 5 else "#f44336" if bid_pct_change < -5 else "#888"
            ask_color = "#4caf50" if ask_pct_change > 5 else "#f44336" if ask_pct_change < -5 else "#888"

            row += f'<td style="color:{bid_color}">{bid_pct_change:+.1f}%</td>'
            row += f'<td style="color:{ask_color}">{ask_pct_change:+.1f}%</td>'
        row += "</tr>"
        rows_html += row

    html = f"""<html><body style="margin:0;background:#0e1117;color:#fafafa;font-family:monospace;font-size:13px">
    <table style="width:100%;border-collapse:collapse;text-align:right">
    <thead style="border-bottom:1px solid #333">{header}</thead>
    <tbody>{rows_html}</tbody>
    </table></body></html>"""

    n_rows = rows_html.count("<tr>")
    components.html(html, height=40 + 30 * n_rows)


def render_cvd_divergence(vd_rows, candle_rows, window=60):
    """
    Detect CVD vs price divergence.
    Compares rate of change of CVD vs rate of change of price over rolling window.
    """
    cutoff = int(time.time()) - 24 * 3600
    vd_filtered = [(t, v) for t, v in vd_rows if t >= cutoff]
    candle_filtered = [(t, o, h, l, c, vb, vs) for t, o, h, l, c, vb, vs in candle_rows if t >= cutoff]

    if len(vd_filtered) < window + 1 or len(candle_filtered) < window + 1:
        st.info("Not enough data for divergence analysis (need 24h+)")
        return

    # Compute CVD
    _, cvd = compute_cvd(vd_filtered)
    ts_price, prices = compute_price_series(candle_filtered)

    # Align lengths
    n = min(len(cvd), len(prices))
    cvd = cvd[:n]
    prices = prices[:n]
    timestamps = ts_price[:n]

    # Rolling rate of change
    cvd_roc = []
    price_roc = []
    roc_ts = []
    for i in range(window, n):
        cvd_change = cvd[i] - cvd[i - window]
        price_start = prices[i - window]
        price_change = ((prices[i] - price_start) / price_start * 100) if price_start > 0 else 0
        # Normalize CVD change to percentage of mean absolute CVD in window
        cvd_abs_mean = np.mean(np.abs(cvd[i-window:i+1]))
        cvd_norm = (cvd_change / cvd_abs_mean * 100) if cvd_abs_mean > 0 else 0

        cvd_roc.append(cvd_norm)
        price_roc.append(price_change)
        roc_ts.append(timestamps[i])

    if not roc_ts:
        return

    # Divergence = CVD direction differs from price direction
    import pandas as pd
    roc_ts_labels = [ts_to_label(t) for t in roc_ts]

    # Simple state classification
    latest_cvd_roc = cvd_roc[-1]
    latest_price_roc = price_roc[-1]

    if latest_cvd_roc > 10 and latest_price_roc < 1:
        state = "CVD rising, price flat → absorption (bids absorbing sells)"
        color = "#4caf50"
    elif latest_cvd_roc < -10 and latest_price_roc > -1:
        state = "CVD falling, price flat → distribution (asks absorbing buys)"
        color = "#f44336"
    elif latest_price_roc > 1 and latest_cvd_roc < -5:
        state = "Price rising, CVD falling → rising on thin air"
        color = "#ff9800"
    elif latest_price_roc < -1 and latest_cvd_roc > 5:
        state = "Price falling, CVD rising → selling on thin air"
        color = "#ff9800"
    else:
        state = "No significant divergence"
        color = "#888"

    st.markdown(f'<span style="color:{color};font-size:14px">**{state}**</span>', unsafe_allow_html=True)
    st.caption(f"60-min CVD RoC: {latest_cvd_roc:+.1f}% | Price RoC: {latest_price_roc:+.2f}%")


# --- Main ---

def main():
    conn = get_conn()

    # Load all data
    vd_rows = get_vd_history(conn, EXCHANGE, SYMBOL)
    stats_rows = get_stats_history(conn, EXCHANGE, SYMBOL)
    candle_rows = get_candle_history(conn, EXCHANGE, SYMBOL)

    st.markdown(f"### FLOW/USD — Binance Spot")

    if not vd_rows or not stats_rows:
        st.error("No data yet. Run backfill.py first, then start collector.py.")
        return

    # Header metrics
    last_stats = stats_rows[-1]
    last_price = last_stats[1]
    last_bids = json.loads(last_stats[2])
    last_asks = json.loads(last_stats[3])

    total_records = len(vd_rows)
    days_of_data = (vd_rows[-1][0] - vd_rows[0][0]) / 86400

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Price", f"${last_price:.4f}")
    col2.metric("Records", f"{total_records:,}")
    col3.metric("Days", f"{days_of_data:.1f}")
    col4.metric("Tick", f"{TICK_SIZE}")

    st.divider()

    # --- MARKET BUYING (CVD) ---
    st.markdown("## Market Activity (CVD)")

    hours = st.selectbox("Time window", [6, 12, 24, 48, 72, 168], index=2, format_func=lambda h: f"{h}h" if h < 48 else f"{h//24}d")

    render_cvd_vs_price(vd_rows, candle_rows, hours=hours)

    # --- CVD vs Price Divergence ---
    st.markdown("### CVD vs Price Divergence")
    render_cvd_divergence(vd_rows, candle_rows)

    st.divider()

    # --- LIMIT ORDER ACTIVITY (Depth) ---
    st.markdown("## Limit Order Activity (Depth Changes)")
    st.caption("How bid and ask depth has changed over time windows. Green = depth increased >5%, Red = decreased >5%.")

    render_depth_change_table(stats_rows)

    # Depth charts for key levels
    valid = valid_level_indices(last_price)
    level_choice = st.selectbox(
        "Depth chart level",
        valid,
        format_func=lambda i: f"{DEPTH_LEVELS[i]}%"
    )

    if level_choice is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Bid depth @ {DEPTH_LEVELS[level_choice]}%**")
            render_depth_chart(stats_rows, level_choice, "bid", hours=hours)
        with col2:
            st.markdown(f"**Ask depth @ {DEPTH_LEVELS[level_choice]}%**")
            render_depth_chart(stats_rows, level_choice, "ask", hours=hours)

    # Auto-refresh
    st.divider()
    st.caption(f"Last update: {ts_to_label(stats_rows[-1][0])} UTC | Auto-refresh: 60s")
    time.sleep(60)
    st.rerun()


main()
