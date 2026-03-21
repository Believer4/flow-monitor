"""
Flow Monitor Dashboard
Tracks market buying (CVD) and limit order activity (depth) for FLOW/USD on Binance spot.
"""
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import altair as alt
import json
import time
from datetime import datetime, timezone
from config import EXCHANGE, SYMBOL, TICK_SIZE, DEPTH_LEVELS, HISTORY_DAYS
from db import (
    get_conn, get_vd_history, get_stats_history, get_candle_history,
    get_trades_history, get_trade_stats, get_depth_events,
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
    col1.metric("CVD (coins)", f"{current_cvd_coins:+,.0f} FLOW")
    col2.metric("CVD (USD)", f"${current_cvd_usd:+,.0f}")

    # Downsample
    ts_price_ds, prices_ds = downsample(ts_price, prices)
    ts_cvd_ds, cvd_coins_ds = downsample(ts_cvd, cvd_coins)

    buy_vols = [r[5] for r in candle_filtered]
    sell_vols = [r[6] for r in candle_filtered]
    ts_vol = [r[0] for r in candle_filtered]
    ts_vol_ds, buy_ds = downsample(ts_vol, buy_vols)
    _, sell_ds = downsample(ts_vol, sell_vols)

    # Price chart with regime boundary markers
    st.markdown("**Price**")
    df_price = pd.DataFrame({"time": [ts_to_datetime(t) for t in ts_price_ds], "Price": prices_ds})
    price_line = make_chart(df_price, "Price", color="#4fc3f7")

    # Detect regimes and add boundary dots
    regimes = detect_regimes(candle_filtered)
    if regimes:
        boundary_times = []
        boundary_prices = []
        boundary_dirs = []
        for start_idx, end_idx, direction in regimes:
            # Mark regime start
            boundary_times.append(ts_to_datetime(candle_filtered[start_idx][0]))
            boundary_prices.append(candle_filtered[start_idx][4])
            boundary_dirs.append(direction)
            # Mark regime end
            boundary_times.append(ts_to_datetime(candle_filtered[end_idx][0]))
            boundary_prices.append(candle_filtered[end_idx][4])
            boundary_dirs.append(direction)

        df_dots = pd.DataFrame({"time": boundary_times, "Price": boundary_prices, "dir": boundary_dirs})
        dots = (
            alt.Chart(df_dots)
            .mark_circle(size=50)
            .encode(
                x="time:T",
                y=alt.Y("Price:Q", scale=alt.Scale(zero=False)),
                color=alt.Color("dir:N",
                    scale=alt.Scale(domain=["up", "down"], range=["#4caf50", "#f44336"]),
                    legend=alt.Legend(title="Regime", orient="top")),
                tooltip=[alt.Tooltip("time:T", format="%m/%d %H:%M"), alt.Tooltip("Price:Q", format=".4f"), "dir:N"],
            )
        )
        st.altair_chart(alt.layer(price_line, dots).properties(height=200), use_container_width=True)
    else:
        st.altair_chart(price_line, use_container_width=True)

    # CVD chart
    st.markdown("**CVD (net coins bought)**")
    df_cvd = pd.DataFrame({"time": [ts_to_datetime(t) for t in ts_cvd_ds], "CVD": cvd_coins_ds})
    # Color based on direction
    cvd_color = "#4caf50" if current_cvd_coins >= 0 else "#f44336"
    st.altair_chart(make_chart(df_cvd, "CVD", color=cvd_color), use_container_width=True)

    # Volume bars — aggregate into hourly buckets so bars are visible
    st.markdown("**Buy / Sell Volume ($) — hourly**")
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


def detect_regimes(candle_rows, threshold_pct=1.0, min_duration_min=5):
    """Segment price action into regimes (directional moves).
    A new regime starts when price reverses >threshold_pct from local extreme."""
    if len(candle_rows) < 2:
        return []

    regimes = []
    regime_start_idx = 0
    direction = None  # "up" or "down"
    local_high = candle_rows[0][4]  # close price
    local_low = candle_rows[0][4]
    high_idx = 0
    low_idx = 0

    for i in range(1, len(candle_rows)):
        price = candle_rows[i][4]

        if price > local_high:
            local_high = price
            high_idx = i
        if price < local_low:
            local_low = price
            low_idx = i

        # Check for reversal from high (was going up, now reversing down)
        if local_high > 0:
            drop_from_high = (local_high - price) / local_high * 100
            if drop_from_high >= threshold_pct and direction != "down":
                # End previous regime at the high point
                if direction is not None:
                    end_idx = high_idx
                    duration_min = (candle_rows[end_idx][0] - candle_rows[regime_start_idx][0]) / 60
                    if duration_min >= min_duration_min:
                        regimes.append((regime_start_idx, end_idx, direction))
                regime_start_idx = high_idx
                direction = "down"
                local_low = price
                low_idx = i

        # Check for reversal from low (was going down, now reversing up)
        if local_low > 0:
            rise_from_low = (price - local_low) / local_low * 100
            if rise_from_low >= threshold_pct and direction != "up":
                if direction is not None:
                    end_idx = low_idx
                    duration_min = (candle_rows[end_idx][0] - candle_rows[regime_start_idx][0]) / 60
                    if duration_min >= min_duration_min:
                        regimes.append((regime_start_idx, end_idx, direction))
                regime_start_idx = low_idx
                direction = "up"
                local_high = price
                high_idx = i

    # Close the last regime
    if direction is not None:
        end_idx = len(candle_rows) - 1
        duration_min = (candle_rows[end_idx][0] - candle_rows[regime_start_idx][0]) / 60
        if duration_min >= min_duration_min:
            regimes.append((regime_start_idx, end_idx, direction))

    return regimes


def render_regime_analysis(candle_rows, stats_rows, hours=24):
    """Show what changed in the book during each price regime."""
    cutoff = int(time.time()) - hours * 3600
    candle_filtered = [r for r in candle_rows if r[0] >= cutoff]
    stats_filtered = [r for r in stats_rows if r[0] >= cutoff]

    if len(candle_filtered) < 20 or len(stats_filtered) < 20:
        return

    regimes = detect_regimes(candle_filtered)
    if not regimes:
        st.caption("No significant price moves detected in this window")
        return

    last_price = candle_filtered[-1][4]
    valid = valid_level_indices(last_price)

    st.markdown("**Price regimes → book changes:**")

    # Build stats lookup by timestamp for depth
    stats_by_ts = {r[0]: r for r in stats_filtered}
    stats_ts_list = [r[0] for r in stats_filtered]

    def get_depth_at(target_ts, window=15):
        """Get 15-min smoothed depth around a timestamp."""
        start = target_ts - window * 60
        nearby = [r for r in stats_filtered if start <= r[0] <= target_ts]
        if not nearby:
            # Fall back to closest available
            nearby = [r for r in stats_filtered if abs(r[0] - target_ts) < 30 * 60]
            if not nearby:
                return None, None
        n = len(nearby)
        bid_avg = [0.0] * 7
        ask_avg = [0.0] * 7
        for row in nearby:
            bids = json.loads(row[2])
            asks = json.loads(row[3])
            for j in range(min(7, len(bids))):
                bid_avg[j] += bids[j]
                ask_avg[j] += asks[j]
        return [v / n for v in bid_avg], [v / n for v in ask_avg]

    # Build table — most recent regime first
    header = "<tr><th>Period</th><th>Dur</th><th>Price</th>"
    for i in valid:
        header += f"<th>Bid {DEPTH_LEVELS[i]}%</th><th>Ask {DEPTH_LEVELS[i]}%</th>"
    header += "<th>Net Vol</th><th>What happened</th></tr>"

    rows_html = ""
    for start_idx, end_idx, direction in reversed(regimes):
        start_ts = candle_filtered[start_idx][0]
        end_ts = candle_filtered[end_idx][0]
        start_price = candle_filtered[start_idx][4]
        end_price = candle_filtered[end_idx][4]
        price_chg = ((end_price - start_price) / start_price * 100) if start_price > 0 else 0
        duration_min = (end_ts - start_ts) / 60

        # Depth at regime boundaries
        bid_start, ask_start = get_depth_at(start_ts)
        bid_end, ask_end = get_depth_at(end_ts)

        if bid_start is None or bid_end is None:
            continue

        # Net volume during this regime
        regime_stats = [r for r in stats_filtered if start_ts <= r[0] <= end_ts]
        buy_vol = sum(r[5] for r in regime_stats)
        sell_vol = sum(r[6] for r in regime_stats)
        net_vol = buy_vol - sell_vol

        # Direction styling
        dir_icon = "+" if direction == "up" else "-"
        row_bg = "rgba(76,175,80,0.08)" if direction == "up" else "rgba(244,67,54,0.08)"
        price_color = "#4caf50" if price_chg >= 0 else "#f44336"

        # Format duration
        if duration_min >= 60:
            dur_str = f"{duration_min / 60:.1f}h"
        else:
            dur_str = f"{duration_min:.0f}m"

        time_str = f"{time.strftime('%m/%d %H:%M', time.gmtime(start_ts))}→{time.strftime('%H:%M', time.gmtime(end_ts))}"

        row = f'<tr style="background:{row_bg}">'
        row += f'<td style="text-align:left;white-space:nowrap">{time_str}</td>'
        row += f'<td>{dur_str}</td>'
        row += f'<td style="color:{price_color};font-weight:bold">{price_chg:+.1f}%</td>'

        # Per-level depth changes
        interp_parts = []
        for i in valid:
            bid_pct = ((bid_end[i] - bid_start[i]) / bid_start[i] * 100) if bid_start[i] > 0 else 0
            ask_pct = ((ask_end[i] - ask_start[i]) / ask_start[i] * 100) if ask_start[i] > 0 else 0
            bc = "#4caf50" if bid_pct > 5 else "#f44336" if bid_pct < -5 else "#888"
            ac = "#4caf50" if ask_pct > 5 else "#f44336" if ask_pct < -5 else "#888"
            row += f'<td style="color:{bc}">{bid_pct:+.0f}%</td>'
            row += f'<td style="color:{ac}">{ask_pct:+.0f}%</td>'

            # Build interpretation from most significant level
            if i == valid[0]:  # Use tightest valid level for interpretation
                if bid_pct < -10:
                    interp_parts.append("bids down")
                elif bid_pct > 10:
                    interp_parts.append("bids up")
                if ask_pct < -10:
                    interp_parts.append("asks down")
                elif ask_pct > 10:
                    interp_parts.append("asks up")

        net_color = "#4caf50" if net_vol > 0 else "#f44336"
        row += f'<td style="color:{net_color}">${net_vol:+,.0f}</td>'

        # Interpretation
        if not interp_parts:
            interp_text = "book stable"
        else:
            interp_text = ", ".join(interp_parts)
            # Add context based on price direction + book changes
            if direction == "down" and "bids down" in interp_parts:
                interp_text += " — bids absorbed/filled"
            elif direction == "down" and "bids up" in interp_parts:
                interp_text += " — buying the dip"
            elif direction == "up" and "asks down" in interp_parts:
                interp_text += " — asks lifted/filled"
            elif direction == "up" and "asks up" in interp_parts:
                interp_text += " — selling into strength"

        row += f'<td style="text-align:left;font-size:11px">{interp_text}</td>'
        row += "</tr>"
        rows_html += row

    html = f"""<html><body style="margin:0;background:#0e1117;color:#fafafa;font-family:monospace;font-size:13px">
    <table style="width:100%;border-collapse:collapse;text-align:right">
    <thead style="border-bottom:1px solid #333">{header}</thead>
    <tbody>{rows_html}</tbody>
    </table></body></html>"""
    n_rows = rows_html.count("<tr>")
    components.html(html, height=50 + 36 * n_rows)


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


def render_depth_change_table(stats_rows):
    if len(stats_rows) < 2:
        st.warning("Not enough stats data")
        return

    last_price = stats_rows[-1][1]
    valid = valid_level_indices(last_price)
    now_ts = stats_rows[-1][0]
    now_bids, now_asks = smooth_depth(stats_rows)

    windows = [(60, "1h"), (240, "4h"), (720, "12h"), (1440, "24h")]
    refs = {}
    for minutes, label in windows:
        target_ts = now_ts - minutes * 60
        bid_ref, ask_ref = smooth_depth_at_time(stats_rows, target_ts)
        if bid_ref is not None:
            refs[label] = (bid_ref, ask_ref)

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

    label = f"{'Bid' if side == 'bid' else 'Ask'} @ {DEPTH_LEVELS[level_idx]}%"
    color = "#4caf50" if side == "bid" else "#f44336"
    df = pd.DataFrame({"time": [ts_to_datetime(t) for t in timestamps], label: smoothed})
    st.altair_chart(make_chart(df, label, color=color, height=200), use_container_width=True)


# ── Depth Events (WS fill/pull tracking) ───────────────────────────────────

def render_depth_events(conn, hours=24):
    """Show recent order book events: fills, pulls, new orders from WS depth tracking."""
    cutoff = int(time.time()) - hours * 3600
    events = get_depth_events(conn, EXCHANGE, SYMBOL, from_ts=cutoff, min_usd=500)

    if not events:
        st.caption("No depth events yet. Start trades_collector.py for live order book tracking.")
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

    html = f"""<html><body style="margin:0;background:#0e1117;color:#fafafa;font-family:monospace;font-size:12px">
    <table style="width:100%;border-collapse:collapse;text-align:right">
    <thead style="border-bottom:1px solid #333">{header}</thead>
    <tbody>{rows_html}</tbody>
    </table></body></html>"""

    n_rows = min(len(events), 50)
    components.html(html, height=40 + 24 * n_rows, scrolling=True)

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

    vd_rows = get_vd_history(conn, EXCHANGE, SYMBOL)
    stats_rows = get_stats_history(conn, EXCHANGE, SYMBOL)
    candle_rows = get_candle_history(conn, EXCHANGE, SYMBOL)

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
    st.caption("Price → CVD (net coins) → Buy/Sell Volume")

    render_market_activity(vd_rows, candle_rows, hours=hours)

    # Divergence + book narrative
    st.markdown("### CVD vs Price Divergence")
    render_divergence(vd_rows, candle_rows, hours=hours)
    render_regime_analysis(candle_rows, stats_rows, hours=hours)

    # Depth events (fill/pull from WS)
    render_depth_events(conn, hours=hours)

    st.divider()

    # ── 2. NOTABLE TRADES ──
    st.markdown("## 2. Notable Trades (z-score > 2σ)")
    st.caption("Trades >2σ vs last 12h of trades. Detected via WebSocket. Includes TWAP pattern detection.")

    render_notable_trades(conn, hours=hours)
    render_trade_intensity(conn, hours=hours)

    st.divider()

    # ── 3. LIMIT ORDER ACTIVITY ──
    st.markdown("## 3. Limit Order Activity (Depth Changes)")
    st.caption("15-min smoothed averages. Green = depth increased >5%, Red = decreased >5%.")

    render_depth_change_table(stats_rows)

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
