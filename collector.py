"""
Collector: polls MMT every 60 seconds for VD, stats, and candles.
Covers all spot exchanges. On startup, checks for gaps and backfills automatically.
"""
import time
import traceback
from config import EXCHANGES, SYMBOL
from mmt_client import MMTClient
from db import get_conn, insert_vd, insert_stats, insert_candles

POLL_INTERVAL = 60  # seconds
CHUNK_SECONDS = 6 * 3600  # 6-hour chunks for backfill


def get_last_timestamp(conn, exchange):
    """Get the most recent timestamp across all tables for an exchange."""
    latest = 0
    for table in ["vd", "stats", "candles"]:
        row = conn.execute(
            f"SELECT MAX(timestamp) FROM {table} WHERE exchange=? AND symbol=?",
            (exchange, SYMBOL)
        ).fetchone()
        if row[0] and row[0] > latest:
            latest = row[0]
    return latest


def _backfill_range(client, conn, exchange, start_ts, end_ts):
    """Backfill VD, stats, candles for a time range."""
    chunk_start = start_ts
    chunks_done = 0
    while chunk_start < end_ts:
        chunk_end = min(chunk_start + CHUNK_SECONDS, end_ts)
        try:
            vd = client.get_vd(exchange, SYMBOL, tf="1m", from_ts=chunk_start, to_ts=chunk_end)
            if vd.get("data"):
                insert_vd(conn, exchange, SYMBOL, vd["data"])

            stats = client.get_stats(exchange, SYMBOL, tf="1m", from_ts=chunk_start, to_ts=chunk_end)
            if stats.get("data"):
                insert_stats(conn, exchange, SYMBOL, stats["data"])

            candles = client.get_candles(exchange, SYMBOL, tf="1m", from_ts=chunk_start, to_ts=chunk_end)
            if candles.get("data"):
                insert_candles(conn, exchange, SYMBOL, candles["data"])

            chunks_done += 1
        except Exception as e:
            print(f"    Error at {chunk_start}: {e}")
            time.sleep(5)

        time.sleep(2)  # rate limit
        chunk_start = chunk_end
    return chunks_done


def fill_gap(client, conn, exchange):
    """Check for trailing + internal gaps and backfill all."""
    last_ts = get_last_timestamp(conn, exchange)
    now = int(time.time())

    if last_ts == 0:
        print(f"  [{exchange}] No existing data — run backfill.py first")
        return

    total_chunks = 0

    # 1) Find internal gaps > 5 min in candles
    rows = conn.execute(
        "SELECT timestamp FROM candles WHERE exchange=? AND symbol=? ORDER BY timestamp",
        (exchange, SYMBOL)
    ).fetchall()
    ts_list = [r[0] for r in rows]
    internal_gaps = []
    for i in range(1, len(ts_list)):
        gap = ts_list[i] - ts_list[i-1]
        if gap > 300:  # > 5 min
            internal_gaps.append((ts_list[i-1], ts_list[i]))

    if internal_gaps:
        print(f"  [{exchange}] Found {len(internal_gaps)} internal gaps — backfilling...")
        for gap_start, gap_end in internal_gaps:
            gap_min = (gap_end - gap_start) / 60
            print(f"    Gap: {time.strftime('%m/%d %H:%M', time.gmtime(gap_start))} → "
                  f"{time.strftime('%m/%d %H:%M', time.gmtime(gap_end))} ({gap_min:.0f}m)")
            total_chunks += _backfill_range(client, conn, exchange, gap_start, gap_end)

    # 2) Trailing gap (last record to now)
    gap_seconds = now - last_ts
    if gap_seconds > 180:  # > 3 min
        print(f"  [{exchange}] Trailing gap: {gap_seconds/60:.0f}m — backfilling...")
        total_chunks += _backfill_range(client, conn, exchange, last_ts, now)

    if total_chunks == 0:
        print(f"  [{exchange}] No gaps")
    else:
        print(f"  [{exchange}] Gap fill complete: {total_chunks} chunks")


def collect_once(client, conn, exchange):
    now = int(time.time())
    from_ts = now - 120

    vd_resp = client.get_vd(exchange, SYMBOL, tf="1m", from_ts=from_ts, to_ts=now)
    if vd_resp.get("data"):
        insert_vd(conn, exchange, SYMBOL, vd_resp["data"])

    stats_resp = client.get_stats(exchange, SYMBOL, tf="1m", from_ts=from_ts, to_ts=now)
    if stats_resp.get("data"):
        insert_stats(conn, exchange, SYMBOL, stats_resp["data"])

    candles_resp = client.get_candles(exchange, SYMBOL, tf="1m", from_ts=from_ts, to_ts=now)
    if candles_resp.get("data"):
        insert_candles(conn, exchange, SYMBOL, candles_resp["data"])

    return len(vd_resp.get("data", [])), len(stats_resp.get("data", []))


def main():
    client = MMTClient()
    conn = get_conn()
    exchanges = list(EXCHANGES.keys())

    print(f"Collector started: {len(exchanges)} exchanges × {SYMBOL}")
    print(f"Exchanges: {', '.join(exchanges)}")
    print(f"Rate budget: {len(exchanges) * 3} weight/min (limit: 100)")

    # Fill gaps for all exchanges
    for exch in exchanges:
        fill_gap(client, conn, exch)

    print(f"\nPolling every {POLL_INTERVAL}s")
    last_poll = time.time()
    while True:
        now = time.time()
        elapsed = now - last_poll

        # Detect sleep/wake: if >5 min since last poll, laptop was asleep
        if elapsed > 300:
            print(f"[{time.strftime('%H:%M:%S')}] Wake detected ({elapsed/60:.0f}m gap) — backfilling...")
            for exch in exchanges:
                try:
                    fill_gap(client, conn, exch)
                except Exception as e:
                    print(f"[{time.strftime('%H:%M:%S')}] {exch} backfill error: {e}")

        last_poll = time.time()
        parts = []
        for exch in exchanges:
            try:
                vd_n, stats_n = collect_once(client, conn, exch)
                parts.append(f"{exch}:{vd_n}vd")
            except Exception as e:
                parts.append(f"{exch}:ERR")
                print(f"[{time.strftime('%H:%M:%S')}] {exch} error: {e}")
        print(f"[{time.strftime('%H:%M:%S')}] {' | '.join(parts)}")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
